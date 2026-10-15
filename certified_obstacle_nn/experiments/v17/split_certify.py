"""v17 Stages 2-3 (roadmap R3): re-evaluate a STORED validated run on its frozen partition with the per-term split,
and report every direct bound with its status.

    python3 -m certified_obstacle_nn.experiments.v17.split_certify --run-dir results/v14_stefan_validated/<ts> \
        --backends numpy arb --workers 16 [--q 3]

For each backend: the validated per-term upper sums
  U_R >= ||R_hat||^2, U_S >= ||S_hat||^2, U_I >= ||u0 - v_hat(0)||^2, U_C >= int mu (v-g)_+       (projected)
  U_R0, U_S0, U_init, U_gradw, U_w, U_wt, U_w0, U_wT                                              (unprojected)
(certification/enclose/engine*.py split_terms, reduced in arb by upper_sum.reduce_upper_sums), the aggregate sums
(checked to reproduce the stored v14 aggregates exactly when the stored run has the same backend), the direct
bounds of reliability.direct_projected_bounds (split: F_dir, G_dir primary, theta = 1/2 check, graph, E_dir theta
scan; no-split fallbacks = v16 numbers), the estimator route, and relative bounds against the analytic reference
norm (validated only for Stefan and disk). Per-term sums are also split by cell class (sign of gamma resolved /
unresolved on the cell). Diagnostic float64 integrals of the same densities ON THE SAME PARTITION (q^D Gauss points
per cell) give the slack U / I of every term and class. Writes results/v17_split/<timestamp>/ (config.json,
summary.csv, cells.npz, weights.pt); never touches the source run.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from certified_obstacle_nn.certification import reliability as rel
from certified_obstacle_nn.certification.enclose import boxes as Bx
from certified_obstacle_nn.certification.enclose import upper_sum as US
from certified_obstacle_nn.experiments import results_io
from certified_obstacle_nn.experiments.v14.common import REPO_ROOT, git_commit, git_dirty, hardware, sha256_json
from certified_obstacle_nn.experiments.v14.disk_validated import _peak_rss_gb
from certified_obstacle_nn.experiments.v17 import cell_quadrature as CQ
from certified_obstacle_nn.experiments.v17.common import load_stored_run

EXPERIMENT = "v17_split"
DIM = {"stefan": 1, "disk2d": 2, "ellipse": 2}
ENCLOSURE_ID = "TM1"  # first-order Taylor models, engine.py / engine2d.py (unchanged since v14)


def _arb_sum(area, vals, mask=None):
    if mask is not None:
        area, vals = area[mask], vals[mask]
    return US._up(US._arb_dot_up(area, vals))


def _lohi(part):
    if hasattr(part, "lo"):
        return part.lo, part.hi
    return np.stack([part.t0, part.x0]), np.stack([part.t1, part.x1])


def _tr_lohi(tp):
    if hasattr(tp, "lo"):
        return tp.lo, tp.hi
    return tp.x0[None, :], tp.x1[None, :]


def diagnostics(r, q: int) -> dict:
    kind = "stefan" if r.problem_name == "stefan" else "2d"
    lo, hi = _lohi(r.part)
    tic = time.time()
    cell = CQ.cell_integrals(kind, r.problem, r.triple, lo, hi, q=q)
    l0, h0 = _tr_lohi(r.tp0)
    lT, hT = _tr_lohi(r.tpT)
    cell.update(CQ.trace_integrals(kind, r.problem, r.triple, l0, h0, 0.0, q=max(q, 3)))
    cell.update(CQ.trace_integrals(kind, r.problem, r.triple, lT, hT, 1.0, q=max(q, 3)))
    cell["seconds"] = time.time() - tic
    return cell


def encloser_for(r, enclosure: str):
    """The stored run's encloser (TM1) or a new one with an opt-in v17 enclosure candidate (Stage 4)."""
    if enclosure == ENCLOSURE_ID:
        return r.encloser
    if r.problem_name == "stefan":
        from certified_obstacle_nn.certification.enclose.engine import TripleEncloser

        return TripleEncloser(r.triple, r.encloser.ad, enclosure=enclosure)
    from certified_obstacle_nn.certification.enclose.engine2d import TripleEncloser2D

    return TripleEncloser2D(r.triple, r.encloser.ad, enclosure=enclosure)


def run(run_dir, backends, workers, q, prec=64, log=print, enclosure=ENCLOSURE_ID, r=None, diag=None,
        partition_info=None, volumes=None):
    """r: a StoredRun (default: loaded from run_dir); its part/tp0/tpT may be replaced by a frozen pilot partition
    (partition_info then describes it). diag: reuse the float64 cell quadrature of the same partition."""
    t_start = time.time()
    r = load_stored_run(run_dir, workers=workers) if r is None else r
    if not r.frozen:
        raise RuntimeError("model not frozen")
    prob = r.problem
    alpha, A_inf, Lam = float(prob.alpha), float(prob.A_inf), float(prob.Lambda)
    La = Lam / alpha
    kap = float(rel.kappa(prob))
    cf = rel.cF2_upper(DIM[r.problem_name])
    try:
        ref = rel.reference_norm(r.problem_name)
    except KeyError:
        ref = None
    stored = pd.read_csv(r.run_dir / "summary.csv", dtype=str).set_index("backend")
    same_partition = r.part.hash() == r.summary["partition_hash"]
    hashes = {"partition_hash": r.part.hash(), "trace0_hash": r.tp0.hash(), "traceT_hash": r.tpT.hash()}
    enc_obj = encloser_for(r, enclosure)
    log(f"{r.run_dir.name}: {r.problem_name} n={r.part.n} traces {r.tp0.n}/{r.tpT.n} enclosure {enclosure}")
    if diag is None:
        diag = diagnostics(r, q)
        log(f"float64 cell quadrature q={q}: {diag['seconds']:.0f}s")
    sign_ref = None
    rows, cells = [], {}
    for bk in backends:
        tic = time.time()
        if volumes is not None and bk in volumes:
            # chunked evaluation (chunked_certify.py): per-cell densities computed by separate jobs on slices of the
            # same frozen partition; traces and the arb reduction are done here
            vol, t_vol = volumes[bk]
            t1 = time.time()
            d0 = Bx.evaluate_trace(enc_obj, r.tp0, 0.0, bk, prec, workers=workers)
            dT = Bx.evaluate_trace(enc_obj, r.tpT, 1.0, bk, prec, workers=workers)
            t2 = time.time()
            us = US.reduce_upper_sums(r.part, vol, r.tp0, d0, r.tpT, dT, backend_info=US.backend_info(bk, prec),
                                      model_frozen=r.frozen, constants_validated=True)
            us.timings = {"leaf_eval_s": t_vol, "trace_eval_s": t2 - t1, "reduction_s": time.time() - t2,
                          "workers": workers, "s_per_cell_wall": t_vol / max(r.part.n, 1)}
        elif r.problem_name == "stefan":
            us, vol, _ = US.certify_partition(enc_obj, r.part, r.tp0, r.tpT, backend=bk, prec=prec, workers=workers,
                                              model_frozen=r.frozen, constants_validated=True)
        else:
            us, vol, _ = Bx.certify_boxes(enc_obj, r.part, r.tp0, r.tpT, backend=bk, prec=prec, workers=workers,
                                          model_frozen=r.frozen, constants_validated=True)
        t_eval = time.time() - tic + (volumes[bk][1] if volumes is not None and bk in volumes else 0.0)
        P = us.parts
        area = r.part.area()
        sgn = vol["sign"]
        if sign_ref is None:
            sign_ref = sgn
        terms = {"U_R": P["vol_Rhat"], "U_S": P["vol_Shat"], "U_I": P["init_mismatch_hat"], "U_C": us.U_c,
                 "U_R0": P["vol_R0"], "U_S0": P["vol_S_term"], "U_init": P["init_mismatch"], "U_gradw": P["vol_gradw"],
                 "U_w": P["vol_w2"], "U_wt": P["vol_wt2"], "U_w0": P["w0"], "U_wT": P["wT"]}
        row = {"problem": r.problem_name, "source_run": str(r.run_dir.relative_to(REPO_ROOT)), "backend": bk,
               "prec_bits": prec if bk == "arb" else 53, "enclosure": enclosure, "n_cells": us.n_cells,
               "n_trace0": us.n_trace0, "n_traceT": us.n_traceT, "status": us.status,
               "status_reasons": ";".join(us.status_reasons), "U_r": us.U_r, "U_p": us.U_p, "U_c": us.U_c,
               "U_r_hat": us.U_r_hat, **terms}
        # identical aggregates to the stored v14 run (same engine, same partition) -- bitwise
        if bk in stored.index and same_partition and enclosure == ENCLOSURE_ID:
            s = stored.loc[bk]
            row["reproduces_stored_aggregates"] = all(float(s[k]) == row[k] for k in ("U_r", "U_p", "U_c", "U_r_hat"))
        else:
            row["reproduces_stored_aggregates"] = "n/a (other backend, partition or enclosure)"
        split_rhat = La * terms["U_R"] + terms["U_S"] + terms["U_I"]
        row["split_consistency_rel"] = abs(split_rhat - us.U_r_hat) / us.U_r_hat
        # per-class sums (cells where the sign of gamma is resolved vs unresolved), validated reductions
        for cls, m in (("res", sgn != 0), ("unres", sgn == 0)):
            row[f"n_cells_{cls}"] = int(m.sum())
            row[f"vol_{cls}"] = float(area[m].sum())
            for nm, key in (("U_R", "Rh_term"), ("U_S", "Sh_term"), ("U_C", "c")):
                row[f"{nm}_{cls}"] = _arb_sum(area, vol[key], m)
        d = rel.direct_projected_bounds(
            U_R=terms["U_R"], U_S=terms["U_S"], U_I=terms["U_I"], U_C=terms["U_C"], cF2_up=cf["cF2_up"],
            status=us.status, alpha=alpha, A_inf=A_inf, U_r_hat=us.U_r_hat, Lambda=Lam, U_gradw=terms["U_gradw"],
            reference_norm_lower=ref["norm_lower"] if ref else None, reference_status=ref["status"] if ref else None)
        row.update({k: v for k, v in d.items() if k not in row and not isinstance(v, (dict, list))})
        Ceq = rel.C_eq_heat(alpha, A_inf, Lam)
        pe = rel.projected_estimator_bounds(U_r_hat=us.U_r_hat, U_c=us.U_c, alpha=alpha, A_inf=A_inf, C_eq=Ceq)
        row.update(pe)
        row["C_eq_heat"] = Ceq
        row["E_U_weighted_Ceq_heat"] = rel.weighted_bound_heat(U_r=us.U_r, U_p=us.U_p, U_c=us.U_c, C_eq=Ceq)
        if ref:
            row["grad_bound_projected_rel"] = rel.relative_upper(pe["grad_bound_projected"], ref["norm_lower"])
            row["reference_N2_formula"] = ref["N2_formula"]
        else:
            row["ratio_status"] = "diagnostic_float64"
        # budget shares of the direct bound components (Stage 3), against N^2 (milestone 2) and N^2/4 (milestone 3)
        N2 = ref["norm_lower"] ** 2 if ref else None
        for nm, val in (("U_S", terms["U_S"]), ("U_I", terms["U_I"]), ("2U_C", 2 * terms["U_C"]), ("L", d["L"])):
            row[f"share_F2_{nm}"] = val / (d["D"] + d["L"])
            if N2:
                row[f"budget_m2_{nm}"] = val / N2
                row[f"budget_m3_{nm}"] = val / (N2 / 4)
        row.update({"t_leaf_eval_s": us.timings["leaf_eval_s"], "t_trace_s": us.timings["trace_eval_s"],
                    "t_reduction_s": us.timings["reduction_s"], "t_eval_total_s": t_eval, "workers": workers,
                    "s_per_cell_wall": us.timings["s_per_cell_wall"]})
        rows.append(row)
        for k in ("rhat", "c", "sign", "R0_term", "Rh_term", "Sh_term", "w2_term", "wt2_term", "R_width"):
            cells[f"{bk}_{k}"] = vol[k]
        log(f"  {bk}: {us.status} U_R={terms['U_R']:.6e} U_S={terms['U_S']:.6e} U_I={terms['U_I']:.3e} "
            f"U_C={terms['U_C']:.6e} F_dir={d['flux_bound_direct_split']:.5f} G_dir={d['grad_bound_direct_split']:.5f} "
            f"E_dir={d['E_dir_split']:.5f} (no-split F/G {d['flux_bound_direct']:.4f}/{d['grad_bound_direct']:.4f}) "
            f"t={t_eval:.0f}s")
    # diagnostics, same for every backend: float64 integrals and slack
    sgn = sign_ref
    I = {"U_R": diag["Rh2"], "U_S": diag["Sh2"], "U_C": diag["C"], "U_R0": diag["R02"], "U_S0": diag["S02"]}
    dg = {"I_R": float(diag["Rh2"].sum()), "I_S": float(diag["Sh2"].sum()), "I_I": float(diag["I0"].sum()),
          "I_C": float(diag["C"].sum()), "I_R0": float(diag["R02"].sum()), "I_S0": float(diag["S02"].sum()),
          "true_grad_err_vhat": math.sqrt(diag["Eg"].sum()), "true_flux_err": math.sqrt(diag["Ef"].sum()),
          "true_terminal_err_vhat": math.sqrt(diag["ZT"].sum()),
          "true_flux_plus_terminal": math.sqrt(diag["Ef"].sum() + diag["ZT"].sum()),
          "norm_grad_u_quad": math.sqrt(diag["Ng"].sum()), "diag_q": q, "t_diag_cells_s": diag["seconds"],
          "diag_status": "diagnostic_float64"}
    for cls, m in (("res", sgn != 0), ("unres", sgn == 0)):
        for nm in ("U_R", "U_S", "U_C"):
            dg[f"I_{nm[2:]}_{cls}"] = float(I[nm][m].sum())
    # sampled-sup sums: sum |K| max_{points of K} density (diagnostic; a LOWER bound of sum |K| sup_K density).
    # slack U/I = (U / sampled-sup sum) [enclosure overestimate, upper estimate] x (sampled-sup sum / I) [majorant]
    area = r.part.area()
    for nm, k in (("U_R", "Rh2"), ("U_S", "Sh2"), ("U_C", "C")):
        dg[f"Ssup_{nm[2:]}"] = float((area * diag["sup_" + k]).sum())
        for cls, m in (("res", sgn != 0), ("unres", sgn == 0)):
            dg[f"Ssup_{nm[2:]}_{cls}"] = float((area[m] * diag["sup_" + k][m]).sum())
    # the D, L split evaluated on the float64 integrals (diagnostic counterparts of the direct bounds)
    Dd = dg["I_S"] + dg["I_I"] + 2 * dg["I_C"]
    Ld = cf["cF2_up"] * dg["I_R"] / alpha
    dg.update({"diag_F_dir": math.sqrt(Dd + Ld), "diag_G_dir": math.sqrt(Ld) + math.sqrt(Dd + Ld)})
    stored_row = stored.iloc[0]
    for k in ("E_true_v", "E_true_vhat", "E_true_vhat_grad", "E_true_flux", "norm_grad_u"):
        if k in stored_row.index and isinstance(stored_row[k], str):
            dg[f"stored_{k}"] = float(stored_row[k])
    for row in rows:
        row.update(dg)
        for nm, inm in (("U_R", "I_R"), ("U_S", "I_S"), ("U_I", "I_I"), ("U_C", "I_C"), ("U_R0", "I_R0"),
                        ("U_S0", "I_S0")):
            row[f"slack_{nm}"] = row[nm] / dg[inm] if dg[inm] > 0 else float("nan")
        for cls in ("res", "unres"):
            for nm in ("U_R", "U_S", "U_C"):
                iv = dg[f"I_{nm[2:]}_{cls}"]
                row[f"slack_{nm}_{cls}"] = row[f"{nm}_{cls}"] / iv if iv > 0 else float("nan")
        for nm in ("U_R", "U_S", "U_C"):
            ss, iv = dg[f"Ssup_{nm[2:]}"], dg[f"I_{nm[2:]}"]
            row[f"enclosure_factor_{nm}"] = row[nm] / ss if ss > 0 else float("nan")
            row[f"majorant_factor_{nm}"] = ss / iv if iv > 0 else float("nan")
        row.update({"checkpoint_sha256": r.checkpoint_sha256, **hashes,
                    "alpha": alpha, "Lambda": Lam, "kappa": kap, "cF2_up": cf["cF2_up"],
                    "reference_norm_lower": ref["norm_lower"] if ref else "",
                    "git_commit": git_commit(), "git_dirty": git_dirty(),
                    "t_total_s": time.time() - t_start, "peak_rss_gb": _peak_rss_gb()})
    config = {"experiment": EXPERIMENT, "source_run": str(r.run_dir.relative_to(REPO_ROOT)),
              "source_config_hash": r.config.get("config_hash"), "problem": r.problem_name,
              "checkpoint": str(r.checkpoint.relative_to(REPO_ROOT)), "checkpoint_sha256": r.checkpoint_sha256,
              "partition": partition_info or {**hashes, "n_cells": int(r.part.n),
                                              "source": "uniform (reconstructed)" if r.problem_name == "stefan"
                                              else "lo/hi of the source cells.npz; traces reconstructed by "
                                                   "boxes.adaptive_traces"},
              "backends": list(backends), "arb_prec_bits": prec, "workers": workers, "enclosure": enclosure,
              "constants": {"alpha": alpha, "Lambda": Lam, "A_norm_upper": A_inf, "kappa": kap,
                            "cF2": cf, "C_eq_heat": rel.C_eq_heat(alpha, A_inf, Lam),
                            "reference_norm": ref or "none (diagnostic ratios only)"},
              "formulas": rel.direct_projected_bounds.__doc__,
              "diagnostic_quadrature": f"q={q} Gauss points per axis in every cell of the same partition; traces q>=3",
              "git_commit": git_commit(), "hardware": hardware(),
              "statuses": {row["backend"]: row["status"] for row in rows}}
    config["config_hash"] = sha256_json({k: v for k, v in config.items() if k != "hardware"})
    for row in rows:
        row["config_hash"] = config["config_hash"]
    cells.update({f"float64_{k}": diag[k] for k in CQ.VOL})
    cells.update({f"float64_sup_{k}": diag["sup_" + k] for k in CQ.VOL})
    return config, rows, cells, r, diag


def write_run(config, rows, cells, r, experiment=EXPERIMENT):
    d = results_io.make_run_dir(experiment)
    results_io.write_config(d, config)
    pd.DataFrame(rows).to_csv(d / "summary.csv", index=False)
    np.savez_compressed(d / "cells.npz", **cells)
    shutil.copyfile(r.checkpoint, d / "weights.pt")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", nargs="+", required=True)
    ap.add_argument("--backends", nargs="+", default=["numpy"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--q", type=int, default=2)
    ap.add_argument("--prec", type=int, default=64)
    ap.add_argument("--enclosure", default=ENCLOSURE_ID)
    a = ap.parse_args()

    def log(msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    for rd in a.run_dir:
        config, rows, cells, r, _ = run(rd, a.backends, a.workers, a.q, a.prec, log=log, enclosure=a.enclosure)
        d = write_run(config, rows, cells, r)
        log(f"wrote {d.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
