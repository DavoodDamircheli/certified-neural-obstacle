"""v20 Task 2 gates on the Stefan 512^2 certificates (seeds 0-2, results/v17_split/20261009T19{1750,2819,3854}Z).

    python3 -m certified_obstacle_nn.experiments.v20.nlower_stefan --workers 20

Per seed:
  * re-evaluates the frozen 512^2 partition (load_stored_run of the split run's source; hashes checked) with the NumPy
    outward backend twice: lower_sums off and on. Gate "no change": with the new output off (and also on), every
    stored per-cell array of the split run's cells.npz (numpy_*: rhat, c, sign, R0_term, Rh_term, Sh_term, w2_term, wt2_term, R_width) and the
    re-reduced U_R, U_S, U_I, U_C equal the stored values bit for bit;
  * lower sums L_tau, L_gradvhat (upper_sum.reduce_lower_sums), G_AR (reliability.grad_bound_AR, cross-checked against
    handoff_v19_ellipse/tables/grad_AR_fullprec.csv), N_lo (reliability.solution_scale_lower) with the stored F_dir;
    gate N_lo <= N_lower (192-bit closed-form lower endpoint of N = sqrt(23/12 - e^2/4)); ratios F_dir/N_lo,
    G_dir/N_lo, G_AR/N_lo next to the published F_dir/N, G_dir/N, G_AR/N;
  * 1000 random cells (seed 20261014): 33^2 points per cell (unscrambled Sobol + 4 corners, as v18 D37), screened in
    float64 (tau^2 and (d_x v_hat)^2); the 4 corners and the TOP_K = 4 smallest float values of each term are
    re-evaluated in 192-bit arb (experiments/v18/whole_cell arb networks and exact Stefan data). Violation: the arb
    sample's UPPER endpoint is below the cell's lower endpoint. Also counted: float64 samples below the lower endpoint
    (screening, all 1089 points).
Writes results/v20_nlower_stefan/<timestamp>/ (config.json, summary.csv, cells.csv.gz).
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from flint import arb, ctx  # noqa: E402

from certified_obstacle_nn.certification import reliability as rel  # noqa: E402
from certified_obstacle_nn.certification.enclose import upper_sum as US  # noqa: E402
from certified_obstacle_nn.certification.enclose.engine import StefanAdapter, TripleEncloser  # noqa: E402
from certified_obstacle_nn.experiments import results_io  # noqa: E402
from certified_obstacle_nn.experiments.v14.common import REPO_ROOT, git_commit, git_dirty, hardware, sha256_json  # noqa: E402
from certified_obstacle_nn.experiments.v17.common import load_stored_run  # noqa: E402
from certified_obstacle_nn.experiments.v18 import whole_cell as WC  # noqa: E402

EXPERIMENT = "v20_nlower_stefan"
RUNS = {0: "results/v17_split/20261009T191750Z", 1: "results/v17_split/20261009T192819Z",
        2: "results/v17_split/20261009T193854Z"}
SEED = 20261014
N_CELLS = 1000
TOP_K = 4
PREC_POINT = 192
STORED_KEYS = ("rhat", "c", "sign", "R0_term", "Rh_term", "Sh_term", "w2_term", "wt2_term", "R_width")

_W: dict = {}


def f64_terms(layers, P):
    """tau^2, (d_x v)^2, (g_x)^2 and gamma = g - v at float64 points P (n, 2) (Stefan; screening only)."""
    t, x = P[:, 0], P[:, 1]
    N, dN = WC.mlp_f64(layers["state"], P, (1,))
    sg, _ = WC.mlp_f64(layers["flux"], P, ())
    N, Nx, sg = N[:, 0], dN[1][:, 0], sg[:, 0]
    B, dB = 4 * x * (1 - x), 4 - 8 * x
    h = np.exp(t) - t - 1
    g, gx = -h * (1 - x), h
    v, vx = B * N, dB * N + B * Nx
    gam = g - v
    return {"tau2": sg * sg, "gvh2": np.where(gam > 0, gx * gx, vx * vx)}


def arb_terms(nets, cols):
    """192-bit arb balls of tau^2 and (d_x v_hat)^2 at point balls cols = [t list, x list]."""
    Nv, dN = WC.arb_mlp(nets["state"], cols, (1,))
    Sv, _ = WC.arb_mlp(nets["flux"], cols, ())
    out = []
    for p in range(len(cols[0])):
        t, x = cols[0][p], cols[1][p]
        d = WC.data_arb("stefan", None, t, x)
        B, dB = 4 * x * (1 - x), 4 - 8 * x
        v = B * Nv[0][p]
        vx = dB * Nv[0][p] + B * dN[1][0][p]
        tau = Sv[0][p]
        gam = d["g"] - v
        vx2, gx2 = vx * vx, d["g_x"] * d["g_x"]
        gv = gx2 if gam > 0 else (vx2 if gam < 0 else vx2.union(gx2))
        out.append({"tau2": tau * tau, "gvh2": gv, "amb": not (gam > 0 or gam < 0)})
    return out


def _init(nets_f64):
    torch.set_num_threads(1)
    ctx.prec = PREC_POINT
    _W["f64"] = nets_f64
    _W["arb"] = {k: WC._arbnet(v) for k, v in nets_f64.items()}


def check_cell(task):
    idx, lo, hi, low = task
    ctx.prec = PREC_POINT
    U = WC._unit_points(2)
    P = lo[None, :] + U * (hi - lo)[None, :]
    vals = f64_terms(_W["f64"], P)
    cand = set(range(4))
    for k in vals:
        cand.update(np.argsort(vals[k])[:TOP_K].tolist())
    cand = sorted(cand)
    pts = arb_terms(_W["arb"], WC._balls_from_points(P[cand]))
    rec = {"cell": int(idx), "amb_points": int(sum(p["amb"] for p in pts))}
    for k in ("tau2", "gvh2"):
        L = arb(float(low[k]))
        rec[f"{k}_lo"] = float(low[k])
        rec[f"{k}_f64_min"] = float(vals[k].min())
        rec[f"{k}_f64_below"] = int((vals[k] < low[k]).sum())
        mins = [p[k] for p in pts]
        rec[f"{k}_arb_min_mid"] = min(float(b.mid()) for b in mins)
        rec[f"{k}_violation"] = bool(any(b < L for b in mins))  # every point of the sample ball below the endpoint
    return rec


def grad_ar_table() -> dict:
    df = pd.read_csv(REPO_ROOT / "handoff_v19_ellipse/tables/grad_AR_fullprec.csv", dtype=str, keep_default_na=False)
    return {r["run"]: r for _, r in df.iterrows()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()

    def log(msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    d = results_io.make_run_dir(EXPERIMENT)
    gar = grad_ar_table()
    ref = rel.reference_norm("stefan")
    cf = rel.cF2_upper(1)
    rows, cell_rows = [], []
    for seed in a.seeds:
        split = REPO_ROOT / RUNS[seed]
        scfg = json.loads((split / "config.json").read_text())
        srow = pd.read_csv(split / "summary.csv", dtype=str).set_index("backend").loc["numpy"]
        r = load_stored_run(REPO_ROOT / scfg["source_run"], workers=a.workers)
        if r.checkpoint_sha256 != srow["checkpoint_sha256"] or r.part.hash() != srow["partition_hash"]:
            raise RuntimeError("stored run identity mismatch")
        cz = np.load(split / "cells.npz")
        row = {"seed": seed, "split_run": RUNS[seed], "cells": r.part.n}
        res = {}
        for mode in ("off", "on"):
            enc = TripleEncloser(r.triple, StefanAdapter(r.problem), lower_sums=(mode == "on"))
            tic = time.time()
            us, vol, _ = US.certify_partition(enc, r.part, r.tp0, r.tpT, backend="numpy", workers=a.workers,
                                              model_frozen=r.frozen, constants_validated=True)
            res[mode] = (us, vol, time.time() - tic)
            same_cells = all(np.array_equal(vol[k], cz[f"numpy_{k}"]) for k in STORED_KEYS)
            P = us.parts
            same_U = (P["vol_Rhat"] == float(srow["U_R"]) and P["vol_Shat"] == float(srow["U_S"])
                      and P["init_mismatch_hat"] == float(srow["U_I"]) and us.U_c == float(srow["U_C"]))
            row[f"bitwise_cells_lower_{mode}"] = same_cells
            row[f"bitwise_U_lower_{mode}"] = same_U
            row[f"eval_s_lower_{mode}"] = res[mode][2]
            log(f"seed {seed} lower_sums {mode}: per-cell arrays bitwise {same_cells}, U sums bitwise {same_U}, "
                f"{res[mode][2]:.0f}s")
        us, vol, t_on = res["on"]
        low = US.reduce_lower_sums(r.part, vol, backend_info=US.backend_info("numpy", 64), model_frozen=r.frozen)
        U = {k: float(srow[k]) for k in ("U_R", "U_S", "U_I", "U_C")}
        F_dir = float(srow["flux_bound_direct_split"])
        G_dir = float(srow["grad_bound_direct_split"])
        G_AR = rel.grad_bound_AR(**U, cF2_up=cf["cF2_up"])
        g19 = gar[RUNS[seed]]
        row["G_AR_equals_v19"] = float.fromhex(g19["G_AR_hex"]) == G_AR
        nl = rel.solution_scale_lower(L_tau=low["L_tau"], L_gradvhat=low["L_gradvhat"], F_dir=F_dir, G_AR=G_AR,
                                      upper_status=srow["status"], lower_status=low["lower_status"])
        N_lower = ref["norm_lower"]
        row.update({"status": srow["status"], "lower_status": low["lower_status"], "L_tau": low["L_tau"],
                    "L_gradvhat": low["L_gradvhat"], **nl, "N_lower_closed_form": N_lower,
                    "N_lo_le_N": nl["N_lo"] <= N_lower, "N_lo_over_N": nl["N_lo"] / N_lower,
                    "F_dir": F_dir, "G_dir": G_dir, "G_AR": G_AR,
                    "F_dir_over_N_published": float(srow["flux_bound_direct_split_rel"]),
                    "G_dir_over_N_published": float(srow["grad_bound_direct_split_rel"]),
                    "G_AR_over_N": rel.relative_upper(G_AR, N_lower)})
        for k, b in (("F_dir", F_dir), ("G_dir", G_dir), ("G_AR", G_AR)):
            row[f"{k}_over_N_lo"] = rel.ratio_over_lower(b, nl["N_lo"])
        for k in ("L_tau", "L_gradvhat", "N_lo", "N_lo_tau", "N_lo_grad", "N_lower_closed_form", "G_AR",
                  "F_dir_over_N_lo", "G_AR_over_N_lo"):
            row[k + "_hex"] = float(row[k]).hex() if isinstance(row[k], float) else ""
        log(f"seed {seed}: L_tau={low['L_tau']:.6f} L_gvh={low['L_gradvhat']:.6f} N_lo={nl['N_lo']:.6f} "
            f"(tau {nl['N_lo_tau']:.6f}, grad {nl['N_lo_grad']:.6f}) N={N_lower:.6f} N_lo/N={row['N_lo_over_N']:.4f} "
            f"F/N_lo={row['F_dir_over_N_lo']} vs F/N={row['F_dir_over_N_published']}")
        # 1000 random cells: lower endpoints vs 192-bit arb samples
        rng = np.random.default_rng(SEED + seed)
        idx = np.sort(rng.choice(r.part.n, N_CELLS, replace=False))
        p = r.part
        tasks = [(int(i), np.array([p.t0[i], p.x0[i]]), np.array([p.t1[i], p.x1[i]]),
                  {"tau2": vol["tau2_lo"][i], "gvh2": vol["gvh2_lo"][i]}) for i in idx]
        nets = {"state": WC._layers(r.triple.state_net.net), "flux": WC._layers(r.triple.flux_net.net)}
        tic = time.time()
        with mp.get_context("fork").Pool(a.workers, initializer=_init, initargs=(nets,)) as pool:
            recs = pool.map(check_cell, tasks, chunksize=4)
        for rc in recs:
            rc["seed"] = seed
        cell_rows += recs
        cr = pd.DataFrame(recs)
        row.update({"sample_cells": N_CELLS, "sample_points_per_cell": WC.N_PTS**2,
                    "violations_tau2": int(cr["tau2_violation"].sum()), "violations_gvh2": int(cr["gvh2_violation"].sum()),
                    "f64_points_below_tau2": int(cr["tau2_f64_below"].sum()),
                    "f64_points_below_gvh2": int(cr["gvh2_f64_below"].sum()),
                    "ambiguous_sign_points": int(cr["amb_points"].sum()),
                    "median_lo_over_f64min_tau2": float(np.median(cr["tau2_lo"] / cr["tau2_f64_min"])),
                    "median_lo_over_f64min_gvh2": float(np.median(cr["gvh2_lo"] / cr["gvh2_f64_min"])),
                    "sample_check_s": time.time() - tic})
        log(f"seed {seed}: sample check violations tau2 {row['violations_tau2']} gvh2 {row['violations_gvh2']} "
            f"(f64 below {row['f64_points_below_tau2']}/{row['f64_points_below_gvh2']}) {row['sample_check_s']:.0f}s")
        rows.append(row)
    pd.DataFrame(rows).to_csv(d / "summary.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(d / "cells.csv.gz", index=False)
    config = {"experiment": EXPERIMENT, "runs": RUNS, "seed": SEED, "n_cells": N_CELLS, "top_k": TOP_K,
              "prec_point": PREC_POINT, "points": "33^2 = 4 corners + unscrambled Sobol (v18 whole_cell._unit_points)",
              "reference_norm": ref, "cF2": cf, "doc": __doc__, "git_commit": git_commit(), "git_dirty": git_dirty(),
              "hardware": hardware()}
    config["config_hash"] = sha256_json({k: v for k, v in config.items() if k != "hardware"})
    results_io.write_config(d, config)
    log(f"wrote {d.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
