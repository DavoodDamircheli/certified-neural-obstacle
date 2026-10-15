"""v20 Task 4: frozen certification of a trained rising1d triple (protocol results/v20_protocols/rising1d_protocol.json).

    python3 -m certified_obstacle_nn.experiments.v20.certify_rising1d --train-run results/v20_rising1d_train/<ts> \
        --n 1024 --workers 20

The Stefan pipeline with the rising1d adapter: uniform dyadic n x n partition of (0,1)^2 and uniform 4096-cell traces
at t = 0, T (upper_sum.uniform_2d / uniform_1d), TM1 per-cell enclosures (engine.TripleEncloser, mean_value off,
lower_sums on), NumPy outward backend, arb reduction of the per-term upper sums (upper_sum.certify_partition) and of
the new lower sums (upper_sum.reduce_lower_sums). Bounds (192-bit arb, reliability):
  F_dir, G_dir, E_dir           reliability.direct_projected_bounds (Theorem 4.10, as for Stefan)
  G_AR                          reliability.grad_bound_AR (Cor. 4.12, v19)
  N_lo                          reliability.solution_scale_lower (validated_lower_bound)
  F_dir/N_lo, G_dir/N_lo, G_AR/N_lo   reliability.ratio_over_lower (rounded up; 'not positive' if N_lo <= 0)
Slack diagnostics (diagnostic_float64): q^2 Gauss points per cell (q = 3) on the same partition, densities
R_hat^2, |S_hat|^2, mu (v - g)_+, R^2, |S|^2, tau^2, (d_x v_hat)^2 (v_hat branch by the sign of g - v at the point,
as experiments/v17/cell_quadrature._fields), t = 0 trace (u0 - v_hat(0))^2 with q = 3.
Writes results/v20_rising1d_cert/<timestamp>/ (config.json, summary.csv, cells.npz, weights.pt).
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
import torch

from certified_obstacle_nn.certification import reliability as rel
from certified_obstacle_nn.certification.enclose import upper_sum as US
from certified_obstacle_nn.certification.enclose.engine import Rising1DAdapter, TripleEncloser
from certified_obstacle_nn.experiments import results_io
from certified_obstacle_nn.experiments.v14.common import (REPO_ROOT, git_commit, git_dirty, hardware, sha256_file,
                                                          sha256_json)
from certified_obstacle_nn.experiments.v14.disk_validated import _peak_rss_gb
from certified_obstacle_nn.experiments.v17 import cell_quadrature as CQ
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.rising1d import Rising1DProblem

EXPERIMENT = "v20_rising1d_cert"
PROTOCOL = REPO_ROOT / "results/v20_protocols/rising1d_protocol.json"
TRACE_N = 4096
Q_SLACK = 3
PREC = 64  # arb backend precision of the Stefan runs (unused by the numpy backend; recorded)
DIAG_VOL = ("Rh2", "Sh2", "C", "R02", "S02", "tau2", "gvh2")


def load_frozen(train_run: Path):
    torch.set_default_dtype(torch.float64)
    problem = Rising1DProblem()
    triple = NeuralTriple(problem, width=64, depth=4, multiplier_head="square")
    triple.load_state_dict(torch.load(train_run / "weights.pt", map_location="cpu"))
    triple = triple.double().eval()
    for q in triple.parameters():
        q.requires_grad_(False)
    return problem, triple


def point_fields(problem, triple, t, x):
    """Float64 point densities (diagnostic). t, x: 1-D tensors with requires_grad."""
    v, tau, mu = (a.reshape(-1) for a in triple(t.reshape(-1, 1), x.reshape(-1, 1)))
    divt = CQ._grad(tau, (x,))[0]
    g = problem.g(t, x)
    f = problem.f(t, x)
    dv = CQ._grad(v, (t, x))
    dg = CQ._grad(g, (t, x))
    with torch.no_grad():
        pos = v > g
        vh_t = torch.where(pos, dv[0], dg[0])
        vh_x = torch.where(pos, dv[1], dg[1])
        Rh = f - vh_t - divt + mu
        R0 = f - dv[0] - divt + mu
        out = {"Rh2": Rh**2, "Sh2": (tau + vh_x) ** 2, "C": mu * torch.clamp(v - g, min=0), "R02": R0**2,
               "S02": (tau + dv[1]) ** 2, "tau2": tau**2, "gvh2": vh_x**2}
    return {k: w.detach().numpy() for k, w in out.items()}


def cell_integrals(problem, triple, lo, hi, q, chunk=200_000):
    n = lo.shape[1]
    nq = q**2
    out = {k: np.zeros(n) for k in DIAG_VOL}
    out.update({"sup_" + k: np.zeros(n) for k in DIAG_VOL})
    out.update({"inf_" + k: np.zeros(n) for k in ("tau2", "gvh2")})
    per = max(1, chunk // nq)
    for s in range(0, n, per):
        e = min(n, s + per)
        P, W = CQ._pts(lo[:, s:e], hi[:, s:e], q)
        t = torch.as_tensor(P[0]).clone().requires_grad_(True)
        x = torch.as_tensor(P[1]).clone().requires_grad_(True)
        F = point_fields(problem, triple, t, x)
        for k in DIAG_VOL:
            out[k][s:e] = (F[k] * W).reshape(e - s, nq).sum(1)
            out["sup_" + k][s:e] = F[k].reshape(e - s, nq).max(1)
        for k in ("tau2", "gvh2"):
            out["inf_" + k][s:e] = F[k].reshape(e - s, nq).min(1)
    return out


def trace0_integral(problem, triple, x0, x1, q=3):
    P, W = CQ._pts(x0[None, :], x1[None, :], q)
    with torch.no_grad():
        x = torch.as_tensor(P[0])
        t = torch.zeros_like(x)
        v = triple(t.reshape(-1, 1), x.reshape(-1, 1))[0].reshape(-1)
        d = (problem.u0(x) - torch.maximum(v, problem.g(t, x))) ** 2
    return float((d.numpy() * W).sum())


def hx(v):
    return float(v).hex() if isinstance(v, float) else ""


def run(train_run: Path, n: int, workers: int, log=print):
    t_start = time.time()
    train_run = Path(train_run)
    if not train_run.is_absolute():
        train_run = REPO_ROOT / train_run
    tcfg = json.loads((train_run / "config.json").read_text())
    ck = train_run / "weights.pt"
    ck_sha = sha256_file(ck)
    if tcfg.get("weights_sha256") and tcfg["weights_sha256"] != ck_sha:
        raise RuntimeError("checkpoint hash mismatch")
    problem, triple = load_frozen(train_run)
    frozen = all(not q.requires_grad for q in triple.parameters()) and not triple.training
    enc = TripleEncloser(triple, Rising1DAdapter(problem), lower_sums=True)
    part = US.uniform_2d(n, n)
    tp0, tpT = US.uniform_1d(TRACE_N), US.uniform_1d(TRACE_N)
    log(f"{train_run.name} seed {tcfg['seed']}: {part.n} cells, traces {TRACE_N}, partition {part.hash()[:12]}")
    tic = time.time()
    us, vol, (tr0, trT) = US.certify_partition(enc, part, tp0, tpT, backend="numpy", prec=PREC, workers=workers,
                                               model_frozen=frozen, constants_validated=True)
    t_eval = time.time() - tic
    tic = time.time()
    low = US.reduce_lower_sums(part, vol, backend_info=US.backend_info("numpy", PREC), model_frozen=frozen)
    t_low = time.time() - tic
    P = us.parts
    terms = {"U_R": P["vol_Rhat"], "U_S": P["vol_Shat"], "U_I": P["init_mismatch_hat"], "U_C": us.U_c,
             "U_R0": P["vol_R0"], "U_S0": P["vol_S_term"], "U_init": P["init_mismatch"], "U_gradw": P["vol_gradw"]}
    cf = rel.cF2_upper(1)
    d = rel.direct_projected_bounds(U_R=terms["U_R"], U_S=terms["U_S"], U_I=terms["U_I"], U_C=terms["U_C"],
                                    cF2_up=cf["cF2_up"], status=us.status, U_r_hat=us.U_r_hat,
                                    Lambda=float(problem.Lambda), U_gradw=terms["U_gradw"])
    G_AR = rel.grad_bound_AR(U_R=terms["U_R"], U_S=terms["U_S"], U_I=terms["U_I"], U_C=terms["U_C"],
                             cF2_up=cf["cF2_up"])
    F_dir, G_dir, E_dir = d["flux_bound_direct_split"], d["grad_bound_direct_split"], d["E_dir_split"]
    nl = rel.solution_scale_lower(L_tau=low["L_tau"], L_gradvhat=low["L_gradvhat"], F_dir=F_dir, G_AR=G_AR,
                                  upper_status=us.status, lower_status=low["lower_status"])
    row = {"problem": "rising1d", "seed": tcfg["seed"], "train_run": str(train_run.relative_to(REPO_ROOT)),
           "cells": part.n, "n": n, "n_trace0": tp0.n, "n_traceT": tpT.n, "backend": "numpy", "enclosure": "TM1",
           "status": us.status, "status_reasons": ";".join(us.status_reasons), "U_r_hat": us.U_r_hat, **terms,
           "F_dir": F_dir, "G_dir": G_dir, "G_AR": G_AR, "E_dir": E_dir, "E_dir_theta": d["E_dir_split_theta"],
           "D": d["D"], "L": d["L"], "cF2_up": cf["cF2_up"],
           "L_tau": low["L_tau"], "L_gradvhat": low["L_gradvhat"], "lower_status": low["lower_status"],
           "lower_status_reasons": low["lower_status_reasons"], **nl}
    for k, b in (("F_dir", F_dir), ("G_dir", G_dir), ("G_AR", G_AR)):
        row[f"{k}_over_N_lo"] = rel.ratio_over_lower(b, nl["N_lo"])
    row["ratio_status"] = nl["N_lo_status"] if nl["N_lo"] > 0 else "not positive"
    for nm, val in (("U_S", terms["U_S"]), ("U_I", terms["U_I"]), ("2U_C", 2 * terms["U_C"]), ("L", d["L"])):
        row[f"share_F2_{nm}"] = val / (d["D"] + d["L"])
    area = part.area()
    sgn = vol["sign"]
    for cls, m in (("res", sgn != 0), ("unres", sgn == 0)):
        row[f"n_cells_{cls}"] = int(m.sum())
        for nm, key in (("U_R", "Rh_term"), ("U_S", "Sh_term"), ("U_C", "c")):
            row[f"{nm}_{cls}"] = US._up(US._arb_dot_up(area[m], vol[key][m]))
    # float64 slack quadrature on the same partition
    tic = time.time()
    lo, hi = np.stack([part.t0, part.x0]), np.stack([part.t1, part.x1])
    diag = cell_integrals(problem, triple, lo, hi, Q_SLACK)
    I0 = trace0_integral(problem, triple, tp0.x0, tp0.x1)
    t_diag = time.time() - tic
    I = {"U_R": diag["Rh2"].sum(), "U_S": diag["Sh2"].sum(), "U_C": diag["C"].sum(), "U_I": I0,
         "U_R0": diag["R02"].sum(), "U_S0": diag["S02"].sum()}
    for nm, iv in I.items():
        row[f"I_{nm[2:]}"] = float(iv)
        row[f"slack_{nm}"] = row[nm] / iv if iv > 0 else float("nan")
    row["I_tau2"], row["I_gradvhat2"] = float(diag["tau2"].sum()), float(diag["gvh2"].sum())
    row["lower_slack_tau"] = row["I_tau2"] / low["L_tau"] if low["L_tau"] > 0 else float("nan")
    row["lower_slack_gradvhat"] = row["I_gradvhat2"] / low["L_gradvhat"] if low["L_gradvhat"] > 0 else float("nan")
    Dd = I["U_S"] + I["U_I"] + 2 * I["U_C"]
    Ld = cf["cF2_up"] * I["U_R"]
    row["diag_F_dir"] = math.sqrt(Dd + Ld)
    ad_ = math.sqrt(I["U_S"]) + math.sqrt(cf["cF2_up"] * I["U_R"])
    row["diag_G_AR"] = (ad_ + math.sqrt(ad_ * ad_ + 2 * (I["U_I"] + 2 * I["U_C"]))) / 2
    # per-cell lower endpoint below the sampled minimum (necessary condition, float64 screening)
    row["lower_cells_above_sampled_min_tau"] = int((vol["tau2_lo"] > diag["inf_tau2"]).sum())
    row["lower_cells_above_sampled_min_gvh"] = int((vol["gvh2_lo"] > diag["inf_gvh2"]).sum())
    row.update({"t_leaf_eval_s": us.timings["leaf_eval_s"], "t_trace_s": us.timings["trace_eval_s"],
                "t_reduction_s": us.timings["reduction_s"], "t_lower_reduction_s": t_low,
                "eval_s": t_eval + t_low, "t_diag_s": t_diag, "workers": workers,
                "checkpoint_sha256": ck_sha, "partition_hash": part.hash(), "trace0_hash": tp0.hash(),
                "traceT_hash": tpT.hash(), "git_commit": git_commit(), "git_dirty": git_dirty(),
                "peak_rss_gb": _peak_rss_gb()})
    for k in ("U_R", "U_S", "U_I", "U_C", "F_dir", "G_dir", "G_AR", "E_dir", "L_tau", "L_gradvhat", "N_lo",
              "N_lo_tau", "N_lo_grad", "F_dir_over_N_lo", "G_dir_over_N_lo", "G_AR_over_N_lo"):
        row[k + "_hex"] = hx(row[k])
    row["t_total_s"] = time.time() - t_start
    config = {"experiment": EXPERIMENT, "problem": "rising1d", "protocol": str(PROTOCOL.relative_to(REPO_ROOT)),
              "protocol_sha256": sha256_file(PROTOCOL), "train_run": row["train_run"], "seed": tcfg["seed"],
              "train_config_hash": tcfg.get("config_hash"), "checkpoint_sha256": ck_sha,
              "partition": {"volume": f"uniform {n} x {n}", "trace_n": TRACE_N, "partition_hash": part.hash(),
                            "trace0_hash": tp0.hash(), "traceT_hash": tpT.hash()},
              "backend": US.backend_info("numpy", PREC), "enclosure": "TM1", "lower_sums": True, "workers": workers,
              "constants": {**rel.problem_constants_record(problem), "cF2": cf},
              "formulas": {"direct": rel.direct_projected_bounds.__doc__, "G_AR": rel.grad_bound_AR.__doc__,
                           "N_lo": rel.solution_scale_lower.__doc__, "lower_sums": US.reduce_lower_sums.__doc__},
              "slack_quadrature": f"q={Q_SLACK} Gauss per axis per cell; t = 0 trace q = 3", "git_commit": git_commit(),
              "hardware": hardware(), "status": us.status, "lower_status": low["lower_status"]}
    config["config_hash"] = sha256_json({k: v for k, v in config.items() if k != "hardware"})
    row["config_hash"] = config["config_hash"]
    cells = {k: vol[k] for k in ("rhat", "c", "sign", "Rh_term", "Sh_term", "R0_term", "S_term", "tau2_lo",
                                 "gvh2_lo")}
    cells.update({"tr0_r0hat": tr0["r0hat"], "tr0_w2": tr0["w2"], "trT_w2": trT["w2"]})
    cells.update({f"float64_{k}": diag[k] for k in DIAG_VOL})
    log(f"  {us.status}/{low['lower_status']} U_R={terms['U_R']:.4e} U_S={terms['U_S']:.4e} U_I={terms['U_I']:.3e} "
        f"U_C={terms['U_C']:.3e} F_dir={F_dir:.5f} G_dir={G_dir:.5f} G_AR={G_AR:.5f} N_lo={nl['N_lo']:.5f} "
        f"(tau {nl['N_lo_tau']:.5f}, grad {nl['N_lo_grad']:.5f}) F/N_lo={row['F_dir_over_N_lo']} "
        f"G_AR/N_lo={row['G_AR_over_N_lo']} eval {t_eval:.0f}s diag {t_diag:.0f}s")
    return config, row, cells, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-run", nargs="+", required=True)
    ap.add_argument("--n", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    torch.set_num_threads(4)

    def log(msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    for tr in a.train_run:
        for n in a.n:
            config, row, cells, ck = run(Path(tr), n, a.workers, log)
            d = results_io.make_run_dir(EXPERIMENT)
            results_io.write_config(d, config)
            pd.DataFrame([row]).to_csv(d / "summary.csv", index=False)
            np.savez_compressed(d / "cells.npz", **cells)
            shutil.copyfile(ck, d / "weights.pt")
            log(f"wrote {d.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
