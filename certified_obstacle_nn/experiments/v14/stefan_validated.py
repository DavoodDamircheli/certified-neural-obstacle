"""v14 Stage 5: validated Stefan certificate for one frozen checkpoint and one
cell budget, every bound variant, float64 diagnostics, costs.

    python3 -m certified_obstacle_nn.experiments.v14.stefan_validated \
        --seed 0 --mode uniform --n 32 --backend arb --workers 20
    python3 -m certified_obstacle_nn.experiments.v14.stefan_validated \
        --seed 0 --mode adaptive --budget 100000 --backend numpy --verify-backend arb --workers 20

Writes results/v14_stefan_validated/<timestamp>/{config.json, summary.csv,
cells.npz, weights.pt} (experiments/results_io.py convention).

Adaptive mode: the partition is CHOSEN with the fast backend (refining the
largest contributions to 18 U_r + 36 U_p + 12 U_c), then FROZEN and hashed,
and every leaf and trace cell is re-evaluated from scratch by the reported
backend(s). Uniform mode uses a uniform dyadic n x n partition.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch

from certified_obstacle_nn.certification import reliability as rel
from certified_obstacle_nn.certification.components import converged_components
from certified_obstacle_nn.certification.enclose import upper_sum as US
from certified_obstacle_nn.certification.enclose.engine import StefanAdapter, TripleEncloser
from certified_obstacle_nn.errors.true_error_v14 import converged_true_error
from certified_obstacle_nn.experiments import results_io
from certified_obstacle_nn.experiments.v14.common import (
    REPO_ROOT, STEFAN_SEEDS, git_commit, git_dirty, hardware, load_frozen_stefan, sha256_file, sha256_json,
)

EXPERIMENT = "v14_stefan_validated"


def _peak_rss_gb():
    s = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    c = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return max(s, c) / 1e6


def diagnostics(problem, triple, n_quad=256):
    comp = converged_components(problem, triple, n_quad, 4)
    raw = converged_true_error(problem, triple, 64, "duffy", projected=False)
    proj = converged_true_error(problem, triple, 256, "duffy", projected=True)
    return comp, raw, proj


def run(seed, mode, n, budget, backend, verify_backend, workers, prec, trace_n, theta, diag=None, log=print):
    t_start = time.time()
    ck = STEFAN_SEEDS[seed]
    problem, triple = load_frozen_stefan(ck)
    frozen = all(not q.requires_grad for q in triple.parameters())
    enc = TripleEncloser(triple, StefanAdapter(problem))
    tic = time.time()
    refine_record = None
    if mode == "uniform":
        part = US.uniform_2d(n, n)
        tp0 = US.uniform_1d(trace_n)
        tpT = US.uniform_1d(trace_n)
    else:
        part, tp0, tpT, refine_record = US.adaptive_partition(
            enc, budget=budget, start=32, target="weighted", theta=theta, backend="numpy", workers=workers,
            trace_budget=trace_n, log=log)
    t_partition = time.time() - tic
    log(f"partition: {part.n} cells, traces {tp0.n}/{tpT.n}, {t_partition:.1f}s, hash {part.hash()[:12]}")

    results = {}
    vols = {}
    for bk in [backend] + ([verify_backend] if verify_backend and verify_backend != backend else []):
        us, vol, _ = US.certify_partition(enc, part, tp0, tpT, backend=bk, prec=prec, workers=workers,
                                          model_frozen=frozen, constants_validated=True)
        log(f"{bk}: U_r={us.U_r:.6e} U_p={us.U_p:.6e} U_c={us.U_c:.6e} U_r_hat={us.U_r_hat:.6e} "
            f"status={us.status} {us.timings}")
        results[bk] = us
        vols[bk] = vol

    if diag is None:
        tic = time.time()
        diag = diagnostics(problem, triple)
        t_diag = time.time() - tic
    else:
        t_diag = 0.0
    comp, raw, proj = diag
    rows = []
    for bk, us in results.items():
        b = rel.all_bounds(problem, us.U_r, us.U_p, us.U_c, status=us.status, U_r_hat=us.U_r_hat,
                           norm_grad_u=raw["norm_grad_u"], norm_sigma=raw["norm_sigma"])
        row = {
            "seed": seed, "mode": mode, "backend": bk, "prec_bits": prec if bk == "arb" else 53,
            "n_cells": us.n_cells, "n_trace0": us.n_trace0, "n_traceT": us.n_traceT, "max_cell_diam": us.max_diam,
            **{k: v for k, v in b.items() if not isinstance(v, (dict, list))},
            "E_true_v": raw["E_true"], "E_true_vhat": proj["E_true"],
            "E_true_grad": math.sqrt(raw["grad"]), "E_true_flux": math.sqrt(raw["flux"]),
            "E_true_init": math.sqrt(raw["init"]), "E_true_term": math.sqrt(raw["term"]),
            "E_true_graph": math.sqrt(raw["graph"]), "E_true_mult_L2": math.sqrt(raw["mult"]),
            "E_true_vhat_grad": math.sqrt(proj["grad"]), "E_true_vhat_graph": math.sqrt(proj["graph"]),
            "E_true_rel_change": raw["rel_change_E_true"], "E_true_vhat_rel_change": proj["rel_change_E_true"],
            "norm_grad_u": raw["norm_grad_u"], "norm_sigma": raw["norm_sigma"],
            "diag_r2": comp["r2"], "diag_p2": comp["p2"], "diag_c2": comp["c2"], "diag_r2_hat": comp["r2_hat"],
            "slack_U_r": us.U_r / comp["r2"], "slack_U_p": us.U_p / comp["p2"], "slack_U_c": us.U_c / comp["c2"],
            "slack_U_r_hat": us.U_r_hat / comp["r2_hat"],
            "ratio_E_U_weighted": b["E_U_weighted"] / raw["E_true"],
            "ratio_E_U_uniform": b["E_U_uniform"] / raw["E_true"],
            "ratio_E_U_216": b["E_U_uniform_216_comparison_convention"] / raw["E_true"],
            "ratio_E_U_best": b["E_U_best"] / raw["E_true"],
            "ratio_E_U_projected_vs_vhat": b["E_U_projected"] / proj["E_true"],
            "ratio_E_U_original": b["E_U_original_via_projection"] / raw["E_true"],
            "status_reasons": ";".join(us.status_reasons),
            "partition_hash": us.partition_hash, "trace0_hash": us.trace0_hash, "traceT_hash": us.traceT_hash,
            "checkpoint_sha256": sha256_file(ck), "git_commit": git_commit(), "git_dirty": git_dirty(),
            "t_partition_s": t_partition, "t_leaf_eval_s": us.timings["leaf_eval_s"],
            "t_trace_s": us.timings["trace_eval_s"], "t_reduction_s": us.timings["reduction_s"],
            "s_per_cell_wall": us.timings["s_per_cell_wall"],
            "s_per_cell_cpu": us.timings["s_per_cell_wall"] * max(workers, 1),
            "workers": workers, "t_diagnostics_s": t_diag,
            "t_total_post_training_s": time.time() - t_start,
            "peak_rss_gb": _peak_rss_gb(),
            "training_time_s": json.loads((ck.parent / "config.json").read_text()).get("training_time_seconds"),
            "network": "4x64 tanh x3 (state, flux, multiplier square head)",
        }
        for k, v in us.parts.items():
            row["part_" + k] = v
        rows.append(row)
    config = {
        "experiment": EXPERIMENT, "seed": seed, "checkpoint": str(ck.relative_to(REPO_ROOT)),
        "checkpoint_sha256": sha256_file(ck), "mode": mode, "n": n, "budget": budget, "theta": theta,
        "backends": list(results), "arb_prec_bits": prec, "workers": workers, "trace_n": trace_n,
        "problem_constants": rel.problem_constants_record(problem),
        "architecture": {"width": 64, "depth": 4, "multiplier_head": "square", "activation": "tanh"},
        "training_config": json.loads((ck.parent / "config.json").read_text()),
        "enclosure_method": "first-order Taylor models with interval coefficients, forward-mode first derivatives",
        "density_bounds": "per-cell sup via TM ranges; traces on separate 1-D partitions",
        "diagnostic_quadrature": {"components": "composite Gauss 256x256 cells q=4 vs 512", "E_true": "Duffy split n=64/128 (raw), 256/512 (projected)"},
        "refinement": refine_record, "hardware": hardware(), "git_commit": git_commit(),
        "statuses": {bk: us.status for bk, us in results.items()},
    }
    config["config_hash"] = sha256_json({k: v for k, v in config.items() if k not in ("hardware",)})
    for r in rows:
        r["config_hash"] = config["config_hash"]
    return config, rows, part, vols, diag


def write_run(config, rows, part, vols, triple_path, results_dir=None):
    run_dir = results_io.make_run_dir(EXPERIMENT) if results_dir is None else results_io.make_run_dir(EXPERIMENT, Path(results_dir))
    results_io.write_config(run_dir, config)
    import pandas as pd

    # summary.csv: one row per backend evaluation of the SAME frozen partition
    pd.DataFrame(rows).to_csv(run_dir / "summary.csv", index=False)
    arrays = {"t0": part.t0, "t1": part.t1, "x0": part.x0, "x1": part.x1}
    for bk, vol in vols.items():
        for k, v in vol.items():
            arrays[f"{bk}_{k}"] = v
    np.savez_compressed(run_dir / "cells.npz", **arrays)
    import shutil

    shutil.copyfile(triple_path, run_dir / "weights.pt")
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=("uniform", "adaptive"), default="uniform")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--backend", default="arb")
    ap.add_argument("--verify-backend", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--prec", type=int, default=64)
    ap.add_argument("--trace-n", type=int, default=4096)
    ap.add_argument("--theta", type=float, default=0.5)
    a = ap.parse_args()
    torch.set_num_threads(1)
    config, rows, part, vols, _ = run(a.seed, a.mode, a.n, a.budget, a.backend, a.verify_backend, a.workers, a.prec,
                                      a.trace_n, a.theta)
    d = write_run(config, rows, part, vols, STEFAN_SEEDS[a.seed])
    for r in rows:
        print({k: r[k] for k in ("backend", "n_cells", "status", "U_r", "U_p", "U_c", "U_r_hat", "E_U_weighted",
                                 "E_U_best", "E_U_projected", "E_U_original_via_projection", "ratio_E_U_weighted",
                                 "grad_bound_rel", "flux_bound_rel", "s_per_cell_wall", "peak_rss_gb")})
    print("run dir:", d)


if __name__ == "__main__":
    main()
