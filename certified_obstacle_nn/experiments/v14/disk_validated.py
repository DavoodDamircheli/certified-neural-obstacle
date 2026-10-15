"""v14 Stage 5 (disk): validated certificate of the stored expanding-disk
checkpoints (baseline and long schedule, seeds 0-2), every bound variant,
float64 diagnostics, costs.

    nohup python3 -m certified_obstacle_nn.experiments.v14.disk_validated \
        --schedules baseline long --seeds 0 1 2 --budgets 100000 1000000 --workers 20 \
        > logs/v14_disk_study.log 2>&1 &

Partition: adaptive dyadic cubes in (t, x, y), chosen with the fast backend
by Doerfler marking on 18 U_r + 36 U_p + 12 U_c (traces refined separately),
then FROZEN, hashed and re-evaluated from scratch by every reported backend
(arb only up to --arb-max-cells). One run directory per (schedule, seed,
budget) under results/v14_disk_validated/ (results_io convention); resumable
through --index.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from certified_obstacle_nn.certification import reliability as rel
from certified_obstacle_nn.certification.enclose import boxes as Bx
from certified_obstacle_nn.certification.enclose.engine2d import DiskAdapter, TripleEncloser2D
from certified_obstacle_nn.errors.diagnostics_2d import converged_diagnostics
from certified_obstacle_nn.experiments import results_io
from certified_obstacle_nn.experiments.expanding_disk_2d.config import RESULTS_ROOT
from certified_obstacle_nn.experiments.expanding_disk_2d.final_common import REFERENCE_6X64, load_run
from certified_obstacle_nn.experiments.v14.common import REPO_ROOT, git_commit, git_dirty, hardware, sha256_file, sha256_json
from certified_obstacle_nn.problems.disk import DiskProblem

EXPERIMENT = "v14_disk_validated"
INDEX = REPO_ROOT / f"results/{EXPERIMENT}/index.jsonl"
RUNS = {("baseline", s): REFERENCE_6X64[s] for s in range(3)}
RUNS.update({("long", s): f"final/long_6x64_N20000_s{s}_k5" for s in range(3)})


def _peak_rss_gb():
    s = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    c = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return max(s, c) / 1e6


def run(schedule, seed, budget, *, workers, prec, arb_max_cells, start, trace_budget, theta, diag=None, log=print):
    t_start = time.time()
    run_dir = RESULTS_ROOT / RUNS[(schedule, seed)]
    ck = run_dir / "checkpoint.pt"
    triple, cfg = load_run(run_dir)
    problem = DiskProblem(cfg.kappa)
    assert (cfg.alpha, cfg.Lambda) == (problem.alpha, problem.Lambda)
    frozen = all(not q.requires_grad for q in triple.parameters()) and not triple.training
    enc = TripleEncloser2D(triple, DiskAdapter(cfg.kappa, cfg.Lambda, cfg.alpha))
    tic = time.time()
    part, tp0, tpT, record = Bx.adaptive_boxes(enc, D=3, budget=budget, start=start, theta=theta, backend="numpy",
                                              workers=workers, trace_start=64, trace_budget=trace_budget, log=log)
    t_partition = time.time() - tic
    log(f"partition: {part.n} cells, traces {tp0.n}/{tpT.n}, {t_partition:.1f}s, hash {part.hash()[:12]}")
    backends = ["numpy"] + (["arb"] if part.n <= arb_max_cells else [])
    results, vols = {}, {}
    for bk in backends:
        us, vol, _ = Bx.certify_boxes(enc, part, tp0, tpT, backend=bk, prec=prec, workers=workers, model_frozen=frozen,
                                      constants_validated=True)
        log(f"{bk}: U_r={us.U_r:.6e} U_p={us.U_p:.6e} U_c={us.U_c:.6e} U_r_hat={us.U_r_hat:.6e} "
            f"status={us.status} {us.timings}")
        results[bk], vols[bk] = us, vol
    t_diag = 0.0
    if diag is None:
        tic = time.time()
        diag = converged_diagnostics(problem, triple)
        t_diag = time.time() - tic
    rows = []
    for bk, us in results.items():
        b = rel.all_bounds(problem, us.U_r, us.U_p, us.U_c, status=us.status, U_r_hat=us.U_r_hat,
                           norm_grad_u=diag["norm_grad_u"], norm_sigma=diag["norm_sigma"])
        row = {
            "problem": "disk2d", "schedule": schedule, "seed": seed, "mode": "adaptive", "budget": budget,
            "backend": bk, "prec_bits": prec if bk == "arb" else 53,
            "n_cells": us.n_cells, "n_trace0": us.n_trace0, "n_traceT": us.n_traceT, "max_cell_diam": us.max_diam,
            **{k: v for k, v in b.items() if not isinstance(v, (dict, list))},
            "E_true_v": diag["E_true"], "E_true_vhat": diag["E_true_vhat"],
            "E_true_grad": math.sqrt(diag["E_grad"]), "E_true_flux": math.sqrt(diag["E_flux"]),
            "E_true_init": math.sqrt(diag["E_0"]), "E_true_term": math.sqrt(diag["E_T"]),
            "E_true_graph": math.sqrt(diag["E_graph"]), "E_true_mult_L2": math.sqrt(diag["E_mult"]),
            "E_true_vhat_grad": math.sqrt(diag["Eh_grad"]), "E_true_vhat_graph": math.sqrt(diag["Eh_graph"]),
            "E_true_rel_change": diag["rel_change"].get("E_true"),
            "E_true_vhat_rel_change": diag["rel_change"].get("E_true_vhat"),
            "norm_grad_u": diag["norm_grad_u"], "norm_sigma": diag["norm_sigma"],
            "diag_r2": diag["r2"], "diag_p2": diag["p2"], "diag_c2": diag["c2"], "diag_r2_hat": diag["r2_hat"],
            "diag_rel_change_max": max(diag["rel_change"].get(k, 0.0) for k in ("r2", "p2", "c2", "r2_hat")),
            "slack_U_r": us.U_r / diag["r2"], "slack_U_p": us.U_p / diag["p2"], "slack_U_c": us.U_c / diag["c2"],
            "slack_U_r_hat": us.U_r_hat / diag["r2_hat"],
            "ratio_E_U_weighted": b["E_U_weighted"] / diag["E_true"],
            "ratio_E_U_uniform": b["E_U_uniform"] / diag["E_true"],
            "ratio_E_U_216": b["E_U_uniform_216_comparison_convention"] / diag["E_true"],
            "ratio_E_U_best": b["E_U_best"] / diag["E_true"],
            "ratio_E_U_projected_vs_vhat": b["E_U_projected"] / diag["E_true_vhat"],
            "ratio_E_U_original": b["E_U_original_via_projection"] / diag["E_true"],
            "status_reasons": ";".join(us.status_reasons),
            "partition_hash": us.partition_hash, "trace0_hash": us.trace0_hash, "traceT_hash": us.traceT_hash,
            "checkpoint": str(ck.relative_to(REPO_ROOT)), "checkpoint_sha256": sha256_file(ck),
            "git_commit": git_commit(), "git_dirty": git_dirty(),
            "t_partition_s": t_partition, "t_leaf_eval_s": us.timings["leaf_eval_s"],
            "t_trace_s": us.timings["trace_eval_s"], "t_reduction_s": us.timings["reduction_s"],
            "s_per_cell_wall": us.timings["s_per_cell_wall"], "s_per_cell_cpu": us.timings["s_per_cell_wall"] * workers,
            "workers": workers, "t_diagnostics_s": t_diag, "t_total_post_training_s": time.time() - t_start,
            "peak_rss_gb": _peak_rss_gb(),
            "training_time_s": _training_time(run_dir),
            "network": f"{cfg.arch} tanh x3 (state, flux R^2, multiplier square head)",
            "refine_stopping_reason": record["stopping_reason"],
        }
        for k, v in us.parts.items():
            row["part_" + k] = v
        rows.append(row)
    config = {
        "experiment": EXPERIMENT, "problem": "disk2d", "schedule": schedule, "seed": seed,
        "checkpoint": str(ck.relative_to(REPO_ROOT)), "checkpoint_sha256": sha256_file(ck),
        "training_config": json.loads((run_dir / "config.json").read_text()),
        "mode": "adaptive", "budget": budget, "start": start, "theta": theta, "trace_budget": trace_budget,
        "backends": list(results), "arb_prec_bits": prec, "workers": workers,
        "problem_constants": rel.problem_constants_record(problem),
        "data_formulas": DiskAdapter.__doc__, "boundary_mask": "B = 16 x(1-x) y(1-y)", "domain": "(0,1) x (0,1)^2",
        "enclosure_method": "first-order Taylor models (t, x, y) with interval coefficients, forward-mode first "
                            "derivatives; branch-aware data (hull on cells straddling psi = 0); decimal constants "
                            "enclosed as exact rationals",
        "diagnostic_quadrature": diag["method"], "refinement": record, "hardware": hardware(),
        "git_commit": git_commit(), "statuses": {bk: us.status for bk, us in results.items()},
    }
    config["config_hash"] = sha256_json({k: v for k, v in config.items() if k != "hardware"})
    for r in rows:
        r["config_hash"] = config["config_hash"]
    return config, rows, part, vols, diag, ck


def _training_time(run_dir):
    for name in ("train_info.json", "metrics.json"):
        p = run_dir / name
        if p.exists():
            d = json.loads(p.read_text())
            for k in ("train_time_s", "training_time_s", "wall_time_s", "time_s", "training_time_seconds"):
                if k in d:
                    return d[k]
    return None


def write_run(config, rows, part, vols, ck):
    run_dir = results_io.make_run_dir(EXPERIMENT)
    results_io.write_config(run_dir, config)
    import pandas as pd

    pd.DataFrame(rows).to_csv(run_dir / "summary.csv", index=False)
    arrays = {"lo": part.lo, "hi": part.hi}
    for bk, vol in vols.items():
        for k, v in vol.items():
            arrays[f"{bk}_{k}"] = v
    np.savez_compressed(run_dir / "cells.npz", **arrays)
    shutil.copyfile(ck, run_dir / "weights.pt")
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedules", nargs="+", default=["baseline", "long"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--budgets", type=int, nargs="+", default=[100000])
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--prec", type=int, default=64)
    ap.add_argument("--arb-max-cells", type=int, default=300000)
    ap.add_argument("--start", type=int, default=16)
    ap.add_argument("--trace-budget", type=int, default=16384)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--index", default=str(INDEX))
    a = ap.parse_args()
    torch.set_num_threads(4)
    index = Path(a.index)
    index.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if index.exists():
        done = {(d["schedule"], d["seed"], d["budget"]) for d in map(json.loads, index.read_text().splitlines())}

    def log(msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    for sched in a.schedules:
        for seed in a.seeds:
            diag = None
            for budget in a.budgets:
                if (sched, seed, budget) in done:
                    log(f"skip {sched} seed {seed} budget {budget} (done)")
                    continue
                log(f"{sched} seed {seed} budget {budget}: start")
                config, rows, part, vols, diag, ck = run(sched, seed, budget, workers=a.workers, prec=a.prec,
                                                         arb_max_cells=a.arb_max_cells, start=a.start,
                                                         trace_budget=a.trace_budget, theta=a.theta, diag=diag, log=log)
                d = write_run(config, rows, part, vols, ck)
                with index.open("a") as fh:
                    fh.write(json.dumps({"schedule": sched, "seed": seed, "budget": budget,
                                         "run_dir": str(d.relative_to(REPO_ROOT))}) + "\n")
                for r in rows:
                    log(f"  {r['backend']}: n={r['n_cells']} {r['status']} E_U_w={r['E_U_weighted']:.4f} "
                        f"ratio={r['ratio_E_U_weighted']:.2f} proj={r['E_U_projected']:.4f} "
                        f"orig={r['E_U_original_via_projection']:.4f} grad_rel={r['grad_bound_rel']:.3f} "
                        f"slack r/p/c={r['slack_U_r']:.2f}/{r['slack_U_p']:.2f}/{r['slack_U_c']:.2f} "
                        f"t_leaf={r['t_leaf_eval_s']:.1f}s")


if __name__ == "__main__":
    main()
