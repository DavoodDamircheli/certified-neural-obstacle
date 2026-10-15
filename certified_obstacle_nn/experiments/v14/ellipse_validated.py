"""v14 Stage 6 (ellipse): validated certificate of the trained expanding-ellipse
checkpoints (results/v14_ellipse_train/, kappa0 = 1/100 main benchmark, seeds
0-2), every bound variant, float64 diagnostics, costs. Same partition strategy
as the disk study (experiments/v14/disk_validated.py, DECISIONS D12): adaptive
dyadic cubes in (t, x, y) chosen with the fast backend by Doerfler marking on
18 U_r + 36 U_p + 12 U_c, traces refined separately, then FROZEN, hashed and
re-evaluated from scratch by every reported backend (arb only up to
--arb-max-cells).

    nohup python3 -m certified_obstacle_nn.experiments.v14.ellipse_validated \
        --seeds 0 1 2 --budgets 100000 1000000 --workers 20 \
        > logs/v14_ellipse_validated.log 2>&1 &

One run directory per (seed, budget) under results/v14_ellipse_validated/
(results_io convention); resumable through --index.
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
from certified_obstacle_nn.certification.enclose import boxes as Bx
from certified_obstacle_nn.certification.enclose.engine2d import EllipseAdapter, TripleEncloser2D
from certified_obstacle_nn.errors.diagnostics_2d import converged_diagnostics
from certified_obstacle_nn.experiments import results_io
from certified_obstacle_nn.experiments.expanding_disk_2d.model import Triple2D
from certified_obstacle_nn.experiments.v14.common import (REPO_ROOT, git_commit, git_dirty, hardware, read_summary,
                                                        sha256_file, sha256_json)
from certified_obstacle_nn.experiments.v14.disk_validated import _peak_rss_gb
from certified_obstacle_nn.problems.ellipse import EllipseProblem

EXPERIMENT = "v14_ellipse_validated"
INDEX = REPO_ROOT / f"results/{EXPERIMENT}/index.jsonl"
TRAIN_ROOT = REPO_ROOT / "results/v14_ellipse_train"


def find_training_run(seed: int, variant: str = "main") -> Path:
    """The (unique) clean training run of `seed` and `variant`; raises if absent or ambiguous."""
    hits = []
    for f in sorted(TRAIN_ROOT.glob("*/summary.csv")):
        s = read_summary(f).iloc[0]
        if int(s["seed"]) == seed and s["variant"] == variant and not bool(s["git_dirty"]):
            hits.append(f.parent)
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one clean ellipse training run for seed {seed} ({variant}), got {hits}")
    return hits[0]


def load_triple(train_dir: Path):
    """Frozen float64 Triple2D (eval mode, no grad) from a v14_ellipse_train run, plus its config and summary."""
    cfg = json.loads((train_dir / "config.json").read_text())
    summ = read_summary(train_dir / "summary.csv").iloc[0]
    ck = train_dir / "weights.pt"
    if sha256_file(ck) != summ["weights_sha256"]:
        raise RuntimeError(f"checkpoint hash mismatch for {ck}")
    torch.set_default_dtype(torch.float64)
    arch = cfg["architecture"]
    triple = Triple2D(arch["depth"], arch["width"], arch["multiplier_head"]).double()
    triple.load_state_dict(torch.load(ck))
    for p in triple.parameters():
        p.requires_grad_(False)
    triple.eval()
    return triple, cfg, summ, ck


def _ratio(a, b):
    """Diagnostic ratio; NaN when the float64 integral is exactly 0 (e.g. no penetration anywhere)."""
    return a / b if b != 0.0 else float("nan")


def run(triple, k0, ck, *, seed, budget, workers, prec, arb_max_cells, start, trace_budget, theta, diag=None,
        training_config=None, training_run=None, training_time_s=None, log=print):
    t_start = time.time()
    problem = EllipseProblem(kappa0=k0[0] / k0[1])
    adapter = EllipseAdapter(*k0)
    assert (problem.alpha, problem.Lambda, problem.kappa) == (1.0, adapter.La, adapter.kappa)
    frozen = all(not q.requires_grad for q in triple.parameters()) and not triple.training
    enc = TripleEncloser2D(triple, adapter)
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
    variant = "main" if tuple(k0) == (1, 100) else f"VARIANT kappa0={k0[0]}/{k0[1]}"
    ck_rel = str(Path(ck).resolve().relative_to(REPO_ROOT)) if Path(ck).resolve().is_relative_to(REPO_ROOT) else str(ck)
    rows = []
    for bk, us in results.items():
        b = rel.all_bounds(problem, us.U_r, us.U_p, us.U_c, status=us.status, U_r_hat=us.U_r_hat,
                           norm_grad_u=diag["norm_grad_u"], norm_sigma=diag["norm_sigma"])
        row = {
            "problem": "ellipse", "schedule": variant, "seed": seed, "mode": "adaptive", "budget": budget,
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
            "slack_U_r": _ratio(us.U_r, diag["r2"]), "slack_U_p": _ratio(us.U_p, diag["p2"]),
            "slack_U_c": _ratio(us.U_c, diag["c2"]), "slack_U_r_hat": _ratio(us.U_r_hat, diag["r2_hat"]),
            "ratio_E_U_weighted": b["E_U_weighted"] / diag["E_true"],
            "ratio_E_U_uniform": b["E_U_uniform"] / diag["E_true"],
            "ratio_E_U_216": b["E_U_uniform_216_comparison_convention"] / diag["E_true"],
            "ratio_E_U_best": b["E_U_best"] / diag["E_true"],
            "ratio_E_U_projected_vs_vhat": b["E_U_projected"] / diag["E_true_vhat"],
            "ratio_E_U_original": b["E_U_original_via_projection"] / diag["E_true"],
            "status_reasons": ";".join(us.status_reasons),
            "partition_hash": us.partition_hash, "trace0_hash": us.trace0_hash, "traceT_hash": us.traceT_hash,
            "checkpoint": ck_rel, "checkpoint_sha256": sha256_file(ck),
            "git_commit": git_commit(), "git_dirty": git_dirty(),
            "t_partition_s": t_partition, "t_leaf_eval_s": us.timings["leaf_eval_s"],
            "t_trace_s": us.timings["trace_eval_s"], "t_reduction_s": us.timings["reduction_s"],
            "s_per_cell_wall": us.timings["s_per_cell_wall"], "s_per_cell_cpu": us.timings["s_per_cell_wall"] * workers,
            "workers": workers, "t_diagnostics_s": t_diag, "t_total_post_training_s": time.time() - t_start,
            "peak_rss_gb": _peak_rss_gb(), "training_time_s": training_time_s,
            "network": "6x64 tanh x3 (state, flux R^2, multiplier square head)",
            "refine_stopping_reason": record["stopping_reason"],
        }
        for k, v in us.parts.items():
            row["part_" + k] = v
        rows.append(row)
    config = {
        "experiment": EXPERIMENT, "problem": "ellipse", "variant": variant, "kappa0": list(k0), "seed": seed,
        "checkpoint": ck_rel, "checkpoint_sha256": sha256_file(ck), "training_run": training_run,
        "training_config": training_config,
        "mode": "adaptive", "budget": budget, "start": start, "theta": theta, "trace_budget": trace_budget,
        "backends": list(results), "arb_prec_bits": prec, "workers": workers,
        "problem_constants": rel.problem_constants_record(problem),
        "data_formulas": EllipseAdapter.__doc__, "boundary_mask": "B = 16 x(1-x) y(1-y)", "domain": "(0,1) x (0,1)^2",
        "enclosure_method": "first-order Taylor models (t, x, y) with interval coefficients, forward-mode first "
                            "derivatives; branch-aware data (hull on cells straddling zeta = 0); decimal constants "
                            "enclosed as exact rationals",
        "diagnostic_quadrature": diag["method"], "refinement": record, "hardware": hardware(),
        "git_commit": git_commit(), "statuses": {bk: us.status for bk, us in results.items()},
    }
    config["config_hash"] = sha256_json({k: v for k, v in config.items() if k != "hardware"})
    for r in rows:
        r["config_hash"] = config["config_hash"]
    return config, rows, part, vols, diag


def write_run(config, rows, part, vols, ck, experiment=EXPERIMENT, results_dir=None):
    run_dir = results_io.make_run_dir(experiment, **({} if results_dir is None else {"results_dir": Path(results_dir)}))
    results_io.write_config(run_dir, config)
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
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--budgets", type=int, nargs="+", default=[100000])
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--prec", type=int, default=64)
    ap.add_argument("--arb-max-cells", type=int, default=150000)
    ap.add_argument("--arb-seeds", type=int, nargs="*", default=[0], help="seeds whose runs may use arb")
    ap.add_argument("--start", type=int, default=32)
    ap.add_argument("--trace-budget", type=int, default=16384)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--index", default=str(INDEX))
    ap.add_argument("--train-run", default=None, help="v19: training run directory (default: find_training_run(seed))")
    ap.add_argument("--experiment", default=EXPERIMENT, help="v19: results/<experiment>/ of the written runs")
    a = ap.parse_args()
    torch.set_num_threads(4)
    index = Path(a.index)
    index.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if index.exists():
        done = {(d["seed"], d["budget"]) for d in map(json.loads, index.read_text().splitlines())}

    def log(msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    for seed in a.seeds:
        train_dir = find_training_run(seed) if a.train_run is None else REPO_ROOT / a.train_run
        triple, tcfg, tsum, ck = load_triple(train_dir)
        k0 = tuple(tcfg["kappa0"])
        diag = None
        for budget in a.budgets:
            if (seed, budget) in done:
                log(f"skip seed {seed} budget {budget} (done)")
                continue
            log(f"ellipse seed {seed} budget {budget}: start ({train_dir.relative_to(REPO_ROOT)})")
            config, rows, part, vols, diag = run(
                triple, k0, ck, seed=seed, budget=budget, workers=a.workers, prec=a.prec,
                arb_max_cells=a.arb_max_cells if seed in a.arb_seeds else 0, start=a.start,
                trace_budget=a.trace_budget, theta=a.theta, diag=diag, training_config=tcfg,
                training_run=str(train_dir.relative_to(REPO_ROOT)), training_time_s=float(tsum["train_time_s"]),
                log=log)
            d = write_run(config, rows, part, vols, ck, experiment=a.experiment)
            with index.open("a") as fh:
                fh.write(json.dumps({"seed": seed, "budget": budget, "run_dir": str(d.relative_to(REPO_ROOT))}) + "\n")
            for r in rows:
                log(f"  {r['backend']}: n={r['n_cells']} {r['status']} E_U_w={r['E_U_weighted']:.4f} "
                    f"ratio={r['ratio_E_U_weighted']:.2f} proj={r['E_U_projected']:.4f} "
                    f"orig={r['E_U_original_via_projection']:.4f} grad_rel={r['grad_bound_rel']:.3f} "
                    f"slack r/p/c={r['slack_U_r']:.2f}/{r['slack_U_p']:.2f}/{r['slack_U_c']:.2f} "
                    f"t_leaf={r['t_leaf_eval_s']:.1f}s")


if __name__ == "__main__":
    main()
