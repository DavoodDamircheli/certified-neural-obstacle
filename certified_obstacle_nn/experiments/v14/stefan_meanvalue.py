"""v14: mean-value enclosure of R (DECISIONS D16) against the TM1 enclosure on the
SAME frozen Stefan partition.

    python3 -m certified_obstacle_nn.experiments.v14.stefan_meanvalue --seed 0 --budget 10000 --workers 8

The adaptive partition is chosen with the TM1 encloser exactly as in
experiments/v14/stefan_validated.py (so its hash can be compared with the
committed v14_stefan_validated run of the same seed and budget), frozen, and
every leaf and trace cell is evaluated from scratch by both encloser variants
and every requested backend. Gate (exit code 1 otherwise), per backend:
  * the TM1 part of the new evaluation reproduces the old R widths bit for bit;
  * per cell, every new density (r, p, c, rhat, R_term) <= the old one, i.e. the
    new R / R_hat enclosures are contained in the old ones;
  * U_r, U_p, U_c, U_r_hat (new) <= (old).
float64 diagnostics (diag_*, E_true) are read from the committed clean
v14_stefan_validated summary for the same checkpoint (matched by SHA-256);
they do not depend on the enclosure.

Writes results/v14_stefan_meanvalue/<timestamp>/{config.json, summary.csv, cells.npz}.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time

import numpy as np
import torch

from certified_obstacle_nn.certification import reliability as rel
from certified_obstacle_nn.certification.enclose import upper_sum as US
from certified_obstacle_nn.certification.enclose.engine import StefanAdapter, TripleEncloser
from certified_obstacle_nn.experiments import results_io
from certified_obstacle_nn.experiments.v14.common import (
    REPO_ROOT, STEFAN_SEEDS, git_commit, git_dirty, hardware, load_frozen_stefan, read_summary, sha256_file, sha256_json,
)

EXPERIMENT = "v14_stefan_meanvalue"
ENCLOSURES = {"tm1": False, "tm1_meanvalue": True}
DIAG_KEYS = ("diag_r2", "diag_p2", "diag_c2", "diag_r2_hat", "E_true_v", "E_true_vhat", "norm_grad_u", "norm_sigma")
DENSITY_KEYS = ("r", "p", "c", "rhat", "R_term")


def committed_diagnostics(ck_sha: str, seed: int) -> dict:
    for f in sorted(glob.glob(str(REPO_ROOT / "results" / "v14_stefan_validated" / "*" / "summary.csv"))):
        d = read_summary(f)
        d = d[(d.seed == seed) & (d.checkpoint_sha256.astype(str) == ck_sha) & (d.git_dirty.astype(str) == "False")]
        if len(d):
            row = d.iloc[0]
            return {"source": f[len(str(REPO_ROOT)) + 1:], **{k: float(row[k]) for k in DIAG_KEYS}}
    raise RuntimeError(f"no clean v14_stefan_validated summary for checkpoint {ck_sha[:12]}")


def committed_partition_hash(ck_sha: str, seed: int, budget: int):
    for f in sorted(glob.glob(str(REPO_ROOT / "results" / "v14_stefan_validated" / "*" / "summary.csv"))):
        d = read_summary(f)
        d = d[(d.seed == seed) & (d.checkpoint_sha256.astype(str) == ck_sha) & (d["mode"] == "adaptive")
              & (d.git_dirty.astype(str) == "False")]
        cfg = json.loads((REPO_ROOT / f).parent.joinpath("config.json").read_text()) if len(d) else {}
        if len(d) and cfg.get("budget") == budget:
            return str(d.iloc[0].partition_hash)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=10000)
    ap.add_argument("--backends", nargs="+", default=["numpy", "arb"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--prec", type=int, default=64)
    ap.add_argument("--trace-n", type=int, default=4096)
    ap.add_argument("--theta", type=float, default=0.5)
    a = ap.parse_args()
    torch.set_num_threads(1)
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)

    ck = STEFAN_SEEDS[a.seed]
    ck_sha = sha256_file(ck)
    problem, triple = load_frozen_stefan(ck)
    frozen = all(not q.requires_grad for q in triple.parameters())
    diag = committed_diagnostics(ck_sha, a.seed)
    encs = {name: TripleEncloser(triple, StefanAdapter(problem), mean_value=mv) for name, mv in ENCLOSURES.items()}

    tic = time.time()
    part, tp0, tpT, record = US.adaptive_partition(encs["tm1"], budget=a.budget, start=32, target="weighted",
                                                   theta=a.theta, backend="numpy", workers=a.workers,
                                                   trace_budget=a.trace_n, log=log)
    t_partition = time.time() - tic
    ref_hash = committed_partition_hash(ck_sha, a.seed, a.budget)
    log(f"partition: {part.n} cells, hash {part.hash()[:12]} (committed run: {(ref_hash or 'none')[:12]}), {t_partition:.0f}s")

    rows, vols, failures = [], {}, []
    for bk in a.backends:
        res = {}
        for name, enc in encs.items():
            us, vol, _ = US.certify_partition(enc, part, tp0, tpT, backend=bk, prec=a.prec, workers=a.workers,
                                              model_frozen=frozen, constants_validated=True)
            res[name] = (us, vol)
            vols[(bk, name)] = vol
            log(f"{bk} {name}: U_r={us.U_r:.6e} U_p={us.U_p:.6e} U_c={us.U_c:.6e} U_r_hat={us.U_r_hat:.6e} "
                f"{us.status} {us.timings['s_per_cell_wall'] * 1e3:.2f} ms/cell")
        (uo, vo), (un, vn) = res["tm1"], res["tm1_meanvalue"]
        checks = {"tm1_widths_reproduced": bool(np.array_equal(vn["R_width_tm1"], vo["R_width"]))}
        for k in DENSITY_KEYS:
            checks[f"cellwise_{k}_new_le_old"] = bool(np.all(vn[k] <= vo[k]))
        checks["R_width_new_le_old"] = bool(np.all(vn["R_width"] <= vo["R_width"]))
        for k in ("U_r", "U_p", "U_c", "U_r_hat"):
            checks[f"{k}_new_le_old"] = bool(getattr(un, k) <= getattr(uo, k))
        failures += [f"{bk}:{k}" for k, ok in checks.items() if not ok]
        better = vn["R_width_mv"] < vn["R_width_tm1"]
        for name, (us, vol) in res.items():
            b = rel.all_bounds(problem, us.U_r, us.U_p, us.U_c, status=us.status, U_r_hat=us.U_r_hat,
                               norm_grad_u=diag["norm_grad_u"], norm_sigma=diag["norm_sigma"])
            rows.append({
                "seed": a.seed, "mode": "adaptive", "budget": a.budget, "enclosure": name, "backend": bk,
                "prec_bits": a.prec if bk == "arb" else 53, "n_cells": us.n_cells, "status": us.status,
                "U_r": us.U_r, "U_p": us.U_p, "U_c": us.U_c, "U_r_hat": us.U_r_hat,
                "slack_U_r": us.U_r / diag["diag_r2"], "slack_U_p": us.U_p / diag["diag_p2"],
                "slack_U_c": us.U_c / diag["diag_c2"], "slack_U_r_hat": us.U_r_hat / diag["diag_r2_hat"],
                "E_U_weighted": b["E_U_weighted"], "E_U_projected": b["E_U_projected"],
                "ratio_E_U_weighted": b["E_U_weighted"] / diag["E_true_v"],
                "ratio_E_U_projected_vs_vhat": b["E_U_projected"] / diag["E_true_vhat"],
                "vol_R_term": us.parts["vol_R_term"], "median_R_width": float(np.median(vol["R_width"])),
                "frac_cells_mv_narrower": float(better.mean()) if name == "tm1_meanvalue" else 0.0,
                "median_R_width_ratio_mv_over_tm1": float(np.median(vn["R_width_mv"] / vn["R_width_tm1"]))
                if name == "tm1_meanvalue" else 1.0,
                "containment_gate": "PASS" if all(checks.values()) else "FAIL",
                "s_per_cell_wall": us.timings["s_per_cell_wall"], "t_leaf_eval_s": us.timings["leaf_eval_s"],
                "workers": a.workers, "partition_hash": us.partition_hash,
                "partition_hash_matches_committed": ref_hash == us.partition_hash if ref_hash else None,
                "checkpoint_sha256": ck_sha, "git_commit": git_commit(), "git_dirty": git_dirty(),
            })
        log(f"{bk} gate: {'PASS' if all(checks.values()) else 'FAIL'} {checks}")

    config = {
        "experiment": EXPERIMENT, "seed": a.seed, "checkpoint": str(ck.relative_to(REPO_ROOT)),
        "checkpoint_sha256": ck_sha, "budget": a.budget, "theta": a.theta, "trace_n": a.trace_n,
        "backends": a.backends, "arb_prec_bits": a.prec, "workers": a.workers,
        "partition": {"chosen_by": "tm1 encloser, numpy, weighted target (as stefan_validated)",
                      "hash": part.hash(), "committed_hash": ref_hash, "t_partition_s": t_partition,
                      "refinement": record},
        "enclosures": {"tm1": "first-order Taylor models (D4)",
                       "tm1_meanvalue": "intersection of TM1 with the mean-value form of R, R_hat (D16)"},
        "diagnostics": diag, "problem_constants": rel.problem_constants_record(problem),
        "hardware": hardware(), "git_commit": git_commit(), "git_dirty": git_dirty(),
        "gate_failures": failures,
    }
    config["config_hash"] = sha256_json({k: v for k, v in config.items() if k != "hardware"})
    for r in rows:
        r["config_hash"] = config["config_hash"]
    run_dir = results_io.make_run_dir(EXPERIMENT)
    results_io.write_config(run_dir, config)
    import pandas as pd

    pd.DataFrame(rows).to_csv(run_dir / "summary.csv", index=False)
    arrays = {"t0": part.t0, "t1": part.t1, "x0": part.x0, "x1": part.x1}
    for (bk, name), vol in vols.items():
        for k, v in vol.items():
            arrays[f"{bk}_{name}_{k}"] = v
    np.savez_compressed(run_dir / "cells.npz", **arrays)
    log(f"run dir: {run_dir}")
    if failures:
        log(f"GATE FAILED: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
