"""Phase 1 smoke test (NEW_RUNS.md infra scaffolding, 2026-09-13).

Exercises the full record -> check -> aggregate -> plot pipeline in
experiments/new_runs/common.py + stefan_run.py end to end, at drastically
reduced training/certification/quadrature resolution, entirely inside
results/new_runs/_smoke/ (deleted at the end of this script -- never part of
the real results tree). Also exercises the negative paths of each sanity
check (deliberately-corrupted inputs, in a separate try/except, never
touching the real rows) so a check that silently never fires is not mistaken
for one that works.

Run: python -m certified_obstacle_nn.experiments.new_runs.smoke_test
"""

from __future__ import annotations

import json
import shutil
from copy import deepcopy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from certified_obstacle_nn.experiments.new_runs.common import (
    NEW_RUNS_DIR,
    aggregate_master_csv,
    check_certificate_inequality,
    check_matched_configs,
    check_no_nan_inf,
    save_figure,
)
from certified_obstacle_nn.experiments.new_runs.stefan_run import train_and_evaluate_stefan_run
from certified_obstacle_nn.problems.stefan import StefanProblem
from certified_obstacle_nn.training.sampler import TrainingSampler

SMOKE_DIR = NEW_RUNS_DIR / "_smoke"

# Drastically reduced vs. any real study (stefan_adaptive.py: n_interior=1500,
# 5 rounds x (1500 Adam + 50 LBFGS); stefan_core.py: n_interior=5000, 5000
# Adam + 300 LBFGS, grid=1024) -- this exists only to prove the pipeline
# runs, not to produce a scientifically meaningful triple.
SMOKE_N_INTERIOR = 300
SMOKE_N_INITIAL = 50
SMOKE_N_ADAM = 50
SMOKE_N_LBFGS = 10
SMOKE_CERT_GRID = 16
SMOKE_N_QP = 3
SMOKE_TE_N = 30
SMOKE_FB_N_T = 80


def _run_one(seed: int):
    problem = StefanProblem()
    sampler = TrainingSampler(problem, n_interior=SMOKE_N_INTERIOR, n_initial=SMOKE_N_INITIAL, seed=seed)
    t, x, x0 = sampler.sample()
    run_dir = SMOKE_DIR / f"seed{seed}"
    return train_and_evaluate_stefan_run(
        run_dir, width=8, depth=2, seed=seed, t=t, x=x, x0=x0,
        n_interior=SMOKE_N_INTERIOR, n_initial=SMOKE_N_INITIAL,
        n_adam_epochs=SMOKE_N_ADAM, n_lbfgs_epochs=SMOKE_N_LBFGS,
        cert_grid=SMOKE_CERT_GRID, n_qp=SMOKE_N_QP,
        true_error_n_t=SMOKE_TE_N, true_error_n_x=SMOKE_TE_N, fb_n_t=SMOKE_FB_N_T,
        experiment="_smoke", subexperiment="pipeline_check", strategy="uniform",
    )


def main() -> None:
    if SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR)
    SMOKE_DIR.mkdir(parents=True)

    print("=== 1. record: running two throwaway Stefan smoke runs (seed 0, 1) ===")
    row0 = _run_one(seed=0)
    row1 = _run_one(seed=1)
    for row in (row0, row1):
        print(f"  seed={row['seed']} status={row['status']} E_true={row['E_true']} "
              f"E_cert_U={row['E_cert_U']} certificate_inequality_violated={row['certificate_inequality_violated']}")
    assert row0["status"] == "ok" and row1["status"] == "ok", "smoke runs must succeed for the pipeline to be exercised end to end"

    print("\n=== 2. check (positive paths): both smoke rows must pass as-is ===")
    check_no_nan_inf(row0, label="smoke seed0")
    check_no_nan_inf(row1, label="smoke seed1")
    config0 = json.loads((SMOKE_DIR / "seed0" / "config.json").read_text())
    config1 = json.loads((SMOKE_DIR / "seed1" / "config.json").read_text())
    check_matched_configs(
        [config0, config1],
        keys=["architecture.width", "architecture.depth", "grid_sizes.n_interior", "grid_sizes.n_initial",
              "optimizer_schedule.adam.lr", "optimizer_schedule.adam.n_epochs", "certification_grid"],
        label="smoke seed0 vs seed1",
    )
    print("  check_no_nan_inf: both rows passed")
    print("  check_matched_configs: seed0/seed1 configs agree on all matched keys")

    print("\n=== 2b. check (negative paths): deliberately-corrupted inputs must be caught ===")
    violated = check_certificate_inequality(E_true=1e6, E_cert_U=1.0, label="deliberately-corrupted smoke check")
    assert violated is True, "check_certificate_inequality must return True (and warn) when E_true > E_cert_U"
    print("  check_certificate_inequality: correctly flagged a deliberately-violated pair (E_true=1e6 > E_cert_U=1.0)")

    bad_row = deepcopy(row0)
    bad_row["E_true"] = float("nan")
    try:
        check_no_nan_inf(bad_row, label="deliberately-corrupted smoke row")
        raise SystemExit("check_no_nan_inf FAILED to catch a deliberately-injected NaN -- infra bug")
    except AssertionError:
        print("  check_no_nan_inf: correctly raised AssertionError on a deliberately-injected NaN")

    mismatched_config = deepcopy(config1)
    mismatched_config["grid_sizes"]["n_interior"] = SMOKE_N_INTERIOR + 1
    try:
        check_matched_configs([config0, mismatched_config], keys=["grid_sizes.n_interior"], label="deliberate mismatch")
        raise SystemExit("check_matched_configs FAILED to catch a deliberately-mismatched key -- infra bug")
    except AssertionError:
        print("  check_matched_configs: correctly raised AssertionError on a deliberately-mismatched n_interior")

    print("\n=== 3. aggregate: smoke master CSV (2 rows) ===")
    smoke_csv_path = SMOKE_DIR / "smoke_master.csv"
    df = aggregate_master_csv([row0, row1], out_path=smoke_csv_path)
    print(f"  wrote {smoke_csv_path} ({len(df)} rows, {len(df.columns)} columns)")
    print(df[["experiment", "strategy", "seed", "status", "E_true", "E_cert_U", "rho_tilde_h", "L_R"]].to_string(index=False))

    print("\n=== 4. plot: fig_smoke_test (PDF + PNG) ===")
    fig, ax = plt.subplots()
    ax.bar([str(row0["seed"]), str(row1["seed"])], [row0["E_true"], row1["E_true"]])
    ax.set_xlabel("seed")
    ax.set_ylabel("E_true")
    ax.set_title("Phase 1 smoke test (throwaway Stefan runs)")
    pdf_path, png_path = save_figure(fig, SMOKE_DIR / "figures" / "fig_smoke_test")
    plt.close(fig)
    assert pdf_path.exists() and png_path.exists()
    print(f"  wrote {pdf_path}")
    print(f"  wrote {png_path}")

    print("\n=== 5. real (non-smoke) master CSV: empty but valid ===")
    real_df = aggregate_master_csv([])
    print(f"  wrote results/new_runs/new_runs_master.csv ({len(real_df)} rows, {len(real_df.columns)} columns)")
    print(f"  columns: {list(real_df.columns)}")

    print("\n=== cleanup: deleting _smoke/ ===")
    shutil.rmtree(SMOKE_DIR)
    print(f"  removed {SMOKE_DIR}")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
