"""NEW_RUNS.md Experiment 4 -- free-boundary threshold sensitivity
(OPTIONAL / CHEAP). No retraining: re-evaluates EXISTING checkpoints only,
via errors/free_boundary.py's own compute_free_boundary_error_sensitivity_sweep
(already built for exactly this -- one shared network-evaluation grid per
checkpoint, reused across every (alpha_u, alpha_lambda) combination, so this
costs one forward pass per checkpoint, not one per threshold).

Per the Phase 0 report: eps_u/eps_lambda are NOT fixed absolute numbers --
they are ALPHA_U/ALPHA_LAMBDA (default 0.02) times, respectively, the
problem's own ||g||_inf and this run's own empirical ||lambda_psi||_inf. So
"epsilon = 0.01/0.02/0.03" (NEW_RUNS.md 4.2) is read as sweeping these two
multipliers together (the diagonal alpha_u=alpha_lambda), which is exactly
option 1 offered there ("vary them together first").

Sources of existing checkpoints (NEW_RUNS.md 4.3: "the existing architecture
runs and/or adaptive runs"), read-only, none retrained:
  - results/stefan_m18/{4x32,4x64,6x64,6x128}/seed{0-4}/  (20 checkpoints,
    the existing Stefan architecture study -- NOT rerun, only re-evaluated)
  - results/new_runs/adaptive_multiseed/strategy_*/seed_*/ (30 checkpoints,
    this task's own Experiment 1)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from certified_obstacle_nn.errors.free_boundary import compute_free_boundary_error_sensitivity_sweep
from certified_obstacle_nn.experiments.new_runs.common import NEW_RUNS_DIR, load_frozen_triple_from_run, save_figure

EXPERIMENT = "threshold_sensitivity"
REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results"

ALPHA_VALUES = [0.01, 0.02, 0.03]
N_T = 400  # matches experiments/stefan_core.py's own FB_N_T=400 (measured-safe resolution for this triple family)
N_SCAN = 401


def _stefan_m18_checkpoints() -> list[dict]:
    out = []
    m18_dir = RESULTS_DIR / "stefan_m18"
    for arch_dir in sorted(m18_dir.iterdir()):
        if not arch_dir.is_dir():
            continue
        for seed_dir in sorted(arch_dir.glob("seed*")):
            weights = seed_dir / "weights.pt"
            config = seed_dir / "config.json"
            if weights.exists() and config.exists():
                seed = json.loads(config.read_text())["seed"]
                out.append({"source": "stefan_m18_architecture", "group": arch_dir.name, "seed": seed,
                            "weights_path": weights, "config_path": config})
    return out


def _adaptive_multiseed_checkpoints() -> list[dict]:
    out = []
    base = NEW_RUNS_DIR / "adaptive_multiseed"
    for strategy_dir in sorted(base.glob("strategy_*")):
        for seed_dir in sorted(strategy_dir.glob("seed_*")):
            weights = seed_dir / "weights.pt"
            config = seed_dir / "config.json"
            if weights.exists() and config.exists():
                seed = json.loads(config.read_text())["seed"]
                out.append({"source": "adaptive_multiseed", "group": strategy_dir.name.replace("strategy_", ""), "seed": seed,
                            "weights_path": weights, "config_path": config})
    return out


def main() -> pd.DataFrame:
    checkpoints = _stefan_m18_checkpoints() + _adaptive_multiseed_checkpoints()
    print(f"Found {len(checkpoints)} existing checkpoints to re-evaluate (no retraining).", flush=True)

    rows = []
    for ckpt in checkpoints:
        problem, triple = load_frozen_triple_from_run(ckpt["config_path"], ckpt["weights_path"])
        free_boundary_fn = getattr(problem, "free_boundary_exact", None)
        if free_boundary_fn is None:
            continue  # American put has no closed-form free boundary -- skip, per module docstring convention
        sweep_rows = compute_free_boundary_error_sensitivity_sweep(
            problem, triple, free_boundary_fn,
            alpha_u_values=ALPHA_VALUES, alpha_lambda_values=ALPHA_VALUES,
            n_t=N_T, n_scan=N_SCAN,
        )
        for r in sweep_rows:
            rows.append({"experiment": ckpt["source"], "group": ckpt["group"], "seed": ckpt["seed"], **r})
        print(f"  {ckpt['source']}/{ckpt['group']}/seed={ckpt['seed']}: done ({len(sweep_rows)} threshold combos)", flush=True)

    df = pd.DataFrame.from_records(rows)
    out_dir = NEW_RUNS_DIR / EXPERIMENT
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "free_boundary_threshold_sensitivity.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nfree_boundary_threshold_sensitivity.csv -> {csv_path} ({len(df)} rows)")

    _make_figure(df, out_dir)
    _write_ranking_check(df, out_dir)
    return df


def _make_figure(df: pd.DataFrame, out_dir: Path) -> None:
    diag = df[df["alpha_u"] == df["alpha_lambda"]].copy()  # "vary them together" convention (NEW_RUNS.md 4.2 option 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, source, title in [
        (axes[0], "stefan_m18_architecture", "Stefan architecture study (existing checkpoints)"),
        (axes[1], "adaptive_multiseed", "Adaptive multi-seed (Experiment 1 checkpoints)"),
    ]:
        sub = diag[diag["experiment"] == source]
        summary = sub.groupby(["group", "alpha_u"])["E_FB_gap_and_multiplier"].mean().reset_index()
        for group, g in summary.groupby("group"):
            g = g.sort_values("alpha_u")
            ax.plot(g["alpha_u"], g["E_FB_gap_and_multiplier"], marker="o", label=group)
        ax.set_xlabel("alpha_u = alpha_lambda")
        ax.set_ylabel("E_FB_gap_and_multiplier (mean over seeds)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_xticks(ALPHA_VALUES)

    fig.suptitle("Experiment 4 -- free-boundary error vs. threshold multiplier")
    fig.tight_layout()
    save_figure(fig, out_dir / "figures" / "fig_free_boundary_threshold_sensitivity")
    plt.close(fig)
    print(f"fig_free_boundary_threshold_sensitivity.{{pdf,png}} -> {out_dir / 'figures'}")


def _write_ranking_check(df: pd.DataFrame, out_dir: Path) -> None:
    diag = df[df["alpha_u"] == df["alpha_lambda"]].copy()
    lines = ["# Experiment 4 -- ranking stability across thresholds\n"]
    for source in ["stefan_m18_architecture", "adaptive_multiseed"]:
        sub = diag[diag["experiment"] == source]
        lines.append(f"\n## {source}\n")
        rankings = {}
        for alpha in ALPHA_VALUES:
            means = sub[sub["alpha_u"] == alpha].groupby("group")["E_FB_gap_and_multiplier"].mean().sort_values()
            rankings[alpha] = list(means.index)
            lines.append(f"alpha={alpha}: ranking (best/lowest E_FB first) = {list(means.index)}\n")
        stable = len(set(tuple(r) for r in rankings.values())) == 1
        lines.append(f"\n**Ranking stable across alpha in {ALPHA_VALUES}: {stable}**\n")
    (out_dir / "RANKING_STABILITY.md").write_text("".join(lines))
    print(f"RANKING_STABILITY.md -> {out_dir / 'RANKING_STABILITY.md'}")


if __name__ == "__main__":
    main()
