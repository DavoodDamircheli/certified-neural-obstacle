"""NEW_RUNS.md Experiment 1 -- adaptive vs. uniform sampling, over 5 seeds
per strategy (REQUIRED priority).

Every constant that defines "the existing adaptive-vs-uniform experiment" --
WIDTH, DEPTH, N_INTERIOR, N_INITIAL, ADAM_LR, LBFGS_LR, CERT_GRID, N_QP,
TRUE_ERROR_N_T/X, OPTION_B_MODE, ADAM/LBFGS_EPOCHS_PER_ROUND, L_MAX,
DORFLER_THETA, and the six-entry METHODS list itself -- is IMPORTED directly
from experiments/stefan_adaptive.py (read-only; that module's own
`if __name__ == "__main__"` guard means importing it here never re-runs its
existing 1-seed sweep or touches results/stefan_adaptive/). This guarantees
byte-identical values rather than a hand-copied set that could silently
drift, satisfying NEW_RUNS.md section 1.1 ("Keep the Existing Experimental
Setup Fixed") by construction.

_run_uniform / _run_adaptive below are the same two functions from
stefan_adaptive.py, generalized to take `seed` as a parameter instead of a
module-level constant, and to write into
results/new_runs/adaptive_multiseed/strategy_<name>/seed_<n>/ instead of
results/stefan_adaptive/<name>/ (existing results/stefan_adaptive/ is never
touched). Everything else -- the training call, the adaptive loop call, the
resulting collocation-point counts and epoch totals -- is identical code to
the existing experiment.

final_loss/best_loss are computed post-hoc via
common.compute_diagnostic_loss, since training/adaptive.py's run_adaptive_loop
does not expose a literal "final training loss" the way a single
trainer.train() call does (see that function's own docstring for why: a
fixed out-of-sample diagnostic loss, comparable across every strategy, is
used instead of two different in-sample definitions per arm).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from certified_obstacle_nn.certification.partition import build_partition
from certified_obstacle_nn.experiments.new_runs.common import (
    NEW_RUNS_DIR,
    aggregate_master_csv,
    build_master_row,
    certify_run,
    check_matched_configs,
    check_no_nan_inf,
    compute_diagnostic_loss,
    compute_residual_rms,
    extract_spectral_norms,
    is_run_complete,
    load_frozen_triple_from_run,
    new_run_dir,
    save_figure,
    true_error_run,
    write_row_csv,
    write_status,
)
from certified_obstacle_nn.experiments.stefan_adaptive import (
    ADAM_EPOCHS_PER_ROUND,
    ADAM_LR,
    CERT_GRID,
    DEPTH,
    L_MAX,
    LBFGS_EPOCHS_PER_ROUND,
    LBFGS_LR,
    METHODS,
    N_INITIAL,
    N_INTERIOR,
    N_QP,
    OPTION_B_MODE,
    TOTAL_ADAM_EPOCHS,
    TOTAL_LBFGS_EPOCHS,
    TRUE_ERROR_N_X,
    TRUE_ERROR_N_T,
    WIDTH,
)
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.stefan import StefanProblem
from certified_obstacle_nn.training.adaptive import run_adaptive_loop
from certified_obstacle_nn.training.trainer import train

EXPERIMENT = "adaptive_multiseed"
SEEDS = [0, 1, 2, 3, 4]  # Phase 0: confirmed canonical set (stefan_core.py, multiplier_ablation.py)
LOSS_WEIGHTS = {"w_r": 1.0, "w_s": 1.0, "w_0": 1.0, "w_p": 1.0, "w_c": 1.0}  # specification section 3 debug default; unchanged by either script

MATCHED_KEYS = [
    "architecture.width", "architecture.depth", "n_points",
    "total_adam_epochs", "total_lbfgs_epochs", "certification_grid",
    "optimizer_schedule", "loss_weights",
]

FIXED_CLOUD_SEED = 0  # Plot D: NOT chosen for visual appeal -- the first/canonical seed in SEEDS, used consistently everywhere else in this file.


def _base_config(seed: int) -> dict:
    return {
        "certification_grid": CERT_GRID,
        "optimizer_schedule": {"adam": {"lr": ADAM_LR}, "lbfgs": {"lr": LBFGS_LR}},
        "loss_weights": dict(LOSS_WEIGHTS),
    }


def _run_uniform(run_dir: Path, seed: int) -> dict:
    torch.manual_seed(seed)
    problem = StefanProblem()
    triple = NeuralTriple(problem, width=WIDTH, depth=DEPTH)

    generator = torch.Generator().manual_seed(seed)
    a, b = problem.omega_bounds
    T = problem.T
    t = (torch.rand(N_INTERIOR, 1, generator=generator, dtype=torch.float64) * T).requires_grad_(True)
    x = (torch.rand(N_INTERIOR, 1, generator=generator, dtype=torch.float64) * (b - a) + a).requires_grad_(True)
    x0 = torch.rand(N_INITIAL, 1, generator=generator, dtype=torch.float64) * (b - a) + a
    assert t.shape[0] == N_INTERIOR, "N_total_uniform must equal N_INTERIOR"

    wall_start = time.perf_counter()
    _, outcome = train(
        problem, triple, t, x, x0,
        n_epochs=TOTAL_ADAM_EPOCHS, lr=ADAM_LR,
        n_lbfgs_epochs=TOTAL_LBFGS_EPOCHS, lbfgs_lr=LBFGS_LR, **LOSS_WEIGHTS,
    )
    training_time = time.perf_counter() - wall_start

    run_dir.mkdir(parents=True, exist_ok=True)
    weights_path = run_dir / "weights.pt"
    config_path = run_dir / "config.json"
    torch.save(triple.state_dict(), weights_path)
    config = {
        "problem": "stefan", "seed": seed,
        "architecture": {"width": WIDTH, "depth": DEPTH, "multiplier_head": triple.multiplier_net.head},
        "method": "uniform", "n_points": N_INTERIOR,
        "total_adam_epochs": TOTAL_ADAM_EPOCHS, "total_lbfgs_epochs": TOTAL_LBFGS_EPOCHS,
        "training_time_seconds": training_time, "lbfgs_outcome_reason": outcome.reason,
        **_base_config(seed),
    }
    config_path.write_text(json.dumps(config, indent=2))

    return {
        "method": "uniform", "n_points": N_INTERIOR, "weights_path": weights_path, "config_path": config_path,
        "training_time": training_time, "config": config,
        "t0": t.detach().clone(), "x0": x.detach().clone(), "t_final": t.detach().clone(), "x_final": x.detach().clone(),
    }


def _run_adaptive(spec: dict, run_dir: Path, seed: int) -> dict:
    torch.manual_seed(seed)
    problem = StefanProblem()
    triple = NeuralTriple(problem, width=WIDTH, depth=DEPTH)
    partition = build_partition(problem, n_t=CERT_GRID, n_x=CERT_GRID)

    final_triple, history = run_adaptive_loop(
        problem, triple, partition,
        method=spec["marking_method"], theta=spec.get("theta", 0.5), q=spec.get("q", 1.0),
        mark_on=spec["mark_on"],
        n_interior=N_INTERIOR, n_initial=N_INITIAL, l_max=L_MAX, seed=seed,
        n_adam_epochs=ADAM_EPOCHS_PER_ROUND, adam_lr=ADAM_LR,
        n_lbfgs_epochs=LBFGS_EPOCHS_PER_ROUND, lbfgs_lr=LBFGS_LR,
        n_qp=N_QP,
    )
    assert history[-1].t_points.shape[0] == N_INTERIOR, "N_total_adaptive must equal N_INTERIOR at every round"
    total_training_time = sum(r.training_time for r in history)

    run_dir.mkdir(parents=True, exist_ok=True)
    weights_path = run_dir / "weights.pt"
    config_path = run_dir / "config.json"
    torch.save(final_triple.state_dict(), weights_path)
    config = {
        "problem": "stefan", "seed": seed,
        "architecture": {"width": WIDTH, "depth": DEPTH, "multiplier_head": final_triple.multiplier_net.head},
        "method": spec["name"], "marking_method": spec["marking_method"], "mark_on": spec["mark_on"],
        "theta": spec.get("theta"), "q": spec.get("q"), "n_points": N_INTERIOR, "l_max": L_MAX,
        "adam_epochs_per_round": ADAM_EPOCHS_PER_ROUND, "lbfgs_epochs_per_round": LBFGS_EPOCHS_PER_ROUND,
        "total_adam_epochs": (L_MAX + 1) * ADAM_EPOCHS_PER_ROUND, "total_lbfgs_epochs": (L_MAX + 1) * LBFGS_EPOCHS_PER_ROUND,
        "training_time_seconds": total_training_time,
        "n_marked_per_round": [r.n_marked for r in history],
        "rho_tilde_estimate_per_round": [r.rho_tilde_estimate for r in history],
        **_base_config(seed),
    }
    config_path.write_text(json.dumps(config, indent=2))

    return {
        "method": spec["name"], "n_points": N_INTERIOR, "weights_path": weights_path, "config_path": config_path,
        "training_time": total_training_time, "config": config,
        "t0": history[0].t_points, "x0": history[0].x_points, "t_final": history[-1].t_points, "x_final": history[-1].x_points,
    }


def _run_one(spec: dict, seed: int) -> tuple[dict, dict]:
    run_dir = new_run_dir(EXPERIMENT, f"strategy_{spec['name']}", f"seed_{seed}")
    if is_run_complete(run_dir):
        import pandas as pd
        row = pd.read_csv(run_dir / "summary.csv").iloc[0].to_dict()
        config = json.loads((run_dir / "config.json").read_text())
        cloud_path = run_dir / "cloud.pt"
        cloud = torch.load(cloud_path, weights_only=True) if cloud_path.exists() else None
        print(f"  [resume] {spec['name']} seed={seed} already complete, loading from disk", flush=True)
        return row, config, cloud

    try:
        run_info = _run_uniform(run_dir, seed) if spec["kind"] == "uniform" else _run_adaptive(spec, run_dir, seed)

        cert_wall, cert_payload = certify_run(
            run_info["weights_path"], run_info["config_path"], run_dir / "certification",
            grid=CERT_GRID, n_qp=N_QP, option_b_mode=OPTION_B_MODE,
        )
        te_wall, te_payload = true_error_run(
            run_info["weights_path"], run_info["config_path"], run_dir / "true_error",
            n_t=TRUE_ERROR_N_T, n_x=TRUE_ERROR_N_X,
        )

        problem, frozen_triple = load_frozen_triple_from_run(run_info["config_path"], run_info["weights_path"])
        # NOTE: run_info["t0"]/["x0"] are the ITERATION-0 interior collocation
        # cloud (Figure-7/Plot-D plotting convention inherited verbatim from
        # stefan_adaptive.py), NOT the initial-time (t=0) sample -- that
        # sample is internal to _run_uniform/run_adaptive_loop and never
        # returned. A fresh, fixed-seed initial-time sample is drawn here
        # instead, purely for this post-hoc residual diagnostic.
        a_, b_ = problem.omega_bounds
        x0_diag = torch.rand(N_INITIAL, 1, generator=torch.Generator().manual_seed(999_999), dtype=torch.float64) * (b_ - a_) + a_
        residual_rms = compute_residual_rms(frozen_triple, problem, run_info["t_final"], run_info["x_final"], x0_diag)
        spectral = extract_spectral_norms(frozen_triple)
        diagnostic_loss = compute_diagnostic_loss(frozen_triple, problem, loss_weights=LOSS_WEIGHTS)

        n_parameters = sum(p.numel() for p in NeuralTriple(problem, width=WIDTH, depth=DEPTH).parameters())

        row = build_master_row(
            experiment=EXPERIMENT, subexperiment=None, strategy=spec["name"], seed=seed,
            architecture_tag=f"{DEPTH}x{WIDTH}", width=WIDTH, depth=DEPTH, n_parameters=n_parameters,
            n_adam_epochs=run_info["config"]["total_adam_epochs"], n_lbfgs_epochs=run_info["config"]["total_lbfgs_epochs"],
            n_interior=N_INTERIOR, n_initial=N_INITIAL, adam_lr=ADAM_LR, lbfgs_lr=LBFGS_LR,
            certification_grid=CERT_GRID, training_time_seconds=run_info["training_time"],
            final_loss=diagnostic_loss, best_loss=diagnostic_loss,
            cert_payload=cert_payload, te_payload=te_payload, residual_rms=residual_rms,
            weights_path=str(run_info["weights_path"]), config_path=str(run_info["config_path"]),
            status="ok", notes="final_loss/best_loss: post-hoc out-of-sample diagnostic loss, not literal in-sample training loss (see module docstring)",
        )
        row.update(spectral)
        write_row_csv(run_dir, row)
        write_status(run_dir, "ok")
        check_no_nan_inf(row, label=f"{spec['name']}/seed{seed}")

        cloud = None
        if seed == FIXED_CLOUD_SEED:
            cloud = {"t0": run_info["t0"], "x0": run_info["x0"], "t_final": run_info["t_final"], "x_final": run_info["x_final"]}
            torch.save(cloud, run_dir / "cloud.pt")
        return row, run_info["config"], cloud

    except Exception as exc:  # noqa: BLE001 -- must be recorded, never silently propagated
        row = build_master_row(
            experiment=EXPERIMENT, subexperiment=None, strategy=spec["name"], seed=seed,
            architecture_tag=f"{DEPTH}x{WIDTH}", width=WIDTH, depth=DEPTH, n_parameters=0,
            n_adam_epochs=TOTAL_ADAM_EPOCHS, n_lbfgs_epochs=TOTAL_LBFGS_EPOCHS,
            n_interior=N_INTERIOR, n_initial=N_INITIAL, adam_lr=ADAM_LR, lbfgs_lr=LBFGS_LR,
            certification_grid=CERT_GRID, training_time_seconds=float("nan"), final_loss=float("nan"),
            status="failed", notes=f"{type(exc).__name__}: {exc}",
        )
        write_row_csv(run_dir, row)
        write_status(run_dir, "failed", notes=f"{type(exc).__name__}: {exc}")
        print(f"  *** FAILED: {spec['name']} seed={seed}: {type(exc).__name__}: {exc}", flush=True)
        return row, {}, None


def _summary_table(df) -> "pd.DataFrame":
    import pandas as pd
    metrics = ["E_true", "rho_tilde_h", "E_cert", "E_cert_U", "E_FB", "training_time_seconds"]
    ok = df[df["status"] == "ok"]
    rows = []
    for strategy, g in ok.groupby("strategy", sort=False):
        row = {"strategy": strategy, "n_seeds": len(g)}
        for m in metrics:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std"] = g[m].std()
            row[f"{m}_min"] = g[m].min()
            row[f"{m}_max"] = g[m].max()
            row[f"{m}_median"] = g[m].median()
        rows.append(row)
    order = [spec["name"] for spec in METHODS]
    out = pd.DataFrame.from_records(rows)
    out["strategy"] = pd.Categorical(out["strategy"], categories=order, ordered=True)
    return out.sort_values("strategy").reset_index(drop=True)


def _fig_by_strategy(df, column: str, title: str, ylabel: str) -> plt.Figure:
    import pandas as pd
    order = [spec["name"] for spec in METHODS]
    ok = df[df["status"] == "ok"]
    data = [ok[ok["strategy"] == s][column].dropna().to_numpy() for s in order]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, tick_labels=order, showmeans=True)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def _fig_clouds(clouds: dict, best_adaptive_name: str) -> plt.Figure:
    problem = StefanProblem()
    t_line = torch.linspace(0.0, problem.T, 200, dtype=torch.float64)
    x_line = problem.free_boundary_exact(t_line)

    panels = [("uniform", "uniform"), (best_adaptive_name, f"best adaptive ({best_adaptive_name})"),
              ("dorfler_total", "dorfler_total"), ("probabilistic_q2", "probabilistic_q2")]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), sharex=True, sharey=True)
    for ax, (name, title) in zip(axes, panels):
        cloud = clouds[name]
        ax.scatter(cloud["x_final"].numpy(), cloud["t_final"].numpy(), s=2, alpha=0.4, color="tab:blue")
        ax.plot(x_line.numpy(), t_line.numpy(), color="red", linewidth=1.5, label="exact free boundary t=x")
        ax.set_xlabel("x")
        ax.set_title(f"{title}\n(final collocation cloud, seed={FIXED_CLOUD_SEED})")
        ax.legend(loc="upper right", fontsize=7)
    axes[0].set_ylabel("t")
    fig.suptitle(f"Experiment 1 Plot D -- representative collocation clouds (fixed seed={FIXED_CLOUD_SEED}, not cherry-picked)")
    fig.tight_layout()
    return fig


def main() -> None:
    exp_dir = new_run_dir(EXPERIMENT)
    rows: list[dict] = []
    configs_by_seed: dict[int, list[dict]] = {seed: [] for seed in SEEDS}
    clouds: dict[str, dict] = {}

    for seed in SEEDS:
        print(f"\n########## seed={seed} ##########", flush=True)
        for spec in METHODS:
            print(f"=== {spec['name']} (seed={seed}) ===", flush=True)
            result = _run_one(spec, seed)
            row, config = result[0], result[1]
            cloud = result[2] if len(result) > 2 else None
            rows.append(row)
            if config:
                configs_by_seed[seed].append(config)
            if cloud is not None:
                clouds[spec["name"]] = cloud
            print(f"  status={row['status']} E_true={row.get('E_true')} E_cert_U={row.get('E_cert_U')} "
                  f"training_time={row.get('training_time_seconds')}", flush=True)

        if len(configs_by_seed[seed]) == len(METHODS):
            check_matched_configs(configs_by_seed[seed], MATCHED_KEYS, label=f"seed={seed}, all 6 strategies")
            print(f"  [check] matched-config assertion passed across all 6 strategies at seed={seed}", flush=True)

    import pandas as pd
    all_runs_path = exp_dir / "adaptive_multiseed_all_runs.csv"
    df = aggregate_master_csv(rows, out_path=all_runs_path)
    print(f"\nadaptive_multiseed_all_runs.csv -> {all_runs_path} ({len(df)} rows)")

    summary = _summary_table(df)
    summary_path = exp_dir / "adaptive_multiseed_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"adaptive_multiseed_summary.csv -> {summary_path}")
    print(summary[["strategy", "n_seeds", "E_true_mean", "E_true_std", "E_cert_U_mean", "E_cert_U_std"]].to_string(index=False))

    figures_dir = exp_dir / "figures"
    ok_df = df[df["status"] == "ok"]
    adaptive_means = summary[summary["strategy"] != "uniform"][["strategy", "E_true_mean"]].set_index("strategy")["E_true_mean"]
    best_adaptive_name = adaptive_means.idxmin() if len(adaptive_means) else "dorfler_total"

    fig_a = _fig_by_strategy(df, "E_true", "Plot A -- true error by sampling strategy", "E_true")
    save_figure(fig_a, figures_dir / "fig_adaptive_true_error")
    plt.close(fig_a)

    fig_b = _fig_by_strategy(df, "E_cert_U", "Plot B -- certified bound (E_cert_U) by sampling strategy", "E_cert_U")
    save_figure(fig_b, figures_dir / "fig_adaptive_certificate")
    plt.close(fig_b)

    fig_c = _fig_by_strategy(df, "E_FB", "Plot C -- free-boundary error by sampling strategy", "E_FB")
    save_figure(fig_c, figures_dir / "fig_adaptive_free_boundary_error")
    plt.close(fig_c)

    if len(clouds) == len(METHODS):
        fig_d = _fig_clouds(clouds, best_adaptive_name)
        save_figure(fig_d, figures_dir / "fig_adaptive_clouds")
        plt.close(fig_d)
        print(f"fig_adaptive_clouds -> {figures_dir / 'fig_adaptive_clouds.pdf'}")
    else:
        print(f"fig_adaptive_clouds SKIPPED -- only {len(clouds)}/{len(METHODS)} seed={FIXED_CLOUD_SEED} clouds available (resumed run?)")

    print(f"\nbest adaptive strategy by mean E_true: {best_adaptive_name}")
    print("\nDone. See INTERPRETATION.md (written separately) for Outcome A/B/C.")


if __name__ == "__main__":
    main()
