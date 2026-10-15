"""One-Stefan-run orchestration built on experiments/new_runs/common.py.

Mirrors experiments/stefan_core.py's own train-in-process /
certify-and-true-error-via-subprocess split exactly (same functions, same
CLI flags) -- the only generalization is that the caller supplies an
already-sampled (t, x, x0) (so Experiment 1's per-strategy sampling logic
lives in that experiment's own script, never here) and an `extra_config`
dict merged into config.json (so `strategy`, `gamma`, etc. are recorded
without this module needing to know about every experiment's own knobs).

Used directly by the Phase 1 smoke test (drastically reduced steps, uniform
sampling, written to a _smoke/ directory) and intended for reuse by
Experiment 1 (adaptive_multiseed) once Phase 2 starts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from certified_obstacle_nn.experiments.new_runs.common import (
    build_master_row,
    certify_run,
    compute_residual_rms,
    extract_spectral_norms,
    load_frozen_triple_from_run,
    true_error_run,
    write_row_csv,
    write_status,
)
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.stefan import StefanProblem
from certified_obstacle_nn.training.trainer import train


def train_and_evaluate_stefan_run(
    run_dir: Path,
    *,
    width: int,
    depth: int,
    seed: int,
    t: torch.Tensor,
    x: torch.Tensor,
    x0: torch.Tensor,
    n_interior: int,
    n_initial: int,
    n_adam_epochs: int,
    adam_lr: float = 1e-3,
    n_lbfgs_epochs: int = 0,
    lbfgs_lr: float = 1.0,
    loss_weights: dict | None = None,
    cert_grid: int = 128,
    n_qp: int = 4,
    option_b_mode: str = "min_apriori_and_partition",
    fb_n_t: int = 400,
    true_error_n_t: int = 60,
    true_error_n_x: int = 60,
    experiment: str,
    subexperiment: str | None = None,
    strategy: str | None = None,
    regularization_type: str | None = None,
    regularization_strength: float | None = None,
    gamma: float | None = None,
    extra_config: dict | None = None,
) -> dict:
    loss_weights = loss_weights or {}
    architecture_tag = f"{depth}x{width}"
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    problem = StefanProblem()
    triple = NeuralTriple(problem, width=width, depth=depth)

    history_csv = run_dir / "history.csv"
    weights_path = run_dir / "weights.pt"
    config_path = run_dir / "config.json"

    try:
        wall_start = time.perf_counter()
        history, outcome = train(
            problem, triple, t, x, x0,
            n_epochs=n_adam_epochs, lr=adam_lr,
            n_lbfgs_epochs=n_lbfgs_epochs, lbfgs_lr=lbfgs_lr,
            csv_path=str(history_csv), **loss_weights,
        )
        training_time = time.perf_counter() - wall_start
        torch.save(triple.state_dict(), weights_path)

        config = {
            "problem": "stefan",
            "seed": seed,
            "architecture": {"width": width, "depth": depth, "multiplier_head": triple.multiplier_net.head},
            "optimizer_schedule": {
                "adam": {"lr": adam_lr, "n_epochs": n_adam_epochs},
                "lbfgs": {
                    "lr": lbfgs_lr, "n_epochs_requested": n_lbfgs_epochs, "line_search": "strong_wolfe",
                    "outcome_reason": outcome.reason, "n_steps_completed": outcome.n_steps_completed,
                },
            },
            "loss_weights": {"w_r": 1.0, "w_s": 1.0, "w_0": 1.0, "w_p": 1.0, "w_c": 1.0, **loss_weights},
            "grid_sizes": {"n_interior": n_interior, "n_initial": n_initial},
            "certification_grid": cert_grid,
            "n_qp": n_qp,
            "option_b_mode": option_b_mode,
            "training_time_seconds": training_time,
            "weights_path": str(weights_path),
            "best_weights_path": str(weights_path),
            "best_checkpoint_criterion": (
                "final state after the full Adam+L-BFGS schedule -- training/trainer.py's train() "
                "has no intermediate/early-stopping checkpoint selection anywhere in this codebase, "
                "so best==final by construction, not a new rule invented for this task"
            ),
            "history_csv_path": str(history_csv),
            "experiment": experiment,
            "subexperiment": subexperiment,
            "strategy": strategy,
            "regularization_type": regularization_type,
            "regularization_strength": regularization_strength,
            "gamma": gamma,
            **(extra_config or {}),
        }
        config_path.write_text(json.dumps(config, indent=2))

        cert_wall, cert_payload = certify_run(
            weights_path, config_path, run_dir / "certification",
            grid=cert_grid, n_qp=n_qp, option_b_mode=option_b_mode,
        )
        te_wall, te_payload = true_error_run(
            weights_path, config_path, run_dir / "true_error",
            n_t=true_error_n_t, n_x=true_error_n_x, fb_n_t=fb_n_t,
        )

        _, frozen_triple = load_frozen_triple_from_run(config_path, weights_path)
        residual_rms = compute_residual_rms(frozen_triple, problem, t, x, x0)
        spectral = extract_spectral_norms(frozen_triple)

        row = build_master_row(
            experiment=experiment, subexperiment=subexperiment, strategy=strategy,
            seed=seed, architecture_tag=architecture_tag, width=width, depth=depth,
            n_parameters=sum(p.numel() for p in triple.parameters()),
            regularization_type=regularization_type, regularization_strength=regularization_strength, gamma=gamma,
            n_adam_epochs=n_adam_epochs, n_lbfgs_epochs=n_lbfgs_epochs,
            n_interior=n_interior, n_initial=n_initial, adam_lr=adam_lr, lbfgs_lr=lbfgs_lr,
            certification_grid=cert_grid, training_time_seconds=training_time,
            final_loss=float(history["loss_total"].iloc[-1]),
            cert_payload=cert_payload, te_payload=te_payload, residual_rms=residual_rms,
            weights_path=str(weights_path), config_path=str(config_path),
            status="ok", notes="",
        )
        row.update(spectral)
        write_row_csv(run_dir, row)
        write_status(run_dir, "ok")
        return row

    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any failure must be RECORDED, not silently propagated and left undocumented
        row = build_master_row(
            experiment=experiment, subexperiment=subexperiment, strategy=strategy,
            seed=seed, architecture_tag=architecture_tag, width=width, depth=depth,
            n_parameters=sum(p.numel() for p in triple.parameters()),
            regularization_type=regularization_type, regularization_strength=regularization_strength, gamma=gamma,
            n_adam_epochs=n_adam_epochs, n_lbfgs_epochs=n_lbfgs_epochs,
            n_interior=n_interior, n_initial=n_initial, adam_lr=adam_lr, lbfgs_lr=lbfgs_lr,
            certification_grid=cert_grid, training_time_seconds=float("nan"),
            final_loss=float("nan"),
            weights_path=str(weights_path) if weights_path.exists() else None, config_path=str(config_path) if config_path.exists() else None,
            status="failed", notes=f"{type(exc).__name__}: {exc}",
        )
        write_row_csv(run_dir, row)
        write_status(run_dir, "failed", notes=f"{type(exc).__name__}: {exc}")
        return row
