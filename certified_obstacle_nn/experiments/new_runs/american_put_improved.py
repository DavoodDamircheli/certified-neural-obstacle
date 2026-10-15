"""NEW_RUNS.md Experiment 2 (sections 2.2-2.9) -- improved American put.

Baseline being improved on: experiments/american_put_baseline.py (4x64, seed
0, plain uniform TrainingSampler, single train() call) -- results/american_put_baseline/,
E_true=3.69, E_cert_U=5.95e7, flagged in the Phase 0 report as not yet a
success case. That run and its results directory are untouched.

Section 2.2 (normalize inputs/outputs): ALREADY satisfied by existing code
(problems/american_put.py's xi -> [-1,1], NormalizedAmericanPutProblem ->
ubar/gbar/fbar by K, both already wired into american_put_baseline.py and
covered by tests/test_x_rescale_chain_rule.py) -- nothing new needed, per the
Phase 0 report.

Section 2.3 (payoff kink) + 2.4 (learned contact set): new,
experiments/new_runs/american_put_sampler.py -- two-phase training per run:
  Phase 1: sample_interior_with_kink_band (static stratified sample, a fixed
           30% of the interior budget drawn from a narrow band around
           x_star=log(K)), trained with training/trainer.py's train()
           directly so the FULL per-epoch loss history is logged
           (losses_phase1.csv) -- satisfies section 2.6 for the bulk of the
           training budget.
  Phase 2: run_contact_set_adaptive (training/adaptive.py's own
           run_adaptive_loop, Dorfler marking on rho_penetration_K -- a
           function of the network's OWN current obstacle-violation gap,
           never the reference), L_MAX+1 short warm-started rounds refining
           collocation density near wherever the network CURRENTLY thinks
           the contact set is. Per-round aggregate rho_tilde_estimate is
           recorded (same limitation as training/adaptive.py's existing
           Stefan experiments: no per-EPOCH loss csv during these rounds,
           only per-round summaries) -- documented, not silently dropped.

Section 2.5: architecture study over {4x64, 6x64, 6x128}, one exploratory
seed (0), before committing to a final multi-seed configuration.

Certification/E_true settings (CERT_GRID, N_QP, OPTION_B_MODE, TRUE_ERROR_N,
TRUE_ERROR_TOLERANCE) are inherited UNCHANGED from american_put_baseline.py,
so results remain comparable to the flagged baseline.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from certified_obstacle_nn.certification.partition import build_partition
from certified_obstacle_nn.errors.exercise_boundary import extract_neural_exercise_boundary
from certified_obstacle_nn.experiments.new_runs.american_put_sampler import (
    BAND_FRACTION,
    BAND_HALFWIDTH,
    run_contact_set_adaptive,
    sample_interior_with_kink_band,
)
from certified_obstacle_nn.experiments.new_runs.common import (
    build_master_row,
    certify_run,
    check_no_nan_inf,
    compute_diagnostic_loss,
    compute_residual_rms,
    extract_spectral_norms,
    is_run_complete,
    load_frozen_triple_from_run,
    new_run_dir,
    true_error_run,
    write_row_csv,
    write_status,
)
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.american_put import AmericanPutProblem, NormalizedAmericanPutProblem, verify_boundary_and_initial_conditions
from certified_obstacle_nn.reference.american_put_reference import FROZEN_DIR, load_reference
from certified_obstacle_nn.training.trainer import train

EXPERIMENT = "american_put_improved"

# Inherited unchanged from experiments/american_put_baseline.py -- keeps this
# experiment's numbers comparable to the flagged baseline run.
CERT_GRID = 256
OPTION_B_MODE = "min_apriori_and_partition"
N_QP = 4
TRUE_ERROR_N = 320
TRUE_ERROR_TOLERANCE = 1e-3
N_INTERIOR = 5000
N_INITIAL = 300
ADAM_LR = 1e-3
LBFGS_LR = 1.0
LOSS_WEIGHTS = {"w_r": 1.0, "w_s": 1.0, "w_0": 1.0, "w_p": 1.0, "w_c": 1.0}

# Phase 1 (static kink-band sample, full per-epoch loss logging).
PHASE1_ADAM_EPOCHS = 4000
PHASE1_LBFGS_EPOCHS = 300

# Phase 2 (contact-set adaptive refinement, warm-started from Phase 1).
# ADAPTIVE_MARKING_GRID is deliberately smaller than the official CERT_GRID=256
# -- an INFORMAL per-round estimate to decide resampling, not the final
# certified quantity (same distinction training/adaptive.py's own docstring
# draws for Stefan); running the full 256-grid raw estimator 4 extra times
# per seed would be needlessly expensive for a per-round marking signal.
ADAPTIVE_L_MAX = 3
ADAPTIVE_ADAM_EPOCHS_PER_ROUND = 250
ADAPTIVE_LBFGS_EPOCHS_PER_ROUND = 20
ADAPTIVE_MARKING_GRID = 64
ADAPTIVE_THETA = 0.5
ADAPTIVE_ALPHA_MARKED_FRACTION = 0.7

ARCHITECTURES = [
    {"tag": "4x64", "width": 64, "depth": 4},
    {"tag": "6x64", "width": 64, "depth": 6},
    {"tag": "6x128", "width": 128, "depth": 6},
]
EXPLORATORY_SEED = 0
FINAL_SEEDS = [0, 1, 2, 3, 4]

S_POINTS = {"S70_below_strike": 70.0, "S100_near_strike": 100.0, "S130_above_strike": 130.0}
TAU_SLICES = [0.0, 0.25, 0.5, 0.75, 1.0]


def _u_lift(problem: AmericanPutProblem, x) -> "np.ndarray | torch.Tensor":
    return problem.C * (problem.R - x) / (problem.R - problem.L)


def _train_one_run(width: int, depth: int, seed: int, run_dir: Path) -> dict:
    problem = AmericanPutProblem()
    verify_boundary_and_initial_conditions(problem)
    normalized_problem = NormalizedAmericanPutProblem()

    torch.manual_seed(seed)
    triple = NeuralTriple(problem, width=width, depth=depth, x_rescale=problem.xi)

    t, x, x0 = sample_interior_with_kink_band(normalized_problem, N_INTERIOR, N_INITIAL, seed)
    losses_csv = run_dir / "losses_phase1.csv"
    wall_start = time.perf_counter()
    history1, outcome1 = train(
        normalized_problem, triple, t, x, x0,
        n_epochs=PHASE1_ADAM_EPOCHS, lr=ADAM_LR,
        n_lbfgs_epochs=PHASE1_LBFGS_EPOCHS, lbfgs_lr=LBFGS_LR,
        csv_path=str(losses_csv), **LOSS_WEIGHTS,
    )
    phase1_time = time.perf_counter() - wall_start

    partition = build_partition(normalized_problem, n_t=ADAPTIVE_MARKING_GRID, n_x=ADAPTIVE_MARKING_GRID)
    wall_start = time.perf_counter()
    final_triple, adaptive_history = run_contact_set_adaptive(
        normalized_problem, triple, partition,
        n_interior=N_INTERIOR, n_initial=N_INITIAL, l_max=ADAPTIVE_L_MAX, seed=seed,
        n_adam_epochs=ADAPTIVE_ADAM_EPOCHS_PER_ROUND, adam_lr=ADAM_LR,
        n_lbfgs_epochs=ADAPTIVE_LBFGS_EPOCHS_PER_ROUND, lbfgs_lr=LBFGS_LR,
        n_qp=N_QP, theta=ADAPTIVE_THETA, alpha_marked_fraction=ADAPTIVE_ALPHA_MARKED_FRACTION,
    )
    phase2_time = time.perf_counter() - wall_start
    total_training_time = phase1_time + phase2_time

    run_dir.mkdir(parents=True, exist_ok=True)
    weights_path = run_dir / "weights.pt"
    config_path = run_dir / "config.json"
    torch.save(final_triple.state_dict(), weights_path)

    total_adam_epochs = PHASE1_ADAM_EPOCHS + (ADAPTIVE_L_MAX + 1) * ADAPTIVE_ADAM_EPOCHS_PER_ROUND
    total_lbfgs_epochs = PHASE1_LBFGS_EPOCHS + (ADAPTIVE_L_MAX + 1) * ADAPTIVE_LBFGS_EPOCHS_PER_ROUND

    config = {
        "problem": "american_put", "seed": seed,
        "architecture": {"width": width, "depth": depth, "multiplier_head": final_triple.multiplier_net.head},
        "sampling": {
            "phase1": "kink_band_stratified", "band_fraction": BAND_FRACTION, "band_halfwidth": BAND_HALFWIDTH,
            "phase2": "contact_set_adaptive_dorfler", "mark_on": "rho_penetration_K",
            "theta": ADAPTIVE_THETA, "alpha_marked_fraction": ADAPTIVE_ALPHA_MARKED_FRACTION,
            "l_max": ADAPTIVE_L_MAX, "marking_grid": ADAPTIVE_MARKING_GRID,
        },
        "optimizer_schedule": {
            "adam": {"lr": ADAM_LR}, "lbfgs": {"lr": LBFGS_LR},
            "phase1_adam_epochs": PHASE1_ADAM_EPOCHS, "phase1_lbfgs_epochs": PHASE1_LBFGS_EPOCHS,
            "phase2_adam_epochs_per_round": ADAPTIVE_ADAM_EPOCHS_PER_ROUND,
            "phase2_lbfgs_epochs_per_round": ADAPTIVE_LBFGS_EPOCHS_PER_ROUND,
            "phase2_rounds": ADAPTIVE_L_MAX + 1,
        },
        "total_adam_epochs": total_adam_epochs, "total_lbfgs_epochs": total_lbfgs_epochs,
        "loss_weights": dict(LOSS_WEIGHTS),
        "grid_sizes": {"n_interior": N_INTERIOR, "n_initial": N_INITIAL},
        "certification_grid": CERT_GRID, "n_qp": N_QP, "option_b_mode": OPTION_B_MODE,
        "training_time_seconds": total_training_time,
        "phase1_training_time_seconds": phase1_time, "phase2_training_time_seconds": phase2_time,
        "weights_path": str(weights_path), "best_weights_path": str(weights_path),
        "best_checkpoint_criterion": "final state after phase1+phase2 -- no early-stopping/intermediate checkpoint selection exists anywhere in this codebase",
        "phase1_lbfgs_outcome_reason": outcome1.reason,
        "phase1_final_loss_total": float(history1["loss_total"].iloc[-1]),
        "phase2_n_marked_per_round": [r.n_marked for r in adaptive_history],
        "phase2_rho_tilde_estimate_per_round": [r.rho_tilde_estimate for r in adaptive_history],
    }
    config_path.write_text(json.dumps(config, indent=2))

    return {
        "weights_path": weights_path, "config_path": config_path, "config": config,
        "training_time": total_training_time, "phase1_final_loss": float(history1["loss_total"].iloc[-1]),
    }


def _evaluate_against_reference(problem: AmericanPutProblem, physical_triple, n_grid: int = 80, n_boundary_tau: int = 100) -> dict:
    """Diagnostic-only (not a certified quantity): dense evaluation of the
    trained network's PHYSICAL price V_theta = u_theta + u_lift against the
    frozen PSOR reference's own V_ref, and of the network's OWN learned
    exercise boundary (errors/exercise_boundary.py) against the reference's.
    Never touches training; called only after a run's weights are frozen."""
    reference = load_reference(FROZEN_DIR)
    a, b = problem.omega_bounds
    T = problem.T

    tau_grid = torch.linspace(0.0, T, n_grid, dtype=torch.float64)
    x_grid = torch.linspace(a, b, n_grid, dtype=torch.float64)
    tau_mesh, x_mesh = torch.meshgrid(tau_grid, x_grid, indexing="ij")
    tau_flat, x_flat = tau_mesh.reshape(-1, 1), x_mesh.reshape(-1, 1)

    with torch.no_grad():
        u_neural = physical_triple.state_net(tau_flat, x_flat).reshape(n_grid, n_grid)
    V_neural = (u_neural + _u_lift(problem, x_mesh)).numpy()

    u_ref_np = reference.u
    V_ref_full = u_ref_np + _u_lift(problem, reference.x_grid)[None, :]
    V_ref = np.empty((n_grid, n_grid))
    for i in range(n_grid):
        V_ref[i, :] = _interp_row(reference, V_ref_full, float(tau_grid[i]), x_grid.numpy())

    diff = V_neural - V_ref
    domain_measure = T * (b - a)
    max_price_error = float(np.abs(diff).max())
    L2_price_error = float(np.sqrt(np.mean(diff ** 2) * domain_measure))

    tau_b = np.linspace(0.0, T, n_boundary_tau)
    neural_boundary = extract_neural_exercise_boundary(problem, physical_triple, tau_b)
    ref_boundary_on_tau_b = np.interp(tau_b, reference.tau_grid, reference.exercise_boundary, left=np.nan, right=np.nan)
    valid = ~(np.isnan(neural_boundary) | np.isnan(ref_boundary_on_tau_b))
    if valid.sum() > 0:
        eb_diff = neural_boundary[valid] - ref_boundary_on_tau_b[valid]
        exercise_boundary_error = float(np.sqrt(np.trapezoid(eb_diff ** 2, tau_b[valid]) / max(tau_b[valid].max() - tau_b[valid].min(), 1e-12)))
    else:
        exercise_boundary_error = float("nan")

    return {
        "max_price_error": max_price_error, "L2_price_error": L2_price_error,
        "exercise_boundary_error": exercise_boundary_error,
        "tau_grid": tau_grid.numpy(), "x_grid": x_grid.numpy(), "V_neural": V_neural, "V_ref": V_ref,
        "tau_b": tau_b, "neural_boundary": neural_boundary, "ref_boundary_on_tau_b": ref_boundary_on_tau_b,
        "valid_boundary_fraction": float(valid.mean()),
    }


def _interp_row(reference, V_ref_full, tau_query: float, x_query: np.ndarray) -> np.ndarray:
    """V_ref at one tau (linear in tau across the two bracketing reference
    rows, then linear in x within that row) -- a cheap bilinear stand-in,
    adequate for this diagnostic surface/error plot (NOT used for any
    certified quantity; E_true's own quadrature uses the differentiable
    bicubic reference/american_put_reference.py::build_reference_callables
    machinery instead, unchanged)."""
    row = np.array([np.interp(tau_query, reference.tau_grid, V_ref_full[:, j]) for j in range(V_ref_full.shape[1])])
    return np.interp(x_query, reference.x_grid, row)


def _price_slices(problem: AmericanPutProblem, physical_triple) -> dict:
    reference = load_reference(FROZEN_DIR)
    rows = []
    for tau in TAU_SLICES:
        for label, S in S_POINTS.items():
            x_val = float(np.log(S))
            t_col = torch.tensor([[tau]], dtype=torch.float64)
            x_col = torch.tensor([[x_val]], dtype=torch.float64)
            with torch.no_grad():
                u_neural = physical_triple.state_net(t_col, x_col).item()
            V_neural = u_neural + float(_u_lift(problem, x_val))
            V_ref = _interp_row(reference, reference.u + _u_lift(problem, reference.x_grid)[None, :], tau, np.array([x_val]))[0]
            rows.append({"tau": tau, "S_label": label, "S": S, "V_neural": V_neural, "V_ref": V_ref, "abs_error": abs(V_neural - V_ref)})
    return rows


def _run_one(width: int, depth: int, seed: int, subexperiment: str) -> dict:
    architecture_tag = f"{depth}x{width}"
    run_dir = new_run_dir(EXPERIMENT, subexperiment, architecture_tag, f"seed_{seed}")

    if is_run_complete(run_dir):
        import pandas as pd
        print(f"  [resume] {architecture_tag} seed={seed} already complete, loading from disk", flush=True)
        return pd.read_csv(run_dir / "summary.csv").iloc[0].to_dict()

    try:
        run_info = _train_one_run(width, depth, seed, run_dir)

        cert_wall, cert_payload = certify_run(
            run_info["weights_path"], run_info["config_path"], run_dir / "certification",
            grid=CERT_GRID, n_qp=N_QP, option_b_mode=OPTION_B_MODE,
        )
        te_wall, te_payload = true_error_run(
            run_info["weights_path"], run_info["config_path"], run_dir / "true_error",
            n_t=TRUE_ERROR_N, n_x=TRUE_ERROR_N,
        )
        E_true = te_payload["result"]["E_true"]

        problem, frozen_triple = load_frozen_triple_from_run(run_info["config_path"], run_info["weights_path"])
        residual_rms = compute_residual_rms(frozen_triple, problem, *_diagnostic_points(problem, seed))
        spectral = extract_spectral_norms(frozen_triple)
        diagnostic_loss = compute_diagnostic_loss(frozen_triple, problem, loss_weights=LOSS_WEIGHTS)

        ref_eval = _evaluate_against_reference(problem, frozen_triple)
        np.savez_compressed(
            run_dir / "reference_comparison.npz",
            tau_grid=ref_eval["tau_grid"], x_grid=ref_eval["x_grid"], V_neural=ref_eval["V_neural"], V_ref=ref_eval["V_ref"],
            tau_b=ref_eval["tau_b"], neural_boundary=ref_eval["neural_boundary"], ref_boundary=ref_eval["ref_boundary_on_tau_b"],
        )
        slices = _price_slices(problem, frozen_triple)
        pd_slices = __import__("pandas").DataFrame.from_records(slices)
        pd_slices.to_csv(run_dir / "price_slices.csv", index=False)

        n_parameters = sum(p.numel() for p in NeuralTriple(problem, width=width, depth=depth).parameters())
        row = build_master_row(
            experiment=EXPERIMENT, subexperiment=subexperiment, strategy="kink_band_plus_contact_adaptive", seed=seed,
            architecture_tag=architecture_tag, width=width, depth=depth, n_parameters=n_parameters,
            n_adam_epochs=run_info["config"]["total_adam_epochs"], n_lbfgs_epochs=run_info["config"]["total_lbfgs_epochs"],
            n_interior=N_INTERIOR, n_initial=N_INITIAL, adam_lr=ADAM_LR, lbfgs_lr=LBFGS_LR,
            certification_grid=CERT_GRID, training_time_seconds=run_info["training_time"],
            final_loss=diagnostic_loss, best_loss=diagnostic_loss, E_ref=E_true,
            cert_payload=cert_payload, te_payload=te_payload, residual_rms=residual_rms,
            weights_path=str(run_info["weights_path"]), config_path=str(run_info["config_path"]),
            status="ok",
            notes="E_ref := E_true (compute_true_error against the frozen PSOR reference; this codebase "
                  "uses one field name for both the exact-solution and reference-solution cases). "
                  "final_loss/best_loss: post-hoc out-of-sample diagnostic loss (see common.compute_diagnostic_loss).",
        )
        row.update(spectral)
        row["exercise_boundary_error"] = ref_eval["exercise_boundary_error"]
        row["max_price_error"] = ref_eval["max_price_error"]
        row["L2_price_error"] = ref_eval["L2_price_error"]
        write_row_csv(run_dir, row)
        write_status(run_dir, "ok")
        check_no_nan_inf(row, label=f"{architecture_tag}/seed{seed}")
        return row

    except Exception as exc:  # noqa: BLE001 -- must be recorded, never silently propagated
        row = build_master_row(
            experiment=EXPERIMENT, subexperiment=subexperiment, strategy="kink_band_plus_contact_adaptive", seed=seed,
            architecture_tag=architecture_tag, width=width, depth=depth, n_parameters=0,
            n_adam_epochs=0, n_lbfgs_epochs=0, n_interior=N_INTERIOR, n_initial=N_INITIAL,
            adam_lr=ADAM_LR, lbfgs_lr=LBFGS_LR, certification_grid=CERT_GRID,
            training_time_seconds=float("nan"), final_loss=float("nan"),
            status="failed", notes=f"{type(exc).__name__}: {exc}",
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        write_row_csv(run_dir, row)
        write_status(run_dir, "failed", notes=f"{type(exc).__name__}: {exc}")
        print(f"  *** FAILED: {architecture_tag} seed={seed}: {type(exc).__name__}: {exc}", flush=True)
        return row


def _diagnostic_points(problem: AmericanPutProblem, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a, b = problem.omega_bounds
    generator = torch.Generator().manual_seed(999_999)
    t = torch.rand(2000, 1, generator=generator, dtype=torch.float64) * problem.T
    x = torch.rand(2000, 1, generator=generator, dtype=torch.float64) * (b - a) + a
    x0 = torch.rand(N_INITIAL, 1, generator=generator, dtype=torch.float64) * (b - a) + a
    return t, x, x0


def main() -> None:
    print("########## exploratory architecture study (1 seed) ##########", flush=True)
    exploratory_rows = []
    for arch in ARCHITECTURES:
        print(f"=== {arch['tag']} (seed={EXPLORATORY_SEED}) ===", flush=True)
        row = _run_one(arch["width"], arch["depth"], EXPLORATORY_SEED, "exploratory")
        exploratory_rows.append(row)
        print(f"  status={row['status']} E_true={row.get('E_true')} E_cert_U={row.get('E_cert_U')} "
              f"exercise_boundary_error={row.get('exercise_boundary_error')} training_time={row.get('training_time_seconds')}", flush=True)

    import pandas as pd
    exp_df = pd.DataFrame.from_records(exploratory_rows)
    exp_dir = new_run_dir(EXPERIMENT)
    exp_df.to_csv(exp_dir / "exploratory_architecture_study.csv", index=False)

    ok_exp = exp_df[exp_df["status"] == "ok"]
    if len(ok_exp) == 0:
        raise RuntimeError("all exploratory architecture runs failed -- cannot select a final configuration")
    best_arch_tag = ok_exp.sort_values("E_true").iloc[0]["architecture"]
    print(f"\nbest exploratory architecture by E_true: {best_arch_tag}")
    best_arch = next(a for a in ARCHITECTURES if a["tag"] == best_arch_tag)

    print(f"\n########## final configuration ({best_arch_tag}, {len(FINAL_SEEDS)} seeds) ##########", flush=True)
    final_rows = []
    for seed in FINAL_SEEDS:
        print(f"=== {best_arch_tag} (seed={seed}) ===", flush=True)
        row = _run_one(best_arch["width"], best_arch["depth"], seed, "final")
        final_rows.append(row)
        print(f"  status={row['status']} E_true={row.get('E_true')} E_cert_U={row.get('E_cert_U')} "
              f"exercise_boundary_error={row.get('exercise_boundary_error')} training_time={row.get('training_time_seconds')}", flush=True)

    final_df = pd.DataFrame.from_records(final_rows)
    final_df.to_csv(exp_dir / "final_configuration_all_seeds.csv", index=False)

    all_rows = exploratory_rows + final_rows
    all_df = pd.DataFrame.from_records(all_rows)
    all_df.to_csv(exp_dir / "all_runs.csv", index=False)

    print(f"\nexploratory_architecture_study.csv, final_configuration_all_seeds.csv, all_runs.csv -> {exp_dir}")
    print(f"Selected final architecture: {best_arch_tag}")


if __name__ == "__main__":
    main()
