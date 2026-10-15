"""NEW_RUNS.md Experiment 3 -- certificate-aware / Lipschitz-controlled
training.

Single baseline architecture (4x64, specification section 1's own default,
matching results/stefan_m18/4x64's own architecture exactly). Everything
except gamma is IDENTICAL to results/stefan_m18/4x64's own training
procedure (same width/depth/N_INTERIOR/N_INITIAL/epochs/LR/loss weights) --
gamma=0's final 5-seed numbers are expected to be statistically consistent
with that already-published M18 4x64 row as an external cross-check (not
re-derived FROM it; a fresh gamma=0 baseline is trained here so the ONLY
factor that can differ across this experiment's three settings is gamma
itself, per NEW_RUNS.md 3.3's "the comparison must isolate the effect of
certificate-aware regularization").

Two stages:
  1. Exploratory (1 seed=0): gamma in {0, 1e-6, 1e-5, 1e-4}, a CHEAPER
     certification grid (128, matching experiments/stefan_adaptive.py's own
     "less than full production but still meaningful" convention) -- purely
     to pick one weak and one moderate/best gamma value, per NEW_RUNS.md 3.5.
  2. Final (5 seeds): the 3 selected settings (gamma=0, weak, best) at the
     FULL M18 production certification grid (1024, matching
     experiments/stefan_core.py's own CERT_GRID exactly) for full
     comparability with the existing M18 4x64 numbers.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from certified_obstacle_nn.certification.network_bounds import compute_triple_bounds, decompose_L_sm_and_L_Psi
from certified_obstacle_nn.experiments.new_runs.certificate_aware_training import train_with_lipschitz_penalty
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
from certified_obstacle_nn.problems.stefan import StefanProblem
from certified_obstacle_nn.training.sampler import TrainingSampler

EXPERIMENT = "certificate_aware"

WIDTH, DEPTH = 64, 4  # specification section 1 default, matches results/stefan_m18/4x64 exactly
N_INTERIOR = 5000
N_INITIAL = 300
ADAM_LR = 1e-3
N_ADAM_EPOCHS = 5000
LBFGS_LR = 1.0
N_LBFGS_EPOCHS = 300
LOSS_WEIGHTS = {"w_r": 1.0, "w_s": 1.0, "w_0": 1.0, "w_p": 1.0, "w_c": 1.0}

N_QP = 4
OPTION_B_MODE = "min_apriori_and_partition"
TRUE_ERROR_N_T = 60
TRUE_ERROR_N_X = 60

EXPLORATORY_GAMMAS = [0.0, 1e-6, 1e-5, 1e-4]
EXPLORATORY_SEED = 0
EXPLORATORY_CERT_GRID = 128

FINAL_CERT_GRID = 1024  # matches experiments/stefan_core.py's own M17/M18 production mesh
FINAL_SEEDS = [0, 1, 2, 3, 4]


def _train_one_run(gamma: float, seed: int, run_dir: Path) -> dict:
    problem = StefanProblem()
    torch.manual_seed(seed)
    triple = NeuralTriple(problem, width=WIDTH, depth=DEPTH)

    sampler = TrainingSampler(problem, n_interior=N_INTERIOR, n_initial=N_INITIAL, seed=seed)
    t, x, x0 = sampler.sample()

    run_dir.mkdir(parents=True, exist_ok=True)
    losses_csv = run_dir / "losses.csv"
    weights_path = run_dir / "weights.pt"
    config_path = run_dir / "config.json"

    wall_start = time.perf_counter()
    history, outcome = train_with_lipschitz_penalty(
        problem, triple, t, x, x0, n_epochs=N_ADAM_EPOCHS, gamma=gamma,
        lr=ADAM_LR, n_lbfgs_epochs=N_LBFGS_EPOCHS, lbfgs_lr=LBFGS_LR,
        csv_path=str(losses_csv), **LOSS_WEIGHTS,
    )
    training_time = time.perf_counter() - wall_start
    torch.save(triple.state_dict(), weights_path)

    config = {
        "problem": "stefan", "seed": seed,
        "architecture": {"width": WIDTH, "depth": DEPTH, "multiplier_head": triple.multiplier_net.head},
        "regularization": {"type": "spectral_product_penalty", "gamma": gamma},
        "optimizer_schedule": {
            "adam": {"lr": ADAM_LR, "n_epochs": N_ADAM_EPOCHS},
            "lbfgs": {"lr": LBFGS_LR, "n_epochs_requested": N_LBFGS_EPOCHS, "outcome_reason": outcome.reason, "n_steps_completed": outcome.n_steps_completed},
        },
        "loss_weights": dict(LOSS_WEIGHTS),
        "grid_sizes": {"n_interior": N_INTERIOR, "n_initial": N_INITIAL},
        "total_adam_epochs": N_ADAM_EPOCHS, "total_lbfgs_epochs": N_LBFGS_EPOCHS,
        "training_time_seconds": training_time,
        "final_loss_total_unregularized": float(history["loss_total_unregularized"].iloc[-1]),
        "final_loss_total_regularized": float(history["loss_total_regularized"].iloc[-1]),
        "final_lipschitz_penalty": float(history["lipschitz_penalty"].iloc[-1]),
        "weights_path": str(weights_path), "best_weights_path": str(weights_path),
        "best_checkpoint_criterion": "final state after the full Adam+L-BFGS schedule -- no early-stopping/intermediate checkpoint selection exists anywhere in this codebase",
    }
    config_path.write_text(json.dumps(config, indent=2))

    return {
        "weights_path": weights_path, "config_path": config_path, "config": config,
        "training_time": training_time,
        "final_loss_unregularized": float(history["loss_total_unregularized"].iloc[-1]),
        "final_lipschitz_penalty": float(history["lipschitz_penalty"].iloc[-1]),
    }


def _run_one(gamma: float, seed: int, subexperiment: str, cert_grid: int) -> dict:
    gamma_tag = f"gamma_{gamma:.0e}" if gamma != 0.0 else "gamma_0"
    run_dir = new_run_dir(EXPERIMENT, subexperiment, gamma_tag, f"seed_{seed}")

    if is_run_complete(run_dir):
        import pandas as pd
        print(f"  [resume] gamma={gamma} seed={seed} already complete, loading from disk", flush=True)
        return pd.read_csv(run_dir / "summary.csv").iloc[0].to_dict()

    try:
        run_info = _train_one_run(gamma, seed, run_dir)

        cert_wall, cert_payload = certify_run(
            run_info["weights_path"], run_info["config_path"], run_dir / "certification",
            grid=cert_grid, n_qp=N_QP, option_b_mode=OPTION_B_MODE,
        )
        te_wall, te_payload = true_error_run(
            run_info["weights_path"], run_info["config_path"], run_dir / "true_error",
            n_t=TRUE_ERROR_N_T, n_x=TRUE_ERROR_N_X,
        )

        problem, frozen_triple = load_frozen_triple_from_run(run_info["config_path"], run_info["weights_path"])
        a, b = problem.omega_bounds
        diag_gen = torch.Generator().manual_seed(999_999)
        t_diag = torch.rand(2000, 1, generator=diag_gen, dtype=torch.float64) * problem.T
        x_diag = torch.rand(2000, 1, generator=diag_gen, dtype=torch.float64) * (b - a) + a
        x0_diag = torch.rand(N_INITIAL, 1, generator=diag_gen, dtype=torch.float64) * (b - a) + a
        residual_rms = compute_residual_rms(frozen_triple, problem, t_diag, x_diag, x0_diag)
        spectral = extract_spectral_norms(frozen_triple)
        diagnostic_loss = compute_diagnostic_loss(frozen_triple, problem, loss_weights=LOSS_WEIGHTS)

        bounds_obj = compute_triple_bounds(problem, frozen_triple)
        decomposition = decompose_L_sm_and_L_Psi(problem, bounds_obj)

        n_parameters = sum(p.numel() for p in NeuralTriple(problem, width=WIDTH, depth=DEPTH).parameters())
        row = build_master_row(
            experiment=EXPERIMENT, subexperiment=subexperiment, strategy=None, seed=seed,
            architecture_tag=f"{DEPTH}x{WIDTH}", width=WIDTH, depth=DEPTH, n_parameters=n_parameters,
            regularization_type="spectral_product_penalty", regularization_strength=gamma, gamma=gamma,
            n_adam_epochs=N_ADAM_EPOCHS, n_lbfgs_epochs=N_LBFGS_EPOCHS,
            n_interior=N_INTERIOR, n_initial=N_INITIAL, adam_lr=ADAM_LR, lbfgs_lr=LBFGS_LR,
            certification_grid=cert_grid, training_time_seconds=run_info["training_time"],
            final_loss=diagnostic_loss, best_loss=diagnostic_loss,
            cert_payload=cert_payload, te_payload=te_payload, residual_rms=residual_rms,
            weights_path=str(run_info["weights_path"]), config_path=str(run_info["config_path"]),
            status="ok",
            notes=(f"training_lipschitz_penalty(final_epoch)={run_info['final_lipschitz_penalty']:.6e}; "
                   f"final_loss/best_loss: post-hoc out-of-sample diagnostic loss (see common.compute_diagnostic_loss)."),
        )
        row.update(spectral)
        row["training_lipschitz_penalty_final"] = run_info["final_lipschitz_penalty"]
        # "all intermediate Lipschitz/derivative bounds used by the certificate" (NEW_RUNS.md 3.6):
        # every scalar field of TripleBounds, plus the L_sm/L_Psi additive decomposition.
        b_asdict = cert_payload["result"]["bounds"]
        for k, v in b_asdict.items():
            if isinstance(v, (int, float)):
                row[f"bound_{k}"] = v
        row["decomp_L_sm_recomputed"] = decomposition["L_sm_recomputed"]
        row["decomp_L_Psi_recomputed"] = decomposition["L_Psi_recomputed"]
        for k, v in decomposition["L_sm_terms"].items():
            row[f"decomp_{k}"] = v
        for k, v in decomposition["L_Psi_terms"].items():
            row[f"decomp_{k}"] = v

        write_row_csv(run_dir, row)
        write_status(run_dir, "ok")
        check_no_nan_inf(row, label=f"gamma={gamma}/seed{seed}")
        return row

    except Exception as exc:  # noqa: BLE001 -- must be recorded, never silently propagated
        row = build_master_row(
            experiment=EXPERIMENT, subexperiment=subexperiment, strategy=None, seed=seed,
            architecture_tag=f"{DEPTH}x{WIDTH}", width=WIDTH, depth=DEPTH, n_parameters=0,
            regularization_type="spectral_product_penalty", regularization_strength=gamma, gamma=gamma,
            n_adam_epochs=N_ADAM_EPOCHS, n_lbfgs_epochs=N_LBFGS_EPOCHS,
            n_interior=N_INTERIOR, n_initial=N_INITIAL, adam_lr=ADAM_LR, lbfgs_lr=LBFGS_LR,
            certification_grid=cert_grid, training_time_seconds=float("nan"), final_loss=float("nan"),
            status="failed", notes=f"{type(exc).__name__}: {exc}",
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        write_row_csv(run_dir, row)
        write_status(run_dir, "failed", notes=f"{type(exc).__name__}: {exc}")
        print(f"  *** FAILED: gamma={gamma} seed={seed}: {type(exc).__name__}: {exc}", flush=True)
        return row


def main() -> None:
    import pandas as pd

    print("########## exploratory gamma sweep (1 seed, grid=128) ##########", flush=True)
    exploratory_rows = []
    for gamma in EXPLORATORY_GAMMAS:
        print(f"=== gamma={gamma} (seed={EXPLORATORY_SEED}) ===", flush=True)
        row = _run_one(gamma, EXPLORATORY_SEED, "exploratory", EXPLORATORY_CERT_GRID)
        exploratory_rows.append(row)
        print(f"  status={row['status']} E_true={row.get('E_true')} E_cert_U={row.get('E_cert_U')} "
              f"L_R={row.get('L_R')} training_time={row.get('training_time_seconds')}", flush=True)

    exp_dir = new_run_dir(EXPERIMENT)
    exp_df = pd.DataFrame.from_records(exploratory_rows)
    exp_df.to_csv(exp_dir / "exploratory_gamma_sweep.csv", index=False)

    ok_exp = exp_df[exp_df["status"] == "ok"].copy()
    if len(ok_exp) == 0:
        raise RuntimeError("all exploratory gamma runs failed -- cannot select final settings")

    baseline_row = ok_exp[ok_exp["gamma"] == 0.0]
    nonzero = ok_exp[ok_exp["gamma"] > 0.0].sort_values("gamma")
    weak_gamma = float(nonzero.iloc[0]["gamma"]) if len(nonzero) >= 1 else 0.0
    best_row = nonzero.sort_values("E_cert_U").iloc[0] if len(nonzero) > 0 else None
    best_gamma = float(best_row["gamma"]) if best_row is not None else weak_gamma
    print(f"\nSelected: baseline gamma=0.0, weak gamma={weak_gamma:.0e}, best/moderate gamma={best_gamma:.0e} (lowest E_cert_U among nonzero exploratory gammas)", flush=True)

    final_gammas = sorted(set([0.0, weak_gamma, best_gamma]))
    print(f"\n########## final multi-seed sweep: gammas={final_gammas}, {len(FINAL_SEEDS)} seeds, grid={FINAL_CERT_GRID} ##########", flush=True)
    final_rows = []
    for gamma in final_gammas:
        for seed in FINAL_SEEDS:
            print(f"=== gamma={gamma} (seed={seed}) ===", flush=True)
            row = _run_one(gamma, seed, "final", FINAL_CERT_GRID)
            final_rows.append(row)
            print(f"  status={row['status']} E_true={row.get('E_true')} E_cert_U={row.get('E_cert_U')} "
                  f"L_R={row.get('L_R')} training_time={row.get('training_time_seconds')}", flush=True)

    final_df = pd.DataFrame.from_records(final_rows)
    final_df.to_csv(exp_dir / "final_gamma_sweep_all_seeds.csv", index=False)

    all_df = pd.DataFrame.from_records(exploratory_rows + final_rows)
    all_df.to_csv(exp_dir / "all_runs.csv", index=False)
    print(f"\nexploratory_gamma_sweep.csv, final_gamma_sweep_all_seeds.csv, all_runs.csv -> {exp_dir}")
    print(f"Final gammas used: {final_gammas}")


if __name__ == "__main__":
    main()
