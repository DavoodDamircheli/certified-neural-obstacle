"""Multiplier-head ablation (2026-09-11): square (M^2), softplus(M),
relu_square ((relu M)^2), specification section 1's alternative-heads footnote
and section 6's own given bounds for them.

certification/network_bounds.py's compute_triple_bounds was extended
(2026-09-11) to branch lambda_inf/lip_lambda on triple.multiplier_net.head:
before that, only 'square' had a CORRECT a priori bound, so a rigorous
Level-2 certificate (E_cert_U) could only ever have been computed for that
one head. This script exists BECAUSE that gap is now closed -- all three
heads go through the identical certify_option_b pipeline, with head-specific
but equally rigorous a priori bounds. specification's own instruction ("never
report a rigorous Level-2 certificate for one head next to a raw estimator
for another") is what this satisfies: there is no raw-estimator-only
fallback anywhere in this script, for any head.

Per specification section 12 invariant 2, training happens in-process (mirrors
experiments/stefan_core.py) but certification and true-error are invoked
through certify_cli.py/true_error_cli.py as separate subprocesses.

Table 6 columns (user-specified 2026-09-11): head, ||lambda-lambda_psi||_L2,
E_FB, E_true, E_cert_U, optimization_failures (aggregated over >=5 seeds).
||lambda-lambda_psi||_L2 is true_error_cli's own norm_lambda_L2 (already
computed by errors/true_error.py, no new math). optimization_failures counts
seeds where training did not produce a usable triple (see
_is_optimization_failure) -- those seeds are EXCLUDED from every other
column's aggregate (a failed run's weights are not meaningfully certifiable),
not silently averaged in as if they were ordinary data points.
"""

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.stefan import StefanProblem
from certified_obstacle_nn.training.sampler import TrainingSampler
from certified_obstacle_nn.training.trainer import train

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
ABLATION_DIR = RESULTS_DIR / "multiplier_ablation"

HEADS = ["square", "softplus", "relu_square"]
SEEDS = [0, 1, 2, 3, 4]  # >= 5 seeds, user-specified

# Identical across every (head, seed) -- matches experiments/stefan_core.py's
# own M18 defaults, for direct comparability with that milestone's results.
WIDTH, DEPTH = 64, 4
N_INTERIOR = 5000
N_INITIAL = 300
N_ADAM_EPOCHS = 5000
ADAM_LR = 1e-3
N_LBFGS_EPOCHS = 300
LBFGS_LR = 1.0

CERT_GRID = 1024  # M17-fixed standard mesh, same as M18
OPTION_B_MODE = "min_apriori_and_partition"
N_QP = 4
TRUE_ERROR_N_T = 60
TRUE_ERROR_N_X = 60


def _is_optimization_failure(final_loss: float, lbfgs_reason: str) -> bool:
    """A failure is a training run that did not produce a usable triple:
    a non-finite final loss (Adam itself diverged, trainer.py has no
    internal NaN guard during the Adam phase), or L-BFGS aborting for an
    adversarial reason (non_finite_loss, runtime_error). "line_search_stalled"
    and "loss_increased" are NORMAL L-BFGS termination modes (it simply
    cannot improve further) seen throughout this project's other milestones
    -- not failures, and "completed"/"skipped" obviously are not either."""
    if not math.isfinite(final_loss):
        return True
    if lbfgs_reason in ("non_finite_loss", "runtime_error"):
        return True
    return False


def _train_one_run(head: str, seed: int, run_dir: Path) -> dict:
    torch.manual_seed(seed)
    problem = StefanProblem()
    triple = NeuralTriple(problem, width=WIDTH, depth=DEPTH, multiplier_head=head)

    sampler = TrainingSampler(problem, n_interior=N_INTERIOR, n_initial=N_INITIAL, seed=seed)
    t, x, x0 = sampler.sample()

    run_dir.mkdir(parents=True, exist_ok=True)
    history_csv = run_dir / "history.csv"
    weights_path = run_dir / "weights.pt"
    config_path = run_dir / "config.json"

    wall_start = time.perf_counter()
    history, outcome = train(
        problem, triple, t, x, x0,
        n_epochs=N_ADAM_EPOCHS, lr=ADAM_LR,
        n_lbfgs_epochs=N_LBFGS_EPOCHS, lbfgs_lr=LBFGS_LR,
        csv_path=str(history_csv),
    )
    training_time = time.perf_counter() - wall_start
    final_loss = float(history["loss_total"].iloc[-1])

    torch.save(triple.state_dict(), weights_path)
    config = {
        "problem": "stefan",
        "seed": seed,
        "architecture": {"width": WIDTH, "depth": DEPTH, "multiplier_head": head},
        "optimizer_schedule": {
            "adam": {"lr": ADAM_LR, "n_epochs": N_ADAM_EPOCHS},
            "lbfgs": {"lr": LBFGS_LR, "n_epochs_requested": N_LBFGS_EPOCHS, "outcome_reason": outcome.reason, "n_steps_completed": outcome.n_steps_completed},
        },
        "training_time_seconds": training_time,
        "final_loss_total": final_loss,
    }
    config_path.write_text(json.dumps(config, indent=2))

    is_failure = _is_optimization_failure(final_loss, outcome.reason)
    return {
        "head": head, "seed": seed, "weights_path": weights_path, "config_path": config_path,
        "training_time": training_time, "final_loss_total": final_loss,
        "lbfgs_outcome_reason": outcome.reason, "is_optimization_failure": is_failure,
    }


def _run_cli(module: str, args: list[str]) -> None:
    result = subprocess.run([sys.executable, "-m", module] + args, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{module} failed (exit {result.returncode}):\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}")


def _certify_and_true_error(weights_path: Path, config_path: Path, run_dir: Path) -> dict:
    cert_dir = run_dir / "certification"
    te_dir = run_dir / "true_error"

    _run_cli("certified_obstacle_nn.scripts.certify_cli", [
        "--weights", str(weights_path), "--config", str(config_path),
        "--grid", str(CERT_GRID), "--n-qp", str(N_QP),
        "--option-b-mode", OPTION_B_MODE, "--out", str(cert_dir),
    ])
    cert_payload = json.loads((cert_dir / f"certificate_grid{CERT_GRID}.json").read_text())

    _run_cli("certified_obstacle_nn.scripts.true_error_cli", [
        "--weights", str(weights_path), "--config", str(config_path),
        "--n-t", str(TRUE_ERROR_N_T), "--n-x", str(TRUE_ERROR_N_X),
        "--out", str(te_dir),
    ])
    te_payload = json.loads((te_dir / f"true_error_nt{TRUE_ERROR_N_T}_nx{TRUE_ERROR_N_X}.json").read_text())

    return {"cert": cert_payload, "true_error": te_payload}


def main() -> pd.DataFrame:
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    per_run_rows = []

    for head in HEADS:
        for seed in SEEDS:
            run_dir = ABLATION_DIR / head / f"seed{seed}"
            print(f"=== {head} seed={seed} ===", flush=True)
            train_info = _train_one_run(head, seed, run_dir)
            print(f"  trained: {train_info['training_time']:.1f}s  final_loss={train_info['final_loss_total']:.4e}  "
                  f"lbfgs={train_info['lbfgs_outcome_reason']}  failure={train_info['is_optimization_failure']}", flush=True)

            row = {
                "head": head, "seed": seed,
                "is_optimization_failure": train_info["is_optimization_failure"],
                "norm_lambda_L2": None, "E_FB": None, "E_true": None, "E_cert_U": None,
            }
            if not train_info["is_optimization_failure"]:
                diag = _certify_and_true_error(train_info["weights_path"], train_info["config_path"], run_dir)
                te = diag["true_error"]["result"]
                fb = diag["true_error"].get("free_boundary_error")
                r = diag["cert"]["result"]
                row.update({
                    "norm_lambda_L2": te["norm_lambda_L2"],
                    "E_FB": fb["E_FB_gap_and_multiplier"] if fb is not None else None,
                    "E_true": te["E_true"],
                    "E_cert_U": r["E_cert_U_new"],
                })
                print(f"  certified: E_true={row['E_true']:.4e}  E_cert_U={row['E_cert_U']:.4e}  "
                      f"norm_lambda_L2={row['norm_lambda_L2']:.4e}", flush=True)
            else:
                print("  SKIPPING certification (optimization failure)", flush=True)
            per_run_rows.append(row)

    per_run_df = pd.DataFrame.from_records(per_run_rows)
    per_run_path = RESULTS_DIR / "multiplier_ablation_per_run.csv"
    per_run_df.to_csv(per_run_path, index=False)

    table6_rows = []
    for head in HEADS:
        head_df = per_run_df[per_run_df["head"] == head]
        n_failures = int(head_df["is_optimization_failure"].sum())
        ok_df = head_df[~head_df["is_optimization_failure"]]
        table6_rows.append({
            "head": head,
            "norm_lambda_L2_mean": ok_df["norm_lambda_L2"].mean(), "norm_lambda_L2_std": ok_df["norm_lambda_L2"].std(),
            "E_FB_mean": ok_df["E_FB"].mean(), "E_FB_std": ok_df["E_FB"].std(),
            "E_true_mean": ok_df["E_true"].mean(), "E_true_std": ok_df["E_true"].std(),
            "E_cert_U_mean": ok_df["E_cert_U"].mean(), "E_cert_U_std": ok_df["E_cert_U"].std(),
            "optimization_failures": n_failures,
            "n_seeds": len(SEEDS),
        })

    table6 = pd.DataFrame.from_records(table6_rows)
    table6_path = RESULTS_DIR / "stefan_table6_multiplier_ablation.csv"
    table6.to_csv(table6_path, index=False)
    print(f"\nTable 6 -> {table6_path}")
    print(table6.to_string(index=False))
    return table6


if __name__ == "__main__":
    main()
