"""Stefan baseline experiment: train the default 4x64 admissible triple
(specification section 1), produce the qualitative solution-comparison figure, and
run the independent raw estimator over a certification partition to produce
the estimator heat map and the free-boundary-localization diagnostic.

Writes a CSV loss history, a CSV of the per-cell estimator, a JSON of the full
config (seed, architecture, optimizer schedule, grid sizes, wall-clock time,
estimator diagnostics), the trained weights, and both figures, all to
results/.

This script is the orchestration layer: it is expected to import from both
training/ and certification/ (that is the whole point of a Train -> Estimate
pipeline script). The actual constraint (specification section 12 invariant 1) is
narrower: certification/ code itself must never import training/ or receive
its sampler, and vice versa -- which is satisfied here since the estimator is
handed a fresh certification/partition.py grid, never the TrainingSampler's
points.
"""

import json
import time
from pathlib import Path

import torch

from certified_obstacle_nn.certification.partition import build_partition
from certified_obstacle_nn.certification.raw_estimator import compute_raw_estimator
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.plots.plot_estimator import plot_estimator
from certified_obstacle_nn.plots.plot_solution import plot_solution
from certified_obstacle_nn.problems.stefan import StefanProblem
from certified_obstacle_nn.training.sampler import TrainingSampler
from certified_obstacle_nn.training.trainer import train

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def main(
    seed: int = 0,
    width: int = 64,
    depth: int = 4,
    n_interior: int = 5000,
    n_initial: int = 300,
    n_adam_epochs: int = 5000,
    adam_lr: float = 1e-3,
    n_lbfgs_epochs: int = 300,
    lbfgs_lr: float = 1.0,
    n_t_cert: int = 50,
    n_x_cert: int = 50,
    n_qp: int = 4,
    band_width: float = 0.05,
):
    torch.manual_seed(seed)
    problem = StefanProblem()
    triple = NeuralTriple(problem, width=width, depth=depth)

    sampler = TrainingSampler(problem, n_interior=n_interior, n_initial=n_initial, seed=seed)
    t, x, x0 = sampler.sample()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "stefan_baseline_history.csv"
    config_path = RESULTS_DIR / "stefan_baseline_config.json"
    weights_path = RESULTS_DIR / "stefan_baseline_weights.pt"
    figure_path = RESULTS_DIR / "stefan_baseline_solution.png"
    estimator_cells_csv_path = RESULTS_DIR / "stefan_baseline_estimator_cells.csv"
    estimator_figure_path = RESULTS_DIR / "stefan_baseline_estimator.png"

    wall_start = time.perf_counter()
    history, outcome = train(
        problem,
        triple,
        t,
        x,
        x0,
        n_epochs=n_adam_epochs,
        lr=adam_lr,
        n_lbfgs_epochs=n_lbfgs_epochs,
        lbfgs_lr=lbfgs_lr,
        csv_path=str(csv_path),
    )
    wall_clock_seconds = time.perf_counter() - wall_start

    torch.save(triple.state_dict(), weights_path)

    config = {
        "seed": seed,
        "problem": "stefan",
        "architecture": {
            "width": width,
            "depth": depth,
            "multiplier_head": triple.multiplier_net.head,
        },
        "optimizer_schedule": {
            "adam": {"lr": adam_lr, "n_epochs": n_adam_epochs},
            "lbfgs": {
                "lr": lbfgs_lr,
                "n_epochs_requested": n_lbfgs_epochs,
                "line_search": "strong_wolfe",
                "outcome_reason": outcome.reason,
                "n_steps_completed": outcome.n_steps_completed,
            },
        },
        "grid_sizes": {"n_interior": n_interior, "n_initial": n_initial},
        "wall_clock_seconds": wall_clock_seconds,
        "weights_path": str(weights_path),
        "history_csv_path": str(csv_path),
    }

    plot_solution(problem, triple, save_path=str(figure_path))

    # Independent estimation phase: a fresh certification partition, never
    # the training sampler's points (specification section 12 invariant 1).
    partition = build_partition(problem, n_t=n_t_cert, n_x=n_x_cert)
    per_cell_df, estimator_summary = compute_raw_estimator(problem, triple, partition, n_qp=n_qp)
    per_cell_df.to_csv(estimator_cells_csv_path, index=False)

    _, ratio_info = plot_estimator(
        per_cell_df,
        partition,
        problem,
        free_boundary_fn=problem.free_boundary_exact,
        band_width=band_width,
        save_path=str(estimator_figure_path),
    )

    config["estimator"] = {
        "n_t_cert": n_t_cert,
        "n_x_cert": n_x_cert,
        "n_qp": n_qp,
        "rho_r": estimator_summary.rho_r,
        "rho_p_tilde": estimator_summary.rho_p_tilde,
        "rho_c": estimator_summary.rho_c,
        "rho_tilde": estimator_summary.rho_tilde,
        "band_localization": ratio_info,
        "per_cell_csv_path": str(estimator_cells_csv_path),
        "figure_path": str(estimator_figure_path),
    }
    config_path.write_text(json.dumps(config, indent=2))

    print(f"final loss_total       = {history['loss_total'].iloc[-1]:.6e}")
    print(f"final loss_pde         = {history['loss_pde'].iloc[-1]:.6e}")
    print(f"final loss_sigma       = {history['loss_sigma'].iloc[-1]:.6e}")
    print(f"final loss_initial     = {history['loss_initial'].iloc[-1]:.6e}")
    print(f"final loss_penetration = {history['loss_penetration'].iloc[-1]:.6e}")
    print(f"final loss_complementarity = {history['loss_complementarity'].iloc[-1]:.6e}")
    print(f"L-BFGS outcome: {outcome}")
    print(f"wall clock: {wall_clock_seconds:.1f}s")
    print(f"rho_r={estimator_summary.rho_r:.6e}  rho_p_tilde={estimator_summary.rho_p_tilde:.6e}  "
          f"rho_c={estimator_summary.rho_c:.6e}  rho_tilde={estimator_summary.rho_tilde:.6e}")
    print(
        f"band |t-x|<{band_width}: mean_inside={ratio_info['mean_inside']:.4e} "
        f"({ratio_info['n_inside']} cells)  mean_outside={ratio_info['mean_outside']:.4e} "
        f"({ratio_info['n_outside']} cells)  RATIO={ratio_info['ratio']:.3f}"
    )
    print(f"history          -> {csv_path}")
    print(f"config           -> {config_path}")
    print(f"weights          -> {weights_path}")
    print(f"solution figure  -> {figure_path}")
    print(f"estimator cells  -> {estimator_cells_csv_path}")
    print(f"estimator figure -> {estimator_figure_path}")

    return history, outcome, triple, per_cell_df, estimator_summary, ratio_info


if __name__ == "__main__":
    main()
