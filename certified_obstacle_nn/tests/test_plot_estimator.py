"""Tests for plots/plot_estimator.py and its band_ratio diagnostic."""

import pandas as pd
import pytest
import torch

from certified_obstacle_nn.certification.partition import build_partition
from certified_obstacle_nn.certification.raw_estimator import compute_raw_estimator
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.plots.plot_estimator import band_ratio, plot_estimator
from certified_obstacle_nn.problems.stefan import StefanProblem


def test_band_ratio_arithmetic_on_synthetic_data():
    df = pd.DataFrame(
        {
            "t_center": [0.5, 0.5, 0.1, 0.9],
            "x_center": [0.5, 0.52, 0.9, 0.1],
            "rho_total_K": [10.0, 10.0, 1.0, 1.0],
        }
    )
    # rows 0,1 have |t-x| < 0.05 (0.0, 0.02); rows 2,3 have |t-x| = 0.8 (outside)
    info = band_ratio(df, band_width=0.05, value_col="rho_total_K")

    assert info["n_inside"] == 2
    assert info["n_outside"] == 2
    assert info["mean_inside"] == pytest.approx(10.0)
    assert info["mean_outside"] == pytest.approx(1.0)
    assert info["ratio"] == pytest.approx(10.0)


def test_plot_estimator_runs_and_saves(tmp_path):
    problem = StefanProblem()
    torch.manual_seed(0)
    triple = NeuralTriple(problem, width=16, depth=2)
    partition = build_partition(problem, n_t=10, n_x=10)
    per_cell_df, _ = compute_raw_estimator(problem, triple, partition, n_qp=3)

    save_path = tmp_path / "estimator.png"
    fig, ratio_info = plot_estimator(
        per_cell_df,
        partition,
        problem,
        free_boundary_fn=problem.free_boundary_exact,
        band_width=0.05,
        save_path=str(save_path),
    )

    assert save_path.exists()
    assert save_path.stat().st_size > 0
    assert ratio_info["ratio"] > 0
    titles = {ax.get_title() for ax in fig.axes if ax.get_title()}
    assert titles == {
        "rho_tilde_K (total)",
        "rho_r_K (PDE + flux + initial)",
        "rho_penetration_K (obstacle)",
        "rho_complementarity_K",
    }


def test_plot_estimator_without_free_boundary_still_works(tmp_path):
    problem = StefanProblem()
    torch.manual_seed(0)
    triple = NeuralTriple(problem, width=16, depth=2)
    partition = build_partition(problem, n_t=8, n_x=8)
    per_cell_df, _ = compute_raw_estimator(problem, triple, partition, n_qp=3)

    fig, ratio_info = plot_estimator(per_cell_df, partition, problem, free_boundary_fn=None)
    assert fig is not None
    assert ratio_info["ratio"] > 0
