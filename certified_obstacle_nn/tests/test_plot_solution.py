"""Smoke test for plots/plot_solution.py.

This only checks that the routine runs end-to-end and produces a figure with
the expected panel structure and a saved file; it cannot and does not assert
anything about rendered pixel content (that was verified visually).
"""

import torch

from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.plots.plot_solution import plot_solution
from certified_obstacle_nn.problems.stefan import StefanProblem


def test_plot_solution_runs_and_saves(tmp_path):
    problem = StefanProblem()
    torch.manual_seed(0)
    triple = NeuralTriple(problem, width=16, depth=2)

    save_path = tmp_path / "solution.png"
    fig = plot_solution(problem, triple, n_t=20, n_x=20, save_path=str(save_path))

    assert save_path.exists()
    assert save_path.stat().st_size > 0

    titles = {ax.get_title() for ax in fig.axes if ax.get_title()}
    assert titles == {
        "u_theta",
        "u exact",
        "u_theta - u exact",
        "sigma_phi",
        "sigma exact",
        "lambda_psi",
        "1_{t<x} exact",
    }
