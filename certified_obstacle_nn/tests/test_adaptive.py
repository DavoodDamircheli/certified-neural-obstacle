"""Tests for training/adaptive.py."""

import numpy as np
import pytest
import torch

from certified_obstacle_nn.certification.partition import build_partition
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.stefan import StefanProblem
from certified_obstacle_nn.training.adaptive import (
    ALPHA_MARKED_FRACTION,
    cell_weights_from_marking,
    dorfler_mark,
    probabilistic_weights,
    resample_points,
    run_adaptive_loop,
)

pytestmark = pytest.mark.legacy


def test_dorfler_marks_smallest_prefix_capturing_theta_energy():
    rho_K = np.array([10.0, 1.0, 1.0, 1.0, 1.0])  # one dominant cell
    mask = dorfler_mark(rho_K, theta=0.5)
    # rho_sq = [100,1,1,1,1], total=104, theta*total=52 -- the single
    # dominant cell (100) alone already exceeds 52, so it alone is marked.
    assert mask.tolist() == [True, False, False, False, False]


def test_dorfler_marks_more_cells_when_energy_is_spread_out():
    rho_K = np.array([1.0, 1.0, 1.0, 1.0])  # uniform energy
    mask = dorfler_mark(rho_K, theta=0.5)
    # total=4, target=2, need cumsum>=2 -> at least 2 cells (each contributes 1).
    assert mask.sum() == 2


def test_dorfler_handles_all_zero_gracefully():
    rho_K = np.zeros(5)
    mask = dorfler_mark(rho_K, theta=0.5)
    assert mask.sum() == 1  # degenerate: marks exactly one cell, does not crash


def test_probabilistic_weights_sum_to_one_and_respect_q():
    rho_K = np.array([1.0, 2.0, 3.0, 4.0])
    w1 = probabilistic_weights(rho_K, q=1.0)
    w2 = probabilistic_weights(rho_K, q=2.0)
    assert w1.sum() == pytest.approx(1.0)
    assert w2.sum() == pytest.approx(1.0)
    # q=2 should concentrate MORE weight on the largest cell than q=1.
    assert w2[-1] > w1[-1]


def test_probabilistic_weights_all_zero_falls_back_to_uniform():
    w = probabilistic_weights(np.zeros(4), q=1.0)
    assert np.allclose(w, 0.25)


def test_cell_weights_from_marking_dorfler_uniform_over_marked_set():
    rho_K = np.array([10.0, 1.0, 1.0, 1.0, 1.0])
    weights = cell_weights_from_marking(rho_K, method="dorfler", theta=0.5)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] == pytest.approx(1.0)  # only cell 0 marked
    assert weights[1:].sum() == pytest.approx(0.0)


def test_cell_weights_from_marking_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown marking method"):
        cell_weights_from_marking(np.array([1.0]), method="bogus")


def test_resample_points_returns_exact_count():
    problem = StefanProblem()
    partition = build_partition(problem, n_t=8, n_x=8)
    weights = np.full(64, 1.0 / 64)
    generator = torch.Generator().manual_seed(0)
    t, x = resample_points(partition, weights, n_total=500, generator=generator)
    assert t.shape == (500, 1)
    assert x.shape == (500, 1)


def test_resample_points_respects_domain_bounds():
    problem = StefanProblem()
    partition = build_partition(problem, n_t=8, n_x=8)
    weights = np.full(64, 1.0 / 64)
    generator = torch.Generator().manual_seed(0)
    t, x = resample_points(partition, weights, n_total=2000, generator=generator)
    a, b = problem.omega_bounds
    assert (t >= 0.0).all() and (t <= problem.T).all()
    assert (x >= a).all() and (x <= b).all()


def test_resample_points_alpha_fraction_concentrates_in_marked_cell():
    """weights fully concentrated on ONE cell: alpha_marked_fraction of the
    points should land inside that cell's rectangle, the rest scattered
    over the whole domain (only a tiny fraction of which also falls in that
    same small cell by chance)."""
    problem = StefanProblem()
    partition = build_partition(problem, n_t=20, n_x=20)  # small cells
    weights = np.zeros(400)
    marked_cell = 210  # arbitrary interior cell
    weights[marked_cell] = 1.0
    generator = torch.Generator().manual_seed(0)
    n_total = 5000
    t, x = resample_points(partition, weights, n_total=n_total, generator=generator, alpha_marked_fraction=0.7)

    t_lo = partition.t_center[marked_cell] - partition.dt / 2
    t_hi = partition.t_center[marked_cell] + partition.dt / 2
    x_lo = partition.x_center[marked_cell] - partition.dx / 2
    x_hi = partition.x_center[marked_cell] + partition.dx / 2
    in_cell = ((t.squeeze(1) >= t_lo) & (t.squeeze(1) <= t_hi) & (x.squeeze(1) >= x_lo) & (x.squeeze(1) <= x_hi))
    fraction_in_cell = in_cell.sum().item() / n_total
    # Expect close to 0.7 plus a tiny sliver from the uniform component
    # (cell area / domain area, negligible at a 20x20 grid: 1/400).
    assert fraction_in_cell == pytest.approx(0.7, abs=0.03)


def test_run_adaptive_loop_dorfler_smoke():
    torch.manual_seed(0)
    problem = StefanProblem()
    triple = NeuralTriple(problem, width=8, depth=2)
    partition = build_partition(problem, n_t=16, n_x=16)

    final_triple, history = run_adaptive_loop(
        problem, triple, partition,
        method="dorfler", theta=0.5,
        n_interior=200, n_initial=30, l_max=2, seed=0,
        n_adam_epochs=20, adam_lr=1e-3, n_lbfgs_epochs=2, lbfgs_lr=1.0,
    )
    assert final_triple is triple
    assert len(history) == 3  # l_max+1 = 3 rounds
    for record in history:
        assert record.t_points.shape == (200, 1)
        assert record.x_points.shape == (200, 1)
    # every parameter must have gradients ENABLED after the loop (not left frozen)
    assert all(p.requires_grad for p in triple.parameters())


def test_run_adaptive_loop_probabilistic_smoke():
    torch.manual_seed(0)
    problem = StefanProblem()
    triple = NeuralTriple(problem, width=8, depth=2)
    partition = build_partition(problem, n_t=16, n_x=16)

    final_triple, history = run_adaptive_loop(
        problem, triple, partition,
        method="probabilistic", q=2.0,
        n_interior=200, n_initial=30, l_max=1, seed=0,
        n_adam_epochs=20, adam_lr=1e-3, n_lbfgs_epochs=2, lbfgs_lr=1.0,
    )
    assert len(history) == 2
    assert history[0].n_marked == 0  # probabilistic never sets a hard marked count
    assert history[1].n_marked == 0


def test_warmstart_actually_updates_parameters_after_marking():
    """Regression test for the freeze/unfreeze gotcha: parameters must
    actually change between iteration 0 and iteration 1 of the SAME loop
    (if compute_raw_estimator's freeze were left in place, Adam would run
    iteration 1 with zero gradients and weights would not move at all)."""
    torch.manual_seed(0)
    problem = StefanProblem()
    triple = NeuralTriple(problem, width=8, depth=2)
    partition = build_partition(problem, n_t=16, n_x=16)

    first_layer_weight_before = triple.state_net.net.layers[0].weight.detach().clone()

    run_adaptive_loop(
        problem, triple, partition,
        method="dorfler", theta=0.5,
        n_interior=200, n_initial=30, l_max=1, seed=0,
        n_adam_epochs=30, adam_lr=1e-2, n_lbfgs_epochs=0, lbfgs_lr=1.0,
    )

    first_layer_weight_after = triple.state_net.net.layers[0].weight.detach().clone()
    assert not torch.allclose(first_layer_weight_before, first_layer_weight_after)


def test_disjointness_violation_would_be_caught():
    """Sanity check that the estimator's own disjointness machinery is
    actually wired up: deliberately passing a collocation point EQUAL to a
    partition cell midpoint must raise, confirming the check inside
    run_adaptive_loop's compute_raw_estimator call is not silently skipped."""
    from certified_obstacle_nn.certification.disjointness import DisjointnessViolationError
    from certified_obstacle_nn.certification.raw_estimator import compute_raw_estimator

    problem = StefanProblem()
    triple = NeuralTriple(problem, width=8, depth=2)
    partition = build_partition(problem, n_t=8, n_x=8)

    # n_qp=1: the single Gauss-Legendre node per cell IS the cell midpoint
    # (leggauss(1) returns node 0 exactly); at the default n_qp=4 the actual
    # quadrature nodes are NOT the midpoint, so this collision would (rightly)
    # go undetected there -- n_qp=1 makes the "colliding point" argument true.
    colliding_t = partition.t_center[0:1].reshape(-1, 1).clone()
    colliding_x = partition.x_center[0:1].reshape(-1, 1).clone()

    with pytest.raises(DisjointnessViolationError):
        compute_raw_estimator(problem, triple, partition, n_qp=1, training_points=(colliding_t, colliding_x))
