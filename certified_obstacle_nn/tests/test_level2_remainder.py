"""Tests for certification/level2_remainder.py.

specification section 7 says to always report E_h_lip, E_h_unr, E_h_dat, E_h_0T
separately -- section 12 invariant 5 forbids merging the estimator and the
certified remainder into one column. For Stefan, Gamma_data is empty
(section 9), so E_h_dat must come out exactly 0, not just approximately.
"""

import pytest
import torch

from certified_obstacle_nn.certification.cell_classifier import CellClassification, classify_cells
from certified_obstacle_nn.certification.level2_remainder import Level2Remainder, compute_level2_remainder
from certified_obstacle_nn.certification.network_bounds import compute_triple_bounds
from certified_obstacle_nn.certification.partition import build_partition
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.stefan import StefanProblem
from certified_obstacle_nn.training.sampler import TrainingSampler
from certified_obstacle_nn.training.trainer import train

pytestmark = pytest.mark.legacy


def _trained_triple(problem, seed=0, n_epochs=200):
    torch.manual_seed(seed)
    triple = NeuralTriple(problem, width=24, depth=3)
    sampler = TrainingSampler(problem, n_interior=1000, n_initial=100, seed=seed)
    t, x, x0 = sampler.sample()
    train(problem, triple, t, x, x0, n_epochs=n_epochs, lr=1e-3)
    return triple


def test_returns_all_components_separately_not_only_sum():
    problem = StefanProblem()
    triple = _trained_triple(problem)
    partition = build_partition(problem, n_t=20, n_x=20)
    bounds = compute_triple_bounds(problem, triple)
    classification = classify_cells(problem, triple, partition, bounds.L_gamma)

    remainder = compute_level2_remainder(problem, triple, partition, bounds, classification)

    assert isinstance(remainder, Level2Remainder)
    for field in ("E_h_lip", "E_h_unr", "E_h_dat", "E_h_0T", "E_h"):
        assert hasattr(remainder, field)

    # the sum must actually be the sum of the reported parts, not a
    # separately computed number that merely happens to be close
    assert remainder.E_h == pytest.approx(remainder.E_h_lip + remainder.E_h_unr + remainder.E_h_dat + remainder.E_h_0T)


def test_stefan_E_h_dat_is_exactly_zero():
    problem = StefanProblem()
    triple = _trained_triple(problem)
    partition = build_partition(problem, n_t=20, n_x=20)
    bounds = compute_triple_bounds(problem, triple)
    classification = classify_cells(problem, triple, partition, bounds.L_gamma)

    assert classification.V_h_dat == 0.0  # Gamma_data is empty for Stefan

    remainder = compute_level2_remainder(problem, triple, partition, bounds, classification)

    assert remainder.E_h_dat == 0.0  # exact, not pytest.approx


def test_components_are_nonnegative():
    problem = StefanProblem()
    triple = _trained_triple(problem)
    partition = build_partition(problem, n_t=15, n_x=15)
    bounds = compute_triple_bounds(problem, triple)
    classification = classify_cells(problem, triple, partition, bounds.L_gamma)

    remainder = compute_level2_remainder(problem, triple, partition, bounds, classification)

    assert remainder.E_h_lip >= 0.0
    assert remainder.E_h_unr >= 0.0
    assert remainder.E_h_dat >= 0.0
    assert remainder.E_h_0T >= 0.0
    assert remainder.E_h >= 0.0


def test_no_unresolved_cells_gives_zero_E_h_unr():
    """Edge case: a classification with no unresolved cells must not error
    and must contribute exactly 0 to E_h_unr."""
    problem = StefanProblem()
    triple = _trained_triple(problem, n_epochs=50)
    partition = build_partition(problem, n_t=10, n_x=10)
    bounds = compute_triple_bounds(problem, triple)

    n_cells = partition.n_t * partition.n_x
    forced_all_resolved = CellClassification(
        data_cut=torch.zeros(n_cells, dtype=torch.bool),
        sign_resolved=torch.ones(n_cells, dtype=torch.bool),
        sign_unresolved=torch.zeros(n_cells, dtype=torch.bool),
        V_h_dat=0.0,
        V_h_unr=0.0,
    )

    remainder = compute_level2_remainder(problem, triple, partition, bounds, forced_all_resolved)
    assert remainder.E_h_unr == 0.0


def test_manual_recomputation_matches_E_h_lip_and_E_h_0T():
    problem = StefanProblem()
    triple = _trained_triple(problem)
    partition = build_partition(problem, n_t=12, n_x=12)
    bounds = compute_triple_bounds(problem, triple)
    classification = classify_cells(problem, triple, partition, bounds.L_gamma)

    remainder = compute_level2_remainder(problem, triple, partition, bounds, classification)

    a, b = problem.omega_bounds
    Q_measure = problem.T * (b - a)
    expected_E_h_lip = (bounds.L_sm + bounds.L_Psi) * partition.h * Q_measure
    expected_E_h_0T = (bounds.L_Phi0 + bounds.L_PhiT) * partition.h * (b - a)

    assert remainder.E_h_lip == pytest.approx(expected_E_h_lip)
    assert remainder.E_h_0T == pytest.approx(expected_E_h_0T)
