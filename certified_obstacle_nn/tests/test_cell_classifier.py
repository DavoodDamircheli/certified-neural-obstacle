"""Tests for certification/cell_classifier.py.

Two regimes: a fully synthetic, hand-computable gamma field (to verify the
classification arithmetic itself exactly, including the data-cut/sign
threshold interaction), and the real Stefan problem with a trained network
(to verify the actual scientific claim: sign-unresolved cells cluster around
the true free boundary t=x, without the classifier ever being told where it
is -- it only ever sees the network's own gamma and the a priori L_gamma).
"""

import pytest

pytestmark = pytest.mark.legacy

import torch

from certified_obstacle_nn.certification.cell_classifier import classify_cells
from certified_obstacle_nn.certification.network_bounds import compute_triple_bounds
from certified_obstacle_nn.certification.partition import build_partition
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.stefan import StefanProblem


class _Net:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, t, x):
        return self.fn(t, x)


class _FixedProblem:
    """g = 0 identically, so gamma = -u_theta is fully controlled by whatever
    state_net returns. gamma_data=(0.5,) exercises data-cut classification.
    """

    T = 1.0
    omega_bounds = (0.0, 1.0)
    gamma_data = (0.5,)

    def g(self, t, x):
        return torch.zeros_like(t)


class _FixedTriple:
    def __init__(self, u_fn):
        self.state_net = _Net(u_fn)

    def eval(self):
        pass

    def parameters(self):
        return iter(())


def test_classify_cells_matches_hand_computed_expectation():
    # u_theta(t,x) = x - t  =>  gamma = g - u_theta = t - x
    problem = _FixedProblem()
    triple = _FixedTriple(lambda t, x: x - t)
    partition = build_partition(problem, n_t=4, n_x=4)  # dt=dx=0.25, h=0.25*sqrt(2)

    L_gamma = 1.0
    # threshold = L_gamma * h = 0.25*sqrt(2) ~= 0.3536
    # gamma(z_K) = t_center - x_center = (i-j)*0.25 for cell (i,j)
    # data-cut: x=0.5 sits exactly on the edge between j=1 ([0.25,0.5]) and
    # j=2 ([0.5,0.75]), so both columns are data-cut regardless of i.
    result = classify_cells(problem, triple, partition, L_gamma)

    i = partition.i_index
    j = partition.j_index

    expected_data_cut = (j == 1) | (j == 2)
    diff = (i - j).abs() * 0.25
    expected_resolved_candidate = diff > (L_gamma * partition.h)
    expected_sign_resolved = expected_resolved_candidate & ~expected_data_cut
    expected_sign_unresolved = (~expected_resolved_candidate) & (~expected_data_cut)

    assert torch.equal(result.data_cut, expected_data_cut)
    assert torch.equal(result.sign_resolved, expected_sign_resolved)
    assert torch.equal(result.sign_unresolved, expected_sign_unresolved)

    assert expected_data_cut.sum().item() == 8  # 2 columns x 4 rows
    assert expected_sign_unresolved.sum().item() == 4
    assert expected_sign_resolved.sum().item() == 4

    assert result.V_h_dat == partition.area * 8
    assert result.V_h_unr == partition.area * 4


def test_masks_partition_every_cell_exactly_once():
    problem = StefanProblem()
    torch.manual_seed(0)
    triple = NeuralTriple(problem, width=16, depth=2)
    partition = build_partition(problem, n_t=15, n_x=17)
    bounds = compute_triple_bounds(problem, triple)

    result = classify_cells(problem, triple, partition, bounds.L_gamma)

    n_cells = partition.n_t * partition.n_x
    counts = result.data_cut.int() + result.sign_resolved.int() + result.sign_unresolved.int()
    assert (counts == 1).all()  # exactly one class per cell, no overlap, no gap
    assert result.data_cut.sum().item() + result.sign_unresolved.sum().item() + result.sign_resolved.sum().item() == n_cells


def test_stefan_has_no_data_cut_cells():
    # Stefan's Gamma_data is empty (specification section 9), so no cell should
    # ever be classified as data-cut, and V_h_dat must be exactly 0.
    problem = StefanProblem()
    torch.manual_seed(0)
    triple = NeuralTriple(problem, width=16, depth=2)
    partition = build_partition(problem, n_t=20, n_x=20)
    bounds = compute_triple_bounds(problem, triple)

    result = classify_cells(problem, triple, partition, bounds.L_gamma)

    assert not result.data_cut.any()
    assert result.V_h_dat == 0.0


def test_unresolved_cells_cluster_near_free_boundary():
    problem = StefanProblem()
    torch.manual_seed(0)
    triple = NeuralTriple(problem, width=24, depth=3)

    # brief training so gamma has genuine spatial structure to classify,
    # rather than whatever a fresh random init happens to produce
    from certified_obstacle_nn.training.sampler import TrainingSampler
    from certified_obstacle_nn.training.trainer import train

    sampler = TrainingSampler(problem, n_interior=1000, n_initial=100, seed=0)
    t, x, x0 = sampler.sample()
    train(problem, triple, t, x, x0, n_epochs=300, lr=1e-3)

    # v14 A2 (2026-10-06): spectral norms are now PROVED upper bounds
    # (Frobenius / sqrt(||W||_1||W||_inf)) instead of float SVD values, which
    # raises L_gamma for this triple from ~16 to ~29.7; at 40x40 every cell is
    # then sign-unresolved and the comparison below would be vacuous (mean of
    # an empty set). 200x200 restores resolved cells (3378 of 40000); the
    # assertion itself is unchanged.
    partition = build_partition(problem, n_t=200, n_x=200)
    bounds = compute_triple_bounds(problem, triple)
    result = classify_cells(problem, triple, partition, bounds.L_gamma)

    dist_from_diagonal = (partition.t_center - partition.x_center).abs()

    assert result.sign_unresolved.any(), "expected at least some unresolved cells to compare against"
    mean_dist_unresolved = dist_from_diagonal[result.sign_unresolved].mean().item()
    mean_dist_resolved = dist_from_diagonal[result.sign_resolved].mean().item()

    print(f"mean |t-x| unresolved cells: {mean_dist_unresolved:.4e}")
    print(f"mean |t-x| resolved cells:   {mean_dist_resolved:.4e}")
    print(f"n_unresolved={result.sign_unresolved.sum().item()}  n_resolved={result.sign_resolved.sum().item()}")

    assert mean_dist_unresolved < mean_dist_resolved
