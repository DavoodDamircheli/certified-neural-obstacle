"""Tests for certification/partition.py and certification/raw_estimator.py.

The central property under test is specification section 4's localization
identity: sum_K rho_total_K^2 == rho_tilde^2. rho_tilde^2 is DEFINED in
raw_estimator.py as that same sum, so the identity holds by construction; the
only slack allowed here is the floating-point round-trip of squaring a value
that was itself produced by sqrt (needed because the DataFrame stores
human-readable norms, not their squares) -- hence the very tight (not loose)
tolerance used below, several orders of magnitude tighter than any real
discretization effect.
"""

import math

import pytest
import torch

from certified_obstacle_nn.certification.partition import build_partition
from certified_obstacle_nn.certification.raw_estimator import compute_raw_estimator
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.stefan import StefanProblem

IDENTITY_RTOL = 1e-10


class _Net:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, t, x):
        return self.fn(t, x)


class _ExactTriple:
    """Duck-typed stand-in wrapping the exact Stefan solution as if it were a
    trained triple, so the estimator can be sanity-checked against a known
    (near-)solution using the exact same certification code path.
    """

    def __init__(self, problem):
        self.state_net = _Net(problem.u_exact)
        self.flux_net = _Net(problem.sigma_exact)
        self.multiplier_net = _Net(problem.lambda_exact)

    def eval(self):
        pass

    def parameters(self):
        return iter(())


def test_partition_geometry_small_grid():
    problem = StefanProblem()
    partition = build_partition(problem, n_t=2, n_x=4)

    assert partition.n_t == 2
    assert partition.n_x == 4
    assert partition.dt == pytest.approx(0.5)
    assert partition.dx == pytest.approx(0.25)
    assert partition.area == pytest.approx(0.125)
    assert partition.h_K == pytest.approx(math.sqrt(0.5**2 + 0.25**2))
    assert partition.h == partition.h_K

    assert len(partition.t_center) == 8
    assert len(partition.x_center) == 8
    # first cell (i=0, j=0) is centered at (0.25, 0.125)
    assert partition.t_center[0].item() == pytest.approx(0.25)
    assert partition.x_center[0].item() == pytest.approx(0.125)
    assert partition.i_index[0].item() == 0
    assert partition.j_index[0].item() == 0
    # last cell (i=1, j=3) is centered at (0.75, 0.875)
    assert partition.t_center[-1].item() == pytest.approx(0.75)
    assert partition.x_center[-1].item() == pytest.approx(0.875)
    assert partition.i_index[-1].item() == 1
    assert partition.j_index[-1].item() == 3


def test_localization_identity_holds_for_random_triple():
    problem = StefanProblem()
    torch.manual_seed(0)
    triple = NeuralTriple(problem, width=16, depth=2)
    partition = build_partition(problem, n_t=10, n_x=12)

    per_cell_df, summary = compute_raw_estimator(problem, triple, partition, n_qp=4)

    recomputed = (per_cell_df["rho_total_K"] ** 2).sum()
    assert recomputed == pytest.approx(summary.rho_tilde_squared, rel=IDENTITY_RTOL)
    assert summary.rho_tilde == pytest.approx(math.sqrt(summary.rho_tilde_squared))

    recomputed_r = (per_cell_df["rho_r_K"] ** 2).sum()
    assert recomputed_r == pytest.approx(summary.rho_r**2, rel=IDENTITY_RTOL)

    recomputed_p = (per_cell_df["rho_penetration_K"] ** 2).sum()
    assert recomputed_p == pytest.approx(summary.rho_p_tilde**2, rel=IDENTITY_RTOL)

    recomputed_c = (per_cell_df["rho_complementarity_K"] ** 2).sum()
    assert recomputed_c == pytest.approx(summary.rho_c**2, rel=IDENTITY_RTOL)

    # rho_r_K^2 = rho_pde_K^2 + rho_sigma_K^2 (+ R_0 term on i=0 cells only)
    pde_plus_sigma = (per_cell_df["rho_pde_K"] ** 2) + (per_cell_df["rho_sigma_K"] ** 2)
    non_initial = per_cell_df["i"] != 0
    assert (per_cell_df.loc[non_initial, "rho_r_K"] ** 2).values == pytest.approx(
        pde_plus_sigma[non_initial].values, rel=IDENTITY_RTOL
    )


def test_localization_identity_holds_for_finer_partition():
    """Same identity, different N_t x N_x -- the claim is 'any partition', not
    just one specific grid size."""
    problem = StefanProblem()
    torch.manual_seed(1)
    triple = NeuralTriple(problem, width=16, depth=2)
    partition = build_partition(problem, n_t=25, n_x=17)

    per_cell_df, summary = compute_raw_estimator(problem, triple, partition, n_qp=3)

    recomputed = (per_cell_df["rho_total_K"] ** 2).sum()
    assert recomputed == pytest.approx(summary.rho_tilde_squared, rel=IDENTITY_RTOL)


def test_estimator_nonnegative_and_parameters_frozen_after_call():
    problem = StefanProblem()
    torch.manual_seed(0)
    triple = NeuralTriple(problem, width=16, depth=2)
    partition = build_partition(problem, n_t=8, n_x=8)

    per_cell_df, summary = compute_raw_estimator(problem, triple, partition, n_qp=4)

    for col in ["rho_pde_K", "rho_sigma_K", "rho_r_K", "rho_penetration_K", "rho_complementarity_K", "rho_total_K"]:
        assert (per_cell_df[col] >= 0).all()

    assert summary.rho_r >= 0 and summary.rho_p_tilde >= 0 and summary.rho_c >= 0 and summary.rho_tilde >= 0

    # specification section 12 invariant 2: parameters must be frozen by certification.
    for p in triple.parameters():
        assert p.requires_grad is False


def test_exact_stefan_solution_gives_small_residual_away_from_free_boundary():
    """Not a strict zero: cells straddling the free boundary t=x see the exact
    solution's genuine discontinuity (lambda jumps) and kink (sigma_x jumps),
    so their local quadrature is imperfect even for the true solution -- same
    root cause documented in specification's true_error quadrature note. Cells
    well away from the diagonal should still be essentially exact.
    """
    problem = StefanProblem()
    triple = _ExactTriple(problem)
    partition = build_partition(problem, n_t=20, n_x=20)

    per_cell_df, summary = compute_raw_estimator(problem, triple, partition, n_qp=4)

    far_from_diagonal = (per_cell_df["t_center"] - per_cell_df["x_center"]).abs() > 2 * partition.h
    assert (per_cell_df.loc[far_from_diagonal, "rho_total_K"] < 1e-8).all()
