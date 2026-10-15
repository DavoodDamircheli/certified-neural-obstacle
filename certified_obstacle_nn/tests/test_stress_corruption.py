"""Tests for certification/stress_corruption.py."""

import math

import pytest
import torch

from certified_obstacle_nn.certification.certificate_option_b import certify_option_b
from certified_obstacle_nn.certification.network_bounds import compute_triple_bounds
from certified_obstacle_nn.certification.partition import build_partition
from certified_obstacle_nn.certification.stress_corruption import (
    CorruptedTriple,
    corrected_bounds_flux_add_linear_t,
    corrected_bounds_lambda_add_relu_bump,
    make_flux_add_linear_t_triple,
    make_lambda_add_relu_bump_triple,
)
from certified_obstacle_nn.errors.true_error import compute_true_error
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.stefan import StefanProblem
from certified_obstacle_nn.training.sampler import TrainingSampler
from certified_obstacle_nn.training.trainer import train

pytestmark = pytest.mark.legacy


@pytest.fixture(scope="module")
def trained_triple():
    torch.manual_seed(0)
    problem = StefanProblem()
    triple = NeuralTriple(problem, width=16, depth=2)
    sampler = TrainingSampler(problem, n_interior=500, n_initial=50, seed=0)
    t, x, x0 = sampler.sample()
    train(problem, triple, t, x, x0, n_epochs=200, lr=1e-3, n_lbfgs_epochs=10)
    triple.eval()
    for p in triple.parameters():
        p.requires_grad_(False)
    return problem, triple


def test_flux_wrapper_adds_exactly_a_coefficient_times_t(trained_triple):
    problem, triple = trained_triple
    a_coefficient = 2.5
    wrapped = make_flux_add_linear_t_triple(triple, a_coefficient)

    t = torch.linspace(0.0, 1.0, 7, dtype=torch.float64).reshape(-1, 1)
    x = torch.linspace(0.0, 1.0, 7, dtype=torch.float64).reshape(-1, 1)

    with torch.no_grad():
        base_out = triple.flux_net(t, x)
        wrapped_out = wrapped.flux_net(t, x)

    assert torch.allclose(wrapped_out, base_out + a_coefficient * t, atol=1e-12)
    # state_net / multiplier_net untouched
    with torch.no_grad():
        assert torch.equal(wrapped.state_net(t, x), triple.state_net(t, x))
        assert torch.equal(wrapped.multiplier_net(t, x), triple.multiplier_net(t, x))


def test_lambda_wrapper_adds_exactly_bump_height_times_relu(trained_triple):
    problem, triple = trained_triple
    bump_height = 3.0
    wrapped = make_lambda_add_relu_bump_triple(triple, bump_height)

    t = torch.linspace(0.0, 1.0, 11, dtype=torch.float64).reshape(-1, 1)
    x = torch.linspace(0.0, 1.0, 11, dtype=torch.float64).reshape(-1, 1)

    with torch.no_grad():
        base_out = triple.multiplier_net(t, x)
        wrapped_out = wrapped.multiplier_net(t, x)
        expected_bump = bump_height * torch.clamp(t - x, min=0.0)

    assert torch.allclose(wrapped_out, base_out + expected_bump, atol=1e-12)
    assert (wrapped_out >= -1e-12).all()  # nonnegativity preserved


def test_corrupted_triple_eval_and_parameters_delegate(trained_triple):
    problem, triple = trained_triple
    wrapped = make_flux_add_linear_t_triple(triple, 1.0)
    wrapped.eval()  # must not raise
    assert list(wrapped.parameters()) == list(triple.parameters())


def test_flux_correction_leaves_div_x_dependent_fields_unchanged(trained_triple):
    problem, triple = trained_triple
    base_bounds = compute_triple_bounds(problem, triple)
    corrected = corrected_bounds_flux_add_linear_t(base_bounds, problem, a_coefficient=5.0)

    # d_x a = 0 exactly: these must be BYTE-IDENTICAL, not just close.
    assert corrected.div_x_sigma_inf == base_bounds.div_x_sigma_inf
    assert corrected.lip_R_sig == base_bounds.lip_R_sig
    assert corrected.lip_div_x_sigma == base_bounds.lip_div_x_sigma
    assert corrected.lip_R_PDE == base_bounds.lip_R_PDE
    assert corrected.m_R == base_bounds.m_R
    assert corrected.L_Psi == base_bounds.L_Psi

    # sigma_inf, m_sig: exact sup-norm-subadditivity correction.
    a_inf = 5.0 * problem.T
    assert corrected.sigma_inf == pytest.approx(base_bounds.sigma_inf + a_inf)
    assert corrected.m_sig == pytest.approx(base_bounds.m_sig + problem.alpha ** -0.5 * a_inf)
    assert corrected.L_sm > base_bounds.L_sm  # T2 term strictly increased


def test_lambda_correction_leaves_L_Psi_and_sigma_fields_unchanged(trained_triple):
    problem, triple = trained_triple
    base_bounds = compute_triple_bounds(problem, triple)
    corrected = corrected_bounds_lambda_add_relu_bump(base_bounds, problem, bump_height=2.0)

    assert corrected.L_Psi == base_bounds.L_Psi
    assert corrected.M_Psi == base_bounds.M_Psi
    assert corrected.sigma_inf == base_bounds.sigma_inf
    assert corrected.m_sig == base_bounds.m_sig
    assert corrected.lip_R_sig == base_bounds.lip_R_sig

    bump_inf = 2.0
    bump_lip = 2.0 * math.sqrt(2.0)
    assert corrected.lambda_inf == pytest.approx(base_bounds.lambda_inf + bump_inf)
    assert corrected.m_R == pytest.approx(base_bounds.m_R + bump_inf)
    assert corrected.m_lam == pytest.approx(base_bounds.m_lam + bump_inf)
    assert corrected.lip_lambda == pytest.approx(base_bounds.lip_lambda + bump_lip)
    assert corrected.lip_R_PDE == pytest.approx(base_bounds.lip_R_PDE + bump_lip)
    assert corrected.L_sm > base_bounds.L_sm


def test_M16_holds_on_corrupted_flux_triple_with_corrected_bounds(trained_triple):
    problem, triple = trained_triple
    a_coefficient = 3.0
    wrapped = make_flux_add_linear_t_triple(triple, a_coefficient)
    base_bounds = compute_triple_bounds(problem, triple)
    corrected = corrected_bounds_flux_add_linear_t(base_bounds, problem, a_coefficient)

    partition = build_partition(problem, n_t=24, n_x=24)
    result = certify_option_b(wrapped, problem, partition, bounds_override=corrected, points_per_chunk=4096, midpoint_chunk=4096)
    te = compute_true_error(problem, wrapped, n_t=20, n_x=20, quadrature_tolerance=1.0, free_boundary_fn=problem.free_boundary_exact)

    assert te.E_true <= result.E_cert_U_new, (
        f"M16 violated on corrupted flux triple: E_true={te.E_true} > E_cert_U_new={result.E_cert_U_new}"
    )


def test_M16_holds_on_corrupted_lambda_triple_with_corrected_bounds(trained_triple):
    problem, triple = trained_triple
    bump_height = 2.0
    wrapped = make_lambda_add_relu_bump_triple(triple, bump_height)
    base_bounds = compute_triple_bounds(problem, triple)
    corrected = corrected_bounds_lambda_add_relu_bump(base_bounds, problem, bump_height)

    partition = build_partition(problem, n_t=24, n_x=24)
    result = certify_option_b(wrapped, problem, partition, bounds_override=corrected, points_per_chunk=4096, midpoint_chunk=4096)
    te = compute_true_error(problem, wrapped, n_t=20, n_x=20, quadrature_tolerance=1.0, free_boundary_fn=problem.free_boundary_exact)

    assert te.E_true <= result.E_cert_U_new, (
        f"M16 violated on corrupted lambda triple: E_true={te.E_true} > E_cert_U_new={result.E_cert_U_new}"
    )


def test_uncorrected_apriori_bound_would_be_smaller_than_corrected(trained_triple):
    """Demonstrates the correction is not vacuous: the UNCORRECTED a priori
    m_R is strictly smaller than the corrected one for a nonzero bump --
    confirms that skipping the correction really would understate m_R_apriori,
    the exact failure mode this module exists to prevent (module docstring)."""
    problem, triple = trained_triple
    base_bounds = compute_triple_bounds(problem, triple)
    corrected = corrected_bounds_lambda_add_relu_bump(base_bounds, problem, bump_height=2.0)
    assert corrected.m_R > base_bounds.m_R
