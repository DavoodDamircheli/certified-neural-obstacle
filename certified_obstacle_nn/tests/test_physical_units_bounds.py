"""Tests for certification/network_bounds.py's physical_units_bounds
(2026-09-12): converting a TripleBounds computed on a NORMALIZED-output
triple into physical units, without needing the physical-unit triple to
expose real, weight-introspectable sub-networks.

Direct-sampling verification mirrors tests/test_network_bounds.py's own
methodology: sample the ACTUAL PhysicalUnitsTriple-wrapped triple's outputs/
residuals and check they never exceed the converted bound.
"""

import pytest

pytestmark = pytest.mark.legacy

import torch

from certified_obstacle_nn.autograd.derivatives import compute_derivatives
from certified_obstacle_nn.certification.network_bounds import compute_triple_bounds, physical_units_bounds
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.models.unit_conversion import PhysicalUnitsTriple
from certified_obstacle_nn.problems.american_put import AmericanPutProblem, NormalizedAmericanPutProblem
from certified_obstacle_nn.residuals.residuals import compute_residuals

N_SAMPLES = 4000


def _trained_normalized_triple(seed=0):
    from certified_obstacle_nn.training.sampler import TrainingSampler
    from certified_obstacle_nn.training.trainer import train

    torch.manual_seed(seed)
    problem = AmericanPutProblem()
    normalized = NormalizedAmericanPutProblem()
    triple = NeuralTriple(problem, width=16, depth=2, x_rescale=problem.xi)
    sampler = TrainingSampler(normalized, n_interior=800, n_initial=80, seed=seed)
    t, x, x0 = sampler.sample()
    train(normalized, triple, t, x, x0, n_epochs=200, lr=1e-3)
    triple.eval()
    return problem, normalized, triple


def _sample_domain(problem, n, seed):
    generator = torch.Generator().manual_seed(seed)
    a, b = problem.omega_bounds
    margin = 1e-3 * (b - a)
    t = torch.rand(n, 1, generator=generator, dtype=torch.float64) * problem.T
    x = torch.rand(n, 1, generator=generator, dtype=torch.float64) * (b - a - 2 * margin) + a + margin
    return t, x


def test_k1_fields_scale_linearly_and_k2_fields_quadratically():
    problem, normalized, base_triple = _trained_normalized_triple()
    bounds_bar = compute_triple_bounds(normalized, base_triple)
    K = problem.K
    bounds_phys = physical_units_bounds(bounds_bar, K)

    for name in ["u_inf", "grad_x_u_inf", "sigma_inf", "lambda_inf", "m_R", "lip_lambda", "L_gamma", "m_0"]:
        assert getattr(bounds_phys, name) == getattr(bounds_bar, name) * K

    for name in ["M_Psi", "L_sm", "L_Psi", "L_Phi0", "L_PhiT"]:
        assert getattr(bounds_phys, name) == getattr(bounds_bar, name) * K ** 2

    # Untouched fields.
    assert bounds_phys.multiplier_head == bounds_bar.multiplier_head
    assert bounds_phys.state_bounds is bounds_bar.state_bounds


def test_physical_sup_norms_hold_under_direct_sampling_of_wrapped_triple():
    problem, normalized, base_triple = _trained_normalized_triple(seed=1)
    K = problem.K
    bounds_bar = compute_triple_bounds(normalized, base_triple)
    bounds_phys = physical_units_bounds(bounds_bar, K)

    physical_triple = PhysicalUnitsTriple(base_triple, K)
    t, x = _sample_domain(problem, N_SAMPLES, seed=2)
    t.requires_grad_(True)
    x.requires_grad_(True)

    u, sigma, lam, u_t, u_x, sigma_x = compute_derivatives(
        physical_triple.state_net, physical_triple.flux_net, physical_triple.multiplier_net, t, x
    )

    assert u.abs().max().item() <= bounds_phys.u_inf * (1.0 + 1e-6)
    assert u_x.abs().max().item() <= bounds_phys.grad_x_u_inf * (1.0 + 1e-6)
    assert u_t.abs().max().item() <= bounds_phys.d_t_u_inf * (1.0 + 1e-6)
    assert sigma.abs().max().item() <= bounds_phys.sigma_inf * (1.0 + 1e-6)
    assert sigma_x.abs().max().item() <= bounds_phys.div_x_sigma_inf * (1.0 + 1e-6)
    assert lam.abs().max().item() <= bounds_phys.lambda_inf * (1.0 + 1e-6)


def test_physical_residual_sup_norms_hold_under_direct_sampling():
    problem, normalized, base_triple = _trained_normalized_triple(seed=2)
    K = problem.K
    bounds_bar = compute_triple_bounds(normalized, base_triple)
    bounds_phys = physical_units_bounds(bounds_bar, K)

    physical_triple = PhysicalUnitsTriple(base_triple, K)
    t, x = _sample_domain(problem, N_SAMPLES, seed=3)
    t.requires_grad_(True)
    x.requires_grad_(True)
    x0 = _sample_domain(problem, 500, seed=4)[1]

    residuals = compute_residuals(
        problem, physical_triple.state_net, physical_triple.flux_net, physical_triple.multiplier_net, t, x, x0
    )
    assert residuals.R_PDE.abs().max().item() <= bounds_phys.m_R * (1.0 + 1e-6)
    assert residuals.R_sig.abs().max().item() <= bounds_phys.m_sig * (1.0 + 1e-6)


def test_physical_bounds_are_K_times_larger_than_normalized_for_K_gt_1():
    problem, normalized, base_triple = _trained_normalized_triple(seed=3)
    K = problem.K
    assert K > 1.0
    bounds_bar = compute_triple_bounds(normalized, base_triple)
    bounds_phys = physical_units_bounds(bounds_bar, K)
    assert bounds_phys.u_inf > bounds_bar.u_inf
    assert bounds_phys.L_sm > bounds_bar.L_sm
