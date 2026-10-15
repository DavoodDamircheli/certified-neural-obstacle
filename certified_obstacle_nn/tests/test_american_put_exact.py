"""Tests for problems/american_put.py, specification section 10.

Two kinds of checks, mirroring tests/test_stefan_exact.py and
tests/test_network_bounds.py's own conventions:
  - exact algebraic identities (f's derivation from u_lift, u0=g) checked via
    autograd against independently-transcribed formulas;
  - sup-norm/Lipschitz CLAIMS checked by direct sampling, away from the
    single measure-zero data interface x=log(K) (a thin band is excluded
    there, exactly as test_stefan_exact.py excludes a band around t=x --
    finite differences / sup-norm claims straddling a genuine kink are not
    evidence of a bug).
"""

import math

import pytest
import torch

from certified_obstacle_nn.problems.american_put import (
    AmericanPutProblem,
    NormalizedAmericanPutProblem,
    verify_boundary_and_initial_conditions,
)

pytestmark = pytest.mark.legacy

N_POINTS = 20000
BAND = 1e-3  # excluded around x_star = log(K)


def _sample_x(problem, n, seed, band=BAND):
    generator = torch.Generator().manual_seed(seed)
    L, R = problem.omega_bounds
    x = torch.rand(n, generator=generator, dtype=torch.float64) * (R - L) + L
    mask = (x - problem.x_star).abs() > band
    return x[mask]


def test_f_matches_minus_b_ulift_x_minus_c_ulift():
    """f(x) = -(b*u_lift_x + c*u_lift(x)) exactly -- the algebraic identity
    the module docstring derives f from, checked against the INDEPENDENTLY
    given closed-form f in specification section 10, not re-derived here."""
    problem = AmericanPutProblem()
    x = _sample_x(problem, N_POINTS, seed=0, band=0.0)  # smooth everywhere (linear), no kink to exclude
    t = torch.zeros_like(x)

    u_lift_x = -problem.C / (problem.R - problem.L)  # constant, exact
    u_lift = problem.C * (problem.R - x) / (problem.R - problem.L)
    expected_f = -(problem.b * u_lift_x + problem.c * u_lift)

    assert torch.allclose(problem.f(t, x), expected_f, atol=1e-10)


def test_u0_equals_g_exactly():
    problem = AmericanPutProblem()
    x = _sample_x(problem, N_POINTS, seed=1, band=0.0)
    t0 = torch.zeros_like(x)
    assert torch.equal(problem.u0(x), problem.g(t0, x))


def test_g_is_continuous_at_kink_but_g_x_jumps_by_K():
    """g itself has NO jump at x_star (the payoff (K-e^x)^+ is continuous),
    only g_x jumps -- by exactly K, since g_x jumps by -e^{x_star} = -K
    (module docstring's own claim)."""
    problem = AmericanPutProblem()
    eps = 1e-7
    t = torch.tensor(0.0, dtype=torch.float64)
    g_left = problem.g(t, torch.tensor(problem.x_star - eps, dtype=torch.float64))
    g_right = problem.g(t, torch.tensor(problem.x_star + eps, dtype=torch.float64))
    assert abs((g_right - g_left).item()) < 1e-5  # continuous

    def g_x_fd(x_val, h=1e-6):
        xt = torch.tensor(x_val, dtype=torch.float64)
        return (problem.g(t, xt + h) - problem.g(t, xt - h)) / (2 * h)

    g_x_left = g_x_fd(problem.x_star - 1e-4)
    g_x_right = g_x_fd(problem.x_star + 1e-4)
    jump = (g_x_left - g_x_right).item()
    assert abs(jump) == pytest.approx(problem.K, rel=1e-3)


def test_g_inf_bound_holds_under_direct_sampling():
    problem = AmericanPutProblem()
    x = _sample_x(problem, N_POINTS, seed=2)
    t = torch.zeros_like(x)
    sampled_max = problem.g(t, x).abs().max().item()
    assert sampled_max <= problem.g_inf * (1.0 + 1e-9)
    assert sampled_max > 0.5 * problem.g_inf  # not wildly loose either


def test_g_x_inf_bound_holds_under_finite_difference():
    problem = AmericanPutProblem()
    x = _sample_x(problem, N_POINTS, seed=3, band=1e-2)
    t = torch.zeros_like(x)
    h = 1e-6
    g_x_fd = (problem.g(t, x + h) - problem.g(t, x - h)) / (2 * h)
    sampled_max = g_x_fd.abs().max().item()
    assert sampled_max <= problem.g_x_inf * (1.0 + 1e-4)


def test_g_xx_inf_bound_holds_away_from_kink():
    problem = AmericanPutProblem()
    x = _sample_x(problem, N_POINTS, seed=4, band=1e-2)
    t = torch.zeros_like(x)
    h = 1e-4
    g_xx_fd = (problem.g(t, x + h) - 2 * problem.g(t, x) + problem.g(t, x - h)) / (h ** 2)
    sampled_max = g_xx_fd.abs().max().item()
    assert sampled_max <= problem.g_xx_inf * (1.0 + 1e-2)


def test_f_inf_and_f_x_inf_bounds_hold():
    problem = AmericanPutProblem()
    x = _sample_x(problem, N_POINTS, seed=5, band=0.0)
    t = torch.zeros_like(x)
    assert problem.f(t, x).abs().max().item() <= problem.f_inf * (1.0 + 1e-9)

    h = 1e-6
    f_x_fd = (problem.f(t, x + h) - problem.f(t, x - h)) / (2 * h)
    assert f_x_fd.abs().max().item() <= problem.f_x_inf * (1.0 + 1e-4)


def test_all_t_derivative_bounds_are_exactly_zero():
    problem = AmericanPutProblem()
    assert problem.g_t_inf == 0.0
    assert problem.g_tx_inf == 0.0
    assert problem.g_tt_inf == 0.0
    assert problem.f_t_inf == 0.0


def test_u0_bounds_match_g_bounds():
    problem = AmericanPutProblem()
    assert problem.u0_inf == problem.g_inf
    assert problem.u0_lip == problem.g_x_inf


def test_lambda_at_least_c_F_squared():
    problem = AmericanPutProblem()
    diam = problem.omega_bounds[1] - problem.omega_bounds[0]
    assert problem.Lambda >= diam ** 2


def test_x_star_is_interior_to_domain():
    problem = AmericanPutProblem()
    L, R = problem.omega_bounds
    assert L < problem.x_star < R
    assert problem.gamma_data == (problem.x_star,)


def test_verify_boundary_and_initial_conditions_passes_for_real_problem():
    verify_boundary_and_initial_conditions(AmericanPutProblem())  # must not raise


def test_verify_boundary_and_initial_conditions_catches_a_broken_lift():
    class _BrokenLiftProblem(AmericanPutProblem):
        def _u_lift(self, x):
            return super()._u_lift(x) + 1.0  # deliberately break the boundary cancellation

    with pytest.raises(AssertionError, match="g\\(t,L\\) != 0"):
        verify_boundary_and_initial_conditions(_BrokenLiftProblem())


def test_xi_maps_L_and_R_to_minus_one_and_one():
    problem = AmericanPutProblem()
    L, R = problem.omega_bounds
    assert problem.xi(torch.tensor(L, dtype=torch.float64)).item() == pytest.approx(-1.0)
    assert problem.xi(torch.tensor(R, dtype=torch.float64)).item() == pytest.approx(1.0)
    midpoint = (L + R) / 2
    assert problem.xi(torch.tensor(midpoint, dtype=torch.float64)).item() == pytest.approx(0.0)


def test_normalized_problem_scales_g_f_u0_by_one_over_K():
    problem = AmericanPutProblem()
    normalized = NormalizedAmericanPutProblem()
    x = _sample_x(problem, 5000, seed=6, band=0.0)
    t = torch.zeros_like(x)

    assert torch.allclose(normalized.g(t, x), problem.g(t, x) / problem.K)
    assert torch.allclose(normalized.f(t, x), problem.f(t, x) / problem.K)
    assert torch.allclose(normalized.u0(x), problem.u0(x) / problem.K)

    # PDE coefficients and domain are UNCHANGED (not solution-scale quantities).
    assert normalized.A == problem.A
    assert normalized.b == problem.b
    assert normalized.c == problem.c
    assert normalized.omega_bounds == problem.omega_bounds

    # The certification sup-norm constants are NOT separately overridden, but
    # DO come out scaled by 1/K too (dynamic dispatch: g_inf calls self.g,
    # which resolves to THIS class's normalized g) -- documented in
    # NormalizedAmericanPutProblem's own docstring as the reason certification
    # must always use the base AmericanPutProblem, never this class.
    assert normalized.g_inf == pytest.approx(problem.g_inf / problem.K)
