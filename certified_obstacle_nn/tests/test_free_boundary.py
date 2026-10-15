"""Tests for errors/free_boundary.py (threshold-based E_FB, M18 2026-09-11
redefinition -- supersedes the earlier zero-crossing-of-gamma version)."""

import math

import pytest
import torch

from certified_obstacle_nn.errors.free_boundary import (
    ALPHA_LAMBDA,
    ALPHA_U,
    compute_free_boundary_error,
    compute_free_boundary_error_sensitivity_sweep,
)
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.stefan import StefanProblem


class _RampProblem:
    """g == 0 identically; domain wide enough that s(t)+delta plus any
    threshold shift never exits it. g_inf is fixed directly (not derived),
    to keep the closed-form E_FB check independent of StefanProblem."""

    T = 1.0
    omega_bounds = (0.0, 2.0)
    g_inf = 3.0

    @staticmethod
    def g(t, x):
        return torch.zeros_like(x)

    @staticmethod
    def free_boundary_exact(t):
        return t


class _RampTriple:
    """u_theta(t,x) = max((t+delta) - x, 0) -- zero on the "contact" side
    (x >= t+delta, matching the sign structure of a well-trained Stefan
    triple: gap=0 in contact, growing linearly away from it), so
    gap = |u_theta - g| = u_theta is exactly piecewise-linear, and the
    leftmost-crossing + linear-interpolation machinery recovers the crossing
    EXACTLY (not just approximately)."""

    class _StateNet:
        def __init__(self, delta):
            self.delta = delta

        def __call__(self, t, x):
            return torch.clamp((t + self.delta) - x, min=0.0)

    class _MultiplierNet:
        def __init__(self, value):
            self.value = value

        def __call__(self, t, x):
            return torch.full_like(x, self.value)

    def __init__(self, delta, lambda_value=1000.0):
        self.state_net = self._StateNet(delta)
        self.multiplier_net = self._MultiplierNet(lambda_value)

    def eval(self):
        pass


def test_gap_only_recovers_exact_shifted_crossing():
    """With gap piecewise-linear, s_theta_gap_only(t) = (t+delta) - eps_u
    exactly (the leftmost x where gap <= eps_u, since gap decreases linearly
    to 0 as x increases through t+delta). lambda_psi is a huge constant here
    so detector (ii) reduces to detector (i) and should agree exactly too."""
    delta = 0.3
    problem = _RampProblem()
    triple = _RampTriple(delta, lambda_value=1000.0)

    result = compute_free_boundary_error(
        problem, triple, problem.free_boundary_exact,
        n_t=50, n_scan=2001, quadrature_tolerance=1e-2,
    )

    eps_u = ALPHA_U * problem.g_inf
    assert result.eps_u == pytest.approx(eps_u, rel=1e-9)
    shift = delta - eps_u
    expected_E_FB = abs(shift) * math.sqrt(problem.T)

    assert result.E_FB_gap_only == pytest.approx(expected_E_FB, rel=1e-4)
    # lambda_psi never binding (constant 1000 >> any eps_lambda) -> detector
    # (ii) collapses to detector (i).
    assert result.E_FB_gap_and_multiplier == pytest.approx(result.E_FB_gap_only, rel=1e-6)


class _NeverActiveMultiplierTriple:
    """State net: same ramp as _RampTriple (gap crosses normally around
    x ~ t+delta). Multiplier net: large (defines a meaningful, nonzero
    eps_lambda) only in a region (x < 0.01) far from where gap's own
    crossing ever occurs, and small everywhere else -- including throughout
    the crossing-relevant region, so criterion (ii) genuinely never holds
    there. (A spatially CONSTANT lambda_psi cannot test this path: eps_lambda
    is itself alpha_lambda times lambda_psi's own empirical max, so a
    constant value always satisfies its own derived threshold trivially --
    lambda_psi must vary in space to test the "never holds" fallback.)"""

    class _StateNet:
        def __init__(self, delta):
            self.delta = delta

        def __call__(self, t, x):
            return torch.clamp((t + self.delta) - x, min=0.0)

    class _MultiplierNet:
        def __call__(self, t, x):
            return torch.where(x < 0.01, torch.full_like(x, 1000.0), torch.full_like(x, 0.001))

    def __init__(self, delta):
        self.state_net = self._StateNet(delta)
        self.multiplier_net = self._MultiplierNet()

    def eval(self):
        pass


def test_multiplier_never_active_falls_back_to_right_edge():
    """lambda_psi is small (0.001) throughout the region where gap's own
    crossing occurs, well below eps_lambda = ALPHA_LAMBDA * 1000 = 20:
    criterion (ii) never holds there, so s_theta_ii must fall back to the
    right domain edge for every t, even though detector (i) alone crosses
    normally."""
    delta = 0.3
    problem = _RampProblem()
    triple = _NeverActiveMultiplierTriple(delta)

    result = compute_free_boundary_error(
        problem, triple, problem.free_boundary_exact,
        n_t=50, n_scan=501, quadrature_tolerance=1e-2,
    )

    a, b = problem.omega_bounds

    # s_theta_ii(t) = b for every t (never holds) -> E_FB_ii^2 = integral (b-t)^2 dt.
    import numpy as np
    nodes, weights = np.polynomial.legendre.leggauss(400)
    t_nodes = 0.5 * (nodes + 1.0) * problem.T
    t_weights = 0.5 * weights * problem.T
    expected_sq = float(np.sum(t_weights * (b - t_nodes) ** 2))

    assert result.E_FB_gap_and_multiplier == pytest.approx(math.sqrt(expected_sq), rel=1e-3)
    # detector (i) is unaffected by lambda_psi and still crosses normally.
    assert result.E_FB_gap_only < result.E_FB_gap_and_multiplier


def test_zero_delta_gives_near_zero_gap_only_E_FB_with_default_alpha():
    problem = _RampProblem()
    # eps_u = ALPHA_U * g_inf = 0.02*3.0 = 0.06 -- pick delta == eps_u so the
    # shift is exactly zero (closed-form E_FB_gap_only == 0). E_FB is then at
    # the float64 noise floor, so (as with the earlier zero-shift test in
    # this repo's history) the RELATIVE self-check needs a large tolerance
    # here specifically -- not a general loosening, see
    # test_gap_only_recovers_exact_shifted_crossing above for the
    # normal-magnitude case at the default 1e-2 tolerance.
    delta = ALPHA_U * problem.g_inf
    triple = _RampTriple(delta, lambda_value=1000.0)

    result = compute_free_boundary_error(
        problem, triple, problem.free_boundary_exact,
        n_t=50, n_scan=2001, quadrature_tolerance=1.0,
    )
    assert result.E_FB_gap_only == pytest.approx(0.0, abs=1e-4)


class _AllContactTriple:
    """gap == 0 everywhere -> criterion holds at x=a itself -> s_theta = a
    for every t, no interpolation needed."""

    class _StateNet:
        def __call__(self, t, x):
            return torch.zeros_like(x)

    class _MultiplierNet:
        def __call__(self, t, x):
            return torch.full_like(x, 1000.0)

    def __init__(self):
        self.state_net = self._StateNet()
        self.multiplier_net = self._MultiplierNet()

    def eval(self):
        pass


def test_all_contact_clamps_to_left_edge():
    problem = _RampProblem()
    triple = _AllContactTriple()
    result = compute_free_boundary_error(
        problem, triple, problem.free_boundary_exact,
        n_t=30, n_scan=101, quadrature_tolerance=1.0,
    )
    a, b = problem.omega_bounds
    # s_theta(t) = a for all t -> E_FB^2 = integral (a-t)^2 dt = integral t^2 dt (a=0) = T^3/3
    expected = math.sqrt(problem.T ** 3 / 3.0)
    assert result.E_FB_gap_only == pytest.approx(expected, rel=1e-6)


def test_real_stefan_problem_and_triple_smoke():
    torch.manual_seed(0)
    problem = StefanProblem()
    triple = NeuralTriple(problem, width=16, depth=2)
    triple.eval()
    for p in triple.parameters():
        p.requires_grad_(False)

    result = compute_free_boundary_error(
        problem, triple, problem.free_boundary_exact,
        n_t=200, n_scan=401, quadrature_tolerance=1.0,
    )
    assert result.E_FB_gap_only >= 0.0
    assert result.E_FB_gap_and_multiplier >= 0.0
    assert math.isfinite(result.E_FB_gap_only)
    assert math.isfinite(result.E_FB_gap_and_multiplier)
    assert result.eps_u == pytest.approx(ALPHA_U * problem.g_inf)
    assert result.eps_lambda == pytest.approx(ALPHA_LAMBDA * result.lambda_psi_inf_empirical)


def test_quadrature_self_check_raises_on_tight_tolerance_for_noisy_case():
    """Oscillatory shift genuinely under-resolves at coarse n_t (same
    technique used elsewhere in this repo's quadrature tests)."""
    class _OscillatingTriple:
        class _StateNet:
            def __call__(self, t, x):
                delta = 0.05 * torch.sin(50.0 * t)
                return torch.clamp((t + delta) - x, min=0.0)

        class _MultiplierNet:
            def __call__(self, t, x):
                return torch.full_like(x, 1000.0)

        def __init__(self):
            self.state_net = self._StateNet()
            self.multiplier_net = self._MultiplierNet()

        def eval(self):
            pass

    problem = _RampProblem()
    triple = _OscillatingTriple()
    with pytest.raises(RuntimeError, match="quadrature self-check failed"):
        compute_free_boundary_error(
            problem, triple, problem.free_boundary_exact,
            n_t=6, n_scan=2001, quadrature_tolerance=1e-8,
        )


def test_sensitivity_sweep_shape_and_content():
    problem = _RampProblem()
    triple = _RampTriple(delta=0.3, lambda_value=1000.0)

    alpha_u_values = [0.01, 0.02, 0.05]
    alpha_lambda_values = [0.01, 0.1]
    rows = compute_free_boundary_error_sensitivity_sweep(
        problem, triple, problem.free_boundary_exact,
        alpha_u_values, alpha_lambda_values, n_t=50, n_scan=1001,
    )

    assert len(rows) == len(alpha_u_values) * len(alpha_lambda_values)
    for row in rows:
        assert set(row.keys()) == {
            "alpha_u", "alpha_lambda", "eps_u", "eps_lambda",
            "E_FB_gap_only", "E_FB_gap_and_multiplier",
        }
        assert row["E_FB_gap_only"] >= 0.0
        assert row["E_FB_gap_and_multiplier"] >= 0.0

    # Different alpha_u values must actually change E_FB_gap_only (not a
    # no-op sweep).
    e_fb_values = {row["alpha_u"]: row["E_FB_gap_only"] for row in rows if row["alpha_lambda"] == 0.01}
    assert len(set(e_fb_values.values())) > 1
