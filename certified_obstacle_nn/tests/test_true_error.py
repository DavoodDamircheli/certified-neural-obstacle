"""Tests for errors/true_error.py.

Two regimes are deliberately kept separate:
  - a fully smooth synthetic problem, to verify the quadrature machinery and
    the returned field arithmetic are correct when there is nothing hard to
    integrate;
  - the real Stefan problem, where t=x is a genuine discontinuity (lambda
    jumps, sigma_x jumps) and a single global tensor-product Gauss rule
    structurally cannot reach 1e-3 relative precision there. That the
    self-check catches this is the intended behavior, not a bug -- see
    errors/true_error.py's module docstring.
"""

import math

import pytest
import torch

from certified_obstacle_nn.errors.true_error import compute_true_error
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.stefan import StefanProblem


class _SmoothProblem:
    """Fully smooth synthetic problem (no free boundary)."""

    T = 1.0
    omega_bounds = (0.0, 1.0)

    @staticmethod
    def u_exact(t, x):
        return torch.sin(math.pi * x) * torch.exp(-t)

    @staticmethod
    def sigma_exact(t, x):
        return torch.cos(t) * x

    @staticmethod
    def lambda_exact(t, x):
        return (x * t) ** 2


class _SmoothTriple:
    """Duck-typed stand-in for NeuralTriple: smooth, slightly-off-from-exact
    callables, no actual network. Exists only to feed compute_true_error a
    case with nonzero but well-behaved (smooth) error.
    """

    class _Net:
        def __init__(self, fn):
            self.fn = fn

        def __call__(self, t, x):
            return self.fn(t, x)

    def __init__(self):
        # perturbations are deliberately oscillatory (not low-degree
        # polynomials) so that a too-coarse Gauss rule genuinely
        # under-resolves them, instead of integrating the difference exactly
        # regardless of n (which a constant or linear offset would).
        self.state_net = self._Net(
            lambda t, x: torch.sin(math.pi * x) * torch.exp(-t) + 0.01 * torch.sin(6 * math.pi * x) * torch.cos(4 * math.pi * t)
        )
        self.flux_net = self._Net(lambda t, x: torch.cos(t) * x + 0.01 * torch.sin(8 * math.pi * x))
        self.multiplier_net = self._Net(lambda t, x: (x * t) ** 2 + 0.005 * torch.cos(10 * math.pi * t))

    def eval(self):
        pass


def test_smooth_problem_passes_self_check_and_fields_are_consistent():
    problem = _SmoothProblem()
    triple = _SmoothTriple()

    result = compute_true_error(problem, triple, n_t=40, n_x=40, quadrature_tolerance=1e-3)

    assert result.quadrature_relative_change < 1e-3
    assert result.E_true > 0

    assert result.E_true_squared == pytest.approx(result.E_true**2)
    expected_total = result.term_grad_u + result.term_u0 + result.term_uT + result.term_sigma + result.term_consistency
    assert result.E_true_squared == pytest.approx(expected_total)

    assert result.norm_u_V**2 == pytest.approx(result.term_grad_u + result.term_u0 + result.term_uT)
    assert result.norm_sigma_L2**2 == pytest.approx(result.term_sigma)
    assert result.norm_lambda_L2 >= 0


def test_deterministic_reproducible():
    problem = _SmoothProblem()
    triple = _SmoothTriple()

    r1 = compute_true_error(problem, triple, n_t=40, n_x=40)
    r2 = compute_true_error(problem, triple, n_t=40, n_x=40)

    assert r1 == r2  # bit-for-bit: deterministic Gauss nodes, no RNG anywhere


def test_coarse_grid_fails_self_check_loudly():
    problem = _SmoothProblem()
    triple = _SmoothTriple()

    with pytest.raises(RuntimeError, match="Quadrature self-check failed"):
        compute_true_error(problem, triple, n_t=2, n_x=2, quadrature_tolerance=1e-12)


def test_stefan_global_quadrature_converges_after_v14_A1_fix():
    """v14 A1 (2026-10-06) REPLACES the former
    test_stefan_free_boundary_defeats_global_quadrature, which asserted that
    the global rule FAILS its self-check for Stefan. That failure was a bug:
    lambda_exact used 1_{t<x} while u_exact/sigma_exact use the contact
    branch on t <= x, so the tensor grid's diagonal points t == x carried a
    residual of exactly 1. With the consistent convention the graph term is
    R_PDE (smooth) and only the C^0 kinks of the grad/flux terms remain, so
    the global rule now passes the unchanged 1e-3 tolerance."""
    problem = StefanProblem()
    torch.manual_seed(0)
    triple = NeuralTriple(problem, width=64, depth=4)

    result = compute_true_error(problem, triple, n_t=80, n_x=80, quadrature_tolerance=1e-3)
    assert result.quadrature_relative_change < 1e-3


def test_free_boundary_split_restores_spectral_convergence():
    """Pre-M18 T1: the SAME hard case as the test above, but with
    free_boundary_fn=problem.free_boundary_exact (s(t)=t, known in closed
    form) -- must now satisfy the ORIGINALLY specified 1e-3 tolerance with
    room to spare (near machine precision, not just barely under), since the
    two regions {t>x}/{t<x} are each smooth and get the full Gauss rule."""
    problem = StefanProblem()
    torch.manual_seed(0)
    triple = NeuralTriple(problem, width=64, depth=4)

    result = compute_true_error(
        problem, triple, n_t=80, n_x=80, quadrature_tolerance=1e-3,
        free_boundary_fn=problem.free_boundary_exact,
    )
    assert result.quadrature_relative_change < 1e-10  # near machine precision, not merely "under tolerance"


class _VerticalKinkProblem:
    """Synthetic American-put-style problem: a genuine kink in x at a FIXED
    x*=1.0 for every t (u_exact continuous, u_x discontinuous there) -- the
    same structural feature as American put's g at x=log(K), independent of
    Stefan's t-dependent free boundary."""

    T = 1.0
    omega_bounds = (0.0, 2.0)
    x_star = 1.0

    @staticmethod
    def u_exact(t, x):
        return torch.exp(-t) * torch.abs(x - 1.0)

    @staticmethod
    def sigma_exact(t, x):
        return -torch.exp(-t) * torch.sign(x - 1.0)

    @staticmethod
    def lambda_exact(t, x):
        return torch.zeros_like(x)


class _VerticalKinkTriple:
    """Smooth (no kink) stand-in triple -- so the ONLY non-smoothness in the
    comparison comes from the reference, exactly mirroring how American
    put's reference (not the network) carries the x=log(K) kink."""

    class _Net:
        def __init__(self, fn):
            self.fn = fn

        def __call__(self, t, x):
            return self.fn(t, x)

    def __init__(self):
        self.state_net = self._Net(lambda t, x: 0.9 * torch.exp(-t) * (x - 1.0) ** 2 / 2.0)
        self.flux_net = self._Net(lambda t, x: -0.9 * torch.exp(-t) * (x - 1.0))
        self.multiplier_net = self._Net(lambda t, x: torch.zeros_like(x))

    def eval(self):
        pass


def test_vertical_interface_defeats_global_quadrature():
    """Same class of failure as Stefan's t=x diagonal (test above), but for a
    FIXED-x kink present at every t -- confirms the self-check catches this
    case too before any split is applied."""
    problem = _VerticalKinkProblem()
    triple = _VerticalKinkTriple()

    with pytest.raises(RuntimeError, match="Quadrature self-check failed"):
        compute_true_error(problem, triple, n_t=40, n_x=40, quadrature_tolerance=1e-6)


def test_vertical_interface_split_restores_spectral_convergence():
    problem = _VerticalKinkProblem()
    triple = _VerticalKinkTriple()

    result = compute_true_error(
        problem, triple, n_t=40, n_x=40, quadrature_tolerance=1e-6,
        vertical_interface_x=problem.x_star,
    )
    assert result.quadrature_relative_change < 1e-10


def test_vertical_interface_and_free_boundary_are_mutually_exclusive():
    problem = StefanProblem()
    torch.manual_seed(0)
    triple = NeuralTriple(problem, width=16, depth=2)

    with pytest.raises(AssertionError, match="mutually exclusive"):
        compute_true_error(
            problem, triple, n_t=10, n_x=10,
            free_boundary_fn=problem.free_boundary_exact, vertical_interface_x=0.5,
        )


def test_optional_reference_callables_override_problem_exact():
    """specification section 12 invariant 4: American put has no closed-form
    solution, so compute_true_error must accept an independent reference
    (e.g. reference/american_put_reference.py's build_reference_callables)
    in place of problem.u_exact/sigma_exact/lambda_exact -- checked here with
    a synthetic override rather than the real frozen reference, to isolate
    the plumbing from the PSOR solver itself."""
    problem = _SmoothProblem()
    triple = _SmoothTriple()

    default_result = compute_true_error(problem, triple, n_t=40, n_x=40)

    def shifted_u_ref(t, x):
        return problem.u_exact(t, x) + 1.0

    override_result = compute_true_error(
        problem, triple, n_t=40, n_x=40,
        u_ref=shifted_u_ref, sigma_ref=problem.sigma_exact, lambda_ref=problem.lambda_exact,
    )
    # a constant shift in u_ref changes term_u0/term_uT (e_u picks up the
    # shift) but leaves term_grad_u (a spatial derivative) unaffected.
    assert override_result.term_grad_u == pytest.approx(default_result.term_grad_u)
    assert override_result.term_u0 != pytest.approx(default_result.term_u0)
