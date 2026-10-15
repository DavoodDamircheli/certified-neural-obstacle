"""Tests for errors/exercise_boundary.py."""

import numpy as np
import pytest
import torch

from certified_obstacle_nn.errors.exercise_boundary import extract_neural_exercise_boundary


class _LinearGapProblem:
    """g == 0 identically; domain wide enough that the known boundary never
    exits it."""

    omega_bounds = (0.0, 10.0)

    @staticmethod
    def g(t, x):
        return torch.zeros_like(x)


class _KnownBoundaryTriple:
    """u(tau,x) = max(x - x_star(tau), 0): zero (in contact, since g=0) for
    x <= x_star(tau), positive (gap>0, no contact) for x > x_star(tau) --
    American-put-style structure (contact on the LOW-x side), with an EXACT,
    known boundary at x_star(tau)."""

    class _StateNet:
        def __init__(self, x_star_fn):
            self.x_star_fn = x_star_fn

        def __call__(self, t, x):
            x_star = self.x_star_fn(t)
            return torch.clamp(x - x_star, min=0.0)

    def __init__(self, x_star_fn):
        self.state_net = self._StateNet(x_star_fn)


def test_recovers_constant_known_boundary_exactly():
    problem = _LinearGapProblem()
    x_star_value = 4.0
    triple = _KnownBoundaryTriple(lambda t: torch.full_like(t, x_star_value))

    tau_values = np.linspace(0.0, 1.0, 5)
    boundary = extract_neural_exercise_boundary(problem, triple, tau_values, n_scan=4001, contact_tol=1e-3)

    assert np.isfinite(boundary).all()
    assert boundary == pytest.approx(x_star_value, abs=1e-2)


def test_recovers_time_varying_known_boundary():
    problem = _LinearGapProblem()
    triple = _KnownBoundaryTriple(lambda t: 2.0 + 3.0 * t)  # x_star(tau) = 2+3*tau

    tau_values = np.linspace(0.0, 1.0, 6)
    boundary = extract_neural_exercise_boundary(problem, triple, tau_values, n_scan=4001, contact_tol=1e-3)

    expected = 2.0 + 3.0 * tau_values
    assert np.isfinite(boundary).all()
    assert boundary == pytest.approx(expected, abs=1e-2)


def test_never_in_contact_returns_nan():
    problem = _LinearGapProblem()
    triple = _KnownBoundaryTriple(lambda t: torch.full_like(t, -100.0))  # x_star far below domain -> never in contact

    tau_values = np.linspace(0.0, 1.0, 3)
    boundary = extract_neural_exercise_boundary(problem, triple, tau_values, n_scan=1001, contact_tol=1e-3)
    assert np.isnan(boundary).all()
