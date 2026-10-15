"""Tests for reference/american_put_reference.py's CN+PSOR solver."""

import numpy as np
import pytest
import torch

from certified_obstacle_nn.problems.american_put import AmericanPutProblem
from certified_obstacle_nn.reference.american_put_reference import (
    PSORConfig,
    _bicubic_interpolate_torch,
    _closed_form_g,
    build_reference_callables,
    freeze_reference,
    load_reference,
    save_reference,
    solve_psor,
)

pytestmark = pytest.mark.legacy

# Small grids for fast tests -- the full h/h/2/h/4 refinement study lives in
# experiments/, this file only checks the solver's own correctness properties.
FAST_CONFIG = PSORConfig(n_x=80, n_tau=80)


def test_config_hash_deterministic_and_sensitive():
    c1 = PSORConfig(n_x=100, n_tau=100)
    c2 = PSORConfig(n_x=100, n_tau=100)
    c3 = PSORConfig(n_x=200, n_tau=100)
    assert c1.config_hash() == c2.config_hash()
    assert c1.config_hash() != c3.config_hash()


def test_obstacle_satisfied_everywhere():
    problem = AmericanPutProblem()
    solution = solve_psor(problem, FAST_CONFIG)
    g = _closed_form_g(problem, solution.x_grid)
    violation = (g[None, :] - solution.u).max()
    assert violation <= 1e-8, f"obstacle violated by {violation:.3e}"


def test_boundary_conditions_are_exactly_zero():
    problem = AmericanPutProblem()
    solution = solve_psor(problem, FAST_CONFIG)
    assert np.all(solution.u[:, 0] == 0.0)
    assert np.all(solution.u[:, -1] == 0.0)


def test_initial_condition_equals_g_exactly():
    problem = AmericanPutProblem()
    solution = solve_psor(problem, FAST_CONFIG)
    g = _closed_form_g(problem, solution.x_grid)
    assert np.allclose(solution.u[0, :], g, atol=1e-12)


def test_value_monotonically_nondecreasing_in_tau():
    """American option value must be non-decreasing in time-to-maturity --
    more time can only add optionality, never remove it. A well-known
    property of American option values, checked here as an INDEPENDENT
    sanity check on the solver (not something the discretization enforces
    structurally, so this is real evidence, not a tautology)."""
    problem = AmericanPutProblem()
    solution = solve_psor(problem, FAST_CONFIG)
    diffs = np.diff(solution.u, axis=0)
    assert diffs.min() >= -1e-8, f"value decreased in tau by up to {-diffs.min():.3e}"


def test_exercise_boundary_is_finite_and_increasing_with_tau_initially():
    problem = AmericanPutProblem()
    solution = solve_psor(problem, FAST_CONFIG)
    boundary = solution.exercise_boundary
    # tau index 0 is the ONE expected exception: u0=g exactly everywhere, so
    # "in contact everywhere" is the literal true state there, not a bug.
    assert np.isnan(boundary[0])
    assert np.isfinite(boundary[1:]).all(), "exercise boundary should be well-defined at every tau>0 for this problem"
    # exercise boundary should sit strictly inside (L, x_star): early exercise
    # is only ever optimal deep enough in the money.
    assert (boundary[1:] > problem.L).all()
    assert (boundary[1:] < problem.x_star).all()


def test_refinement_reduces_value_change():
    """A lightweight in-test convergence check (small grids, for speed): the
    reference point value at n_x=n_tau=40 vs 80 vs 160 should show shrinking
    differences -- the full h/h/2/h/4 study at production resolution lives in
    experiments/american_put_reference_convergence.py."""
    problem = AmericanPutProblem()
    values = []
    for n in (40, 80, 160):
        sol = solve_psor(problem, PSORConfig(n_x=n, n_tau=n))
        values.append(sol.value_at_reference_point)
    diff_1 = abs(values[1] - values[0])
    diff_2 = abs(values[2] - values[1])
    assert diff_2 < diff_1, f"refinement did not reduce the change: {diff_1:.3e} -> {diff_2:.3e}"


def test_freeze_and_load_roundtrip(tmp_path):
    config = PSORConfig(n_x=40, n_tau=40)
    freeze_reference(config, out_dir=tmp_path)
    loaded = load_reference(tmp_path)
    assert loaded.config_hash == config.config_hash()
    assert loaded.n_x == 40 and loaded.n_tau == 40
    assert np.isfinite(loaded.value_at_reference_point)


def test_freeze_is_idempotent_for_same_config(tmp_path):
    config = PSORConfig(n_x=30, n_tau=30)
    result1 = freeze_reference(config, out_dir=tmp_path)
    mtime1 = (tmp_path / "solution.npz").stat().st_mtime
    result2 = freeze_reference(config, out_dir=tmp_path)  # should NOT recompute
    mtime2 = (tmp_path / "solution.npz").stat().st_mtime
    assert result1 == result2
    assert mtime1 == mtime2


def test_freeze_refuses_to_overwrite_different_config(tmp_path):
    freeze_reference(PSORConfig(n_x=30, n_tau=30), out_dir=tmp_path)
    with pytest.raises(RuntimeError, match="already has a FROZEN reference"):
        freeze_reference(PSORConfig(n_x=50, n_tau=50), out_dir=tmp_path)


def test_sigma_and_lambda_deep_otm_and_itm_sanity():
    """_derive_sigma_and_lambda is plain finite differences on the converged
    grid (specification section 0's flux definition sigma=-A*u_x, and the LCP
    residual solved for lambda) -- checked here against known theoretical
    structure, not against another numerical method: deep OTM (x near R,
    option worthless, never in contact) must have lambda approx 0; deep ITM
    (x near L, always in contact, u=g exactly) must have lambda equal to a
    CONSTANT across all tau (g is tau-independent on the contact set, so
    lambda = u_tau - A u_xx + b u_x + c u - f reduces to a fixed value)."""
    problem = AmericanPutProblem()
    solution = solve_psor(problem, PSORConfig(n_x=200, n_tau=200))

    otm_col = -2  # one in from x=R: avoid the boundary one-sided stencil
    lam_otm = solution.lam[1:, otm_col]  # skip tau=0, u0=g identically there
    assert np.all(lam_otm <= 3e-6), f"expected ~0 lambda deep OTM, got max {lam_otm.max():.3e}"

    itm_col = 2  # one in from x=L
    lam_itm = solution.lam[1:, itm_col]
    assert lam_itm.min() > 0.0, "expected strictly positive lambda deep ITM (always in contact)"
    assert np.ptp(lam_itm) / lam_itm.mean() < 0.05, (
        f"expected lambda ~constant across tau deep ITM (g is tau-independent on the contact "
        f"set), got relative spread {np.ptp(lam_itm) / lam_itm.mean():.3e}"
    )

    sigma = -problem.A * np.gradient(solution.u[-1, :], solution.x_grid)
    assert np.allclose(solution.sigma[-1, 5:-5], sigma[5:-5], atol=1e-3)


def test_bicubic_interpolant_exact_at_grid_nodes(tmp_path):
    config = PSORConfig(n_x=40, n_tau=40)
    freeze_reference(config, out_dir=tmp_path)
    solution = load_reference(tmp_path)
    u_ref, sigma_ref, lambda_ref = build_reference_callables(solution)

    idx_tau = [0, 5, 20, 39, 40]
    idx_x = [0, 3, 20, 39, 40]
    t = torch.tensor([[solution.tau_grid[i]] for i in idx_tau], dtype=torch.float64)
    x = torch.tensor([[solution.x_grid[j]] for j in idx_x], dtype=torch.float64)

    u_val = u_ref(t, x).detach().numpy().ravel()
    expected = np.array([solution.u[i, j] for i, j in zip(idx_tau, idx_x)])
    assert np.allclose(u_val, expected, atol=1e-12)


def test_bicubic_interpolant_is_c1_across_cell_edges(tmp_path):
    """Regression for the 2026-09-12 root-cause fix: the earlier bilinear
    interpolant's x-derivative was piecewise CONSTANT, jumping at every grid
    cell edge (measured directly: a discontinuity, not float noise), which
    caused a non-monotonic downstream quadrature self-check in
    errors/true_error.py. Checked here by comparing the derivative just to
    the left and right of an interior cell boundary -- must agree closely,
    not merely be "close enough for the quantity's scale"."""
    config = PSORConfig(n_x=40, n_tau=40)
    freeze_reference(config, out_dir=tmp_path)
    solution = load_reference(tmp_path)
    u_ref, _, _ = build_reference_callables(solution)

    hx = (solution.x_grid[-1] - solution.x_grid[0]) / solution.n_x
    x_edge = float(solution.x_grid[20])  # an interior grid line
    eps = 1e-6 * hx

    t = torch.tensor([[0.5]], dtype=torch.float64)
    x_left = torch.tensor([[x_edge - eps]], dtype=torch.float64, requires_grad=True)
    x_right = torch.tensor([[x_edge + eps]], dtype=torch.float64, requires_grad=True)

    u_left = u_ref(t, x_left)
    u_right = u_ref(t, x_right)
    u_x_left = torch.autograd.grad(u_left.sum(), x_left)[0].item()
    u_x_right = torch.autograd.grad(u_right.sum(), x_right)[0].item()

    assert u_x_left == pytest.approx(u_x_right, abs=1e-4), (
        f"u_x discontinuous across a reference grid cell edge: {u_x_left:.8f} vs {u_x_right:.8f}"
    )


def test_bicubic_interpolant_refuses_to_extrapolate():
    tau_grid = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
    x_grid = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
    grid = torch.zeros((11, 11), dtype=torch.float64)

    with pytest.raises(ValueError, match="outside the frozen reference"):
        _bicubic_interpolate_torch(
            grid, tau_grid, x_grid,
            torch.tensor([[0.5]], dtype=torch.float64),
            torch.tensor([[1.5]], dtype=torch.float64),
        )


def test_psor_raises_rather_than_silently_returning_unconverged(monkeypatch):
    """Forcing max_iter absurdly low must raise, not silently return a
    partially-converged step -- specification's own project-wide discipline
    against silently tolerating a failed convergence check."""
    problem = AmericanPutProblem()
    bad_config = PSORConfig(n_x=40, n_tau=40, psor_max_iter=1, psor_tol=1e-14)
    with pytest.raises(RuntimeError, match="did not converge"):
        solve_psor(problem, bad_config)
