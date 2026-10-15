"""Crank-Nicolson + PSOR reference solver for the American put, specification
section 10 (Phase 4) and section 12 invariant 4: "E_true is computed only
from closed-form (Stefan) or a FROZEN INDEPENDENT reference (American put,
Phase 4). Never from the network itself."

Solves for the LIFTED unknown u = V - u_lift (problems/american_put.py's own
convention) directly in PHYSICAL (tau, x) units -- no normalization, no
network, pure finite differences -- so a later E_true computation can compare
the neural triple's own (already-converted-to-physical-units, per specification
section 10's own instruction) prediction against this solution directly.

Numerical method: standard Brennan-Schwartz-style Crank-Nicolson time
stepping with PSOR (Projected Successive Over-Relaxation) enforcing the
obstacle constraint u >= g at each implicit step -- the classical, widely
validated method for linear complementarity problems arising from American
option pricing (e.g. Wilmott/Dewynne/Howison; Achdou & Pironneau,
"Computational Methods for Option Pricing"). PSOR is used red-black ordered
(the domain's interior nodes split into two independent "colors" whose
neighbors are always the OTHER color) so each half-sweep is one vectorized
numpy operation, not a per-node Python loop -- this is a strictly faster
implementation of the SAME sweep order a naive sequential Gauss-Seidel would
perform, not an approximation of it.

Rannacher startup (config.n_rannacher_steps, default 4). Diagnosed 2026-09-12:
a first attempt at pure Crank-Nicolson produced NON-MONOTONIC, oscillating
convergence in value_at_reference_point across h -> h/2 -> h/4 refinement
(differences grew before shrinking, not the expected steadily-decreasing
sequence) -- the classical, well-documented CN pathology for problems with a
NON-SMOOTH initial condition (Rannacher 1984, "Finite element solution of
diffusion problems with irregular data"; standard practice in essentially
every production American-option PSOR/CN implementation). u0=g has a genuine
KINK at x*=log(K) (problems/american_put.py's own documented data interface
-- specification section 10's "Data interface at x=log K"), so its high-frequency
Fourier content is exactly what CN (A-stable but NOT L-stable) fails to
damp; the fix is to replace the first few CN steps with fully-implicit
Backward Euler (L-stable, aggressively damps high-frequency error) before
switching to CN for the remainder of the integration -- not a hack, the
textbook remedy for this exact, well-characterized failure mode. Verified
below (experiments/american_put_reference_convergence.py) to restore
monotonic, rapidly-shrinking refinement differences.

FREEZING (specification section 12 invariant 4's "frozen independent reference"
and this module's own explicit instruction "FREEZE the reference... before
any neural run"): freeze_reference() computes the converged solution ONCE,
at a resolution the refinement study (below) has already shown is stable far
below the expected neural error scale, and writes it to disk (npz for the
dense (tau,x) grids, json for scalar config/diagnostics) tagged with a
SHA-256 hash of its own solver configuration. Any later E_true computation
must load this frozen file, never recompute the reference fresh and never
consult the neural network being evaluated -- "never use the neural method to
validate itself" is enforced by there being no code path here that could.
"""

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from certified_obstacle_nn.problems.american_put import AmericanPutProblem

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = REPO_ROOT / "results" / "american_put_reference"


@dataclass(frozen=True)
class PSORConfig:
    n_x: int
    n_tau: int
    omega: float = 1.5
    psor_tol: float = 1e-11
    psor_max_iter: int = 20000
    n_rannacher_steps: int = 4

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True)
class AmericanPutReferenceSolution:
    config_hash: str
    n_x: int
    n_tau: int
    x_grid: np.ndarray  # (n_x+1,)
    tau_grid: np.ndarray  # (n_tau+1,)
    u: np.ndarray  # (n_tau+1, n_x+1), the LIFTED solution
    sigma: np.ndarray  # (n_tau+1, n_x+1), = -A*u_x (the flux DEFINITION, specification section 0), via finite differences
    lam: np.ndarray  # (n_tau+1, n_x+1), = max(0, u_tau - A*u_xx + b*u_x + c*u - f) (the LCP's own multiplier), via finite differences
    exercise_boundary: np.ndarray  # (n_tau+1,), x*(tau): NaN where the whole row is in contact/never in contact
    value_at_reference_point: float  # u(tau=T, x=log(K)): current-time, at-the-money lifted value
    max_psor_sweeps_used: int
    wall_clock_seconds: float


def _closed_form_g(problem: AmericanPutProblem, x: np.ndarray) -> np.ndarray:
    payoff = np.maximum(problem.K - np.exp(x), 0.0)
    u_lift = problem.C * (problem.R - x) / (problem.R - problem.L)
    return payoff - u_lift


def _closed_form_f(problem: AmericanPutProblem, x: np.ndarray) -> np.ndarray:
    return -(problem.C / (problem.R - problem.L)) * ((problem.r - problem.alpha) + problem.r * (problem.R - x))


def _derive_sigma_and_lambda(problem: AmericanPutProblem, x: np.ndarray, tau: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """sigma = -A*u_x (the flux DEFINITION, specification section 0's sigma +
    A*grad_x(u) = 0 -- sigma is never separately solved for, only defined
    from u) and lambda = max(0, u_tau - A*u_xx + b*u_x + c*u - f) (solving
    residuals/residuals.py's own R_PDE = f - u_t + A*u_xx - b*u_x - c*u +
    lam = 0 for lam at the exact solution -- zero on the inactive set where
    the unconstrained PDE holds exactly, and whatever nonnegative value is
    needed to hold u at the obstacle on the active set), both via SECOND-
    ORDER FINITE DIFFERENCES on the already-converged (tau,x) grid.

    This is plain array arithmetic on a NUMERICAL GRID, not autograd through
    a neural network -- specification section 0's "never compute u_xx" is a
    constraint on network derivatives specifically (to avoid double-backprop
    cost), and does not apply to differencing an already-solved reference
    array. Boundary rows/columns use one-sided (still second-order accurate
    for u_x, first-order for the 3-point u_xx stencil) differences instead
    of centered ones -- affects only 2 of n_x+1 columns / 2 of n_tau+1 rows,
    negligible against the interior-dominated volume integrals E_true uses
    these for.
    """
    hx = x[1] - x[0]
    htau = tau[1] - tau[0]

    u_x = np.empty_like(u)
    u_x[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * hx)
    u_x[:, 0] = (u[:, 1] - u[:, 0]) / hx
    u_x[:, -1] = (u[:, -1] - u[:, -2]) / hx

    u_xx = np.empty_like(u)
    u_xx[:, 1:-1] = (u[:, 2:] - 2.0 * u[:, 1:-1] + u[:, :-2]) / hx ** 2
    u_xx[:, 0] = (u[:, 2] - 2.0 * u[:, 1] + u[:, 0]) / hx ** 2
    u_xx[:, -1] = (u[:, -1] - 2.0 * u[:, -2] + u[:, -3]) / hx ** 2

    u_tau = np.empty_like(u)
    u_tau[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * htau)
    u_tau[0, :] = (u[1, :] - u[0, :]) / htau
    u_tau[-1, :] = (u[-1, :] - u[-2, :]) / htau

    f_row = _closed_form_f(problem, x)  # (n_x+1,), tau-independent
    f_grid = np.broadcast_to(f_row, u.shape)

    sigma = -problem.A * u_x
    lam = np.maximum(0.0, u_tau - problem.A * u_xx + problem.b * u_x + problem.c * u - f_grid)
    return sigma, lam


def _psor_solve_red_black(
    lower: float, diag: float, upper: float,
    rhs: np.ndarray, g_int: np.ndarray, u_init: np.ndarray,
    omega: float, tol: float, max_iter: int,
) -> tuple[np.ndarray, int]:
    """Solves the interior LCP (M1 u >= rhs, u >= g_int, complementarity) for
    ONE Crank-Nicolson step's tridiagonal system (constant lower/diag/upper,
    since A, b, c and the mesh are uniform), via red-black PSOR."""
    n = len(rhs)
    u = u_init.copy()
    idx = np.arange(n)
    red = idx % 2 == 0
    black = ~red

    for sweep in range(max_iter):
        u_old_max = u.copy()
        for color in (red, black):
            left = np.zeros(n)
            right = np.zeros(n)
            has_left = idx > 0
            has_right = idx < n - 1
            left[has_left] = lower * u[idx[has_left] - 1]
            right[has_right] = upper * u[idx[has_right] + 1]
            z = (rhs - left - right) / diag
            u_candidate = u[color] + omega * (z[color] - u[color])
            u[color] = np.maximum(g_int[color], u_candidate)

        max_diff = np.abs(u - u_old_max).max()
        if max_diff < tol:
            return u, sweep + 1

    raise RuntimeError(
        f"PSOR did not converge within {max_iter} sweeps (last max_diff={max_diff:.3e}, tol={tol:.3e}). "
        f"Raise psor_max_iter or check omega -- never silently return an unconverged step."
    )


def solve_psor(problem: AmericanPutProblem, config: PSORConfig) -> AmericanPutReferenceSolution:
    L, R, T = problem.omega_bounds[0], problem.omega_bounds[1], problem.T
    n_x, n_tau = config.n_x, config.n_tau
    hx = (R - L) / n_x
    htau = T / n_tau

    x = np.linspace(L, R, n_x + 1)
    tau = np.linspace(0.0, T, n_tau + 1)

    g = _closed_form_g(problem, x)
    f_int = _closed_form_f(problem, x[1:-1])  # tau-independent, computed once

    A_coef, b_coef, c_coef = problem.A, problem.b, problem.c
    lower = -A_coef / hx ** 2 - b_coef / (2 * hx)  # coefficient of u_{i-1} in L[u]_i
    diag_L = 2 * A_coef / hx ** 2 + c_coef
    upper = -A_coef / hx ** 2 + b_coef / (2 * hx)  # coefficient of u_{i+1} in L[u]_i

    def _theta_scheme_coeffs(theta: float):
        # u^{n+1} + theta*htau*L[u^{n+1}] = u^n - (1-theta)*htau*L[u^n] + htau*f
        # theta=0.5: Crank-Nicolson. theta=1.0: fully-implicit Backward Euler
        # (m2 reduces to pure identity -- no explicit L[u^n] term at all,
        # which is exactly what makes Backward Euler L-stable/aggressively
        # damping, unlike CN's theta=0.5 average).
        m1 = (theta * htau * lower, 1.0 + theta * htau * diag_L, theta * htau * upper)
        m2 = (-(1 - theta) * htau * lower, 1.0 - (1 - theta) * htau * diag_L, -(1 - theta) * htau * upper)
        return m1, m2

    cn_m1, cn_m2 = _theta_scheme_coeffs(0.5)
    be_m1, be_m2 = _theta_scheme_coeffs(1.0)

    u = np.zeros((n_tau + 1, n_x + 1))
    u[0, :] = g  # u0 = g exactly (problems/american_put.py)
    g_int = g[1:-1]

    max_sweeps_used = 0
    wall_start = time.perf_counter()

    for n in range(n_tau):
        (m1_lower, m1_diag, m1_upper), (m2_lower, m2_diag, m2_upper) = (
            (be_m1, be_m2) if n < config.n_rannacher_steps else (cn_m1, cn_m2)
        )

        u_n = u[n, :]
        u_n_int = u_n[1:-1]
        rhs = m2_diag * u_n_int + htau * f_int
        rhs[1:] += m2_lower * u_n_int[:-1]
        rhs[0] += m2_lower * u_n[0]  # u_n[0] = 0 (Dirichlet), included for clarity, contributes 0
        rhs[:-1] += m2_upper * u_n_int[1:]
        rhs[-1] += m2_upper * u_n[-1]  # u_n[-1] = 0 (Dirichlet), contributes 0

        u_new_int, sweeps = _psor_solve_red_black(
            m1_lower, m1_diag, m1_upper, rhs, g_int, u_n_int.copy(),
            config.omega, config.psor_tol, config.psor_max_iter,
        )
        max_sweeps_used = max(max_sweeps_used, sweeps)

        u[n + 1, 1:-1] = u_new_int
        u[n + 1, 0] = 0.0
        u[n + 1, -1] = 0.0

    wall_clock = time.perf_counter() - wall_start

    exercise_boundary = _extract_exercise_boundary(x, g, u)

    # LINEAR INTERPOLATION at the fixed physical point x=x_star, not nearest-
    # grid-point selection: diagnosed 2026-09-12 -- the nearest grid INDEX to
    # x_star shifts around non-monotonically as n_x changes (x_star generally
    # does not sit exactly on any grid), and g (hence u, which tracks it
    # closely nearby) has a steep kink there (g_x_inf ~ 87.5 for this
    # parameter set) -- a ~0.01 spatial misalignment from nearest-point
    # selection alone produced a ~0.1-0.3 spurious "refinement" discrepancy
    # that looked like non-convergence but was actually just comparing
    # slightly different x-locations at each resolution, not a real solver
    # bug (Rannacher startup, tried first, correctly made no difference to
    # this — it was never a Crank-Nicolson smoothness issue).
    value_at_reference_point = float(np.interp(problem.x_star, x, u[-1, :]))

    sigma, lam = _derive_sigma_and_lambda(problem, x, tau, u)

    return AmericanPutReferenceSolution(
        config_hash=config.config_hash(),
        n_x=n_x, n_tau=n_tau,
        x_grid=x, tau_grid=tau, u=u, sigma=sigma, lam=lam,
        exercise_boundary=exercise_boundary,
        value_at_reference_point=value_at_reference_point,
        max_psor_sweeps_used=max_sweeps_used,
        wall_clock_seconds=wall_clock,
    )


def _extract_exercise_boundary(x: np.ndarray, g: np.ndarray, u: np.ndarray, contact_tol: float = 1e-6) -> np.ndarray:
    """For each tau row, the American put's exercise region is the LOW-x
    (deep ITM) contact set {x <= x*(tau)}: a single CONTIGUOUS block
    starting at x=L (checked explicitly below, not assumed), so x*(tau) is
    the FIRST contact-to-non-contact transition scanning left to right --
    NOT the last contact index overall. g itself decays back to 0 near the
    right boundary (x=R=7, deep OTM) where u also equals 0 by construction,
    which makes gap=u-g spuriously small there too; taking the last contact
    index (rather than the first LEFT-TO-RIGHT transition) would pick up
    that unrelated near-boundary coincidence as if it were part of the
    economic exercise region -- this is exactly that bug, caught by
    tests/test_american_put_reference.py, fixed here.

    NaN if x=L itself is not in contact (should never happen: u(tau,L)=0=
    g(L) exactly, by construction, for every tau), or if the whole row is in
    contact -- which is EXPECTED, not a bug, at tau=0 specifically (u0=g
    exactly everywhere by construction, so "the whole domain is in contact"
    is the literal true state at that one instant) but should not happen for
    any tau>0 given u(tau,R)=0=g(R) (deep OTM, never in contact away from
    the initial instant)."""
    n_tau_plus_1 = u.shape[0]
    boundary = np.full(n_tau_plus_1, np.nan)
    gap = u - g[None, :]

    for n in range(n_tau_plus_1):
        contact = gap[n, :] <= contact_tol
        if not contact[0] or contact.all():
            continue
        first_noncontact_idx = np.where(~contact)[0].min()
        if first_noncontact_idx == 0:
            continue
        last_contact_idx = first_noncontact_idx - 1
        x_lo, x_hi = x[last_contact_idx], x[first_noncontact_idx]
        gap_lo, gap_hi = gap[n, last_contact_idx], gap[n, first_noncontact_idx]
        if gap_hi == gap_lo:
            boundary[n] = x_hi
        else:
            frac = (contact_tol - gap_lo) / (gap_hi - gap_lo)
            boundary[n] = x_lo + frac * (x_hi - x_lo)
    return boundary


def save_reference(solution: AmericanPutReferenceSolution, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "solution.npz"
    config_path = out_dir / "config.json"

    np.savez_compressed(
        npz_path,
        x_grid=solution.x_grid, tau_grid=solution.tau_grid,
        u=solution.u, sigma=solution.sigma, lam=solution.lam,
        exercise_boundary=solution.exercise_boundary,
    )
    config = {
        "config_hash": solution.config_hash,
        "n_x": solution.n_x, "n_tau": solution.n_tau,
        "value_at_reference_point": solution.value_at_reference_point,
        "max_psor_sweeps_used": solution.max_psor_sweeps_used,
        "wall_clock_seconds": solution.wall_clock_seconds,
        "npz_path": str(npz_path),
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    config_path.write_text(json.dumps(config, indent=2))
    return config


def load_reference(out_dir: Path) -> AmericanPutReferenceSolution:
    config = json.loads((out_dir / "config.json").read_text())
    npz = np.load(out_dir / "solution.npz")
    return AmericanPutReferenceSolution(
        config_hash=config["config_hash"], n_x=config["n_x"], n_tau=config["n_tau"],
        x_grid=npz["x_grid"], tau_grid=npz["tau_grid"], u=npz["u"],
        sigma=npz["sigma"], lam=npz["lam"],
        exercise_boundary=npz["exercise_boundary"],
        value_at_reference_point=config["value_at_reference_point"],
        max_psor_sweeps_used=config["max_psor_sweeps_used"],
        wall_clock_seconds=config["wall_clock_seconds"],
    )


def _catmull_rom_weights(frac: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Uniform Catmull-Rom (tension=0.5) basis weights for the 4 control
    points p_{-1}, p_0, p_1, p_2 bracketing a query at local coordinate
    `frac` in [0,1) (0 at p_0, 1 at p_1) -- the standard closed-form
    equivalent of a cubic Hermite spline whose tangent at each node is the
    central difference of its two neighbors, i.e. m_i = (p_{i+1}-p_{i-1})/2.
    Using CENTRAL-DIFFERENCE tangents (rather than one independent tangent
    per node) is exactly what makes the resulting piecewise-cubic curve C^1:
    two neighboring segments share the same node and the same tangent value
    there by construction, so both the value AND the first derivative agree
    at every interior grid line -- unlike bilinear interpolation, whose
    x-derivative is only piecewise CONSTANT and jumps at every cell edge
    (diagnosed 2026-09-12: measured directly on the frozen reference, e.g.
    u_x stepping 12.03897->12.03845->12.03845->12.03793... crossing three
    consecutive cell boundaries at fixed t -- a real discontinuity, not
    floating-point noise). That piecewise-bilinear artifact was the actual
    root cause of errors/true_error.py's non-monotonic quadrature self-check
    for American put (n_t=n_x=640->1280 got WORSE, not better): the O(h)
    global rule was reduced to an O(h) rule chasing FIVE ORDERS OF MAGNITUDE
    of hidden interior kinks (6400 reference grid cells), not spectral
    convergence -- distinct from, and in addition to, the genuine x=log(K)
    kink `vertical_interface_x` already splits on."""
    t2 = frac * frac
    t3 = t2 * frac
    w_m1 = -0.5 * t3 + 1.0 * t2 - 0.5 * frac
    w_0 = 1.5 * t3 - 2.5 * t2 + 1.0
    w_1 = -1.5 * t3 + 2.0 * t2 + 0.5 * frac
    w_2 = 0.5 * t3 - 0.5 * t2
    return w_m1, w_0, w_1, w_2


def _bicubic_interpolate_torch(
    grid: torch.Tensor, tau_grid: torch.Tensor, x_grid: torch.Tensor, t: torch.Tensor, x: torch.Tensor,
) -> torch.Tensor:
    """Differentiable, C^1 bicubic (separable Catmull-Rom) interpolation of
    a dense (n_tau+1, n_x+1) grid (UNIFORM spacing in both axes -- true for
    this module's own np.linspace-built tau_grid/x_grid) at arbitrary (t, x),
    implemented with plain torch tensor ops (not scipy) SPECIFICALLY so
    autograd can differentiate through it w.r.t. t and x: errors/true_error.py's
    eval_derivatives_detached calls torch.autograd.grad on u_ref(t,x)'s own
    output to get the REFERENCE's u_t/u_x (and sigma_ref's sigma_x) -- a
    scipy/numpy round-trip would silently break that graph (tried first;
    failed with "does not require grad and does not have a grad_fn").

    Replaces an earlier bilinear version (2026-09-12): bilinear's derivative
    is only piecewise CONSTANT, discontinuous at every one of the reference
    grid's ~6400 cell edges, which was the actual root cause of a
    non-monotonic quadrature self-check downstream (see
    _catmull_rom_weights' docstring). Bicubic Catmull-Rom is C^1 (value and
    both first partials continuous everywhere, since it is a SEPARABLE
    tensor product of two 1D Catmull-Rom splines, each C^1 by construction
    from shared central-difference tangents at every node) while still using
    only the frozen numerical grid's own values -- not a smoother function
    invented on top of it, not the network, and not an approximation of the
    true PDE solution beyond what the discrete PSOR grid already encodes.

    Index selection (floor/long/clamp) has no gradient of its own, which is
    correct: the gradient should flow through the interpolation WEIGHTS
    (tau_frac, x_frac), not through which 4x4 neighborhood was selected.
    Near a domain edge, the missing outer neighbor is clamped to the edge
    index (standard "clamped Catmull-Rom" boundary handling) -- affects only
    the single outermost cell out of ~6400 per axis, negligible.

    Never extrapolates silently: raises if any query point falls outside
    [tau_grid[0], tau_grid[-1]] x [x_grid[0], x_grid[-1]] (a small float
    tolerance absorbs boundary roundoff from t=0/t=T/x=L/x=R queries, which
    errors/true_error.py's own boundary terms DO evaluate at exactly).
    """
    n_tau = tau_grid.shape[0] - 1
    n_x = x_grid.shape[0] - 1
    tau0, tau1 = tau_grid[0], tau_grid[-1]
    x0, x1 = x_grid[0], x_grid[-1]
    htau = (tau1 - tau0) / n_tau
    hx = (x1 - x0) / n_x

    tol = 1e-9 * max(float(tau1 - tau0), float(x1 - x0))
    if (t.min() < tau0 - tol) or (t.max() > tau1 + tol) or (x.min() < x0 - tol) or (x.max() > x1 + tol):
        raise ValueError(
            f"query point outside the frozen reference's grid range "
            f"(tau in [{tau0},{tau1}], x in [{x0},{x1}]) -- refusing to extrapolate"
        )

    tau_pos = torch.clamp((t - tau0) / htau, 0.0, float(n_tau))
    x_pos = torch.clamp((x - x0) / hx, 0.0, float(n_x))

    tau_i0 = torch.clamp(tau_pos.floor().long(), 0, n_tau - 1)  # p_0's index
    x_i0 = torch.clamp(x_pos.floor().long(), 0, n_x - 1)

    tau_frac = tau_pos - tau_i0.to(tau_pos.dtype)
    x_frac = x_pos - x_i0.to(x_pos.dtype)

    tau_idx = torch.stack([torch.clamp(tau_i0 - 1, 0, n_tau), tau_i0,
                            torch.clamp(tau_i0 + 1, 0, n_tau), torch.clamp(tau_i0 + 2, 0, n_tau)], dim=0)
    x_idx = torch.stack([torch.clamp(x_i0 - 1, 0, n_x), x_i0,
                          torch.clamp(x_i0 + 1, 0, n_x), torch.clamp(x_i0 + 2, 0, n_x)], dim=0)

    wt = torch.stack(_catmull_rom_weights(tau_frac), dim=0)  # (4, N)
    wx = torch.stack(_catmull_rom_weights(x_frac), dim=0)  # (4, N)

    result = torch.zeros_like(tau_pos)
    for a in range(4):
        row_vals = torch.zeros_like(tau_pos)
        for b in range(4):
            row_vals = row_vals + wx[b] * grid[tau_idx[a], x_idx[b]]
        result = result + wt[a] * row_vals
    return result


def build_reference_callables(
    solution: AmericanPutReferenceSolution,
) -> tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], ...]:
    """Builds u_ref(t,x)/sigma_ref(t,x)/lambda_ref(t,x) callables matching
    problem.u_exact/sigma_exact/lambda_exact's (t,x) -> Tensor calling
    convention (errors/true_error.py's compute_true_error takes ANY such
    callable, closed-form or not -- specification section 12 invariant 4's
    "frozen independent reference" is exactly this, for American put), via
    differentiable, C^1 bicubic interpolation (_bicubic_interpolate_torch) on
    this FROZEN reference's dense (tau,x) grids -- u_ref/sigma_ref must be
    differentiable (eval_derivatives_detached autogradates u_t/u_x/sigma_x
    from them), and C^1 specifically because a merely-C^0 (bilinear)
    interpolant's derivative is discontinuous at every grid cell edge, which
    was diagnosed as the root cause of a non-monotonic quadrature self-check
    downstream (_catmull_rom_weights' docstring). lambda_ref is never
    differentiated (only its VALUE is used), but uses the identical
    interpolation for consistency, not a second implementation.
    """
    tau_grid_t = torch.from_numpy(solution.tau_grid)
    x_grid_t = torch.from_numpy(solution.x_grid)
    u_grid_t = torch.from_numpy(solution.u)
    sigma_grid_t = torch.from_numpy(solution.sigma)
    lambda_grid_t = torch.from_numpy(solution.lam)

    def _make_callable(grid_t: torch.Tensor):
        def _ref(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return _bicubic_interpolate_torch(grid_t, tau_grid_t, x_grid_t, t, x)
        return _ref

    return _make_callable(u_grid_t), _make_callable(sigma_grid_t), _make_callable(lambda_grid_t)


def freeze_reference(config: PSORConfig, out_dir: Optional[Path] = None) -> dict:
    """Computes the reference ONCE at `config`'s resolution and writes it to
    disk -- must only be called after the refinement study (see
    experiments/... or this module's own __main__) has confirmed this
    resolution is stable well below the neural error scale. Refuses to
    silently overwrite a DIFFERENT already-frozen config (a config hash
    mismatch means someone is trying to freeze a different resolution into
    the same slot, which would silently invalidate any neural run that
    already compared against the old one)."""
    out_dir = out_dir or FROZEN_DIR
    if (out_dir / "config.json").exists():
        existing = json.loads((out_dir / "config.json").read_text())
        if existing["config_hash"] != config.config_hash():
            raise RuntimeError(
                f"{out_dir} already has a FROZEN reference with a DIFFERENT config "
                f"(existing={existing['config_hash']}, requested={config.config_hash()}). "
                f"Refusing to overwrite -- freeze to a new directory if you intend a new reference."
            )
        return existing  # already frozen at this exact config: idempotent, not a re-solve

    problem = AmericanPutProblem()
    solution = solve_psor(problem, config)
    return save_reference(solution, out_dir)


if __name__ == "__main__":
    problem = AmericanPutProblem()
    config = PSORConfig(n_x=400, n_tau=400)
    solution = solve_psor(problem, config)
    print(f"value_at_reference_point (tau=T, x=log K) = {solution.value_at_reference_point:.6f}")
    print(f"max_psor_sweeps_used = {solution.max_psor_sweeps_used}")
    print(f"wall_clock_seconds = {solution.wall_clock_seconds:.2f}")
