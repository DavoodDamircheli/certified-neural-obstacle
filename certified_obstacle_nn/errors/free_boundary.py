"""Free-boundary location error E_FB (M18, threshold-based redefinition
2026-09-11 -- supersedes the earlier zero-crossing-of-gamma definition used
for the first M18 sweep; that version's tests/usage are replaced wholesale,
not kept alongside this one).

Two independent CONTACT DETECTORS, both built only from quantities the
neural triple itself produces (never the exact solution):

  (i)  gap-only:         |u_theta(t,x) - g(t,x)| <= eps_u
  (ii) gap + multiplier: (i)  AND  lambda_psi(t,x) >= eps_lambda

For each t on a fine Gauss-Legendre grid, s_theta(t) = min{x : criterion(t,x)
holds} -- the leftmost x (scanning left to right) where the network's own
detector first reports contact, refined by LINEAR interpolation (not
bisection -- an explicit, simpler choice for this milestone) between the
last non-contact scan point and the first contact scan point. Both criteria
are threshold crossings of a scalar MARGIN (positive where the criterion
holds): margin_i = eps_u - gap; margin_ii = min(eps_u - gap, lambda_psi -
eps_lambda) (the min of the two individual margins gives min>=0 exactly iff
both individual conditions hold, i.e. the AND -- a single well-defined
quantity to linearly interpolate the zero-crossing of, rather than two
independent ones). Fallback conventions: if the criterion holds already at
the leftmost scan point (x=a), s_theta=a (nothing to interpolate); if it
never holds anywhere in the scan, s_theta=b (predicted no contact).

E_FB = ||s_theta - s||_{L2(0,T)}, s(t)=t for Stefan, integrated via the same
deterministic Gauss-Legendre convention as errors/true_error.py.

Thresholds are FIXED ONCE, never retuned per run:
    eps_u      = ALPHA_U      * ||g||_inf           (problem.g_inf: EXACT,
                                                       closed-form, run-independent)
    eps_lambda = ALPHA_LAMBDA * ||lambda_psi||_inf   (EMPIRICAL max over this
                                                       run's own scan grid --
                                                       reused, not a second
                                                       pass; a tighter,
                                                       more meaningful
                                                       reference for a
                                                       THRESHOLD criterion
                                                       than the a priori
                                                       certification bound
                                                       Sigma_Mps^2 would be)
ALPHA_U, ALPHA_LAMBDA are module-level constants (below), identical across
every architecture and seed. Their choice is an engineering judgment call,
not a specification formula -- compute_free_boundary_error_sensitivity_sweep
below recomputes E_FB over a grid of alternative (alpha_u, alpha_lambda)
values (reusing the same network-evaluation grid, so the sweep costs no
extra forward passes) as a separate appendix artifact, so the "official"
Table-1 numbers are not laundered through an unexamined threshold choice.

float64 throughout. The model must already be frozen (triple.eval(), no
grad) before this is called -- mirrors certification/'s invariants
(specification section 12 invariants 2-3): this is a REPORTED diagnostic, never
trained on.

free_boundary_fn=None (no known closed-form interface, e.g. American put,
Phase 4): there is nothing to compare s_theta against, so callers should
skip this computation entirely (mirrors the module's own earlier version).
"""

import math
from typing import Callable, NamedTuple

import numpy as np
import torch

from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.base_problem import BaseProblem

FreeBoundaryFn = Callable[[torch.Tensor], torch.Tensor]

# Fixed once, identical for every M18 run (architecture x seed) -- never
# retuned. 2% of the relevant sup-norm: tight enough to be a meaningful
# contact discriminator, loose enough to tolerate ordinary training noise
# (the sensitivity sweep below characterizes how much this choice matters).
ALPHA_U = 0.02
ALPHA_LAMBDA = 0.02


class FreeBoundaryErrorResult(NamedTuple):
    E_FB_gap_only: float
    E_FB_gap_and_multiplier: float
    E_FB_gap_only_squared: float
    E_FB_gap_and_multiplier_squared: float
    eps_u: float
    eps_lambda: float
    alpha_u: float
    alpha_lambda: float
    g_inf: float
    lambda_psi_inf_empirical: float
    quadrature_relative_change_gap_only: float
    quadrature_relative_change_gap_and_multiplier: float
    n_t: int
    n_scan: int


def _gauss_legendre(n: int, lo: float, hi: float) -> tuple[torch.Tensor, torch.Tensor]:
    nodes, weights = np.polynomial.legendre.leggauss(n)
    scale = (hi - lo) / 2.0
    mapped_nodes = scale * nodes + (hi + lo) / 2.0
    mapped_weights = scale * weights
    return (
        torch.tensor(mapped_nodes, dtype=torch.float64),
        torch.tensor(mapped_weights, dtype=torch.float64),
    )


def _evaluate_scan_grid(
    problem: BaseProblem, triple: NeuralTriple, t_nodes: torch.Tensor, a: float, b: float, n_scan: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """u_theta, g, lambda_psi on the full (n_t, n_scan) grid, one forward
    pass each -- shared by both detectors and by the sensitivity sweep
    (threshold combinations are cheap re-combinations of this same grid,
    never a re-evaluation of the network)."""
    n_t = t_nodes.shape[0]
    x_scan = torch.linspace(a, b, n_scan, dtype=torch.float64)

    t_grid = t_nodes[:, None].expand(n_t, n_scan).reshape(-1, 1)
    x_grid = x_scan[None, :].expand(n_t, n_scan).reshape(-1, 1)
    with torch.no_grad():
        u_theta = triple.state_net(t_grid, x_grid).reshape(n_t, n_scan)
        g = problem.g(t_grid, x_grid).reshape(n_t, n_scan)
        lambda_psi = triple.multiplier_net(t_grid, x_grid).reshape(n_t, n_scan)

    return x_scan, u_theta, g, lambda_psi


def _locate_leftmost_crossing(margin_grid: torch.Tensor, x_scan: torch.Tensor) -> torch.Tensor:
    """s_theta(t) for every row of margin_grid (n_t, n_scan): leftmost x
    where margin >= 0, linearly interpolated between the last margin<0 point
    and the first margin>=0 point. margin<0 everywhere -> s_theta = x_scan[-1]
    (never in contact); margin>=0 at x_scan[0] -> s_theta = x_scan[0]
    (nothing to interpolate)."""
    holds = margin_grid >= 0.0
    any_holds = holds.any(dim=1)
    first_idx = holds.to(torch.float64).argmax(dim=1)  # first True index per row (0 if none, handled via any_holds)
    prev_idx = (first_idx - 1).clamp(min=0)

    x_j = x_scan[first_idx]
    x_jm1 = x_scan[prev_idx]
    m_j = torch.gather(margin_grid, 1, first_idx.unsqueeze(1)).squeeze(1)
    m_jm1 = torch.gather(margin_grid, 1, prev_idx.unsqueeze(1)).squeeze(1)

    denom = m_j - m_jm1
    safe_denom = torch.where(denom.abs() < 1e-300, torch.ones_like(denom), denom)
    frac = (0.0 - m_jm1) / safe_denom
    s_interp = x_jm1 + (x_j - x_jm1) * frac

    s_theta = torch.where(first_idx == 0, x_scan[0].expand_as(s_interp), s_interp)
    s_theta = torch.where(~any_holds, torch.full_like(s_theta, x_scan[-1]), s_theta)
    return s_theta


def _e_fb_from_margins(
    margin_i_grid: torch.Tensor, margin_ii_grid: torch.Tensor,
    x_scan: torch.Tensor, t_nodes: torch.Tensor, t_weights: torch.Tensor,
    free_boundary_fn: FreeBoundaryFn,
) -> tuple[float, float]:
    s_i = _locate_leftmost_crossing(margin_i_grid, x_scan)
    s_ii = _locate_leftmost_crossing(margin_ii_grid, x_scan)
    s_exact = free_boundary_fn(t_nodes)
    e_fb_sq_i = (t_weights * (s_i - s_exact).pow(2)).sum().item()
    e_fb_sq_ii = (t_weights * (s_ii - s_exact).pow(2)).sum().item()
    return e_fb_sq_i, e_fb_sq_ii


def _e_fb_squared_at_resolution(
    problem: BaseProblem, triple: NeuralTriple, free_boundary_fn: FreeBoundaryFn,
    n_t: int, n_scan: int, eps_u: float, eps_lambda: float,
) -> tuple[float, float]:
    a, b = problem.omega_bounds
    T = problem.T
    t_nodes, t_weights = _gauss_legendre(n_t, 0.0, T)
    x_scan, u_theta, g, lambda_psi = _evaluate_scan_grid(problem, triple, t_nodes, a, b, n_scan)

    gap = (u_theta - g).abs()
    margin_i = eps_u - gap
    margin_ii = torch.minimum(eps_u - gap, lambda_psi - eps_lambda)

    return _e_fb_from_margins(margin_i, margin_ii, x_scan, t_nodes, t_weights, free_boundary_fn)


def compute_free_boundary_error(
    problem: BaseProblem,
    triple: NeuralTriple,
    free_boundary_fn: FreeBoundaryFn,
    n_t: int = 200,
    n_scan: int = 401,
    quadrature_tolerance: float = 1e-2,
    alpha_u: float = ALPHA_U,
    alpha_lambda: float = ALPHA_LAMBDA,
) -> FreeBoundaryErrorResult:
    triple.eval()
    a, b = problem.omega_bounds
    T = problem.T

    # eps_u/eps_lambda fixed once per run: g_inf is problem data (exact,
    # run-independent); lambda_psi_inf is this run's own empirical max,
    # taken over a dense reference scan (2x the requested n_t/n_scan) so the
    # threshold itself does not depend on the self-check's own resolution.
    ref_t_nodes, _ = _gauss_legendre(2 * n_t, 0.0, T)
    _, _, _, lambda_psi_ref = _evaluate_scan_grid(problem, triple, ref_t_nodes, a, b, 2 * n_scan)
    lambda_psi_inf_empirical = lambda_psi_ref.max().item()

    eps_u = alpha_u * problem.g_inf
    eps_lambda = alpha_lambda * lambda_psi_inf_empirical

    e_fb_sq_i_1x, e_fb_sq_ii_1x = _e_fb_squared_at_resolution(
        problem, triple, free_boundary_fn, n_t, n_scan, eps_u, eps_lambda,
    )
    e_fb_sq_i_2x, e_fb_sq_ii_2x = _e_fb_squared_at_resolution(
        problem, triple, free_boundary_fn, 2 * n_t, n_scan, eps_u, eps_lambda,
    )

    def _rel_change(sq_1x, sq_2x):
        e_1x = math.sqrt(sq_1x)
        e_2x = math.sqrt(sq_2x)
        return abs(e_2x - e_1x) / max(e_2x, 1e-300)

    rel_change_i = _rel_change(e_fb_sq_i_1x, e_fb_sq_i_2x)
    rel_change_ii = _rel_change(e_fb_sq_ii_1x, e_fb_sq_ii_2x)

    if rel_change_i > quadrature_tolerance or rel_change_ii > quadrature_tolerance:
        raise RuntimeError(
            f"E_FB quadrature self-check failed: gap-only rel change={rel_change_i:.3e}, "
            f"gap+multiplier rel change={rel_change_ii:.3e} (tolerance={quadrature_tolerance:.0e}) "
            f"when n_t doubled ({n_t} -> {2 * n_t}). Result is not trustworthy at this "
            f"resolution -- do not use it."
        )

    return FreeBoundaryErrorResult(
        E_FB_gap_only=math.sqrt(e_fb_sq_i_2x),
        E_FB_gap_and_multiplier=math.sqrt(e_fb_sq_ii_2x),
        E_FB_gap_only_squared=e_fb_sq_i_2x,
        E_FB_gap_and_multiplier_squared=e_fb_sq_ii_2x,
        eps_u=eps_u,
        eps_lambda=eps_lambda,
        alpha_u=alpha_u,
        alpha_lambda=alpha_lambda,
        g_inf=problem.g_inf,
        lambda_psi_inf_empirical=lambda_psi_inf_empirical,
        quadrature_relative_change_gap_only=rel_change_i,
        quadrature_relative_change_gap_and_multiplier=rel_change_ii,
        n_t=2 * n_t,
        n_scan=n_scan,
    )


def compute_free_boundary_error_sensitivity_sweep(
    problem: BaseProblem,
    triple: NeuralTriple,
    free_boundary_fn: FreeBoundaryFn,
    alpha_u_values: list[float],
    alpha_lambda_values: list[float],
    n_t: int = 200,
    n_scan: int = 401,
) -> list[dict]:
    """Appendix-only sensitivity sweep over BOTH thresholds. Reuses ONE
    (n_t, n_scan) network-evaluation grid across the entire (alpha_u,
    alpha_lambda) cross-product -- eps_u/eps_lambda only rescale the already-
    computed gap/lambda_psi grids, so this costs one forward pass total, not
    one per threshold combination. No quadrature self-check here (this is a
    diagnostic sweep, not a certified/reported number); n_t/n_scan should be
    set to values already known-converged for this triple via
    compute_free_boundary_error."""
    triple.eval()
    a, b = problem.omega_bounds
    T = problem.T
    t_nodes, t_weights = _gauss_legendre(n_t, 0.0, T)
    x_scan, u_theta, g, lambda_psi = _evaluate_scan_grid(problem, triple, t_nodes, a, b, n_scan)
    gap = (u_theta - g).abs()

    lambda_psi_inf_empirical = lambda_psi.max().item()
    g_inf = problem.g_inf

    rows = []
    for alpha_u in alpha_u_values:
        eps_u = alpha_u * g_inf
        margin_i = eps_u - gap
        for alpha_lambda in alpha_lambda_values:
            eps_lambda = alpha_lambda * lambda_psi_inf_empirical
            margin_ii = torch.minimum(margin_i, lambda_psi - eps_lambda)
            e_fb_sq_i, e_fb_sq_ii = _e_fb_from_margins(margin_i, margin_ii, x_scan, t_nodes, t_weights, free_boundary_fn)
            rows.append({
                "alpha_u": alpha_u,
                "alpha_lambda": alpha_lambda,
                "eps_u": eps_u,
                "eps_lambda": eps_lambda,
                "E_FB_gap_only": math.sqrt(e_fb_sq_i),
                "E_FB_gap_and_multiplier": math.sqrt(e_fb_sq_ii),
            })
    return rows
