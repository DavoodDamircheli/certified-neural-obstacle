"""True (weak-norm) error, specification section 8.

E_true^2 = ||grad_x e_u||_{L2(Q)}^2 + ||e_u(0)||_{L2(Om)}^2 + ||e_u(T)||_{L2(Om)}^2
         + ||e_s||_{L2(Q)}^2 + ||d_t e_u + div_x e_s - e_l||_{L2(Q)}^2

with e_u = u - u_theta, e_s = sigma - sigma_phi, e_l = lambda - lambda_psi. `u`,
`sigma`, `lambda` here are the independent reference (closed-form for Stefan,
a frozen numerical reference for American put) -- never the network's own
estimate of itself (specification section 12 invariant 4).

This is the ONLY error norm the theorem actually bounds: E_true <= E_cert_U.
Computed via a deterministic tensor-product Gauss-Legendre rule -- never
Monte Carlo, never the training collocation points (this function does not
take, or need, the training sampler at all).

A quadrature self-check recomputes everything at 2x resolution and compares.
If the relative change in E_true exceeds `quadrature_tolerance`, this raises
rather than returning a number that might be quadrature noise.

Free-boundary-aware quadrature (diagnosed 2026-09-10, this session's Pre-M18
milestone T1). A single GLOBAL tensor-product rule over Q only gets O(h) (not
spectral) convergence for Stefan, and the per-term breakdown of E_true^2
shows exactly why: at n_t=n_x=100 vs 2x, term_consistency's relative change
is 92.0% while every other term is under 0.06% (term_lambda_raw, a
diagnostic-only quantity not part of E_true^2, is 1.8%). term_consistency
contains e_l = lambda - lambda_psi, and Stefan's exact lambda = 1_{t<x} JUMPS
across the free boundary t=x (also sigma_x jumps there); u, u_x, u_t, and
sigma itself are all continuous there (only their cross terms inside
term_consistency are not) -- consistent with term_consistency alone carrying
the non-convergence.

Since s(t)=t is known in EXACT closed form (not estimated), the fix is not an
approximation: split Q along the interface into {t>x} and {t<x} and integrate
each with the FULL Gauss-Legendre rule separately, where R is smooth on each
piece (`free_boundary_fn` below, `problem.free_boundary_exact` for Stefan).
Both regions get their own full-order rule, so this is 2x the interior points
of the old global grid at the same (n_t, n_x) -- an intentional, correct cost
of doing the quadrature right, not a resolution change to hide behind.
`free_boundary_fn=None` (the default) falls back to the old global rule
unchanged, for any problem without a known closed-form interface (e.g.
American put, Phase 4 -- its free boundary is not known a priori).

Streaming (docs/history/MEMORY_OOM_DEBUGGING_AND_SAFE_RUN_POLICY.md): every point cloud
here -- the global grid (free_boundary_fn=None) or each of the two regions
(free_boundary_fn given) -- is generated and consumed in ROW CHUNKS, the full
grid never materialized at once, and the base-resolution / refined-resolution
passes run strictly SEQUENTIALLY with an explicit `gc.collect()` between them
(only the reduced scalar terms are ever compared between resolutions).

Derivatives throughout use eval_derivatives_detached (create_graph=False),
not compute_derivatives (create_graph=True, training-only) -- this evaluates
BOTH the network triple's sub-networks and the independent reference solution
through the identical code path, and the reference (e.g. problem.u_exact) is
a bare callable, not a NeuralTriple with parameters to assert frozen, so
eval_triple_derivatives (which requires exactly that) does not fit either
side of this comparison.
"""

import gc
import math
from typing import Callable, Iterator, NamedTuple, Optional

import numpy as np
import torch

from certified_obstacle_nn.autograd.derivatives import eval_derivatives_detached
from certified_obstacle_nn.certification.memory import MemoryGuard, report_memory
from certified_obstacle_nn.certification.raw_estimator import (
    DEFAULT_CHECK_EVERY_N_CHUNKS,
    DEFAULT_MAX_RSS_GB,
    DEFAULT_POINTS_PER_CHUNK,
)
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.base_problem import BaseProblem

NetFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
FreeBoundaryFn = Callable[[torch.Tensor], torch.Tensor]


class TrueErrorResult(NamedTuple):
    E_true: float
    E_true_squared: float
    term_grad_u: float
    term_u0: float
    term_uT: float
    term_sigma: float
    term_consistency: float
    norm_u_V: float
    norm_sigma_L2: float
    norm_lambda_L2: float
    quadrature_relative_change: float
    n_t: int
    n_x: int


def _gauss_legendre(n: int, lo: float, hi: float) -> tuple[torch.Tensor, torch.Tensor]:
    """n-point Gauss-Legendre nodes/weights mapped from [-1,1] to [lo,hi]."""
    nodes, weights = np.polynomial.legendre.leggauss(n)
    scale = (hi - lo) / 2.0
    mapped_nodes = scale * nodes + (hi + lo) / 2.0
    mapped_weights = scale * weights
    return (
        torch.tensor(mapped_nodes, dtype=torch.float64),
        torch.tensor(mapped_weights, dtype=torch.float64),
    )


def _gauss_legendre_per_row_variable_bounds(
    lo_per_row: torch.Tensor, hi_per_row: torch.Tensor, n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Like _gauss_legendre, but a DIFFERENT [lo,hi] sub-interval per row --
    generalizes the fixed-edges per-row GL mapping pattern used elsewhere
    (e.g. raw_estimator.py's boundary terms) to a free-boundary-CONFORMING
    sub-interval per row (e.g. t in (x_i, T) or (0, x_i) for each x_i). This
    is an AFFINE, hence perfectly smooth, reparametrization per row, so
    Gauss-Legendre's spectral convergence for a smooth integrand is fully
    preserved within each row's own sub-interval -- not an approximation.
    """
    ref_nodes, ref_weights = np.polynomial.legendre.leggauss(n)
    ref_nodes = torch.tensor(ref_nodes, dtype=torch.float64)
    ref_weights = torch.tensor(ref_weights, dtype=torch.float64)

    half = (hi_per_row - lo_per_row) / 2.0
    mid = (hi_per_row + lo_per_row) / 2.0

    nodes = mid[:, None] + half[:, None] * ref_nodes[None, :]
    weights = half[:, None] * ref_weights[None, :]
    return nodes, weights


def _iter_dense_grid_chunks(
    t_nodes: torch.Tensor, t_weights: torch.Tensor, x_nodes: torch.Tensor, x_weights: torch.Tensor, points_per_chunk: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yields (t, x, w) row-blocks of the full (n_t, n_x) tensor-product
    Gauss-Legendre grid -- the full grid is never materialized, only
    `rows_per_chunk` rows (x `n_x`) at a time. Used when free_boundary_fn is
    None (no known interface to split on)."""
    n_t = t_nodes.shape[0]
    n_x = x_nodes.shape[0]
    rows_per_chunk = max(1, points_per_chunk // n_x)

    for start in range(0, n_t, rows_per_chunk):
        end = min(start + rows_per_chunk, n_t)
        t_block = t_nodes[start:end]
        tw_block = t_weights[start:end]

        t_grid, x_grid = torch.meshgrid(t_block, x_nodes, indexing="ij")
        w_grid = torch.outer(tw_block, x_weights)

        yield t_grid.reshape(-1, 1), x_grid.reshape(-1, 1), w_grid.reshape(-1, 1)


def _iter_region_chunks(
    x_nodes: torch.Tensor,
    x_weights: torch.Tensor,
    t_lo_fn: Callable[[torch.Tensor], torch.Tensor],
    t_hi_fn: Callable[[torch.Tensor], torch.Tensor],
    n_t: int,
    points_per_chunk: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yields (t, x, w) row-blocks of a region {t_lo_fn(x) < t < t_hi_fn(x)},
    one FULL n_t-point Gauss-Legendre rule per x-row (not shared across
    rows, since the t-bounds vary per row) -- the full region's grid is
    never materialized, only `rows_per_chunk` x-rows (x n_t) at a time.
    A degenerate (zero-width) row (t_lo==t_hi, e.g. at x=0 for one region and
    x=1 for the other) contributes exactly zero: correct, not a special case.
    """
    n_x = x_nodes.shape[0]
    rows_per_chunk = max(1, points_per_chunk // n_t)

    for start in range(0, n_x, rows_per_chunk):
        end = min(start + rows_per_chunk, n_x)
        x_block = x_nodes[start:end]
        xw_block = x_weights[start:end]

        t_lo_block = t_lo_fn(x_block)
        t_hi_block = t_hi_fn(x_block)
        t_nodes_block, t_weights_block = _gauss_legendre_per_row_variable_bounds(t_lo_block, t_hi_block, n_t)  # (rows, n_t)

        x_grid = x_block[:, None].expand(-1, n_t)
        w_grid = xw_block[:, None] * t_weights_block  # (rows, n_t)

        yield t_nodes_block.reshape(-1, 1), x_grid.reshape(-1, 1), w_grid.reshape(-1, 1)


def _accumulate_region(
    problem: BaseProblem,
    u_theta: NetFn,
    sigma_phi: NetFn,
    lambda_psi: NetFn,
    u_ref: NetFn,
    sigma_ref: NetFn,
    lambda_ref: NetFn,
    chunks: Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    guard: MemoryGuard,
    check_every_n_chunks: int,
    term_grad_u_parts: list,
    term_sigma_parts: list,
    term_consistency_parts: list,
    term_lambda_raw_parts: list,
) -> None:
    """Consumes one region's (or the whole global grid's) chunk iterator,
    appending each chunk's contribution to the shared part-lists -- the same
    accumulation logic regardless of whether `chunks` came from
    _iter_dense_grid_chunks or _iter_region_chunks, so splitting into two
    regions is purely a matter of calling this twice with different
    iterators, not a second formula."""
    for chunk_idx, (t_flat, x_flat, w_flat) in enumerate(chunks):
        u_th, sig_th, lam_th, u_t_th, u_x_th, sig_x_th = eval_derivatives_detached(u_theta, sigma_phi, lambda_psi, t_flat, x_flat)
        u_rf, sig_rf, lam_rf, u_t_rf, u_x_rf, sig_x_rf = eval_derivatives_detached(u_ref, sigma_ref, lambda_ref, t_flat, x_flat)

        e_s = sig_rf - sig_th
        e_l = lam_rf - lam_th
        grad_x_e_u = u_x_rf - u_x_th
        consistency = (u_t_rf - u_t_th) + (sig_x_rf - sig_x_th) - e_l

        term_grad_u_parts.append((w_flat * grad_x_e_u.pow(2)).sum().item())
        term_sigma_parts.append((w_flat * e_s.pow(2)).sum().item())
        term_consistency_parts.append((w_flat * consistency.pow(2)).sum().item())
        term_lambda_raw_parts.append((w_flat * e_l.pow(2)).sum().item())

        del (
            t_flat, x_flat, w_flat,
            u_th, sig_th, lam_th, u_t_th, u_x_th, sig_x_th,
            u_rf, sig_rf, lam_rf, u_t_rf, u_x_rf, sig_x_rf,
            e_s, e_l, grad_x_e_u, consistency,
        )

        if (chunk_idx + 1) % check_every_n_chunks == 0:
            guard.check(f"true_error chunk {chunk_idx + 1}")


def _volume_and_boundary_terms(
    problem: BaseProblem,
    u_theta: NetFn,
    sigma_phi: NetFn,
    lambda_psi: NetFn,
    u_ref: NetFn,
    sigma_ref: NetFn,
    lambda_ref: NetFn,
    n_t: int,
    n_x: int,
    *,
    points_per_chunk: int,
    max_rss_gb: float,
    check_every_n_chunks: int,
    free_boundary_fn: Optional[FreeBoundaryFn] = None,
    vertical_interface_x: Optional[float] = None,
) -> dict:
    """vertical_interface_x (2026-09-12, American put): the obstacle g has a
    kink at a FIXED x=x* (specification section 10's log(K) data interface),
    independent of t -- unlike Stefan's t=x diagonal (free_boundary_fn,
    mutually exclusive with this), this needs only a two-RECTANGLE split
    ([a,x*)x[0,T] and (x*,b]x[0,T], both fixed, no per-row-varying t-bounds
    machinery) reusing the SAME dense-grid rule twice. Diagnosed 2026-09-12:
    American put's reference (an interpolant of a frozen numerical solution,
    not a smooth closed form) showed a genuinely NON-MONOTONIC quadrature
    self-check (n_t=n_x=640 gave 1.1e-3 relative change, doubling to 1280
    gave 2.5e-2 -- WORSE, not better) with a single global rule -- the same
    class of failure Stefan's t=x kink caused, fixed the same way: split at
    the KNOWN interface, don't chase a moving target with more points. (A
    SECOND, distinct source of the same symptom was also found and fixed at
    its root in reference/american_put_reference.py: the reference was
    originally interpolated bilinearly, whose derivative is discontinuous at
    every one of the reference grid's ~6400 cell edges -- replaced with a C^1
    bicubic interpolant, see that module's _catmull_rom_weights docstring.
    Both fixes are needed; neither alone fully restores clean convergence.)
    Because g's own kink sits at x=x* for EVERY t (unlike Stefan, where t=0/T
    only meet the diagonal at a single measure-zero corner), the t=0/t=T
    boundary terms ALSO need the same split here for spectral convergence of
    those 1D integrals -- not true for Stefan, handled below."""
    a, b = problem.omega_bounds
    T = problem.T

    guard = MemoryGuard(
        max_rss_gb=max_rss_gb, label_prefix="true_error",
        chunk_size=points_per_chunk, grid=(n_t, n_x),
    )

    term_grad_u_parts: list[float] = []
    term_sigma_parts: list[float] = []
    term_consistency_parts: list[float] = []
    term_lambda_raw_parts: list[float] = []
    accumulate_args = (
        problem, u_theta, sigma_phi, lambda_psi, u_ref, sigma_ref, lambda_ref,
    )

    if vertical_interface_x is not None:
        assert free_boundary_fn is None, "free_boundary_fn and vertical_interface_x are mutually exclusive interface conventions"
        t_nodes, t_weights = _gauss_legendre(n_t, 0.0, T)
        x_nodes_left, x_weights_left = _gauss_legendre(n_x, a, vertical_interface_x)
        x_nodes_right, x_weights_right = _gauss_legendre(n_x, vertical_interface_x, b)
        for x_nodes, x_weights in ((x_nodes_left, x_weights_left), (x_nodes_right, x_weights_right)):
            chunks = _iter_dense_grid_chunks(t_nodes, t_weights, x_nodes, x_weights, points_per_chunk)
            _accumulate_region(
                *accumulate_args, chunks, guard, check_every_n_chunks,
                term_grad_u_parts, term_sigma_parts, term_consistency_parts, term_lambda_raw_parts,
            )
    elif free_boundary_fn is None:
        t_nodes, t_weights = _gauss_legendre(n_t, 0.0, T)
        x_nodes, x_weights = _gauss_legendre(n_x, a, b)
        chunks = _iter_dense_grid_chunks(t_nodes, t_weights, x_nodes, x_weights, points_per_chunk)
        _accumulate_region(
            *accumulate_args, chunks, guard, check_every_n_chunks,
            term_grad_u_parts, term_sigma_parts, term_consistency_parts, term_lambda_raw_parts,
        )
    else:
        # Free-boundary-aware split: {t > x} and {t < x}, each with its own
        # FULL n_t-point rule per x-row (module docstring). free_boundary_fn
        # gives x as a function of t (established convention, e.g.
        # problem.free_boundary_exact); Stefan's interface t=x is self-
        # inverse, so the SAME callable also gives t as a function of x here
        # -- true only because of that specific symmetry, not assumed for a
        # general (asymmetric) interface.
        x_nodes, x_weights = _gauss_legendre(n_x, a, b)

        def t_lo_active(x):
            return free_boundary_fn(x)

        def t_hi_active(x):
            return torch.full_like(x, T)

        def t_lo_inactive(x):
            return torch.zeros_like(x)

        def t_hi_inactive(x):
            return free_boundary_fn(x)

        chunks_active = _iter_region_chunks(x_nodes, x_weights, t_lo_active, t_hi_active, n_t, points_per_chunk)
        _accumulate_region(
            *accumulate_args, chunks_active, guard, check_every_n_chunks,
            term_grad_u_parts, term_sigma_parts, term_consistency_parts, term_lambda_raw_parts,
        )
        chunks_inactive = _iter_region_chunks(x_nodes, x_weights, t_lo_inactive, t_hi_inactive, n_t, points_per_chunk)
        _accumulate_region(
            *accumulate_args, chunks_inactive, guard, check_every_n_chunks,
            term_grad_u_parts, term_sigma_parts, term_consistency_parts, term_lambda_raw_parts,
        )

    term_grad_u = math.fsum(term_grad_u_parts)
    term_sigma = math.fsum(term_sigma_parts)
    term_consistency = math.fsum(term_consistency_parts)
    term_lambda_raw = math.fsum(term_lambda_raw_parts)

    # Boundary terms: O(n_x) only, not O(n_t*n_x) -- cheap regardless of grid
    # resolution, hence not chunked (same reasoning as raw_estimator.py's
    # R_0/w(0)/w(T) boundary pass). For Stefan (free_boundary_fn or neither),
    # t=0 and t=T do not sit ON the interface (except at the single,
    # measure-zero corner point x=0,t=0 or x=1,t=T), so these terms need no
    # splitting. For American put (vertical_interface_x), g's kink sits at
    # x=x* for EVERY t, including t=0/T exactly -- u0=g(0,.) has the SAME
    # kink -- so these 1D integrals are split the same way the volume terms
    # are (module docstring).
    if vertical_interface_x is not None:
        x_node_sets = [(x_nodes_left, x_weights_left), (x_nodes_right, x_weights_right)]
    else:
        x_node_sets = [(x_nodes, x_weights)]

    term_u0 = 0.0
    term_uT = 0.0
    with torch.no_grad():
        for xn, xw in x_node_sets:
            x_col = xn.reshape(-1, 1)
            x_w_col = xw.reshape(-1, 1)
            zeros_t = torch.zeros_like(x_col)
            T_t = torch.full_like(x_col, T)

            e_u0 = u_ref(zeros_t, x_col) - u_theta(zeros_t, x_col)
            term_u0 += (x_w_col * e_u0.pow(2)).sum().item()

            e_uT = u_ref(T_t, x_col) - u_theta(T_t, x_col)
            term_uT += (x_w_col * e_uT.pow(2)).sum().item()

    return {
        "term_grad_u": term_grad_u,
        "term_u0": term_u0,
        "term_uT": term_uT,
        "term_sigma": term_sigma,
        "term_consistency": term_consistency,
        "term_lambda_raw": term_lambda_raw,
    }


def _e_true_squared(terms: dict) -> float:
    return terms["term_grad_u"] + terms["term_u0"] + terms["term_uT"] + terms["term_sigma"] + terms["term_consistency"]


def compute_true_error(
    problem: BaseProblem,
    triple: NeuralTriple,
    n_t: int = 60,
    n_x: int = 60,
    quadrature_tolerance: float = 1e-3,
    *,
    points_per_chunk: int = DEFAULT_POINTS_PER_CHUNK,
    max_rss_gb: float = DEFAULT_MAX_RSS_GB,
    check_every_n_chunks: int = DEFAULT_CHECK_EVERY_N_CHUNKS,
    free_boundary_fn: Optional[FreeBoundaryFn] = None,
    vertical_interface_x: Optional[float] = None,
    u_ref: Optional[NetFn] = None,
    sigma_ref: Optional[NetFn] = None,
    lambda_ref: Optional[NetFn] = None,
) -> TrueErrorResult:
    """`free_boundary_fn` (x as a function of t, e.g. problem.free_boundary_exact
    for Stefan): when given, integrates {t>x} and {t<x} separately with the
    full Gauss rule on each (module docstring) -- restores the intended
    quadrature_tolerance=1e-3 self-check for problems with a known closed-
    form interface. Default None preserves the old global-grid behavior
    exactly, for problems without one (American put, Phase 4).

    `u_ref`/`sigma_ref`/`lambda_ref` (2026-09-12, specification section 12
    invariant 4: "E_true is computed only from closed-form (Stefan) or a
    FROZEN INDEPENDENT reference (American put, Phase 4)"): optional
    (t,x)->Tensor callables, e.g.
    reference/american_put_reference.py's build_reference_callables (an
    interpolated frozen numerical solution, not closed-form). Default None
    for each falls back to problem.u_exact/sigma_exact/lambda_exact
    respectively -- preserves Stefan's exact existing behavior unchanged.

    `vertical_interface_x` (American put's x=log(K)): mutually exclusive
    with `free_boundary_fn` -- see _volume_and_boundary_terms's own
    docstring for why this needed a genuinely different split (a fixed x*
    for every t, not a t-dependent curve)."""
    triple.eval()
    u_ref = u_ref if u_ref is not None else problem.u_exact
    sigma_ref = sigma_ref if sigma_ref is not None else problem.sigma_exact
    lambda_ref = lambda_ref if lambda_ref is not None else problem.lambda_exact

    terms_1x = _volume_and_boundary_terms(
        problem, triple.state_net, triple.flux_net, triple.multiplier_net,
        u_ref, sigma_ref, lambda_ref,
        n_t, n_x,
        points_per_chunk=points_per_chunk, max_rss_gb=max_rss_gb, check_every_n_chunks=check_every_n_chunks,
        free_boundary_fn=free_boundary_fn, vertical_interface_x=vertical_interface_x,
    )

    # Sequential release: force the base grid's chunk-local tensors (and any
    # autograd-graph reference cycles create_graph=False can still leave
    # behind transiently) to actually be collected before the refined grid
    # -- at 4x the points -- is built, rather than trusting refcounting alone
    # to have already done it by the time this line runs.
    gc.collect()
    report_memory("true_error: base grid released, before refined grid")

    terms_2x = _volume_and_boundary_terms(
        problem, triple.state_net, triple.flux_net, triple.multiplier_net,
        u_ref, sigma_ref, lambda_ref,
        2 * n_t, 2 * n_x,
        points_per_chunk=points_per_chunk, max_rss_gb=max_rss_gb, check_every_n_chunks=check_every_n_chunks,
        free_boundary_fn=free_boundary_fn, vertical_interface_x=vertical_interface_x,
    )

    E_true_1x = math.sqrt(_e_true_squared(terms_1x))
    E_true_2x = math.sqrt(_e_true_squared(terms_2x))
    relative_change = abs(E_true_2x - E_true_1x) / max(E_true_2x, 1e-300)

    if relative_change > quadrature_tolerance:
        raise RuntimeError(
            f"Quadrature self-check failed: E_true changed by {relative_change:.3e} "
            f"(> {quadrature_tolerance:.0e}) when resolution doubled "
            f"({n_t}x{n_x} -> {2 * n_t}x{2 * n_x}). E_true_1x={E_true_1x:.6e}, "
            f"E_true_2x={E_true_2x:.6e}. Result is not trustworthy at this resolution "
            f"-- do not use it."
        )

    norm_u_V = math.sqrt(terms_2x["term_grad_u"] + terms_2x["term_u0"] + terms_2x["term_uT"])
    norm_sigma_L2 = math.sqrt(terms_2x["term_sigma"])
    norm_lambda_L2 = math.sqrt(terms_2x["term_lambda_raw"])

    return TrueErrorResult(
        E_true=E_true_2x,
        E_true_squared=E_true_2x ** 2,
        term_grad_u=terms_2x["term_grad_u"],
        term_u0=terms_2x["term_u0"],
        term_uT=terms_2x["term_uT"],
        term_sigma=terms_2x["term_sigma"],
        term_consistency=terms_2x["term_consistency"],
        norm_u_V=norm_u_V,
        norm_sigma_L2=norm_sigma_L2,
        norm_lambda_L2=norm_lambda_L2,
        quadrature_relative_change=relative_change,
        n_t=2 * n_t,
        n_x=2 * n_x,
    )
