"""Raw (Level-1) computable estimator, specification section 4, localized exactly
onto the tensor-product partition built by certification/partition.py.

`rho_tilde^2` is DEFINED here as `sum_K rho_total_K^2` -- it is not computed
independently and then merely cross-checked against that sum. specification
section 4's localization property ("rho_tilde^2 = sum_K rho_tilde_K^2 over any
partition") therefore holds by construction: `rho_tilde_squared` (and rho_r,
rho_p_tilde, rho_c) is computed FROM the completed per-cell numpy arrays
(math.fsum over rho_total_K**2, ..., after every chunk has been written), never
as a running total accumulated chunk-by-chunk in parallel. That makes the
identity a fact about the partition/reduction logic, not something that has to
survive surviving two different summation orders by luck -- see
tests/test_raw_estimator.py and tests/test_certificate_streaming.py.

Streamed evaluation (docs/history/MEMORY_OOM_DEBUGGING_AND_SAFE_RUN_POLICY.md): the
interior (t, x) quadrature cloud is never materialized in full. It is
generated and consumed chunk-by-chunk via
certification/quadrature.py's iter_cell_chunks, using
autograd/derivatives.py's eval_triple_derivatives (create_graph=False) and
certification/densities.py's phi_smooth_from_residuals /
phi_interface_from_residuals (reused, not re-derived -- specification's own
formulas, computed once per chunk from residuals this module already has to
compute anyway). Per-cell results are written into per-cell numpy arrays
preallocated ONCE before the loop; nothing per-POINT is ever retained past the
chunk that produced it, and nothing accumulates in a Python list across
chunks.

The R_0 / w(0) / w(T) boundary terms are a separate, small (O(n_x), not
O(n_t*n_x*q^2)) 1D line integral at the fixed times t=0 and t=T -- they are
not quadrature nodes of any interior cell (Gauss-Legendre nodes are always
strictly interior to a cell) and so cannot be folded into the chunked interior
loop; they are computed once, unchunked, since their cost does not grow with
the certification grid's resolution in t.

specification section 12 invariants respected here: this module is never handed
the training sampler itself (only a plain (t, x) tensor pair via
`training_points`, forwarded to certification/disjointness.py's
iter_disjoint_checked so the check rides along inside this module's own single
pass over the grid -- a second full pass would be twice the compute for no
benefit); triple.eval() is called and every parameter is explicitly frozen
before any evaluation; float64 throughout; there is no optimizer, no
.backward(), no parameter touch anywhere below; no autograd graph escapes a
chunk (asserted explicitly, not just implied by eval_triple_derivatives'
own contract).
"""

import math
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
import torch

from certified_obstacle_nn.autograd.derivatives import eval_gap_derivatives, eval_triple_derivatives
from certified_obstacle_nn.certification.cells_io import save_per_cell_arrays
from certified_obstacle_nn.certification.densities import phi_interface_from_residuals, phi_smooth_from_residuals
from certified_obstacle_nn.certification.disjointness import TrainPoints, iter_disjoint_checked
from certified_obstacle_nn.certification.memory import (
    CHUNK_BYTES_FACTOR,
    GIB,
    MemoryGuard,
    certification_stage,
    estimate_chunk_bytes,
    report_memory,
)
from certified_obstacle_nn.certification.partition import Partition
from certified_obstacle_nn.certification.quadrature import QuadratureRule, build_quadrature_rule, iter_cell_chunks
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.base_problem import BaseProblem
from certified_obstacle_nn.residuals.residuals import assemble_residuals

DEFAULT_POINTS_PER_CHUNK = 65536
DEFAULT_MAX_RSS_GB = 8.0
DEFAULT_CHECK_EVERY_N_CHUNKS = 10


class EstimatorSummary(NamedTuple):
    rho_r: float
    rho_p_tilde: float
    rho_c: float
    rho_tilde: float
    rho_tilde_squared: float


def _freeze(triple: NeuralTriple) -> None:
    triple.eval()
    for p in triple.parameters():
        p.requires_grad_(False)


def _triple_architecture(triple: NeuralTriple) -> tuple[int, int]:
    """(depth, width) of the triple's sub-networks, for the estimate_chunk_bytes
    diagnostic below only -- assumes all three (state/flux/multiplier) share
    the same depth/width, true for every NeuralTriple this repo constructs
    (specification section 1: one width/depth per triple). Duck-typed triples
    without a real MLP (e.g. tests/test_raw_estimator.py's _ExactTriple) don't
    have `.net.layers`, so this returns (0, 0) rather than raising -- the
    logging this feeds is a diagnostic, never load-bearing."""
    net = getattr(triple.state_net, "net", None)
    layers = getattr(net, "layers", None)
    if not layers:
        return 0, 0
    depth = len(layers) - 1  # hidden_depth: layers = hidden_depth hidden + 1 output
    width = layers[0].out_features
    return depth, width


def _map_1d(edges: torch.Tensor, rule: QuadratureRule) -> tuple[torch.Tensor, torch.Tensor]:
    """Same affine [-1,1] -> [lo,hi] mapping certification/quadrature.py uses
    per axis, applied along ONE axis only -- for the t=0 / t=T boundary line
    integrals, which are not interior cell quadrature nodes."""
    lo, hi = edges[:-1], edges[1:]
    half = (hi - lo) / 2.0
    mid = (hi + lo) / 2.0
    nodes = mid[:, None] + half[:, None] * rule.nodes[None, :]
    weights = half[:, None] * rule.weights[None, :]
    return nodes, weights


def _boundary_contributions(
    problem: BaseProblem, triple: NeuralTriple, partition: Partition, rule: QuadratureRule
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """R_0, w(0), w(T) need no derivatives at all (pointwise evaluations of
    u_theta and g only), so this is a plain torch.no_grad() pass -- cheap
    regardless of grid resolution (O(n_x), not O(n_t*n_x*q^2)), hence not
    chunked. Returns three (n_x,) numpy arrays: r0_sq, w0_sq, wT_sq."""
    x_nodes, x_weights = _map_1d(partition.x_edges, rule)  # (n_x, q)
    x0_flat = x_nodes.reshape(-1, 1)
    weights_flat = x_weights.reshape(-1, 1)
    zeros_t0 = torch.zeros_like(x0_flat)
    T_tN = torch.full_like(x0_flat, problem.T)

    with torch.no_grad():
        u0_pred = triple.state_net(zeros_t0, x0_flat)
        uT_pred = triple.state_net(T_tN, x0_flat)
        R_0 = problem.u0(x0_flat) - u0_pred
        w0 = torch.relu(problem.g(zeros_t0, x0_flat) - u0_pred)
        wT = torch.relu(problem.g(T_tN, x0_flat) - uT_pred)

        r0_sq = (weights_flat * R_0.pow(2)).reshape(partition.n_x, rule.q).sum(dim=1)
        w0_sq = (weights_flat * w0.pow(2)).reshape(partition.n_x, rule.q).sum(dim=1)
        wT_sq = (weights_flat * wT.pow(2)).reshape(partition.n_x, rule.q).sum(dim=1)

    return r0_sq.numpy(), w0_sq.numpy(), wT_sq.numpy()


def compute_raw_estimator(
    problem: BaseProblem,
    triple: NeuralTriple,
    partition: Partition,
    n_qp: int = 4,
    *,
    training_points: Optional[TrainPoints] = None,
    points_per_chunk: int = DEFAULT_POINTS_PER_CHUNK,
    max_rss_gb: float = DEFAULT_MAX_RSS_GB,
    check_every_n_chunks: int = DEFAULT_CHECK_EVERY_N_CHUNKS,
    save_cells_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, EstimatorSummary]:
    """`save_cells_dir`, if given, additionally persists the per-cell
    DataFrame to disk (certification/cells_io.py: csv below 1e6 cells, npz
    above) before returning -- default None preserves the exact prior
    behavior (per_cell_df only ever returned in memory, never written)."""
    _freeze(triple)

    n_t, n_x = partition.n_t, partition.n_x
    n_cells = n_t * n_x
    rule = build_quadrature_rule(q=n_qp)

    guard = MemoryGuard(
        max_rss_gb=max_rss_gb, label_prefix="raw_estimator",
        chunk_size=points_per_chunk, n_qp=n_qp, grid=(n_t, n_x),
    )
    depth, width = _triple_architecture(triple)

    # 1. Preallocate every per-cell result array ONCE. Written into by cell_id
    # below, never appended-to-and-concatenated.
    pde_sq_K = np.zeros(n_cells, dtype=np.float64)
    sigma_sq_K = np.zeros(n_cells, dtype=np.float64)
    r0_sq_K = np.zeros(n_cells, dtype=np.float64)
    penetration_interior_sq_K = np.zeros(n_cells, dtype=np.float64)
    w0_sq_K = np.zeros(n_cells, dtype=np.float64)
    wT_sq_K = np.zeros(n_cells, dtype=np.float64)
    complementarity_sq_K = np.zeros(n_cells, dtype=np.float64)
    phi_sm_K = np.zeros(n_cells, dtype=np.float64)
    phi_int_K = np.zeros(n_cells, dtype=np.float64)

    with certification_stage("raw_estimator:interior_quadrature_loop"):
        chunks = iter_cell_chunks(partition, rule, points_per_chunk)
        if training_points is not None:
            chunks = iter_disjoint_checked(chunks, training_points, tol=0.0)

        for chunk_idx, chunk in enumerate(chunks):
            cell_ids_np = chunk.cell_ids.numpy()
            n_cells_in_chunk = chunk.cell_ids.shape[0]
            n_points_in_chunk = chunk.t.shape[0]
            points_per_cell = n_points_in_chunk // n_cells_in_chunk

            # Predicted-vs-measured memory model, first chunk only (policy doc
            # section 15): validates/calibrates CHUNK_BYTES_FACTOR on the
            # actual architecture and chunk size in use, rather than trusting
            # the model blind. Not printed for every chunk -- that would be
            # exactly the inner-loop noise certification_stage's own docstring
            # says to avoid.
            if chunk_idx == 0 and depth > 0:
                predicted_bytes = estimate_chunk_bytes(n_points_in_chunk, depth, width, n_networks=3)
                rss_before_gib = report_memory("raw_estimator: first chunk, before")

            u, sigma, lam, u_t, u_x, sigma_x = eval_triple_derivatives(triple, chunk.t, chunk.x)
            # R_0 is a boundary-only term (handled separately below); x0/u0_pred
            # here are inert placeholders, never used downstream.
            residuals = assemble_residuals(
                problem, chunk.t, chunk.x, chunk.t, u, sigma, lam, u_t, u_x, sigma_x, u0_pred=u
            )
            gamma_t, gamma_x = eval_gap_derivatives(problem, chunk.t, chunk.x, u_t, u_x)

            # 7. no autograd graph may escape a chunk into the accumulation path.
            assert not residuals.R_PDE.requires_grad and residuals.R_PDE.grad_fn is None, (
                "certification invariant violated: an autograd graph escaped the chunk"
            )

            # 3. densities via the M13 functions themselves, not re-derived inline.
            active = (residuals.gamma > 0).to(gamma_t.dtype)
            w_t = active * gamma_t
            w_x = active * gamma_x

            pde_density = (problem.Lambda / problem.alpha) * residuals.R_PDE.pow(2)
            sigma_density = residuals.R_sig.pow(2)
            penetration_density = (
                problem.A * w_x.pow(2) + problem.kappa * residuals.w.pow(2) + (problem.Lambda / problem.alpha) * w_t.pow(2)
            )
            complementarity_density = residuals.R_com
            phi_sm_density = phi_smooth_from_residuals(problem, residuals)
            phi_int_density = phi_interface_from_residuals(problem, residuals, gamma_t, gamma_x)

            def _reduce(density: torch.Tensor) -> np.ndarray:
                return (chunk.w * density).reshape(n_cells_in_chunk, points_per_cell).sum(dim=1).numpy()

            pde_sq_K[cell_ids_np] = _reduce(pde_density)
            sigma_sq_K[cell_ids_np] = _reduce(sigma_density)
            penetration_interior_sq_K[cell_ids_np] = _reduce(penetration_density)
            complementarity_sq_K[cell_ids_np] = _reduce(complementarity_density)
            phi_sm_K[cell_ids_np] = _reduce(phi_sm_density)
            phi_int_K[cell_ids_np] = _reduce(phi_int_density)

            if chunk_idx == 0 and depth > 0:
                rss_after_gib = report_memory("raw_estimator: first chunk, after")
                measured_bytes = max((rss_after_gib - rss_before_gib) * GIB, 1.0)
                implied_C = measured_bytes / (n_points_in_chunk * 3 * depth * width * 8)
                print(
                    f"[raw_estimator] first-chunk memory model: n_points={n_points_in_chunk} "
                    f"predicted={predicted_bytes / GIB * 1024:.1f} MiB "
                    f"measured={measured_bytes / GIB * 1024:.1f} MiB "
                    f"implied_C={implied_C:.3f} (default CHUNK_BYTES_FACTOR={CHUNK_BYTES_FACTOR} in "
                    "certification/memory.py -- if this is consistently far off across runs, update "
                    "it there per docs/history/MEMORY_OOM_DEBUGGING_AND_SAFE_RUN_POLICY.md section 15, not here).",
                    flush=True,
                )

            del (
                u, sigma, lam, u_t, u_x, sigma_x, residuals, gamma_t, gamma_x, active, w_t, w_x,
                pde_density, sigma_density, penetration_density, complementarity_density,
                phi_sm_density, phi_int_density,
            )

            if (chunk_idx + 1) % check_every_n_chunks == 0:
                guard.check(f"interior chunk {chunk_idx + 1}")

    with certification_stage("raw_estimator:boundary_terms"):
        r0_sq_row0, w0_sq_row0, wT_sq_rowlast = _boundary_contributions(problem, triple, partition, rule)
        r0_sq_K[0:n_x] = r0_sq_row0
        w0_sq_K[0:n_x] = w0_sq_row0
        wT_sq_K[(n_t - 1) * n_x : n_t * n_x] = wT_sq_rowlast

    # 2. Totals FROM the completed per-cell arrays, math.fsum for a
    # summation-order-independent (hence chunk-size-independent) result.
    r_sq_K = pde_sq_K + sigma_sq_K + r0_sq_K
    penetration_sq_K = penetration_interior_sq_K + w0_sq_K + wT_sq_K
    total_sq_K = r_sq_K + penetration_sq_K + complementarity_sq_K

    rho_pde_K = np.sqrt(pde_sq_K)
    rho_sigma_K = np.sqrt(sigma_sq_K)
    rho_r_K = np.sqrt(r_sq_K)
    rho_penetration_K = np.sqrt(penetration_sq_K)
    rho_complementarity_K = np.sqrt(complementarity_sq_K)
    rho_total_K = np.sqrt(total_sq_K)

    rho_tilde_squared = math.fsum((rho_total_K ** 2).tolist())
    rho_r_squared = math.fsum((rho_r_K ** 2).tolist())
    rho_p_tilde_squared = math.fsum((rho_penetration_K ** 2).tolist())
    rho_c_squared = math.fsum((rho_complementarity_K ** 2).tolist())

    rho_r = math.sqrt(rho_r_squared)
    rho_p_tilde = math.sqrt(rho_p_tilde_squared)
    rho_c = math.sqrt(rho_c_squared)
    rho_tilde = math.sqrt(rho_tilde_squared)

    # 8. DataFrame built ONCE from the completed numpy arrays.
    per_cell_df = pd.DataFrame(
        {
            "cell_id": np.arange(n_cells),
            "i": partition.i_index.numpy(),
            "j": partition.j_index.numpy(),
            "t_center": partition.t_center.numpy(),
            "x_center": partition.x_center.numpy(),
            "area": partition.area,
            "rho_pde_K": rho_pde_K,
            "rho_sigma_K": rho_sigma_K,
            "rho_r_K": rho_r_K,
            "rho_penetration_K": rho_penetration_K,
            "rho_complementarity_K": rho_complementarity_K,
            "rho_total_K": rho_total_K,
            "phi_sm_K": phi_sm_K,
            "phi_int_K": phi_int_K,
        }
    )

    summary = EstimatorSummary(
        rho_r=rho_r,
        rho_p_tilde=rho_p_tilde,
        rho_c=rho_c,
        rho_tilde=rho_tilde,
        rho_tilde_squared=rho_tilde_squared,
    )

    if save_cells_dir is not None:
        save_per_cell_arrays(per_cell_df, save_cells_dir, n_cells)

    return per_cell_df, summary
