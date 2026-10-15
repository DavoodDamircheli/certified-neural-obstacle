"""Chunked quadrature-point generation for certification.

`raw_estimator.py` currently materializes the ENTIRE (n_t*n_qp, n_x*n_qp)
quadrature cloud as one tensor and runs one batched forward+autograd pass
over it -- root cause H1 of the M15 OOM
(docs/history/MEMORY_OOM_DEBUGGING_AND_SAFE_RUN_POLICY.md section 1). This module provides
the piece that lets that be fixed without changing any formula: a generator
that yields the SAME points in the SAME order, just grouped into
memory-bounded contiguous blocks of cells, computed on demand rather than all
at once.

This module only builds geometry (mapped physical points + quadrature
weights), exactly like partition.py only builds cell geometry -- it knows
nothing about residuals, models, or training, and is never handed the
training sampler (specification section 12 invariant 1).
"""

from dataclasses import dataclass
from typing import Iterator, NamedTuple

import numpy as np
import torch

from certified_obstacle_nn.certification.partition import Partition


@dataclass(frozen=True)
class QuadratureRule:
    """1D Gauss-Legendre rule on the reference interval [-1, 1], float64.
    Built once (via numpy.polynomial.legendre.leggauss) and reused for every
    cell and every axis (t and x both use the same 1D rule; the 2D rule is
    their tensor product, specification section 4's quadrature)."""

    q: int
    nodes: torch.Tensor  # (q,) float64, in [-1, 1]
    weights: torch.Tensor  # (q,) float64, sum to 2.0


def build_quadrature_rule(q: int = 4) -> QuadratureRule:
    nodes, weights = np.polynomial.legendre.leggauss(q)
    return QuadratureRule(
        q=q,
        nodes=torch.tensor(nodes, dtype=torch.float64),
        weights=torch.tensor(weights, dtype=torch.float64),
    )


class CellChunk(NamedTuple):
    cell_ids: torch.Tensor  # (n_cells,) int64, indices into the partition's flattened cell ordering
    t: torch.Tensor  # (n_cells*q*q, 1) float64, physical quadrature points
    x: torch.Tensor  # (n_cells*q*q, 1) float64
    w: torch.Tensor  # (n_cells*q*q, 1) float64, quadrature weight INCLUDING the |K|/4 Jacobian
    area: float  # |K|, uniform across cells (Partition.area)


def _map_cell_block(partition: Partition, rule: QuadratureRule, cell_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Physical (t, x, w) for exactly the cells in `cell_ids`, tensor-product
    of the 1D rule along t and x. Affine map from [-1,1] to a cell of width
    dt (resp. dx): node -> center + (dt/2)*node, weight -> (dt/2)*w_ref; the
    2D weight is therefore w_t*w_x = (dt/2)(dx/2)*w_ref_t*w_ref_x =
    (|K|/4)*w_ref_t*w_ref_x, since |K| = dt*dx on this uniform partition.
    """
    q = rule.q
    n_cells = cell_ids.shape[0]
    dt, dx = partition.dt, partition.dx

    t_c = partition.t_center[cell_ids]  # (n_cells,)
    x_c = partition.x_center[cell_ids]  # (n_cells,)

    t_nodes = t_c[:, None] + (dt / 2.0) * rule.nodes[None, :]  # (n_cells, q)
    x_nodes = x_c[:, None] + (dx / 2.0) * rule.nodes[None, :]  # (n_cells, q)
    w_t = (dt / 2.0) * rule.weights  # (q,)
    w_x = (dx / 2.0) * rule.weights  # (q,)

    t_full = t_nodes[:, :, None].expand(n_cells, q, q)
    x_full = x_nodes[:, None, :].expand(n_cells, q, q)
    w_full = (w_t[None, :, None] * w_x[None, None, :]).expand(n_cells, q, q)

    return t_full.reshape(-1, 1), x_full.reshape(-1, 1), w_full.reshape(-1, 1)


def iter_cell_chunks(partition: Partition, rule: QuadratureRule, points_per_chunk: int) -> Iterator[CellChunk]:
    """Yields CellChunks covering every cell of `partition` exactly once, in
    the partition's own flattened (i-slowest, j-fastest) order, grouped into
    contiguous blocks of cells sized so each chunk has at most
    `points_per_chunk` quadrature points.

    - Points are generated per chunk from `rule` and that chunk's cell
      centers -- the full (N_t*q, N_x*q) point cloud is never materialized
      and never built via a dense meshgrid over the whole domain; each
      iteration only ever holds one block's worth of points.
    - The generator holds no reference to previously yielded chunks: each
      iteration creates fresh local tensors and the loop variable is
      reassigned, so nothing accumulates across iterations.
    - Iteration order (which cells, in what order, with what points/weights)
      depends only on `partition` and `rule`, never on `points_per_chunk` --
      that only changes how many cells are grouped into one chunk. This is
      what makes results bitwise-reproducible across chunk sizes (see
      tests/test_quadrature.py).
    """
    q = rule.q
    points_per_cell = q * q
    if points_per_chunk < points_per_cell:
        raise ValueError(
            f"points_per_chunk={points_per_chunk} is smaller than a single cell's "
            f"{points_per_cell} quadrature points (q={q}); a chunk must contain at "
            "least one whole cell."
        )
    n_cells_per_block = points_per_chunk // points_per_cell
    n_total_cells = partition.n_t * partition.n_x

    for start in range(0, n_total_cells, n_cells_per_block):
        end = min(start + n_cells_per_block, n_total_cells)
        cell_ids = torch.arange(start, end, dtype=torch.int64)

        t_flat, x_flat, w_flat = _map_cell_block(partition, rule, cell_ids)

        yield CellChunk(cell_ids=cell_ids, t=t_flat, x=x_flat, w=w_flat, area=partition.area)
