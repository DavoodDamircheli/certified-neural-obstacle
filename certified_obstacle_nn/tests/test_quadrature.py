"""Tests for certification/quadrature.py.

The headline property under test is chunk-size independence: iter_cell_chunks
must produce the same points/weights/cell coverage regardless of how many
cells get grouped into one chunk, and anything built from it (here, a small
test-only chunked residual aggregation using eval_triple_derivatives) must
therefore agree to near machine precision across chunk sizes. That is the
whole point of chunking a computation that used to be one unchunked batch.
"""

import math

import numpy as np
import pytest
import torch

from certified_obstacle_nn.autograd.derivatives import eval_triple_derivatives
from certified_obstacle_nn.certification.partition import build_partition
from certified_obstacle_nn.certification.quadrature import (
    build_quadrature_rule,
    iter_cell_chunks,
)
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.stefan import StefanProblem


def test_quadrature_rule_matches_numpy_leggauss():
    rule = build_quadrature_rule(q=4)
    ref_nodes, ref_weights = np.polynomial.legendre.leggauss(4)

    assert rule.q == 4
    assert rule.nodes.dtype == torch.float64
    assert rule.weights.dtype == torch.float64
    assert torch.allclose(rule.nodes, torch.tensor(ref_nodes, dtype=torch.float64))
    assert torch.allclose(rule.weights, torch.tensor(ref_weights, dtype=torch.float64))
    assert rule.weights.sum().item() == pytest.approx(2.0)  # reference interval has length 2


def test_iter_cell_chunks_covers_every_cell_exactly_once_in_order():
    problem = StefanProblem()
    partition = build_partition(problem, n_t=5, n_x=5)  # 25 cells, not a multiple of any tidy block size
    rule = build_quadrature_rule(q=4)  # 16 points/cell

    seen_cell_ids = []
    for chunk in iter_cell_chunks(partition, rule, points_per_chunk=100):  # 6 cells/chunk -> uneven last chunk
        seen_cell_ids.append(chunk.cell_ids)
        assert chunk.t.shape[0] <= 100
        assert chunk.t.shape == chunk.x.shape == chunk.w.shape
        assert chunk.area == partition.area

    all_ids = torch.cat(seen_cell_ids)
    assert torch.equal(all_ids, torch.arange(25, dtype=torch.int64))  # exactly once, in order


def test_iter_cell_chunks_respects_points_per_chunk_and_rejects_undersized_chunks():
    problem = StefanProblem()
    partition = build_partition(problem, n_t=10, n_x=10)
    rule = build_quadrature_rule(q=4)  # 16 points/cell

    with pytest.raises(ValueError):
        next(iter_cell_chunks(partition, rule, points_per_chunk=15))  # smaller than one cell's 16 points


def test_iter_cell_chunks_total_weight_matches_domain_measure():
    """Gauss-Legendre integrates the constant function 1 exactly, so the sum
    of every chunk's weights must equal |Q| regardless of chunk size."""
    problem = StefanProblem()
    partition = build_partition(problem, n_t=12, n_x=7)
    rule = build_quadrature_rule(q=4)
    Q_measure = problem.T * (problem.omega_bounds[1] - problem.omega_bounds[0])

    for points_per_chunk in (16, 100, 10_000):
        total_weight = sum(chunk.w.sum().item() for chunk in iter_cell_chunks(partition, rule, points_per_chunk))
        assert total_weight == pytest.approx(Q_measure, rel=1e-12)


def _chunked_residual_aggregate(problem, model, partition, rule, points_per_chunk):
    """Test-only helper: aggregates a handful of residual-based scalars via
    the chunked path (iter_cell_chunks + eval_triple_derivatives), following
    residuals.py's own formulas for R_PDE/R_sig/R_com (not re-deriving them)
    but deliberately not the full rho_tilde^2 (the penetration term needs
    d_t w / grad_x w, which needs its own derivative machinery beyond what
    eval_triple_derivatives provides -- out of scope for this test, whose
    job is only to validate chunk-size independence of point generation +
    eval_triple_derivatives, not to re-implement raw_estimator.py)."""
    n_points = 0
    total_weight = 0.0
    sum_R_PDE_sq_w = 0.0
    sum_R_sig_sq_w = 0.0
    sum_R_com_w = 0.0
    sum_w_sq_w = 0.0

    for chunk in iter_cell_chunks(partition, rule, points_per_chunk):
        u, sig, lam, u_t, u_x, sig_x = eval_triple_derivatives(model, chunk.t, chunk.x)
        g = problem.g(chunk.t, chunk.x)
        gamma = g - u
        w_obs = torch.relu(gamma)
        gamma_minus = torch.relu(-gamma)

        R_PDE = problem.f(chunk.t, chunk.x) - (u_t + sig_x) - problem.b * u_x - problem.c * u + lam
        R_sig = sig / math.sqrt(problem.A) + math.sqrt(problem.A) * u_x
        R_com = lam * gamma_minus

        n_points += chunk.t.shape[0]
        total_weight += chunk.w.sum().item()
        sum_R_PDE_sq_w += (chunk.w * R_PDE.pow(2)).sum().item()
        sum_R_sig_sq_w += (chunk.w * R_sig.pow(2)).sum().item()
        sum_R_com_w += (chunk.w * R_com).sum().item()
        sum_w_sq_w += (chunk.w * w_obs.pow(2)).sum().item()

    return {
        "n_points": n_points,
        "total_weight": total_weight,
        "sum_R_PDE_sq_w": sum_R_PDE_sq_w,
        "sum_R_sig_sq_w": sum_R_sig_sq_w,
        "sum_R_com_w": sum_R_com_w,
        "sum_w_sq_w": sum_w_sq_w,
    }


def test_certify_reproducible_across_chunk_sizes_4096_and_65536():
    """The exact scenario docs/history/MEMORY_OOM_DEBUGGING_AND_SAFE_RUN_POLICY.md and the
    quadrature spec call for: grid chosen so n_t*n_x*q^2 == 65536 exactly (a
    64x64 grid at q=4), so points_per_chunk=65536 yields ONE chunk and
    points_per_chunk=4096 yields 16 chunks of 256 cells each -- genuinely
    different groupings of the identical point set. Every reported scalar
    must agree to < 1e-14 relative.
    """
    problem = StefanProblem()
    torch.manual_seed(0)
    model = NeuralTriple(problem, width=16, depth=2)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    partition = build_partition(problem, n_t=64, n_x=64)
    rule = build_quadrature_rule(q=4)
    assert partition.n_t * partition.n_x * rule.q * rule.q == 65536

    result_small_chunks = _chunked_residual_aggregate(problem, model, partition, rule, points_per_chunk=4096)
    result_one_chunk = _chunked_residual_aggregate(problem, model, partition, rule, points_per_chunk=65536)

    assert result_small_chunks["n_points"] == result_one_chunk["n_points"] == 65536

    for key in ("total_weight", "sum_R_PDE_sq_w", "sum_R_sig_sq_w", "sum_R_com_w", "sum_w_sq_w"):
        a, b = result_small_chunks[key], result_one_chunk[key]
        rel_diff = abs(a - b) / max(abs(a), abs(b), 1e-300)
        assert rel_diff < 1e-14, f"{key}: {a!r} vs {b!r}, rel_diff={rel_diff:.3e}"
