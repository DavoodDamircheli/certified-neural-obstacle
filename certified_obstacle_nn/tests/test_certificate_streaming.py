"""Tests for the M15-streaming rewrite of raw_estimator.py / certificate.py.

Four properties are checked here, each guarding a specific failure mode of
the chunked rewrite:

1. Chunk-size independence: every scalar certify() reports must be the same
   (to < 1e-14 relative) whether the interior quadrature cloud is processed
   in one chunk or many -- chunking must change memory, never the answer.
2. Agreement with the pre-refactor (monolithic, unchunked) implementation, to
   < 1e-12 relative on every reported scalar. `_monolithic_reference_raw_estimator`
   below is that original implementation, kept ONLY in this test file (not in
   production code, which has already moved to the chunked version) purely as
   a fixed comparison point for this one test.
3. The M10 localization identity, sum_K rho_total_K^2 == rho_tilde^2 -- still
   holds, now by construction from the per-cell numpy arrays rather than a
   separately-accumulated running total (raw_estimator.py's own docstring).
4. Peak RSS at a 256x256 grid, default settings, stays under 4 GiB. This is
   the regression test for the M15 OOM itself (RSS ~13.3 GB at this same grid
   size, pre-streaming).
"""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from certified_obstacle_nn.certification.certificate import certify
from certified_obstacle_nn.certification.memory import peak_rss
from certified_obstacle_nn.certification.partition import Partition, build_partition
from certified_obstacle_nn.certification.raw_estimator import EstimatorSummary, compute_raw_estimator
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.base_problem import BaseProblem
from certified_obstacle_nn.problems.stefan import StefanProblem
from certified_obstacle_nn.residuals.residuals import compute_residuals

pytestmark = pytest.mark.legacy

CHUNK_REPRODUCIBILITY_RTOL = 1e-14
MONOLITHIC_AGREEMENT_RTOL = 1e-12
MAX_RSS_GB_256_GRID = 4.0


# ---------------------------------------------------------------------------
# 1. Pre-refactor reference implementation (test-only; NOT used by production
#    code, which now always streams). Faithfully reproduces the monolithic
#    raw_estimator.compute_raw_estimator as it existed before the M15
#    streaming rewrite: builds the full (n_t*n_qp, n_x*n_qp) quadrature cloud
#    as one tensor, one batched compute_residuals (create_graph=True) call.
# ---------------------------------------------------------------------------
def _gauss_legendre_per_row(edges: torch.Tensor, n_qp: int) -> tuple[torch.Tensor, torch.Tensor]:
    ref_nodes, ref_weights = np.polynomial.legendre.leggauss(n_qp)
    ref_nodes = torch.tensor(ref_nodes, dtype=torch.float64)
    ref_weights = torch.tensor(ref_weights, dtype=torch.float64)
    lo, hi = edges[:-1], edges[1:]
    half = (hi - lo) / 2.0
    mid = (hi + lo) / 2.0
    nodes = mid[:, None] + half[:, None] * ref_nodes[None, :]
    weights = half[:, None] * ref_weights[None, :]
    return nodes, weights


def _monolithic_reference_raw_estimator(
    problem: BaseProblem, triple: NeuralTriple, partition: Partition, n_qp: int = 4
) -> tuple[pd.DataFrame, EstimatorSummary]:
    triple.eval()
    for p in triple.parameters():
        p.requires_grad_(False)

    n_t, n_x = partition.n_t, partition.n_x

    t_nodes_per_row, t_weights_per_row = _gauss_legendre_per_row(partition.t_edges, n_qp)
    x_nodes_per_col, x_weights_per_col = _gauss_legendre_per_row(partition.x_edges, n_qp)

    t_full = t_nodes_per_row[:, :, None, None].expand(n_t, n_qp, n_x, n_qp)
    x_full = x_nodes_per_col[None, None, :, :].expand(n_t, n_qp, n_x, n_qp)
    w_full = t_weights_per_row[:, :, None, None] * x_weights_per_col[None, None, :, :]

    t_flat = t_full.reshape(-1, 1).clone().requires_grad_(True)
    x_flat = x_full.reshape(-1, 1).clone().requires_grad_(True)
    w_flat = w_full.reshape(-1, 1)

    x0_flat = x_nodes_per_col.reshape(-1, 1)
    x0_weights_flat = x_weights_per_col.reshape(-1, 1)

    residuals = compute_residuals(
        problem, triple.state_net, triple.flux_net, triple.multiplier_net, t_flat, x_flat, x0_flat
    )

    (w_t,) = torch.autograd.grad(residuals.w, t_flat, grad_outputs=torch.ones_like(residuals.w), retain_graph=True)
    (w_x,) = torch.autograd.grad(residuals.w, x_flat, grad_outputs=torch.ones_like(residuals.w), retain_graph=True)

    with torch.no_grad():
        pde_density = (problem.Lambda / problem.alpha) * residuals.R_PDE.pow(2)
        sigma_density = residuals.R_sig.pow(2)
        penetration_density = (
            problem.A * w_x.pow(2) + problem.kappa * residuals.w.pow(2) + (problem.Lambda / problem.alpha) * w_t.pow(2)
        )
        complementarity_density = residuals.R_com

        def _reduce(density: torch.Tensor) -> torch.Tensor:
            return (w_flat * density).reshape(n_t, n_qp, n_x, n_qp).sum(dim=(1, 3))

        pde_sq = _reduce(pde_density)
        sigma_sq = _reduce(sigma_density)
        penetration_interior_sq = _reduce(penetration_density)
        complementarity_sq = _reduce(complementarity_density)

        r0_contrib_per_x = (x0_weights_flat * residuals.R_0.pow(2)).reshape(n_x, n_qp).sum(dim=1)

        zeros_t0 = torch.zeros_like(x0_flat)
        T_tN = torch.full_like(x0_flat, problem.T)
        w0 = torch.relu(problem.g(zeros_t0, x0_flat) - triple.state_net(zeros_t0, x0_flat))
        wT = torch.relu(problem.g(T_tN, x0_flat) - triple.state_net(T_tN, x0_flat))
        w0_contrib_per_x = (x0_weights_flat * w0.pow(2)).reshape(n_x, n_qp).sum(dim=1)
        wT_contrib_per_x = (x0_weights_flat * wT.pow(2)).reshape(n_x, n_qp).sum(dim=1)

        r0_sq = torch.zeros(n_t, n_x, dtype=torch.float64)
        r0_sq[0, :] = r0_contrib_per_x
        w0_sq = torch.zeros(n_t, n_x, dtype=torch.float64)
        w0_sq[0, :] = w0_contrib_per_x
        wT_sq = torch.zeros(n_t, n_x, dtype=torch.float64)
        wT_sq[-1, :] = wT_contrib_per_x

        r_sq = pde_sq + sigma_sq + r0_sq
        penetration_sq = penetration_interior_sq + w0_sq + wT_sq
        total_sq = r_sq + penetration_sq + complementarity_sq

        rho_tilde_squared = total_sq.sum().item()
        rho_r = math.sqrt(r_sq.sum().item())
        rho_p_tilde = math.sqrt(penetration_sq.sum().item())
        rho_c = math.sqrt(complementarity_sq.sum().item())
        rho_tilde = math.sqrt(rho_tilde_squared)

        per_cell_df = pd.DataFrame(
            {
                "cell_id": np.arange(n_t * n_x),
                "i": partition.i_index.numpy(),
                "j": partition.j_index.numpy(),
                "rho_pde_K": pde_sq.reshape(-1).sqrt().numpy(),
                "rho_sigma_K": sigma_sq.reshape(-1).sqrt().numpy(),
                "rho_r_K": r_sq.reshape(-1).sqrt().numpy(),
                "rho_penetration_K": penetration_sq.reshape(-1).sqrt().numpy(),
                "rho_complementarity_K": complementarity_sq.reshape(-1).sqrt().numpy(),
                "rho_total_K": total_sq.reshape(-1).sqrt().numpy(),
            }
        )

    summary = EstimatorSummary(
        rho_r=rho_r, rho_p_tilde=rho_p_tilde, rho_c=rho_c, rho_tilde=rho_tilde, rho_tilde_squared=rho_tilde_squared
    )
    return per_cell_df, summary


def _rel_diff(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


# ---------------------------------------------------------------------------
# Test 1: chunk-size independence of every certify() scalar.
# ---------------------------------------------------------------------------
def test_certify_bitwise_stable_across_points_per_chunk():
    problem = StefanProblem()
    torch.manual_seed(0)
    triple = NeuralTriple(problem, width=16, depth=2)
    partition = build_partition(problem, n_t=64, n_x=64)  # 64*64*4^2 = 65536 interior points

    result_small = certify(triple, problem, partition, n_qp=4, points_per_chunk=4096)
    result_large = certify(triple, problem, partition, n_qp=4, points_per_chunk=65536)

    scalar_fields = ["rho_r", "rho_p_tilde", "rho_c", "rho_tilde_h", "V_h_unr", "V_h_dat", "E_cert", "E_cert_U"]
    for field in scalar_fields:
        a, b = getattr(result_small, field), getattr(result_large, field)
        rel = _rel_diff(a, b)
        assert rel < CHUNK_REPRODUCIBILITY_RTOL, f"{field}: {a!r} vs {b!r}, rel_diff={rel:.3e}"

    for field in ("E_h_lip", "E_h_unr", "E_h_dat", "E_h_0T", "E_h"):
        a = getattr(result_small.remainder, field)
        b = getattr(result_large.remainder, field)
        rel = _rel_diff(a, b)
        assert rel < CHUNK_REPRODUCIBILITY_RTOL, f"remainder.{field}: {a!r} vs {b!r}, rel_diff={rel:.3e}"


# ---------------------------------------------------------------------------
# Test 2: agreement with the pre-refactor monolithic implementation.
# ---------------------------------------------------------------------------
def test_streaming_matches_monolithic_reference_at_32_grid():
    problem = StefanProblem()
    torch.manual_seed(1)
    triple_a = NeuralTriple(problem, width=16, depth=2)
    torch.manual_seed(1)
    triple_b = NeuralTriple(problem, width=16, depth=2)  # identical init, separate object

    partition = build_partition(problem, n_t=32, n_x=32)

    df_stream, summary_stream = compute_raw_estimator(problem, triple_a, partition, n_qp=4)
    df_mono, summary_mono = _monolithic_reference_raw_estimator(problem, triple_b, partition, n_qp=4)

    for field in EstimatorSummary._fields:
        a, b = getattr(summary_stream, field), getattr(summary_mono, field)
        rel = _rel_diff(a, b)
        assert rel < MONOLITHIC_AGREEMENT_RTOL, f"{field}: streaming={a!r} monolithic={b!r} rel_diff={rel:.3e}"

    shared_cols = ["rho_pde_K", "rho_sigma_K", "rho_r_K", "rho_penetration_K", "rho_complementarity_K", "rho_total_K"]
    df_stream_sorted = df_stream.sort_values("cell_id").reset_index(drop=True)
    df_mono_sorted = df_mono.sort_values("cell_id").reset_index(drop=True)
    for col in shared_cols:
        a = df_stream_sorted[col].to_numpy()
        b = df_mono_sorted[col].to_numpy()
        rel = np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300)
        assert (rel < MONOLITHIC_AGREEMENT_RTOL).all(), f"{col}: max rel_diff = {rel.max():.3e}"


# ---------------------------------------------------------------------------
# Test 3: the M10 localization identity.
# ---------------------------------------------------------------------------
def test_M10_localization_identity_holds_for_streaming_implementation():
    problem = StefanProblem()
    torch.manual_seed(2)
    triple = NeuralTriple(problem, width=16, depth=2)
    partition = build_partition(problem, n_t=20, n_x=15)

    per_cell_df, summary = compute_raw_estimator(problem, triple, partition, n_qp=4)

    recomputed = math.fsum((per_cell_df["rho_total_K"] ** 2).tolist())
    assert recomputed == pytest.approx(summary.rho_tilde_squared, rel=1e-12)
    assert summary.rho_tilde == pytest.approx(math.sqrt(summary.rho_tilde_squared))


# ---------------------------------------------------------------------------
# Test 4: peak RSS at 256x256, default settings, < 4 GiB. Regression test for
# the M15 OOM (measured ~13.3 GB RSS at this same grid, pre-streaming).
# ---------------------------------------------------------------------------
def test_peak_rss_at_256_grid_default_settings_under_4gib():
    problem = StefanProblem()
    torch.manual_seed(3)
    triple = NeuralTriple(problem, width=64, depth=4)  # specification section 1 default architecture
    partition = build_partition(problem, n_t=256, n_x=256)

    rss_before = peak_rss()
    result = certify(triple, problem, partition)  # default n_qp, points_per_chunk, max_rss_gb
    rss_after = peak_rss()

    increase_gib = (rss_after - rss_before) / (1024 ** 3)
    print(f"certify() at 256x256, default settings: peak RSS increase = {increase_gib:.3f} GiB")
    print(f"rho_tilde_h={result.rho_tilde_h:.5e}  E_h={result.remainder.E_h:.5e}  E_cert_U={result.E_cert_U:.5e}")

    assert increase_gib < MAX_RSS_GB_256_GRID
