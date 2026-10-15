"""Tests for certification/certified_residual_bound.py and certify_option_b(),
per docs/history/OPTION_B_CERTIFIED_RESIDUAL_BOUND_IMPLEMENTATION_PLAN.md section 14 and
its addendum.

Tests 1/2/3 test the LEMMA itself (|R_PDE(z)| <= |R_PDE(z_K)| + L_R*r_K for
every z in a cell K) against synthetic R functions with EXACTLY known
Lipschitz constants, using a real (possibly single-cell) certification
Partition so r_K comes from the actual geometry code, not a hand re-derived
number. Test 3 additionally pins the geometry factor exactly (addendum A1):
r_K = h_K/2, verified via an EXACT Cauchy-Schwarz-tight corner construction,
not just an inequality.
"""

import math

import pytest
import torch

from certified_obstacle_nn.certification.certified_residual_bound import (
    DEFAULT_MODE,
    VALID_MODES,
    cell_radius_euclidean,
    combine_certified_residual_bound,
)
from certified_obstacle_nn.certification.certificate_option_b import certify_option_b
from certified_obstacle_nn.certification.level2_remainder import _run_midpoint_pass
from certified_obstacle_nn.certification.network_bounds import compute_triple_bounds
from certified_obstacle_nn.certification.partition import Partition, build_partition
from certified_obstacle_nn.certification.cell_classifier import classify_cells
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.stefan import StefanProblem

pytestmark = pytest.mark.legacy


def _make_single_cell_partition(dt: float, dx: float) -> Partition:
    t_edges = torch.tensor([0.0, dt], dtype=torch.float64)
    x_edges = torch.tensor([0.0, dx], dtype=torch.float64)
    t_center = torch.tensor([dt / 2.0], dtype=torch.float64)
    x_center = torch.tensor([dx / 2.0], dtype=torch.float64)
    i_index = torch.tensor([0], dtype=torch.int64)
    j_index = torch.tensor([0], dtype=torch.int64)
    h_K = math.sqrt(dt**2 + dx**2)
    return Partition(
        n_t=1, n_x=1, t_edges=t_edges, x_edges=x_edges,
        t_center=t_center, x_center=x_center, i_index=i_index, j_index=j_index,
        dt=dt, dx=dx, area=dt * dx, h_K=h_K, h=h_K,
    )


# ---------------------------------------------------------------------------
# Test 1 -- constant function, L_R = 0.
# ---------------------------------------------------------------------------
def test_1_constant_function():
    partition = _make_single_cell_partition(0.1, 0.1)
    r_K = cell_radius_euclidean(partition)
    c = -3.7

    result = combine_certified_residual_bound(
        m_R_apriori=1e9, L_R=0.0, r_K=r_K, max_R_mid=abs(c),
        n_data_cut_cells=0, argmax_cell_id=0, argmax_t=0.0, argmax_x=0.0, mode="partition",
    )
    assert result.lip_term == 0.0
    assert result.m_R_h == pytest.approx(abs(c))


# ---------------------------------------------------------------------------
# Test 2 -- linear in one spatial variable ("1D"): the midpoint+radius bound
# must dominate the EXACT cell supremum.
# ---------------------------------------------------------------------------
def test_2_linear_1d_bound_dominates_exact_supremum():
    dt, dx = 0.15, 0.05
    partition = _make_single_cell_partition(dt, dx)
    r_K = cell_radius_euclidean(partition)
    a = 4.0

    def R(x):
        return a * x

    t0, x0 = partition.t_center.item(), partition.x_center.item()
    max_R_mid = abs(R(x0))
    L_R = abs(a)  # exact Lipschitz constant of R w.r.t. the Euclidean (t,x) metric

    result = combine_certified_residual_bound(
        m_R_apriori=1e9, L_R=L_R, r_K=r_K, max_R_mid=max_R_mid,
        n_data_cut_cells=0, argmax_cell_id=0, argmax_t=t0, argmax_x=x0, mode="partition",
    )

    exact_sup = max(abs(R(x0 - dx / 2.0)), abs(R(x0 + dx / 2.0)))  # R depends only on x -> extremized at x-edges
    assert result.m_R_h >= exact_sup - 1e-12


# ---------------------------------------------------------------------------
# Test 3 -- linear in 2D, addendum A1: pin the geometry factor exactly.
# ---------------------------------------------------------------------------
def test_3_linear_2d_geometry_factor_is_exactly_one_half():
    dt, dx = 0.12, 0.08
    partition = _make_single_cell_partition(dt, dx)
    r_K = cell_radius_euclidean(partition)
    h_K = partition.h_K

    assert r_K == pytest.approx(h_K / 2.0, rel=1e-12)
    # Distinguish from the two flagged-wrong conventions (addendum A1): using
    # h_K directly (2x conservative but still safe) and h_K/sqrt(2) (a
    # unit error if specification's h_K -- the cell DIAGONAL -- is mistaken for a
    # side length, for which h/sqrt(2) would instead be the correct formula).
    assert r_K != pytest.approx(h_K, rel=1e-9)
    assert r_K != pytest.approx(h_K / math.sqrt(2), rel=1e-9)

    # Exact equality construction: R's gradient (a,b) is chosen proportional
    # to (dt,dx), i.e. pointing exactly along the cell diagonal, so the
    # Cauchy-Schwarz bound |R(z)-R(z_K)| <= L_R*||z-z_K|| is TIGHT at the
    # farthest corner -- this pins r_K's numeric value exactly (an equality),
    # not merely a safe-but-loose inequality.
    a, b = dt, dx
    L_R = math.hypot(a, b)
    R_at_farthest_corner = a * (dt / 2.0) + b * (dx / 2.0)  # R(z_K) = 0 by construction

    assert R_at_farthest_corner == pytest.approx(L_R * r_K, rel=1e-12)
    assert R_at_farthest_corner < L_R * h_K  # confirms h_K alone would be (safely) 2x loose here

    # Full combine_certified_residual_bound call, same conclusion end to end.
    result = combine_certified_residual_bound(
        m_R_apriori=1e9, L_R=L_R, r_K=r_K, max_R_mid=0.0,  # R(z_K)=0
        n_data_cut_cells=0, argmax_cell_id=0, argmax_t=0.0, argmax_x=0.0, mode="partition",
    )
    assert result.m_R_h == pytest.approx(R_at_farthest_corner, rel=1e-12)


# ---------------------------------------------------------------------------
# Test 4 -- m_R_cert <= m_R_apriori for the RECOMMENDED mode
# (min_apriori_and_partition), by construction, across realistic numbers in
# both regimes (sharpening helps / sharpening would hurt). The other two
# modes deliberately pick one side unconditionally and do NOT carry this
# guarantee (apriori mode gives exact equality; partition mode is the raw a
# posteriori bound alone, which can exceed m_R_apriori on a coarse grid) --
# checked explicitly too, so the distinction itself is under test.
# ---------------------------------------------------------------------------
def test_4_m_R_cert_never_exceeds_m_R_apriori_for_min_mode():
    for max_R_mid, L_R, r_K, m_R_apriori in [
        (0.05, 700.0, 0.02, 44.3),  # m_R_h << m_R_apriori (sharpening wins)
        (1000.0, 700.0, 0.02, 44.3),  # m_R_h >> m_R_apriori (apriori wins)
    ]:
        result = combine_certified_residual_bound(
            m_R_apriori=m_R_apriori, L_R=L_R, r_K=r_K, max_R_mid=max_R_mid,
            n_data_cut_cells=0, argmax_cell_id=0, argmax_t=0.0, argmax_x=0.0, mode="min_apriori_and_partition",
        )
        assert result.m_R_cert <= m_R_apriori + 1e-9

        apriori_result = combine_certified_residual_bound(
            m_R_apriori=m_R_apriori, L_R=L_R, r_K=r_K, max_R_mid=max_R_mid,
            n_data_cut_cells=0, argmax_cell_id=0, argmax_t=0.0, argmax_x=0.0, mode="apriori",
        )
        assert apriori_result.m_R_cert == pytest.approx(m_R_apriori)

    # partition mode with m_R_h >> m_R_apriori: deliberately NOT capped.
    partition_result = combine_certified_residual_bound(
        m_R_apriori=44.3, L_R=700.0, r_K=0.02, max_R_mid=1000.0,
        n_data_cut_cells=0, argmax_cell_id=0, argmax_t=0.0, argmax_x=0.0, mode="partition",
    )
    assert partition_result.m_R_cert > 44.3


# ---------------------------------------------------------------------------
# Test 5 -- decomposition consistency: L_sm^h + L_Psi reproduces E_h_lip_new.
# ---------------------------------------------------------------------------
def test_5_L_sm_new_plus_L_Psi_reproduces_E_h_lip_new():
    problem = StefanProblem()
    torch.manual_seed(0)
    triple = NeuralTriple(problem, width=16, depth=2)
    partition = build_partition(problem, n_t=10, n_x=10)

    result = certify_option_b(triple, problem, partition)

    a, b = problem.omega_bounds
    Q_measure = problem.T * (b - a)
    L_total_h = result.L_sm_new + result.L_Psi
    E_h_lip_new_recomputed = L_total_h * result.h * Q_measure

    assert E_h_lip_new_recomputed == pytest.approx(result.E_h_lip_new, rel=1e-9)


# ---------------------------------------------------------------------------
# Addendum A4 -- Stefan's Gamma_data is empty: the exclusion-aware path must
# agree EXACTLY with an unrestricted (no data-cut exclusion) computation.
# ---------------------------------------------------------------------------
def test_A4_data_cut_exclusion_is_a_no_op_for_stefan():
    problem = StefanProblem()
    torch.manual_seed(1)
    triple = NeuralTriple(problem, width=16, depth=2)
    partition = build_partition(problem, n_t=12, n_x=12)
    bounds = compute_triple_bounds(problem, triple)
    classification = classify_cells(problem, triple, partition, bounds.L_gamma)

    assert not classification.data_cut.any()  # Stefan: Gamma_data is empty

    pass_result = _run_midpoint_pass(
        problem, triple, partition, points_per_chunk=65536, max_rss_gb=8.0, check_every_n_chunks=10,
        classification=classification, track_R_PDE_max=True,
    )
    assert pass_result["n_data_cut_cells"] == 0

    # Reference: max |R_PDE(z_K)| over literally every cell, computed without
    # any data-cut awareness at all, via the same combined evaluator.
    from certified_obstacle_nn.certification.densities import psi_majorant_and_R_PDE
    from certified_obstacle_nn.certification.quadrature import build_quadrature_rule, iter_cell_chunks

    triple.eval()
    max_R_mid_unrestricted = 0.0
    for chunk in iter_cell_chunks(partition, build_quadrature_rule(q=1), 65536):
        _, R_PDE_vals = psi_majorant_and_R_PDE(problem, triple, chunk.t, chunk.x)
        max_R_mid_unrestricted = max(max_R_mid_unrestricted, R_PDE_vals.abs().max().item())

    assert pass_result["max_R_mid"] == pytest.approx(max_R_mid_unrestricted, rel=1e-12)


# ---------------------------------------------------------------------------
# Section 13 assertions -- non-negativity / geometry must actually be wired.
# ---------------------------------------------------------------------------
def test_section13_rejects_negative_radius():
    with pytest.raises(AssertionError):
        combine_certified_residual_bound(
            m_R_apriori=1.0, L_R=1.0, r_K=-0.01, max_R_mid=0.1,
            n_data_cut_cells=0, argmax_cell_id=0, argmax_t=0.0, argmax_x=0.0, mode="partition",
        )


def test_valid_modes_are_exactly_the_three_plan_modes():
    assert set(VALID_MODES) == {"apriori", "partition", "min_apriori_and_partition"}
    assert DEFAULT_MODE == "min_apriori_and_partition"


def test_rejects_unknown_mode():
    with pytest.raises(ValueError):
        combine_certified_residual_bound(
            m_R_apriori=1.0, L_R=1.0, r_K=0.01, max_R_mid=0.1,
            n_data_cut_cells=0, argmax_cell_id=0, argmax_t=0.0, argmax_x=0.0, mode="not_a_real_mode",
        )


# ---------------------------------------------------------------------------
# certify_option_b: apriori mode must degenerate exactly to the unsharpened
# certificate (mode="apriori" ignores the partition-based bound entirely).
# ---------------------------------------------------------------------------
def test_certify_option_b_apriori_mode_matches_old_values_exactly():
    problem = StefanProblem()
    torch.manual_seed(2)
    triple = NeuralTriple(problem, width=16, depth=2)
    partition = build_partition(problem, n_t=8, n_x=8)

    result = certify_option_b(triple, problem, partition, option_b_mode="apriori")

    assert result.residual_bound.m_R_cert == pytest.approx(result.bounds.m_R)
    assert result.T1_new == pytest.approx(result.T1_old)
    assert result.L_sm_new == pytest.approx(result.L_sm_old)
    assert result.E_h_new == pytest.approx(result.E_h_old)
    assert result.E_cert_U_new == pytest.approx(result.E_cert_U_old)


def test_certify_option_b_min_mode_never_worsens_the_certificate():
    problem = StefanProblem()
    torch.manual_seed(3)
    triple = NeuralTriple(problem, width=16, depth=2)
    partition = build_partition(problem, n_t=8, n_x=8)

    result = certify_option_b(triple, problem, partition, option_b_mode=DEFAULT_MODE)

    assert result.E_h_new <= result.E_h_old + 1e-9
    assert result.E_cert_U_new <= result.E_cert_U_old + 1e-9
