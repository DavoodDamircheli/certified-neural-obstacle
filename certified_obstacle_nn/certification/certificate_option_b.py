"""Option B top-level entry point: certify() with the a posteriori sharpened
m_R (docs/history/OPTION_B_CERTIFIED_RESIDUAL_BOUND_IMPLEMENTATION_PLAN.md), reporting old
(fully a priori) and new (sharpened) values side by side.

Deliberately NOT built by calling certificate.certify() and then separately
recomputing Option B's pieces -- certify() runs its own
compute_level2_remainder (one midpoint pass) internally, and Option B's m_R_h
needs its own midpoint pass too; calling both would be two traversals of the
partition where addendum A5 requires one. This module re-runs certify()'s own
orchestration (network bounds once, raw estimator once, cell classification
once) but calls compute_level2_remainder_and_residual_bound ONCE in place of
certify()'s two separate calls (classify_cells's own pass is O(n_cells),
no_grad, unchunked already -- not the expensive traversal A5 is about).

L_R = Lip(R_PDE) is never sharpened by sampling (plan section 18): it stays
exactly bounds.lip_R_PDE, the existing analytic bound, untouched. Only m_R is
sharpened; T1, L_sm, E_h, and the final certificate are recomputed from that
one substitution, reusing every other already-computed piece
(decompose_L_sm_and_L_Psi's T2..T5, bounds.L_Psi, remainder's E_h_unr/E_h_dat/
E_h_0T) rather than re-deriving anything.
"""

import math
from dataclasses import dataclass
from pathlib import Path

from certified_obstacle_nn.certification.cell_classifier import classify_cells
from certified_obstacle_nn.certification.certificate import _assert_frozen, _compute_C_eq, _freeze
from certified_obstacle_nn.certification.certified_residual_bound import (
    NORM_CONSISTENCY_REPORT,
    CertifiedResidualBound,
    DEFAULT_MODE,
    cell_radius_euclidean,
    combine_certified_residual_bound,
)
from certified_obstacle_nn.certification.disjointness import TrainPoints
from certified_obstacle_nn.certification.level2_remainder import (
    DEFAULT_MIDPOINT_CHUNK,
    compute_level2_remainder_and_residual_bound,
)
from certified_obstacle_nn.certification.memory import certification_stage
from certified_obstacle_nn.certification.network_bounds import TripleBounds, compute_triple_bounds, decompose_L_sm_and_L_Psi
from certified_obstacle_nn.certification.partition import Partition
from certified_obstacle_nn.certification.raw_estimator import DEFAULT_MAX_RSS_GB, DEFAULT_POINTS_PER_CHUNK, compute_raw_estimator
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.base_problem import BaseProblem


@dataclass(frozen=True)
class OptionBCertificateResult:
    # geometry / provenance
    h: float
    option_b_mode: str

    # raw estimator (unaffected by Option B). rho_r/rho_p_tilde/rho_c are the
    # specification section 4 split of rho_tilde_h (rho_tilde_h^2 = rho_r^2 +
    # rho_p_tilde^2 + rho_c^2) -- certify() (certificate.py) already threads
    # these through from EstimatorSummary; added here (M18) so
    # certify_option_b() reports the same split, not just the combined value.
    rho_tilde_h: float
    rho_tilde_h_sq: float
    rho_r: float
    rho_p_tilde: float
    rho_c: float
    V_h_unr: float
    V_h_dat: float

    # network + triple bounds (m_R_apriori = bounds.m_R, L_R = bounds.lip_R_PDE)
    bounds: TripleBounds

    # Option B's sharpened residual bound
    residual_bound: CertifiedResidualBound

    # T1 / L_sm, old (apriori) vs new (sharpened)
    T1_old: float
    T1_new: float
    L_sm_old: float
    L_sm_new: float
    L_Psi: float  # unchanged by Option B; reported for completeness

    # E_h, old vs new
    E_h_lip_old: float
    E_h_lip_new: float
    E_h_unr: float  # unchanged by Option B (does not depend on m_R)
    E_h_dat: float  # unchanged by Option B
    E_h_0T: float  # unchanged by Option B
    E_h_old: float
    E_h_new: float
    E_h_old_over_rho2: float
    E_h_new_over_rho2: float

    # final certificate, old vs new
    C_rel: float
    C_eq: float
    E_cert_old: float
    E_cert_U_old: float
    E_cert_new: float
    E_cert_U_new: float


def certify_option_b(
    model: NeuralTriple,
    problem: BaseProblem,
    partition: Partition,
    training_points: TrainPoints | None = None,
    n_qp: int = 4,
    *,
    points_per_chunk: int = DEFAULT_POINTS_PER_CHUNK,
    midpoint_chunk: int = DEFAULT_MIDPOINT_CHUNK,
    max_rss_gb: float = DEFAULT_MAX_RSS_GB,
    option_b_mode: str = DEFAULT_MODE,
    save_cells_dir: Path | None = None,
    bounds_override: TripleBounds | None = None,
) -> OptionBCertificateResult:
    """bounds_override (experiments/stefan_stress.py S2/S3, 2026-09-11): when
    given, used INSTEAD of compute_triple_bounds(problem, model) -- the only
    legitimate reason to do this is that `model` is a post-hoc-corrupted
    triple (certification/stress_corruption.py) whose a priori Lipschitz
    bounds must be hand-corrected for the corruption (weight-based
    compute_triple_bounds cannot see a post-hoc output addition at all).
    Default None preserves the exact prior behavior for every other caller."""
    _freeze(model)
    _assert_frozen(model)

    with certification_stage("certify_option_b:network_bounds"):
        bounds = bounds_override if bounds_override is not None else compute_triple_bounds(problem, model)

    with certification_stage("certify_option_b:raw_estimator"):
        _, estimator_summary = compute_raw_estimator(
            problem, model, partition, n_qp=n_qp,
            training_points=training_points, points_per_chunk=points_per_chunk, max_rss_gb=max_rss_gb,
            save_cells_dir=save_cells_dir,
        )
        rho_tilde_h = estimator_summary.rho_tilde
        rho_r = estimator_summary.rho_r
        rho_p_tilde = estimator_summary.rho_p_tilde
        rho_c = estimator_summary.rho_c

    with certification_stage("certify_option_b:cell_classifier"):
        classification = classify_cells(problem, model, partition, bounds.L_gamma)

    with certification_stage("certify_option_b:level2_remainder_and_residual_bound"):
        remainder, pass_result = compute_level2_remainder_and_residual_bound(
            problem, model, partition, bounds, classification,
            points_per_chunk=midpoint_chunk, max_rss_gb=max_rss_gb,
        )

    # Addendum A1: report the norm check in the log BEFORE r_K is used.
    r_K = cell_radius_euclidean(partition)
    print(NORM_CONSISTENCY_REPORT, flush=True)
    print(f"r_K = h_K/2 = {r_K:.6e}  (h_K = {partition.h_K:.6e})", flush=True)

    decomposition = decompose_L_sm_and_L_Psi(problem, bounds)
    T1_old = decomposition["L_sm_terms"]["T1_pde_residual"]
    other_terms_sum = (
        decomposition["L_sm_terms"]["T2_sig_residual"]
        + decomposition["L_sm_terms"]["T3_kappa_gap"]
        + decomposition["L_sm_terms"]["T4_lambda_Lgamma"]
        + decomposition["L_sm_terms"]["T5_gap_lip_lambda"]
    )
    assert math.isclose(T1_old + other_terms_sum, bounds.L_sm, rel_tol=1e-9), (
        "decompose_L_sm_and_L_Psi drifted from compute_triple_bounds's own L_sm -- see "
        "tests/test_network_bounds.py::test_decompose_L_sm_and_L_Psi_sums_back_exactly"
    )

    residual_bound = combine_certified_residual_bound(
        m_R_apriori=bounds.m_R,
        L_R=bounds.lip_R_PDE,
        r_K=r_K,
        max_R_mid=pass_result["max_R_mid"],
        n_data_cut_cells=pass_result["n_data_cut_cells"],
        argmax_cell_id=pass_result["argmax_cell_id"],
        argmax_t=pass_result["argmax_t"],
        argmax_x=pass_result["argmax_x"],
        mode=option_b_mode,
    )

    Lambda_over_alpha = problem.Lambda / problem.alpha
    T1_new = Lambda_over_alpha * 2.0 * residual_bound.m_R_cert * bounds.lip_R_PDE
    L_sm_new = T1_new + other_terms_sum
    L_sm_old = bounds.L_sm

    a, b = problem.omega_bounds
    omega_measure = b - a
    Q_measure = problem.T * omega_measure
    h = partition.h

    E_h_lip_old = (L_sm_old + bounds.L_Psi) * h * Q_measure
    E_h_lip_new = (L_sm_new + bounds.L_Psi) * h * Q_measure

    E_h_old = E_h_lip_old + remainder.E_h_unr + remainder.E_h_dat + remainder.E_h_0T
    E_h_new = E_h_lip_new + remainder.E_h_unr + remainder.E_h_dat + remainder.E_h_0T

    assert math.isclose(E_h_old, remainder.E_h, rel_tol=1e-9), (
        "E_h_old must equal the ordinary (unsharpened) Level2Remainder.E_h exactly -- "
        "it is built from the SAME L_sm_old bounds.L_sm already used there"
    )

    rho_tilde_h_sq = rho_tilde_h**2
    E_h_old_over_rho2 = E_h_old / rho_tilde_h_sq
    E_h_new_over_rho2 = E_h_new / rho_tilde_h_sq

    C_rel = 216.0
    C_eq = _compute_C_eq(problem)
    inner_old = rho_tilde_h_sq + E_h_old
    inner_new = rho_tilde_h_sq + E_h_new
    E_cert_old = math.sqrt(C_rel * inner_old)
    E_cert_U_old = math.sqrt(C_eq * C_rel * inner_old)
    E_cert_new = math.sqrt(C_rel * inner_new)
    E_cert_U_new = math.sqrt(C_eq * C_rel * inner_new)

    _assert_frozen(model)  # certify_option_b() must never touch parameters, start to finish

    return OptionBCertificateResult(
        h=h,
        option_b_mode=option_b_mode,
        rho_tilde_h=rho_tilde_h,
        rho_tilde_h_sq=rho_tilde_h_sq,
        rho_r=rho_r,
        rho_p_tilde=rho_p_tilde,
        rho_c=rho_c,
        V_h_unr=classification.V_h_unr,
        V_h_dat=classification.V_h_dat,
        bounds=bounds,
        residual_bound=residual_bound,
        T1_old=T1_old,
        T1_new=T1_new,
        L_sm_old=L_sm_old,
        L_sm_new=L_sm_new,
        L_Psi=bounds.L_Psi,
        E_h_lip_old=E_h_lip_old,
        E_h_lip_new=E_h_lip_new,
        E_h_unr=remainder.E_h_unr,
        E_h_dat=remainder.E_h_dat,
        E_h_0T=remainder.E_h_0T,
        E_h_old=E_h_old,
        E_h_new=E_h_new,
        E_h_old_over_rho2=E_h_old_over_rho2,
        E_h_new_over_rho2=E_h_new_over_rho2,
        C_rel=C_rel,
        C_eq=C_eq,
        E_cert_old=E_cert_old,
        E_cert_U_old=E_cert_U_old,
        E_cert_new=E_cert_new,
        E_cert_U_new=E_cert_U_new,
    )
