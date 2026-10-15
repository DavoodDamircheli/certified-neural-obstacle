"""Option B: a posteriori certified residual supremum bound, per
docs/history/OPTION_B_CERTIFIED_RESIDUAL_BOUND_IMPLEMENTATION_PLAN.md and its addendum.

Lemma (plan section 4). Let R_PDE be L_R-Lipschitz on a certification cell K
(w.r.t. the SAME norm L_R was derived in -- see the norm-consistency note
below), and z_K the cell center. Then for every z in K,

    |R_PDE(z)| <= |R_PDE(z)) - R_PDE(z_K)| + |R_PDE(z_K)|
               <= L_R ||z - z_K|| + |R_PDE(z_K)|
               <= L_R r_K + |R_PDE(z_K)|                    (r_K = sup_{z in K} ||z-z_K||)

so ||R_PDE||_{L^inf(K)} <= |R_PDE(z_K)| + L_R r_K, and taking the max over
cells, ||R_PDE||_{L^inf(Q)} <= max_K (|R_PDE(z_K)| + L_R r_K) =: m_R^h.

This module contains ONLY the closed-form combination of already-computed
quantities (m_R_apriori from network_bounds.py, and a per-partition running
max of |R_PDE| at cell midpoints computed elsewhere -- see
level2_remainder.py's _run_midpoint_pass, addendum A5: this module performs
no network evaluation and no partition traversal of its own).

Addendum A1 -- geometry, stated explicitly. specification section 7 defines
h_K = sqrt(dt^2 + dx^2) as the cell DIAGONAL (partition.py's own h_K field),
not a side length. The Euclidean center-to-corner radius of a rectangle with
side lengths dt, dx is

    r_K = sqrt((dt/2)^2 + (dx/2)^2) = (1/2) sqrt(dt^2 + dx^2) = h_K / 2,

NOT h_K (2x conservative if used directly) and NOT h_K/sqrt(2) (wrong: that
would be the center-to-EDGE-MIDPOINT distance for a SQUARE cell in one
specific degenerate case, not the center-to-CORNER supremum a general
rectangular cell needs -- using it invalidates the lemma above, since r_K
must upper bound ||z-z_K|| for every z in K, including the corners).

Norm consistency (also A1): L_R = bounds.lip_R_PDE is built (specification
section 6's addendum) as a sum of pieces each derived via the Euclidean
combination sqrt(M_t^2 + M_x^2) of separate time/space Lipschitz bounds, and
tests/test_network_bounds.py's dense-sampling validation of every such piece
uses exactly dist = sqrt(dt^2+dx^2) (its `_empirical_scalar_lipschitz`
helper). r_K above is the Euclidean center-to-corner radius. Both are
Euclidean -- they match. NORM_CONSISTENCY_REPORT below is the fixed string
this fact is logged with; it is a static mathematical fact about how L_R was
derived and tested, not something computed at runtsime, so it is a constant,
not a check.
"""

from dataclasses import dataclass

from certified_obstacle_nn.certification.partition import Partition

VALID_MODES = ("apriori", "partition", "min_apriori_and_partition")
DEFAULT_MODE = "min_apriori_and_partition"

NORM_CONSISTENCY_REPORT = (
    "norm check: L_R = Lip(R_PDE) is derived and dense-sampling-validated w.r.t. the "
    "EUCLIDEAN (t,x) distance sqrt(dt^2+dx^2) (specification section 6 addendum's combination "
    "rule; tests/test_network_bounds.py's _empirical_scalar_lipschitz uses this same "
    "distance). r_K = h_K/2 is the EUCLIDEAN center-to-corner radius (h_K = sqrt(dt^2+dx^2) "
    "is the cell DIAGONAL per specification section 7 / partition.py, so half of it is the "
    "center-to-corner distance). NORMS MATCH: both Euclidean."
)


def cell_radius_euclidean(partition: Partition) -> float:
    """r_K = h_K / 2 -- see module docstring's addendum A1 derivation.
    Uniform across cells on this repo's tensor-product partition (h_K is
    itself already uniform, per partition.py)."""
    return partition.h_K / 2.0


@dataclass(frozen=True)
class CertifiedResidualBound:
    mode: str
    m_R_apriori: float
    L_R: float
    r_K: float
    max_R_mid: float  # max_K |R_PDE(z_K)| over NON-data-cut cells only
    lip_term: float  # L_R * r_K
    m_R_h_nondata: float  # max_R_mid + lip_term
    n_data_cut_cells: int
    m_R_h: float  # after data-cut fallback (A4)
    m_R_cert: float  # per `mode`
    lip_term_fraction: float  # lip_term / m_R_h_nondata: which of the 2 terms in m_R_h_nondata dominates
    argmax_cell_id: int
    argmax_t: float
    argmax_x: float
    dist_argmax_from_diagonal: float  # |argmax_t - argmax_x|, Stefan's free boundary is t=x


def combine_certified_residual_bound(
    *,
    m_R_apriori: float,
    L_R: float,
    r_K: float,
    max_R_mid: float,
    n_data_cut_cells: int,
    argmax_cell_id: int,
    argmax_t: float,
    argmax_x: float,
    mode: str = DEFAULT_MODE,
) -> CertifiedResidualBound:
    """Pure combination logic -- no network evaluation, no partition
    traversal (`max_R_mid`/`argmax_*` are supplied, already computed by
    level2_remainder.py's shared midpoint pass, addendum A5).

    Addendum A4 -- data-cut cells. The lemma requires R_PDE Lipschitz on K,
    which the theory does not guarantee on a cell meeting a data interface
    Gamma_data. `max_R_mid` is therefore computed over non-data-cut cells
    ONLY (by the caller). If ANY data-cut cells exist, this function falls
    back CONSERVATIVELY to the full a priori bound for m_R_h (never worse
    than m_R_apriori, per the plan's own guiding principle #1) rather than
    attempting a locally-precise per-cell exclusion -- that refinement is
    deferred to when American put (Gamma_data = {log K}, nonempty) actually
    needs it (M23/M24), not required for Stefan (Gamma_data empty) now. When
    n_data_cut_cells == 0 (Stefan), this reduces to exactly
    m_R_h = m_R_h_nondata with no fallback applied -- the two paths
    "agree exactly" the addendum requires, verified in
    tests/test_certified_residual_bound.py.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

    lip_term = L_R * r_K
    m_R_h_nondata = max_R_mid + lip_term
    lip_term_fraction = lip_term / m_R_h_nondata if m_R_h_nondata > 0 else float("nan")

    if n_data_cut_cells > 0:
        m_R_h = max(m_R_h_nondata, m_R_apriori)
    else:
        m_R_h = m_R_h_nondata

    if mode == "apriori":
        m_R_cert = m_R_apriori
    elif mode == "partition":
        m_R_cert = m_R_h
    else:  # min_apriori_and_partition
        m_R_cert = min(m_R_apriori, m_R_h)

    assert m_R_apriori >= 0.0
    assert m_R_h >= 0.0
    assert m_R_cert >= 0.0
    assert r_K >= 0.0
    # Plan section 13's "m_R_cert <= m_R_apriori" / "m_R_cert <= m_R_h" pair is
    # only universally true for the recommended min_apriori_and_partition
    # mode (where it holds by construction, min(...) <= both operands) -- the
    # apriori/partition modes deliberately pick ONE side and ignore the
    # other, which need not dominate it (e.g. m_R_h can exceed m_R_apriori on
    # a coarse grid where the a posteriori term hasn't shrunk yet). Assert
    # only what each mode actually guarantees.
    if mode == "min_apriori_and_partition":
        assert m_R_cert <= m_R_apriori + 1e-9
        assert m_R_cert <= m_R_h + 1e-9
    elif mode == "apriori":
        assert m_R_cert == m_R_apriori
    else:  # partition
        assert m_R_cert == m_R_h

    return CertifiedResidualBound(
        mode=mode,
        m_R_apriori=m_R_apriori,
        L_R=L_R,
        r_K=r_K,
        max_R_mid=max_R_mid,
        lip_term=lip_term,
        m_R_h_nondata=m_R_h_nondata,
        n_data_cut_cells=n_data_cut_cells,
        m_R_h=m_R_h,
        m_R_cert=m_R_cert,
        lip_term_fraction=lip_term_fraction,
        argmax_cell_id=argmax_cell_id,
        argmax_t=argmax_t,
        argmax_x=argmax_x,
        dist_argmax_from_diagonal=abs(argmax_t - argmax_x),
    )
