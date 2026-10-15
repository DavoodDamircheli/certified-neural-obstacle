"""Top-level certification entry point, specification section 8.

certify() glues together the raw estimator (rho_tilde_h, M9/M10), the network
bounds (M12), and the Level-2 remainder (E_h, M14) into the final certified
numbers:

  E_cert    = sqrt( C_rel * (rho_tilde_h^2 + E_h) )
  E_cert_U  = sqrt( C_eq * C_rel * (rho_tilde_h^2 + E_h) )

with C_rel = 216 and C_eq computed from the problem's own A, b, c, alpha,
Lambda per section 8's formula (== 2 for Stefan, asserted in
tests/test_certificate.py).

Signature note: the requested signature was certify(model, problem,
partition), but asserting "partition points are disjoint from the training
set" is impossible without being given some representation of that training
set to compare against. Extended with an optional `training_points` keyword
(a plain (t, x) tensor pair, e.g. TrainingSampler.interior_points) -- not the
sampler object itself, per specification section 12 invariant 1 ("certification
code must not receive the training sampler"). If omitted, the disjointness
check is skipped rather than silently assumed to pass.

Streaming note (docs/history/MEMORY_OOM_DEBUGGING_AND_SAFE_RUN_POLICY.md): `points_per_chunk`,
`midpoint_chunk`, and `max_rss_gb` are threaded straight through to
compute_raw_estimator (interior quadrature, `points_per_chunk`) and
compute_level2_remainder (E_h's midpoint pass, `midpoint_chunk` -- a SEPARATE
knob, not a reuse of `points_per_chunk`: a midpoint is one point per cell
versus quad_order^2 for the interior loop, so the same "points per chunk"
budget covers a very different number of cells in each pass; see policy
section 14). certify() itself never materializes a grid-sized tensor. The
disjointness check used to live here as its own O(N_train x N_cell) broadcast
pass (_assert_disjoint_from_training, now deleted); it has moved into
compute_raw_estimator, which runs it inline against the SAME chunk generator
the estimator itself consumes (certification/disjointness.py's
iter_disjoint_checked) rather than as a second, separate full pass over the
grid -- `training_points` is simply forwarded there now.
"""

import math
from dataclasses import dataclass
from pathlib import Path

from certified_obstacle_nn.certification import reliability
from certified_obstacle_nn.certification.cell_classifier import classify_cells
from certified_obstacle_nn.certification.component_split import split_global_remainder
from certified_obstacle_nn.certification.disjointness import TrainPoints
from certified_obstacle_nn.certification.level2_remainder import (
    DEFAULT_MIDPOINT_CHUNK,
    Level2Remainder,
    compute_level2_remainder,
)
from certified_obstacle_nn.certification.memory import certification_stage
from certified_obstacle_nn.certification.network_bounds import TripleBounds, compute_triple_bounds
from certified_obstacle_nn.certification.partition import Partition
from certified_obstacle_nn.certification.raw_estimator import (
    DEFAULT_MAX_RSS_GB,
    DEFAULT_POINTS_PER_CHUNK,
    compute_raw_estimator,
)
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.base_problem import BaseProblem


@dataclass(frozen=True)
class CertificateResult:
    # raw estimator (M9/M10)
    rho_r: float
    rho_p_tilde: float
    rho_c: float
    rho_tilde_h: float

    # network + triple bounds (M12)
    bounds: TripleBounds

    # cell classification, specification section 7
    V_h_unr: float
    V_h_dat: float

    # Level-2 remainder (M14)
    remainder: Level2Remainder

    # final certified numbers, specification section 8
    C_rel: float
    C_eq: float
    E_cert: float
    E_cert_U: float

    # v14 additions (A3/A4). C_rel above stays the historical 216 comparison
    # convention so old numbers reproduce; the per-problem factor and the
    # component split live here.
    C_rel_v14: float = float("nan")
    E_cert_U_v14: float = float("nan")
    components: dict | None = None


def _freeze(model: NeuralTriple) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def _assert_frozen(model: NeuralTriple) -> None:
    for p in model.parameters():
        assert not p.requires_grad, "certification invariant violated: a parameter has requires_grad=True"


def _compute_C_eq(problem: BaseProblem) -> float:
    """C_eq from the specification section 8 formula (shared with v14 reliability.py)."""
    return reliability.C_eq(problem)


def certify(
    model: NeuralTriple,
    problem: BaseProblem,
    partition: Partition,
    training_points: TrainPoints | None = None,
    n_qp: int = 4,
    *,
    points_per_chunk: int = DEFAULT_POINTS_PER_CHUNK,
    midpoint_chunk: int = DEFAULT_MIDPOINT_CHUNK,
    max_rss_gb: float = DEFAULT_MAX_RSS_GB,
    save_cells_dir: Path | None = None,
) -> CertificateResult:
    _freeze(model)
    _assert_frozen(model)

    with certification_stage("certify:network_bounds"):
        bounds = compute_triple_bounds(problem, model)

    with certification_stage("certify:raw_estimator"):
        _, estimator_summary = compute_raw_estimator(
            problem, model, partition, n_qp=n_qp,
            training_points=training_points, points_per_chunk=points_per_chunk, max_rss_gb=max_rss_gb,
            save_cells_dir=save_cells_dir,
        )
        rho_tilde_h = estimator_summary.rho_tilde

    with certification_stage("certify:cell_classifier"):
        classification = classify_cells(problem, model, partition, bounds.L_gamma)

    with certification_stage("certify:level2_remainder"):
        remainder = compute_level2_remainder(
            problem, model, partition, bounds, classification,
            points_per_chunk=midpoint_chunk, max_rss_gb=max_rss_gb,
        )

    C_rel = float(reliability.HISTORICAL_C_REL)  # historical comparison convention (v14 A3)
    C_eq = _compute_C_eq(problem)
    components = split_global_remainder(
        problem, bounds, remainder, partition.h,
        estimator_summary.rho_r, estimator_summary.rho_p_tilde, estimator_summary.rho_c,
    )
    C_rel_v14 = float(reliability.C_rel(problem))

    inner = rho_tilde_h**2 + remainder.E_h
    E_cert = math.sqrt(C_rel * inner)
    E_cert_U = math.sqrt(C_eq * C_rel * inner)

    _assert_frozen(model)  # certify() must never touch parameters, start to finish

    return CertificateResult(
        rho_r=estimator_summary.rho_r,
        rho_p_tilde=estimator_summary.rho_p_tilde,
        rho_c=estimator_summary.rho_c,
        rho_tilde_h=rho_tilde_h,
        bounds=bounds,
        V_h_unr=classification.V_h_unr,
        V_h_dat=classification.V_h_dat,
        remainder=remainder,
        C_rel=C_rel,
        C_eq=C_eq,
        E_cert=E_cert,
        E_cert_U=E_cert_U,
        C_rel_v14=C_rel_v14,
        E_cert_U_v14=math.sqrt(C_eq * C_rel_v14 * inner),
        components=components,
    )
