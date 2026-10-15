"""Level-2 guaranteed remainder, specification section 7.

E_h = E_h_lip + E_h_unr + E_h_dat + E_h_0T combines a Lipschitz-sweep term
(Phi_sm + Psi are Lipschitz everywhere, so a crude h*|Q| sweep bounds their
contribution), an unresolved-cell term (evaluated via the Psi majorant, since
sign-unresolved cells may straddle the free boundary where Phi_int loses
regularity), a data-interface term (exact via the closed-form Gamma_data), and
a time-boundary term.

Section 7 says "always report the four components separately", and specification
section 12 invariant 5 forbids merging the estimator and the certified
remainder into one column -- so this module always returns all four plus the
sum, never only the sum.

E_h_unr needs Psi at cell MIDPOINTS only (one point per cell, not the q^2
quadrature points raw_estimator.py uses) -- deliberately its own chunked pass
over certification/quadrature.py's iter_cell_chunks with a q=1 rule (a q=1
Gauss-Legendre rule has exactly one node, at 0, with weight 2, which maps to
exactly the cell midpoint with weight = |K| -- "the same machinery" as the
interior quadrature loop, not a bespoke midpoint scheme), never folded into
that interior loop: the two have different point sets (quadrature nodes are
always strictly interior to a cell; the midpoint is a different point), and
conflating them has already caused sign-classification bugs elsewhere.

docs/history/OPTION_B_CERTIFIED_RESIDUAL_BOUND_IMPLEMENTATION_PLAN.md addendum A5: Option
B's m_R_h needs exactly the derivatives (u_t, u_x, sigma_x, lam -- hence
R_PDE) this module's midpoint pass ALREADY computes per cell to get Psi(z_K).
`_run_midpoint_pass` is the single shared traversal: `compute_level2_remainder`
(unchanged public behavior) uses it with residual-tracking off;
`compute_level2_remainder_and_residual_bound` (new) uses it with
residual-tracking on, paying for the R_PDE side of the computation only once,
never a second traversal of the partition.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from certified_obstacle_nn.certification.cell_classifier import CellClassification
from certified_obstacle_nn.certification.densities import psi_majorant, psi_majorant_and_R_PDE
from certified_obstacle_nn.certification.memory import MemoryGuard, certification_stage
from certified_obstacle_nn.certification.network_bounds import TripleBounds
from certified_obstacle_nn.certification.partition import Partition
from certified_obstacle_nn.certification.quadrature import build_quadrature_rule, iter_cell_chunks
from certified_obstacle_nn.certification.raw_estimator import DEFAULT_CHECK_EVERY_N_CHUNKS, DEFAULT_MAX_RSS_GB
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.base_problem import BaseProblem

# Deliberately a separate constant from raw_estimator.DEFAULT_POINTS_PER_CHUNK
# (currently the same value, 65536, but not the same knob): a midpoint is one
# point per cell versus quad_order^2 for the interior loop, so "points per
# chunk" covers a very different number of CELLS in each pass. Configs (see
# configs/*.yaml) set these independently.
DEFAULT_MIDPOINT_CHUNK = 65536


@dataclass(frozen=True)
class Level2Remainder:
    E_h_lip: float
    E_h_unr: float
    E_h_dat: float
    E_h_0T: float
    E_h: float


def compute_level2_remainder(
    problem: BaseProblem,
    triple: NeuralTriple,
    partition: Partition,
    bounds: TripleBounds,
    classification: CellClassification,
    *,
    points_per_chunk: int = DEFAULT_MIDPOINT_CHUNK,
    max_rss_gb: float = DEFAULT_MAX_RSS_GB,
    check_every_n_chunks: int = DEFAULT_CHECK_EVERY_N_CHUNKS,
) -> Level2Remainder:
    a, b = problem.omega_bounds
    omega_measure = b - a
    Q_measure = problem.T * omega_measure
    h = partition.h

    E_h_lip = (bounds.L_sm + bounds.L_Psi) * h * Q_measure

    unresolved_np = classification.sign_unresolved.numpy()
    if unresolved_np.any():
        pass_result = _run_midpoint_pass(
            problem, triple, partition, points_per_chunk, max_rss_gb, check_every_n_chunks,
            classification=None, track_R_PDE_max=False,
        )
        E_h_unr = _E_h_unr_from_psi_K(pass_result["psi_K"], unresolved_np, bounds.L_Psi, partition)
    else:
        E_h_unr = 0.0

    E_h_dat = bounds.M_Psi * classification.V_h_dat

    E_h_0T = (bounds.L_Phi0 + bounds.L_PhiT) * h * omega_measure

    E_h = E_h_lip + E_h_unr + E_h_dat + E_h_0T

    return Level2Remainder(E_h_lip=E_h_lip, E_h_unr=E_h_unr, E_h_dat=E_h_dat, E_h_0T=E_h_0T, E_h=E_h)


def compute_level2_remainder_and_residual_bound(
    problem: BaseProblem,
    triple: NeuralTriple,
    partition: Partition,
    bounds: TripleBounds,
    classification: CellClassification,
    *,
    points_per_chunk: int = DEFAULT_MIDPOINT_CHUNK,
    max_rss_gb: float = DEFAULT_MAX_RSS_GB,
    check_every_n_chunks: int = DEFAULT_CHECK_EVERY_N_CHUNKS,
) -> tuple[Level2Remainder, dict]:
    """Option B entry point (OPTION_B_..._PLAN.md addendum A5): runs the
    SAME midpoint pass as compute_level2_remainder, but with residual
    tracking on, and UNCONDITIONALLY (m_R_h needs it regardless of whether
    there are any sign-unresolved cells, unlike E_h_unr) -- returns the
    ordinary (unchanged-formula) Level2Remainder alongside the raw dict
    certification/certified_residual_bound.py's combine_certified_residual_bound
    needs (max_R_mid, argmax_*, n_data_cut_cells). Never calls
    compute_level2_remainder itself -- that would be a second traversal.
    """
    a, b = problem.omega_bounds
    omega_measure = b - a
    Q_measure = problem.T * omega_measure
    h = partition.h

    E_h_lip = (bounds.L_sm + bounds.L_Psi) * h * Q_measure

    unresolved_np = classification.sign_unresolved.numpy()
    pass_result = _run_midpoint_pass(
        problem, triple, partition, points_per_chunk, max_rss_gb, check_every_n_chunks,
        classification=classification, track_R_PDE_max=True,
    )
    E_h_unr = _E_h_unr_from_psi_K(pass_result["psi_K"], unresolved_np, bounds.L_Psi, partition) if unresolved_np.any() else 0.0

    E_h_dat = bounds.M_Psi * classification.V_h_dat
    E_h_0T = (bounds.L_Phi0 + bounds.L_PhiT) * h * omega_measure
    E_h = E_h_lip + E_h_unr + E_h_dat + E_h_0T

    remainder = Level2Remainder(E_h_lip=E_h_lip, E_h_unr=E_h_unr, E_h_dat=E_h_dat, E_h_0T=E_h_0T, E_h=E_h)
    return remainder, pass_result


def _E_h_unr_from_psi_K(psi_K: np.ndarray, unresolved_np: np.ndarray, L_Psi: float, partition: Partition) -> float:
    contrib = partition.area * (psi_K[unresolved_np] + L_Psi * partition.h)
    return math.fsum(contrib.tolist())


def _run_midpoint_pass(
    problem: BaseProblem,
    triple: NeuralTriple,
    partition: Partition,
    points_per_chunk: int,
    max_rss_gb: float,
    check_every_n_chunks: int,
    classification: Optional[CellClassification],
    track_R_PDE_max: bool,
) -> dict:
    """The shared M15c chunked midpoint pass (q=1 rule). Always computes
    psi_K (E_h_unr's raw material). When track_R_PDE_max=True, ALSO keeps a
    running max of |R_PDE(z_K)| over non-data-cut cells plus its argmax
    (cell_id, t, x) -- O(1) extra state per chunk, using
    psi_majorant_and_R_PDE instead of psi_majorant so this is the SAME
    network evaluation, not an extra one. Never stores a per-cell residual
    array (OPTION_B_..._PLAN.md section 7 / addendum A5: "keep a running
    maximum plus the argmax index only").
    """
    triple.eval()
    for p in triple.parameters():
        p.requires_grad_(False)

    n_cells = partition.n_t * partition.n_x
    psi_K = np.zeros(n_cells, dtype=np.float64)

    data_cut_np = classification.data_cut.numpy() if (track_R_PDE_max and classification is not None) else None
    max_R_mid = 0.0
    argmax_cell_id = -1
    argmax_t = float("nan")
    argmax_x = float("nan")
    n_data_cut_cells = int(data_cut_np.sum()) if data_cut_np is not None else 0

    midpoint_rule = build_quadrature_rule(q=1)  # 1 point/cell, at the midpoint, weight=|K|
    guard = MemoryGuard(
        max_rss_gb=max_rss_gb, label_prefix="level2_remainder",
        chunk_size=points_per_chunk, grid=(partition.n_t, partition.n_x),
    )

    with certification_stage("level2_remainder:midpoint_pass"):
        for chunk_idx, chunk in enumerate(iter_cell_chunks(partition, midpoint_rule, points_per_chunk)):
            cell_ids_np = chunk.cell_ids.numpy()

            if track_R_PDE_max:
                psi_vals, R_PDE_vals = psi_majorant_and_R_PDE(problem, triple, chunk.t, chunk.x)
            else:
                psi_vals, R_PDE_vals = psi_majorant(problem, triple, chunk.t, chunk.x), None

            psi_vals = psi_vals.reshape(-1)
            assert not psi_vals.requires_grad and psi_vals.grad_fn is None, (
                "certification invariant violated: an autograd graph escaped the chunk"
            )
            psi_K[cell_ids_np] = psi_vals.numpy()

            if track_R_PDE_max:
                assert not R_PDE_vals.requires_grad and R_PDE_vals.grad_fn is None, (
                    "certification invariant violated: an autograd graph escaped the chunk"
                )
                # q=1: one point per cell, so cell_ids_np indexes points 1:1.
                nondata_local = ~data_cut_np[cell_ids_np]
                if nondata_local.any():
                    R_abs_local = R_PDE_vals.reshape(-1).abs().numpy()[nondata_local]
                    local_argmax = int(R_abs_local.argmax())
                    local_max = float(R_abs_local[local_argmax])
                    if local_max > max_R_mid:
                        max_R_mid = local_max
                        kept_global_ids = cell_ids_np[nondata_local]
                        kept_t = chunk.t.reshape(-1).numpy()[nondata_local]
                        kept_x = chunk.x.reshape(-1).numpy()[nondata_local]
                        argmax_cell_id = int(kept_global_ids[local_argmax])
                        argmax_t = float(kept_t[local_argmax])
                        argmax_x = float(kept_x[local_argmax])

            del chunk, psi_vals, R_PDE_vals

            if (chunk_idx + 1) % check_every_n_chunks == 0:
                guard.check(f"midpoint chunk {chunk_idx + 1}")

    return {
        "psi_K": psi_K,
        "max_R_mid": max_R_mid,
        "argmax_cell_id": argmax_cell_id,
        "argmax_t": argmax_t,
        "argmax_x": argmax_x,
        "n_data_cut_cells": n_data_cut_cells,
    }
