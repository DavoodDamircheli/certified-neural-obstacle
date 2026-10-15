"""O(N_train)-memory disjointness check, specification section 12 invariant 1.

The original in-process check (certificate.py's `_assert_disjoint_from_training`,
now superseded here) built an (N_train, N_cell, 2) broadcast comparison --
root cause H3 of the M15 OOM investigation
(docs/history/MEMORY_OOM_DEBUGGING_AND_SAFE_RUN_POLICY.md section 1). It was secondary to
H1/H2 there, but it is still a real O(N_train * N_cert) allocation that grows
without bound as the certification grid refines, which is exactly backwards:
refining the grid should never make the SAFETY CHECK more expensive than the
computation it is guarding.

FORBIDDEN in this module, and the reason each one is forbidden:
    torch.cdist(cert, train)                       -- O(N_cert * N_train) output
    cert[:, None, :] - train[None, :, :]            -- same, via broadcasting
    scipy.spatial.distance.cdist(cert, train)       -- same, numpy version
    any pairwise broadcast/materialization over both sets at once

Required implementation instead, chosen by `tol`:
    tol == 0  -- exact-match mode. A Python set of rounded key tuples is built
                 ONCE from the training points (np.round(pts, 12) -> tuple);
                 each streamed certification chunk is then tested for set
                 membership, O(chunk size) per chunk, O(1) average per point.
    tol > 0   -- scipy.spatial.cKDTree built ONCE on the training points;
                 each streamed chunk is queried with
                 `tree.query(chunk, k=1, distance_upper_bound=tol)`, which lets
                 the tree prove "nothing within tol" without computing an exact
                 distance for far-away points. A returned distance is finite
                 (hence a violation) exactly when a training point was found
                 within tol; otherwise scipy returns +inf, never a value >tol
                 that would need a second check.

Either way, the persistent state is O(N_train) (the set or the tree), and
per-chunk work is O(chunk size) -- total memory never depends on N_cert.

A collision is a scientific-validity error (the certificate would be provably
invalid, not just imprecise), so it is reported, not merely asserted: which
cells it fell in, how many there were, and the minimum distance observed --
see `_DisjointnessChecker.summary()` and `DisjointnessViolationError`.

Two ways to run the check, both O(N_train) memory:
    assert_disjoint(chunks_iter, train_points, tol)         -- consumes the
        iterator itself; for a standalone check (tests, an explicit
        pre-certification validation pass).
    iter_disjoint_checked(chunks_iter, train_points, tol)   -- wraps the SAME
        generator the estimator consumes and yields each chunk through
        unchanged after checking it, so the check rides along inside the
        estimator's own single pass over the grid instead of requiring a
        second full pass (two passes over the grid would be twice the
        compute for no benefit).

Both accept `train_points=None` and are then a no-op (resp. transparent
passthrough) -- matching certificate.py's existing documented default of
skipping the check when no training points were supplied.
"""

import math
from typing import Iterable, Iterator, Optional

import numpy as np
import torch

TrainPoints = tuple[torch.Tensor, torch.Tensor]


class DisjointnessViolationError(RuntimeError):
    """Raised when certification quadrature points coincide with (tol=0) or
    fall within `tol` of (tol>0) the training collocation set. specification
    section 12 invariant 1: this makes the certificate invalid, not merely
    imprecise, so this is always raised loudly with a full report attached,
    never downgraded to a warning."""


class _DisjointnessChecker:
    _MAX_EXAMPLES = 10

    def __init__(self, train_points: TrainPoints, tol: float):
        if tol < 0:
            raise ValueError(f"tol must be >= 0, got {tol}")
        self.tol = tol

        t_train, x_train = train_points
        train_np = (
            torch.stack([t_train.detach().reshape(-1), x_train.detach().reshape(-1)], dim=-1)
            .to(torch.float64)
            .numpy()
        )
        self.n_train = train_np.shape[0]

        self.n_cert_checked = 0
        self.n_collisions = 0
        self.min_distance = math.inf
        self.collisions_by_cell: dict[int, int] = {}
        self.example_collisions: list[dict] = []

        if tol == 0.0:
            self._train_key_set = {tuple(row) for row in np.round(train_np, 12)}
            self._tree = None
        else:
            from scipy.spatial import cKDTree  # imported lazily: only needed for tol>0

            self._tree = cKDTree(train_np)
            self._train_key_set = None

    def check_chunk(self, chunk) -> None:
        t = chunk.t.detach().reshape(-1).to(torch.float64).numpy()
        x = chunk.x.detach().reshape(-1).to(torch.float64).numpy()
        n_points = t.shape[0]
        self.n_cert_checked += n_points

        cell_ids_np = chunk.cell_ids.detach().numpy()
        n_cells = cell_ids_np.shape[0]
        assert n_points % n_cells == 0, (
            "disjointness checker assumes every cell in a chunk contributes the same "
            "number of quadrature points (true for certification/quadrature.py's "
            "iter_cell_chunks); got a chunk that violates this"
        )
        points_per_cell = n_points // n_cells
        point_cell_ids = np.repeat(cell_ids_np, points_per_cell)

        pts = np.stack([t, x], axis=-1)

        if self._train_key_set is not None:
            keys = np.round(pts, 12)
            for i in range(n_points):
                if tuple(keys[i]) in self._train_key_set:
                    self._record_collision(t[i], x[i], 0.0, int(point_cell_ids[i]))
        else:
            distances, _ = self._tree.query(pts, k=1, distance_upper_bound=self.tol)
            hit_idx = np.nonzero(np.isfinite(distances))[0]
            for i in hit_idx:
                self._record_collision(t[i], x[i], float(distances[i]), int(point_cell_ids[i]))

    def _record_collision(self, t_val: float, x_val: float, distance: float, cell_id: int) -> None:
        self.n_collisions += 1
        self.min_distance = min(self.min_distance, distance)
        self.collisions_by_cell[cell_id] = self.collisions_by_cell.get(cell_id, 0) + 1
        if len(self.example_collisions) < self._MAX_EXAMPLES:
            self.example_collisions.append({"t": t_val, "x": x_val, "distance": distance, "cell_id": cell_id})

    def summary(self) -> str:
        mode = "exact-match (tol=0)" if self.tol == 0.0 else f"KD-tree (tol={self.tol})"
        if self.n_collisions > 0:
            min_dist_str = f"{self.min_distance:.6e}"
        elif self.tol == 0.0:
            min_dist_str = "N/A (exact-match mode does not compute a distance for non-colliding points)"
        else:
            min_dist_str = f"> {self.tol} for every checked point (none within tol)"
        lines = [
            f"disjointness check [{mode}]: {self.n_cert_checked} certification point(s) checked "
            f"against {self.n_train} training point(s) -- {self.n_collisions} collision(s).",
            f"  min distance found: {min_dist_str}",
        ]
        if self.collisions_by_cell:
            lines.append(f"  affected cells (cell_id: n_colliding_points): {self.collisions_by_cell}")
        return "\n".join(lines)

    def raise_if_violated(self) -> None:
        report = self.summary()
        print(report, flush=True)
        if self.n_collisions == 0:
            return
        examples = "\n".join(
            f"    cert point (t={e['t']:.12f}, x={e['x']:.12f}) in cell {e['cell_id']}, "
            f"distance={e['distance']:.6e}"
            for e in self.example_collisions
        )
        n_more = self.n_collisions - len(self.example_collisions)
        more_note = f"\n    ... and {n_more} more" if n_more > 0 else ""
        raise DisjointnessViolationError(
            "certification invariant violated (specification section 12 invariant 1): "
            f"{self.n_collisions} certification quadrature point(s) coincide with "
            f"(or fall within tol={self.tol} of) a training collocation point.\n"
            f"{report}\n  examples:\n{examples}{more_note}"
        )


def assert_disjoint(
    cert_points_iter: Iterable,
    train_points: Optional[TrainPoints],
    tol: float = 0.0,
) -> None:
    """Consumes `cert_points_iter` (an iterable of quadrature chunks with
    `.cell_ids`, `.t`, `.x` -- e.g. certification/quadrature.py's
    iter_cell_chunks) fully, raising DisjointnessViolationError if any
    certification point coincides with (tol=0) or falls within tol (tol>0)
    of a training point. O(N_train) memory, chunk-streamed, never an
    (N_cert x N_train) array. `train_points=None` is a no-op (matches
    certificate.py's documented default of skipping the check when no
    training points were supplied)."""
    if train_points is None:
        for _ in cert_points_iter:
            pass
        return
    checker = _DisjointnessChecker(train_points, tol)
    for chunk in cert_points_iter:
        checker.check_chunk(chunk)
    checker.raise_if_violated()


def iter_disjoint_checked(
    cert_chunks_iter: Iterable,
    train_points: Optional[TrainPoints],
    tol: float = 0.0,
) -> Iterator:
    """Wraps `cert_chunks_iter`, yielding every chunk through unchanged after
    checking it, so a caller (e.g. a chunked raw_estimator) gets the
    disjointness check for free inside its own single pass over the grid --
    it must consume THIS generator instead of the raw one, not both, or the
    point of avoiding a second full pass is lost. Raises
    DisjointnessViolationError once the wrapped iterator is exhausted, before
    the caller's own post-loop code (e.g. computing a final certificate
    value) runs. `train_points=None` is a transparent passthrough."""
    if train_points is None:
        yield from cert_chunks_iter
        return
    checker = _DisjointnessChecker(train_points, tol)
    for chunk in cert_chunks_iter:
        checker.check_chunk(chunk)
        yield chunk
    checker.raise_if_violated()
