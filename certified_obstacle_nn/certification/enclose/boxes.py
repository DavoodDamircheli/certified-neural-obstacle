"""Dimension-generic dyadic box partitions of the unit cube [0,1]^D, their
validated evaluation and adaptive refinement (space-time dimension D = 3 for
the 2-D-space benchmarks; trace partitions are D = 2 spatial partitions).

Cells are dyadic cubes  prod_j [i_j 2^-k, (i_j + 1) 2^-k]  (equal level k in
every coordinate). Exact coverage check without a dense occupancy grid:

  (a) every cell is a dyadic cube inside [0,1]^D (exact float tests),
  (b) no cell contains another cell (for dyadic cubes, two cells with
      overlapping interiors are nested: checked by looking up every coarser
      ancestor of every cell),
  (c) the exact (Fraction) sum of the volumes is 1.

(a)+(b) give pairwise disjoint interiors; with (c) the union of the closed
cells has full measure in [0,1]^D and is closed, hence equals [0,1]^D. So
sum_K |K| sup_K Phi >= int Phi for every density Phi bounded on each cell.

Encloser interface (certification/enclose/engine2d.py):
  densities(bk, lo, hi)              lo, hi: (D, B) cell corners -> dict of (B,) arrays
  trace_densities(bk, tval, lo, hi)  spatial corners (D-1, B)    -> {"w2", "r0", "r0hat"}
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from certified_obstacle_nn.certification.enclose import upper_sum as US


@dataclass
class BoxPartition:
    lo: np.ndarray  # (D, N)
    hi: np.ndarray  # (D, N)

    @property
    def D(self) -> int:
        return self.lo.shape[0]

    @property
    def n(self) -> int:
        return self.lo.shape[1]

    def width(self):
        return self.hi[0] - self.lo[0]

    def volume(self):
        v = np.ones(self.n)
        for j in range(self.D):
            v = v * (self.hi[j] - self.lo[j])  # exact: powers of two
        return v

    area = volume  # interface of upper_sum.reduce_upper_sums
    length = volume

    def diam(self):
        return np.sqrt(((self.hi - self.lo) ** 2).sum(0))

    def level(self):
        return np.rint(-np.log2(self.width())).astype(np.int64)

    def hash(self) -> str:
        h = hashlib.sha256()
        o = np.lexsort(self.lo[::-1])
        for arr in (self.lo, self.hi):
            h.update(np.ascontiguousarray(arr[:, o]).tobytes())
        return h.hexdigest()

    def verify_coverage(self, max_level: int = 30) -> tuple[bool, str]:
        lo, hi = self.lo, self.hi
        if not (np.all(lo >= 0.0) and np.all(hi <= 1.0)):
            return False, "cell outside the unit cube"
        w = hi[0] - lo[0]
        if not np.all(hi - lo == w[None, :]):
            return False, "cell is not a cube"
        lev = self.level()
        if np.any(lev < 0) or np.any(lev > max_level) or not np.all(np.ldexp(1.0, -lev) == w):
            return False, "cell width is not a power of two"
        idx = np.ldexp(lo, lev[None, :])
        if not np.all(idx == np.floor(idx)):
            return False, "cell corner not on its dyadic lattice"
        idx = idx.astype(np.int64)
        # encode (level, index) -> int key; max_level <= 30 so 20 bits per coordinate suffice
        # for level <= 20; use Python ints via structured unique rows instead (no overflow)
        keys_by_level = {}
        for L in np.unique(lev):
            sel = lev == L
            keys_by_level[int(L)] = _row_keys(idx[:, sel])
        for L in np.unique(lev):
            sel = lev == L
            I = idx[:, sel]
            for Lc in keys_by_level:
                if Lc >= L:
                    continue
                anc = _row_keys(I >> (L - Lc))
                if np.isin(anc, keys_by_level[Lc]).any():
                    return False, f"a level-{L} cell lies inside a level-{Lc} cell"
            if np.unique(keys_by_level[int(L)]).size != keys_by_level[int(L)].size:
                return False, f"duplicate level-{L} cell"
        vol = Fraction(0)
        for L in np.unique(lev):
            vol += Fraction(int((lev == L).sum()), 2 ** (int(L) * self.D))
        if vol != 1:
            return False, f"volume sum {vol}"
        return True, "ok"

    def subset(self, idx):
        return BoxPartition(self.lo[:, idx], self.hi[:, idx])


def _row_keys(I: np.ndarray) -> np.ndarray:
    """Exact injective int64 key of index columns (indices < 2^20 per axis)."""
    if I.size and I.max() >= 2**20:
        raise ValueError("dyadic level too deep for the key encoding")
    k = np.zeros(I.shape[1], dtype=np.int64)
    for j in range(I.shape[0]):
        k = (k << 20) | I[j]
    return k


def uniform(D: int, n: int) -> BoxPartition:
    if n & (n - 1):
        raise ValueError("dyadic partitions need powers of two")
    e = np.arange(n + 1) / n
    grids = np.meshgrid(*([np.arange(n)] * D), indexing="ij")
    I = np.stack([g.ravel() for g in grids])
    return BoxPartition(e[I], e[I + 1])


def refine(part: BoxPartition, mark: np.ndarray) -> BoxPartition:
    """Split marked cubes into their 2^D dyadic children (exact midpoints)."""
    keep = ~mark
    lo, hi = part.lo[:, mark], part.hi[:, mark]
    mid = 0.5 * (lo + hi)
    los, his = [part.lo[:, keep]], [part.hi[:, keep]]
    D = part.D
    for c in range(2**D):
        bits = [(c >> j) & 1 for j in range(D)]
        los.append(np.stack([mid[j] if bits[j] else lo[j] for j in range(D)]))
        his.append(np.stack([hi[j] if bits[j] else mid[j] for j in range(D)]))
    return BoxPartition(np.concatenate(los, 1), np.concatenate(his, 1))


# --------------------------------------------------------------------- evaluation
def _eval_vol(args):
    enc, bkname, prec, lo, hi = args
    return enc.densities(US._backend(bkname, prec), lo, hi)


def _eval_tr(args):
    enc, bkname, prec, tval, lo, hi = args
    return enc.trace_densities(US._backend(bkname, prec), tval, lo, hi)


def _run(fn, jobs, workers):
    from certified_obstacle_nn.certification.enclose import intervals_np

    intervals_np.tables()  # build once (before fork)
    if workers <= 1 or len(jobs) == 1:
        return [fn(j) for j in jobs]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as ex:
        return list(ex.map(fn, jobs, chunksize=1))


def evaluate_volume(enc, part: BoxPartition, backend: str, prec: int = 64, batch: int = 256, workers: int = 1):
    jobs = [(enc, backend, prec, part.lo[:, s:e], part.hi[:, s:e]) for s, e in US._chunks(part.n, batch)]
    res = _run(_eval_vol, jobs, workers)
    return {k: np.concatenate([r[k] for r in res]) for k in US.VOL_KEYS}


def evaluate_trace(enc, tpart: BoxPartition, tval: float, backend: str, prec: int = 64, batch: int = 1024,
                   workers: int = 1):
    jobs = [(enc, backend, prec, tval, tpart.lo[:, s:e], tpart.hi[:, s:e]) for s, e in US._chunks(tpart.n, batch)]
    res = _run(_eval_tr, jobs, workers)
    return {k: np.concatenate([r[k] for r in res]) for k in ("w2", "r0", "r0hat")}


def certify_boxes(enc, part: BoxPartition, tr0: BoxPartition, trT: BoxPartition, *, backend: str, prec: int = 64,
                  workers: int = 1, model_frozen: bool, constants_validated: bool, T: float = 1.0, batch: int = 256):
    t0 = time.time()
    vol = evaluate_volume(enc, part, backend, prec, batch=batch, workers=workers)
    t1 = time.time()
    d0 = evaluate_trace(enc, tr0, 0.0, backend, prec, workers=workers)
    dT = evaluate_trace(enc, trT, T, backend, prec, workers=workers)
    t2 = time.time()
    us = US.reduce_upper_sums(part, vol, tr0, d0, trT, dT, backend_info=US.backend_info(backend, prec),
                              model_frozen=model_frozen, constants_validated=constants_validated)
    t3 = time.time()
    us.timings = {"leaf_eval_s": t1 - t0, "trace_eval_s": t2 - t1, "reduction_s": t3 - t2, "workers": workers,
                  "s_per_cell_wall": (t1 - t0) / max(part.n, 1)}
    return us, vol, (d0, dT)


def adaptive_boxes(enc, *, D: int, budget: int, start: int, theta: float = 0.5, backend: str = "numpy",
                   workers: int = 1, trace_start: int = 64, trace_budget: int = 16384, T: float = 1.0,
                   max_level: int = 12, log=None, weights=(18.0, 36.0, 12.0)):
    """Doerfler marking on the weighted majorant wr U_r + wp U_p + wc U_c
    (diffusion only: 18/36/12), capped so the leaf count never exceeds
    `budget`; traces refined separately (own budget). The returned partitions
    are frozen; every leaf is re-evaluated from scratch by certify_boxes."""
    wr, wp, wc = weights
    part = uniform(D, start)
    hist, stop = [], None
    split = 2**D - 1
    vol = evaluate_volume(enc, part, backend, workers=workers)
    while True:
        contrib = part.volume() * (wr * vol["r"] + wp * vol["p"] + wc * vol["c"])
        lev = part.level()
        hist.append({"n_cells": int(part.n), "estimate": float(contrib.sum()), "max_level": int(lev.max())})
        if log:
            log(f"adaptive: n={part.n} estimate={contrib.sum():.6e} max_level={lev.max()}")
        room = (budget - part.n) // split
        if room <= 0:
            stop = "cell budget reached"
            break
        refinable = lev < max_level
        if not refinable.any():
            stop = "max level reached everywhere"
            break
        order = np.argsort(-np.where(refinable, contrib, -1.0))
        csum = np.cumsum(contrib[order])
        k = int(np.searchsorted(csum, theta * csum[-1]) + 1)
        k = max(1, min(k, room, int(refinable.sum())))
        mark = np.zeros(part.n, dtype=bool)
        mark[order[:k]] = True
        part = refine(part, mark)
        # refine() keeps the unmarked cells first, in order; only the children are new
        n_keep = int((~mark).sum())
        new = evaluate_volume(enc, part.subset(np.arange(n_keep, part.n)), backend, workers=workers)
        vol = {key: np.concatenate([vol[key][~mark], new[key]]) for key in vol}
    tparts = adaptive_traces(enc, D=D, theta=theta, backend=backend, workers=workers, trace_start=trace_start,
                             trace_budget=trace_budget, T=T, weights=weights)
    lev = part.level()
    record = {"stopping_reason": stop, "history": hist,
              "level_histogram": {int(l): int((lev == l).sum()) for l in np.unique(lev)},
              "max_diam": float(part.diam().max()), "theta": theta, "budget": budget, "weights": list(weights),
              "trace_budget": trace_budget, "n_trace0": tparts[0.0].n, "n_traceT": tparts[T].n}
    return part, tparts[0.0], tparts[T], record


def adaptive_traces(enc, *, D: int, theta: float = 0.5, backend: str = "numpy", workers: int = 1,
                    trace_start: int = 64, trace_budget: int = 16384, T: float = 1.0, weights=(18.0, 36.0, 12.0)):
    """The trace part of adaptive_boxes: separate (D-1)-dimensional refinement of the
    t = 0 and t = T trace partitions (independent of the volume partition, so a
    stored run's trace partitions can be reconstructed from the same inputs and
    checked against its trace0_hash / traceT_hash)."""
    wr, wp, _ = weights
    tparts = {}
    tsplit = 2 ** (D - 1) - 1
    for tval in (0.0, T):
        tp = uniform(D - 1, trace_start)
        while True:
            tr = evaluate_trace(enc, tp, tval, backend, workers=workers)
            c = tp.volume() * ((wr * tr["r0"] + wp * tr["w2"]) if tval == 0.0 else wp * tr["w2"])
            room = (trace_budget - tp.n) // tsplit
            if room <= 0 or c.sum() == 0.0 or not (tp.level() < 20).any():
                break
            order = np.argsort(-c)
            csum = np.cumsum(c[order])
            k = int(np.searchsorted(csum, theta * csum[-1]) + 1)
            k = max(1, min(k, room))
            mark = np.zeros(tp.n, dtype=bool)
            mark[order[:k]] = True
            tp = refine(tp, mark)
        tparts[tval] = tp
    return tparts
