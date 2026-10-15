"""Validated upper sums U_r, U_p, U_c, U_r_hat on a frozen dyadic partition.

    U_r     = sum_K |K| r_K    + sum_{S in T0} |S| r0_S
    U_p     = sum_K |K| p_K    + sum_{S in T0} |S| w2_S(0) + sum_{S in TT} |S| w2_S(T)
    U_c     = sum_K |K| c_K
    U_r_hat = sum_K |K| rhat_K + sum_{S in T0} |S| r0hat_S

(per-cell densities from certification/enclose/engine.py). Correctness of the
upper sum: the cells are closed boxes with disjoint interiors covering Q
(resp. Omega for the traces), boundaries are null sets, and each density is
bounded above on its cell, so int_Q Phi = sum_K int_K Phi <= sum_K |K| sup_K Phi.
Traces are SPATIAL integrals on their own 1-D partitions (never multiplied by a
space-time volume).

Global reduction is done in arb (exact conversion of every float cell bound and
every dyadic cell measure, ball sums and products, upper endpoint), so the
reduction never rounds a bound down.

Status logic (``status`` field): ``validated_upper_bound`` only if
  * the backend is a validated backend (arb or numpy_outward_float64),
  * every volume cell and both trace partitions were evaluated and their
    coverage of Q / Omega was verified exactly (dyadic occupancy + exact area),
  * no per-cell value is NaN or infinite,
  * every constant used is exact (adapter-checked) and flagged validated,
  * the model parameters are frozen.
Any failure downgrades to ``unvalidated_enclosure`` (or ``diagnostic_float64``
for NaN) and records the reason; it never silently passes.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from flint import arb, ctx

VALIDATED_BACKENDS = ("arb", "numpy_outward_float64")


# --------------------------------------------------------------------- partition
@dataclass
class Partition2D:
    t0: np.ndarray
    t1: np.ndarray
    x0: np.ndarray
    x1: np.ndarray
    T: float = 1.0
    a: float = 0.0
    b: float = 1.0

    @property
    def n(self):
        return self.t0.size

    def area(self):
        return (self.t1 - self.t0) * (self.x1 - self.x0)  # exact for dyadic cells

    def diam(self):
        return np.hypot(self.t1 - self.t0, self.x1 - self.x0)

    def hash(self) -> str:
        h = hashlib.sha256()
        o = np.lexsort((self.x0, self.t0))
        for arr in (self.t0, self.t1, self.x0, self.x1):
            h.update(np.ascontiguousarray(arr[o]).tobytes())
        return h.hexdigest()

    def subset(self, idx):
        return Partition2D(self.t0[idx], self.t1[idx], self.x0[idx], self.x1[idx], self.T, self.a, self.b)


@dataclass
class Partition1D:
    x0: np.ndarray
    x1: np.ndarray
    a: float = 0.0
    b: float = 1.0

    @property
    def n(self):
        return self.x0.size

    def length(self):
        return self.x1 - self.x0

    def hash(self) -> str:
        h = hashlib.sha256()
        o = np.argsort(self.x0)
        for arr in (self.x0, self.x1):
            h.update(np.ascontiguousarray(arr[o]).tobytes())
        return h.hexdigest()


def uniform_2d(nt: int, nx: int, T=1.0, a=0.0, b=1.0) -> Partition2D:
    for n in (nt, nx):
        if n & (n - 1):
            raise ValueError("dyadic partitions need powers of two")
    te = T * (np.arange(nt + 1) / nt)
    xe = a + (b - a) * (np.arange(nx + 1) / nx)
    T0, X0 = np.meshgrid(te[:-1], xe[:-1], indexing="ij")
    T1, X1 = np.meshgrid(te[1:], xe[1:], indexing="ij")
    return Partition2D(T0.ravel(), T1.ravel(), X0.ravel(), X1.ravel(), T, a, b)


def uniform_1d(n: int, a=0.0, b=1.0) -> Partition1D:
    if n & (n - 1):
        raise ValueError("dyadic partitions need powers of two")
    e = a + (b - a) * (np.arange(n + 1) / n)
    return Partition1D(e[:-1].copy(), e[1:].copy(), a, b)


def refine_2d(part: Partition2D, mark: np.ndarray) -> Partition2D:
    """Quad-split marked cells into four exact halves (dyadic midpoints)."""
    keep = ~mark
    t0, t1, x0, x1 = part.t0[mark], part.t1[mark], part.x0[mark], part.x1[mark]
    tm = 0.5 * (t0 + t1)
    xm = 0.5 * (x0 + x1)
    nt0 = np.concatenate([part.t0[keep], t0, t0, tm, tm])
    nt1 = np.concatenate([part.t1[keep], tm, tm, t1, t1])
    nx0 = np.concatenate([part.x0[keep], x0, xm, x0, xm])
    nx1 = np.concatenate([part.x1[keep], xm, x1, xm, x1])
    return Partition2D(nt0, nt1, nx0, nx1, part.T, part.a, part.b)


def refine_1d(part: Partition1D, mark: np.ndarray) -> Partition1D:
    keep = ~mark
    x0, x1 = part.x0[mark], part.x1[mark]
    xm = 0.5 * (x0 + x1)
    return Partition1D(np.concatenate([part.x0[keep], x0, xm]), np.concatenate([part.x1[keep], xm, x1]), part.a, part.b)


def verify_coverage_2d(part: Partition2D, max_level: int = 12) -> tuple[bool, str]:
    """Exact check that the cells tile Q: integer occupancy at the finest dyadic
    level (each fine square covered exactly once) and exact area sum."""
    if part.T != 1.0 or (part.a, part.b) != (0.0, 1.0):
        return False, "coverage check implemented for Q = (0,1)^2"
    level = 0
    for arr in (part.t0, part.t1, part.x0, part.x1):
        for v in np.unique(arr):
            fr = Fraction(float(v))
            if fr.denominator & (fr.denominator - 1):
                return False, "non-dyadic endpoint"
            level = max(level, fr.denominator.bit_length() - 1)
    if level > max_level:
        return False, f"finest level {level} > {max_level} (occupancy check too large)"
    N = 2**level
    i0 = np.rint(part.t0 * N).astype(np.int64)
    i1 = np.rint(part.t1 * N).astype(np.int64)
    j0 = np.rint(part.x0 * N).astype(np.int64)
    j1 = np.rint(part.x1 * N).astype(np.int64)
    if not (np.all(i0 * 1.0 == part.t0 * N) and np.all(j1 * 1.0 == part.x1 * N)):
        return False, "endpoint not on the dyadic lattice"
    D = np.zeros((N + 1, N + 1), dtype=np.int64)
    np.add.at(D, (i0, j0), 1)
    np.add.at(D, (i1, j0), -1)
    np.add.at(D, (i0, j1), -1)
    np.add.at(D, (i1, j1), 1)
    occ = D.cumsum(0).cumsum(1)[:N, :N]
    if not np.all(occ == 1):
        return False, f"occupancy min {occ.min()} max {occ.max()}"
    area = sum((Fraction(float(a)) for a in part.area()), Fraction(0))
    if area != 1:
        return False, f"area sum {area}"
    return True, "ok"


def verify_coverage_1d(part: Partition1D) -> tuple[bool, str]:
    o = np.argsort(part.x0)
    x0, x1 = part.x0[o], part.x1[o]
    if x0[0] != part.a or x1[-1] != part.b or np.any(x1[:-1] != x0[1:]) or np.any(x1 <= x0):
        return False, "trace cells do not tile Omega"
    return True, "ok"


# --------------------------------------------------------------------- evaluation
VOL_KEYS = ("r", "p", "c", "rhat", "R_term", "S_term", "gradw", "w_term", "wt_term", "sign", "R_width",
            "R0_term", "Rh_term", "Sh_term", "w2_term", "wt2_term")  # last five: v17 unweighted per-term split


def _eval_vol_chunk(args):
    encloser, bkname, prec, t0, t1, x0, x1 = args
    bk = _backend(bkname, prec)
    return encloser.densities(bk, t0, t1, x0, x1)


def _eval_tr_chunk(args):
    encloser, bkname, prec, tval, x0, x1 = args
    bk = _backend(bkname, prec)
    return encloser.trace_densities(bk, tval, x0, x1)


def _backend(name, prec):
    if name == "arb":
        from certified_obstacle_nn.certification.enclose import intervals_arb as B

        B.set_precision(prec)
        return B
    from certified_obstacle_nn.certification.enclose import intervals_np as B

    return B


def _chunks(n, size):
    return [(s, min(s + size, n)) for s in range(0, n, size)]


def evaluate_volume(encloser, part: Partition2D, backend: str, prec: int = 64, batch: int = 512, workers: int = 1):
    jobs = [(encloser, backend, prec, part.t0[s:e], part.t1[s:e], part.x0[s:e], part.x1[s:e]) for s, e in _chunks(part.n, batch)]
    res = _run(_eval_vol_chunk, jobs, workers)
    keys = VOL_KEYS + tuple(k for k in res[0] if k not in VOL_KEYS)  # optional diagnostics (e.g. R_width_mv)
    return {k: np.concatenate([r[k] for r in res]) for k in keys}


def evaluate_trace(encloser, tpart: Partition1D, tval: float, backend: str, prec: int = 64, batch: int = 2048, workers: int = 1):
    jobs = [(encloser, backend, prec, tval, tpart.x0[s:e], tpart.x1[s:e]) for s, e in _chunks(tpart.n, batch)]
    res = _run(_eval_tr_chunk, jobs, workers)
    return {k: np.concatenate([r[k] for r in res]) for k in ("w2", "r0", "r0hat")}


def _run(fn, jobs, workers):
    if workers <= 1 or len(jobs) == 1:
        if fn is _eval_vol_chunk or fn is _eval_tr_chunk:
            from certified_obstacle_nn.certification.enclose import intervals_np

            intervals_np.tables()
        return [fn(j) for j in jobs]
    import multiprocessing as mp

    from certified_obstacle_nn.certification.enclose import intervals_np

    intervals_np.tables()  # build once before fork
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as ex:
        return list(ex.map(fn, jobs))


# --------------------------------------------------------------------- reduction
def _arb_dot_up(weights: np.ndarray, vals: np.ndarray) -> arb:
    old = ctx.prec
    ctx.prec = 128
    try:
        s = arb(0)
        for w, v in zip(weights.tolist(), vals.tolist()):
            if v != 0.0:
                s += arb(w) * arb(v)
        return s
    finally:
        ctx.prec = old


def _up(x: arb) -> float:
    from certified_obstacle_nn.certification.reliability import up

    return up(x)


@dataclass
class UpperSums:
    U_r: float
    U_p: float
    U_c: float
    U_r_hat: float
    parts: dict
    status: str
    status_reasons: list
    backend: dict
    partition_hash: str
    trace0_hash: str
    traceT_hash: str
    n_cells: int
    n_trace0: int
    n_traceT: int
    max_diam: float
    timings: dict = field(default_factory=dict)


def reduce_upper_sums(part, vol, tr0_part, tr0, trT_part, trT, *, backend_info, model_frozen: bool,
                      constants_validated: bool, timings=None) -> UpperSums:
    reasons = []
    area = part.area()
    L0 = tr0_part.length()
    LT = trT_part.length()
    for name, arrs in (("volume", [vol[k] for k in ("r", "p", "c", "rhat") + VOL_KEYS[-5:] if k in vol]),
                       ("trace0", [tr0[k] for k in ("w2", "r0", "r0hat")]),
                       ("traceT", [trT["w2"]])):
        for a_ in arrs:
            if not np.all(np.isfinite(a_)):
                reasons.append(f"non-finite value in {name}")
                break
    ok, msg = part.verify_coverage() if hasattr(part, "verify_coverage") else verify_coverage_2d(part)
    if not ok:
        reasons.append("volume coverage: " + msg)
    for nm, tp in (("trace0", tr0_part), ("traceT", trT_part)):
        ok, msg = tp.verify_coverage() if hasattr(tp, "verify_coverage") else verify_coverage_1d(tp)
        if not ok:
            reasons.append(f"{nm} coverage: {msg}")
    if backend_info.get("backend") not in VALIDATED_BACKENDS:
        reasons.append(f"backend {backend_info.get('backend')} not validated")
    if not model_frozen:
        reasons.append("model parameters not frozen")
    if not constants_validated:
        reasons.append("problem constants not validated")

    parts = {
        "vol_r": _arb_dot_up(area, vol["r"]),
        "vol_p": _arb_dot_up(area, vol["p"]),
        "vol_c": _arb_dot_up(area, vol["c"]),
        "vol_rhat": _arb_dot_up(area, vol["rhat"]),
        "vol_R_term": _arb_dot_up(area, vol["R_term"]),
        "vol_S_term": _arb_dot_up(area, vol["S_term"]),
        "vol_gradw": _arb_dot_up(area, vol["gradw"]),
        "vol_w_term": _arb_dot_up(area, vol["w_term"]),
        "vol_wt_term": _arb_dot_up(area, vol["wt_term"]),
        # v17 (roadmap R3 Stage 1 item 2): unweighted per-term sums
        "vol_R0": _arb_dot_up(area, vol["R0_term"]),
        "vol_Rhat": _arb_dot_up(area, vol["Rh_term"]),
        "vol_Shat": _arb_dot_up(area, vol["Sh_term"]),
        "vol_w2": _arb_dot_up(area, vol["w2_term"]),
        "vol_wt2": _arb_dot_up(area, vol["wt2_term"]),
        "init_mismatch": _arb_dot_up(L0, tr0["r0"]),
        "init_mismatch_hat": _arb_dot_up(L0, tr0["r0hat"]),
        "w0": _arb_dot_up(L0, tr0["w2"]),
        "wT": _arb_dot_up(LT, trT["w2"]),
    }
    Ur = parts["vol_r"] + parts["init_mismatch"]
    Up = parts["vol_p"] + parts["w0"] + parts["wT"]
    Uc = parts["vol_c"]
    Urh = parts["vol_rhat"] + parts["init_mismatch_hat"]
    if any(np.isnan(v) for v in (vol["r"].sum(), vol["p"].sum())):
        status = "diagnostic_float64"
    elif reasons:
        status = "unvalidated_enclosure"
    else:
        status = "validated_upper_bound"
    return UpperSums(
        U_r=_up(Ur), U_p=_up(Up), U_c=_up(Uc), U_r_hat=_up(Urh),
        parts={k: _up(v) for k, v in parts.items()},
        status=status, status_reasons=reasons, backend=backend_info,
        partition_hash=part.hash(), trace0_hash=tr0_part.hash(), traceT_hash=trT_part.hash(),
        n_cells=part.n, n_trace0=tr0_part.n, n_traceT=trT_part.n,
        max_diam=float(part.diam().max()), timings=timings or {},
    )


LOWER_KEYS = ("tau2_lo", "gvh2_lo")


def reduce_lower_sums(part, vol, *, backend_info, model_frozen: bool) -> dict:
    """v20 Task 2: validated LOWER sums L_tau = sum_K |K| tau2_lo_K <= ||tau||^2 and
    L_gradvhat = sum_K |K| gvh2_lo_K <= ||d_x v_hat||^2 (per-cell lower endpoints from engine.lower_terms, rounded
    downward). Exact dyadic volumes, arb ball sum (same exact products as the upper sums), LOWER endpoint of the
    ball rounded downward. Status validated_lower_bound only under the conditions of the upper sums (validated
    backend, exact coverage, finite non-negative values, frozen model); else unvalidated_enclosure /
    diagnostic_float64. Correctness: int_Q Phi = sum_K int_K Phi >= sum_K |K| inf_K Phi."""
    from certified_obstacle_nn.certification.reliability import down

    reasons = []
    for k in LOWER_KEYS:
        if k not in vol:
            raise KeyError(f"{k} missing: evaluate with TripleEncloser(..., lower_sums=True)")
        if not np.all(np.isfinite(vol[k])):
            reasons.append(f"non-finite value in {k}")
        elif np.any(vol[k] < 0):
            reasons.append(f"negative value in {k}")
    ok, msg = part.verify_coverage() if hasattr(part, "verify_coverage") else verify_coverage_2d(part)
    if not ok:
        reasons.append("volume coverage: " + msg)
    if backend_info.get("backend") not in VALIDATED_BACKENDS:
        reasons.append(f"backend {backend_info.get('backend')} not validated")
    if not model_frozen:
        reasons.append("model parameters not frozen")
    area = part.area()
    out = {"L_tau": down(_arb_dot_up(area, vol["tau2_lo"])), "L_gradvhat": down(_arb_dot_up(area, vol["gvh2_lo"]))}
    if any(np.isnan(vol[k]).any() for k in LOWER_KEYS):
        status = "diagnostic_float64"
    elif reasons:
        status = "unvalidated_enclosure"
    else:
        status = "validated_lower_bound"
    out.update({"lower_status": status, "lower_status_reasons": ";".join(reasons)})
    return out


def backend_info(name: str, prec: int) -> dict:
    if name == "arb":
        import flint

        return {"backend": "arb", "library": "python-flint", "version": flint.__version__, "prec_bits": prec}
    from certified_obstacle_nn.certification.enclose import intervals_np as B

    return {"backend": "numpy_outward_float64", "numpy": np.__version__,
            "tanh_table": f"arb-built, step 2^-{B.GRID_EXP}, |z|<={B.SAT}, 3rd-order Taylor + saturation"}


def certify_partition(encloser, part, tr0_part, trT_part, *, backend: str, prec: int = 64, workers: int = 1,
                      model_frozen: bool, constants_validated: bool, T: float = 1.0, batch: int = 512):
    t0 = time.time()
    vol = evaluate_volume(encloser, part, backend, prec, batch=batch, workers=workers)
    t1 = time.time()
    tr0 = evaluate_trace(encloser, tr0_part, 0.0, backend, prec, workers=workers)
    trT = evaluate_trace(encloser, trT_part, T, backend, prec, workers=workers)
    t2 = time.time()
    us = reduce_upper_sums(part, vol, tr0_part, tr0, trT_part, trT, backend_info=backend_info(backend, prec),
                           model_frozen=model_frozen, constants_validated=constants_validated)
    t3 = time.time()
    us.timings = {"leaf_eval_s": t1 - t0, "trace_eval_s": t2 - t1, "reduction_s": t3 - t2, "workers": workers,
                  "s_per_cell_wall": (t1 - t0) / max(part.n, 1)}
    return us, vol, (tr0, trT)


# --------------------------------------------------------------------- adaptive
WEIGHTS_HEAT = {"r": 18.0, "p": 36.0, "c": 12.0}


def adaptive_partition(encloser, *, budget: int, start: int = 32, target: str = "weighted", theta: float = 0.5,
                       backend: str = "numpy", workers: int = 1, trace_budget: int = 4096, T: float = 1.0,
                       max_level: int = 12, log=None):
    """Bisect (quad-split) the cells with the largest contribution to the
    weighted majorant 18 U_r + 36 U_p + 12 U_c (target='weighted') or to
    U_r_hat + U_c (target='projected') -- Doerfler marking with fraction theta,
    capped so the leaf count never exceeds `budget`. Traces are refined
    separately (1-D, same rule, own budget). Returns the frozen partitions and
    a refinement record (stopping reasons, histogram, max diameter)."""
    part = uniform_2d(start, start, T)
    hist = []
    stop = None
    while True:
        vol = evaluate_volume(encloser, part, backend, workers=workers)
        area = part.area()
        if target == "weighted":
            contrib = area * (18 * vol["r"] + 36 * vol["p"] + 12 * vol["c"])
        else:
            contrib = area * (vol["rhat"] + vol["c"])
        level = np.rint(-np.log2(part.t1 - part.t0)).astype(int)
        hist.append({"n_cells": int(part.n), "estimate": float(contrib.sum()), "max_level": int(level.max())})
        if log:
            log(f"adaptive: n={part.n} estimate={contrib.sum():.6e} max_level={level.max()}")
        room = (budget - part.n) // 3
        if room <= 0:
            stop = "cell budget reached"
            break
        refinable = level < max_level
        if not refinable.any():
            stop = "max level reached everywhere"
            break
        order = np.argsort(-np.where(refinable, contrib, -1.0))
        csum = np.cumsum(contrib[order])
        k = int(np.searchsorted(csum, theta * csum[-1]) + 1)
        k = max(1, min(k, room, int(refinable.sum())))
        mark = np.zeros(part.n, dtype=bool)
        mark[order[:k]] = True
        part = refine_2d(part, mark)
    # traces: separate 1-D refinement
    tparts = {}
    for tval in (0.0, T):
        tp = uniform_1d(256)
        while True:
            tr = evaluate_trace(encloser, tp, tval, backend, workers=workers)
            L = tp.length()
            c = L * (18 * tr["r0"] + 36 * tr["w2"]) if tval == 0.0 else L * 36 * tr["w2"]
            room = trace_budget - tp.n
            lev = np.rint(-np.log2(tp.length())).astype(int)
            if room <= 0 or not (lev < 20).any():
                break
            order = np.argsort(-c)
            csum = np.cumsum(c[order])
            k = int(np.searchsorted(csum, theta * csum[-1]) + 1)
            k = max(1, min(k, room))
            mark = np.zeros(tp.n, dtype=bool)
            mark[order[:k]] = True
            tp = refine_1d(tp, mark)
        tparts[tval] = tp
    level = np.rint(-np.log2(part.t1 - part.t0)).astype(int)
    lev_hist = {int(l): int((level == l).sum()) for l in np.unique(level)}
    record = {"stopping_reason": stop, "history": hist, "level_histogram": lev_hist,
              "max_diam": float(part.diam().max()), "target": target, "theta": theta, "budget": budget,
              "trace_budget": trace_budget, "n_trace0": tparts[0.0].n, "n_traceT": tparts[T].n}
    return part, tparts[0.0], tparts[T], record
