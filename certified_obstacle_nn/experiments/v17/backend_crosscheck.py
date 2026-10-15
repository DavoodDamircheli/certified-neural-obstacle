"""v17b Task 2: reference-backend (arb, 192 bits) cross-check of a protocol-v1 run on cell subsets.

    python3 -m certified_obstacle_nn.experiments.v17.backend_crosscheck --run results/v17_split/20261010T005538Z \
        --random 10000 --worst 1000 --traces 1000 --seed 20261011 --workers 4

Cell sets (indices into the frozen protocol partition, recorded in the output):
  random     a fixed random subset (numpy default_rng(seed), without replacement);
  worst_R    the cells with the largest TM1 R-range width (R_width of the stored chunk files);
  worst_gam  the cells with the largest TM1 gamma-range width (recomputed here, value channels only, numpy).
Traces: a fixed random subset of each frozen trace partition (t = 0 and t = T).

On every cell both backends re-evaluate the per-term densities (numpy outward float64; arb 192-bit balls), and arb
also evaluates every term POINTWISE at the cell centre (degenerate box, 192 bits). Per term
(R0 = sup R^2, Rh = sup R_hat^2, S = sup |S|^2, Sh = sup |S_hat|^2, c = sup lambda (v - g)_+; traces w2, r0, r0hat):
  * rel_diff = |numpy_upper - arb_upper| / arb_upper (both are validated upper bounds of the same supremum);
  * gate: numpy_upper >= arb_lower, where arb_lower is the lower endpoint of the 192-bit ball of the term's value at
    the cell centre (a validated lower bound of the cell supremum, DECISIONS D25). A cell with numpy_upper < arb_lower
    would mean the numpy enclosure misses a validated value: gate FAILED.
The numpy re-evaluation is also compared with the stored chunk densities of the run (must be bit-identical).
Writes results/v17_backend_crosscheck/<ts>/{crosscheck.json, cells.npz}.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from certified_obstacle_nn.certification.enclose import boxes as Bx
from certified_obstacle_nn.certification.enclose import tm as T
from certified_obstacle_nn.certification.enclose import upper_sum as US
from certified_obstacle_nn.certification.enclose.engine2d import enclose_mlp_vec
from certified_obstacle_nn.experiments import results_io
from certified_obstacle_nn.experiments.v14.common import REPO_ROOT
from certified_obstacle_nn.experiments.v17 import chunked_certify as CC

VOL_TERMS = {"R0": "R0_term", "Rh": "Rh_term", "S": "S_term", "Sh": "Sh_term", "c": "c"}
TR_TERMS = ("w2", "r0", "r0hat")


def _cat(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _dn(x):
    return np.nextafter(x, -np.inf)


def _sq_dn(lo, hi):
    """Lower bound of z^2 over z in [lo, hi] (float endpoints), rounded down."""
    d = np.maximum(np.maximum(lo, -hi), 0.0)
    return np.maximum(_dn(d * d), 0.0)


def _gam_width_chunk(args):
    enc, lo, hi = args
    from certified_obstacle_nn.certification.enclose import intervals_np as bk

    c, r = 0.5 * (lo + hi), 0.5 * (hi - lo)
    t = T.tm_variable(bk, c[0], r[0], 1, 3)
    x = T.tm_variable(bk, c[1], r[1], 2, 3)
    y = T.tm_variable(bk, c[2], r[2], 3, 3)
    N, _ = enclose_mlp_vec(enc.Ws, [t, x, y], ())
    Bm, _, _, _ = enc.ad.mask(x, y)
    g = (enc.ad.data(t, x, y)["g"] - T.tm_mul(Bm, N[0])).range()
    return {"gam_width": g.upper() - g.lower()}


def gamma_width(enc, part, workers, batch=4096):
    jobs = [(enc, part.lo[:, s:e], part.hi[:, s:e]) for s, e in US._chunks(part.n, batch)]
    return _cat(Bx._run(_gam_width_chunk, jobs, workers))["gam_width"]


def _point_lower_chunk(args):
    """Validated (arb) lower bounds of each volume term's value at the cell centres."""
    enc, prec, c = args
    bk = US._backend("arb", prec)
    F = enc.fields(bk, c, c)
    rg = {k: F[k].range() for k in ("R", "S1", "S2", "gam", "lam", "Rhat_pos", "Shat1_pos", "Shat2_pos")}
    L = {k: _sq_dn(v.lower(), v.upper()) for k, v in rg.items() if k not in ("gam", "lam")}
    g_lo, g_hi = rg["gam"].lower(), rg["gam"].upper()
    S = _dn(L["S1"] + L["S2"])
    Sp = _dn(L["Shat1_pos"] + L["Shat2_pos"])
    neg, pos = g_hi < 0, g_lo > 0
    return {"R0": L["R"], "S": S,
            "Rh": np.where(neg, L["R"], np.where(pos, L["Rhat_pos"], np.minimum(L["R"], L["Rhat_pos"]))),
            "Sh": np.where(neg, S, np.where(pos, Sp, np.minimum(S, Sp))),
            "c": np.maximum(_dn(np.maximum(rg["lam"].lower(), 0.0) * np.maximum(-g_hi, 0.0)), 0.0)}


def _trace_point_lower_chunk(args):
    enc, prec, tval, c = args
    bk = US._backend("arb", prec)
    z = np.zeros_like(c[0])
    tt = T.tm_variable(bk, np.full_like(c[0], tval), z, 1, 3)
    x = T.tm_variable(bk, c[0], z, 2, 3)
    y = T.tm_variable(bk, c[1], z, 3, 3)
    N, _ = enclose_mlp_vec(enc.Ws, [tt, x, y], ())
    Bm, _, _, _ = enc.ad.mask(x, y)
    v = T.tm_mul(Bm, N[0])
    d = enc.ad.data_trace(tt, x, y)
    gam, vm, gm = (d["g"] - v).range(), (v - d["u0"]).range(), (d["g"] - d["u0"]).range()
    return {"w2": _sq_dn(np.maximum(gam.lower(), 0.0), np.maximum(gam.upper(), 0.0)),
            "r0": _sq_dn(vm.lower(), vm.upper()),
            "r0hat": _sq_dn(np.maximum(vm.lower(), gm.lower()), np.maximum(vm.upper(), gm.upper()))}


def _cmp(nu, au, al):
    rd = np.abs(nu - au) / np.where(au > 0, au, 1.0)
    rd = np.where((au == 0) & (nu == 0), 0.0, rd)
    viol = nu < al
    return {"max_rel_diff": float(rd.max()), "median_rel_diff": float(np.median(rd)),
            "frac_numpy_below_arb_upper": float(np.mean(nu < au)), "n_cells": int(nu.size),
            "n_violations_numpy_upper_below_arb_lower": int(viol.sum()),
            "min_margin_numpy_upper_minus_arb_lower": float((nu - al).min()),
            "max_arb_lower_over_numpy_upper": float(np.max(np.where(nu > 0, al / np.where(nu > 0, nu, 1), 0.0)))}


def run(run_dir, n_random, n_worst, n_traces, seed, workers, prec=192, log=print):
    cfg = json.loads((REPO_ROOT / run_dir / "config.json").read_text())
    job = REPO_ROOT / cfg["partition"]["chunked_job"]
    meta, r = CC._load(job, workers)
    enc = r.encloser
    nch = len(meta["edges"]) - 1
    zs = [np.load(job / f"chunk_{i:03d}.npz") for i in range(nch)]
    stored = {k: np.concatenate([z[k] for z in zs]) for k in list(VOL_TERMS.values()) + ["R_width"]}
    rng = np.random.default_rng(seed)
    sets = {"random": np.sort(rng.choice(r.part.n, size=n_random, replace=False))}
    out = {"run": run_dir, "chunked_job": cfg["partition"]["chunked_job"], "partition_hash": meta["partition_hash"],
           "checkpoint_sha256": meta["checkpoint_sha256"], "n_cells_partition": int(r.part.n), "seed": seed,
           "arb_prec_bits": prec, "numpy_backend": "numpy_outward_float64", "sets": {}, "timing_s": {}}
    if n_worst:
        sets["worst_R"] = np.sort(np.argsort(-stored["R_width"], kind="stable")[:n_worst])
        tic = time.time()
        gw = gamma_width(enc, r.part, workers)
        out["timing_s"]["gamma_width_all_cells"] = time.time() - tic
        sets["worst_gam"] = np.sort(np.argsort(-gw, kind="stable")[:n_worst])
        log(f"gamma widths of {r.part.n} cells in {out['timing_s']['gamma_width_all_cells']:.0f}s")
    cells = {}
    for name, idx in sets.items():
        sub = r.part.subset(idx)
        tic = time.time()
        vn = Bx.evaluate_volume(enc, sub, "numpy", workers=workers)
        tn = time.time() - tic
        tic = time.time()
        va = Bx.evaluate_volume(enc, sub, "arb", prec, batch=16, workers=workers)
        ta = time.time() - tic
        cen = 0.5 * (sub.lo + sub.hi)
        tic = time.time()
        pl = _cat(Bx._run(_point_lower_chunk, [(enc, prec, cen[:, s:e]) for s, e in US._chunks(sub.n, 16)], workers))
        tp = time.time() - tic
        rec = {"n": int(idx.size), "numpy_s": tn, "arb_s": ta, "arb_point_s": tp,
               "numpy_reeval_equals_stored": all(bool(np.array_equal(vn[k], stored[k][idx]))
                                                 for k in VOL_TERMS.values()),
               "signs_agree": bool(np.array_equal(vn["sign"], va["sign"])), "terms": {}}
        for t, k in VOL_TERMS.items():
            rec["terms"][t] = _cmp(vn[k], va[k], pl[t])
            cells[f"{name}_{t}_numpy"], cells[f"{name}_{t}_arb_up"], cells[f"{name}_{t}_arb_lo"] = vn[k], va[k], pl[t]
        cells[f"{name}_idx"] = idx
        out["sets"][name] = rec
        log(f"{name}: {idx.size} cells numpy {tn:.0f}s arb {ta:.0f}s points {tp:.0f}s; violations "
            f"{sum(v['n_violations_numpy_upper_below_arb_lower'] for v in rec['terms'].values())}")
    for tval, tp_, nm in ((0.0, r.tp0, "trace0"), (1.0, r.tpT, "traceT")):
        idx = np.sort(rng.choice(tp_.n, size=min(n_traces, tp_.n), replace=False))
        sub = tp_.subset(idx)
        tn_ = Bx.evaluate_trace(enc, sub, tval, "numpy", workers=workers)
        ta_ = Bx.evaluate_trace(enc, sub, tval, "arb", prec, batch=64, workers=workers)
        cen = 0.5 * (sub.lo + sub.hi)
        pl = _cat(Bx._run(_trace_point_lower_chunk, [(enc, prec, tval, cen[:, s:e]) for s, e in US._chunks(sub.n, 64)],
                          workers))
        out["sets"][nm] = {"n": int(idx.size), "terms": {k: _cmp(tn_[k], ta_[k], pl[k]) for k in TR_TERMS}}
        for k in TR_TERMS:
            cells[f"{nm}_{k}_numpy"], cells[f"{nm}_{k}_arb_up"], cells[f"{nm}_{k}_arb_lo"] = tn_[k], ta_[k], pl[k]
        cells[f"{nm}_idx"] = idx
    nviol = sum(v["n_violations_numpy_upper_below_arb_lower"] for s in out["sets"].values()
                for v in s["terms"].values())
    out["gate"] = "PASS" if nviol == 0 else "FAILED"
    out["n_violations_total"] = nviol
    d = results_io.make_run_dir("v17_backend_crosscheck")
    np.savez_compressed(d / "cells.npz", **cells)
    (d / "crosscheck.json").write_text(json.dumps(out, indent=1))
    log(f"gate {out['gate']} ({nviol} violations) -> {d.relative_to(REPO_ROOT)}")
    return d, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--random", type=int, default=10000)
    ap.add_argument("--worst", type=int, default=1000)
    ap.add_argument("--traces", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--prec", type=int, default=192)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    run(a.run, a.random, a.worst, a.traces, a.seed, a.workers, a.prec,
        log=lambda m: print(time.strftime("%H:%M:%S"), m, flush=True))


if __name__ == "__main__":
    main()
