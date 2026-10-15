"""Adaptive branch-and-bound for the upper-sum certificate (TODO sections 15-18, prompt section 5).

State: a list of space-time boxes (structure of arrays lo, hi [N, m]) covering Q exactly, and a
separate list of spatial boxes covering Omega for the t = 0 / t = T terms. Every box carries
rigorous per-box bounds Phibar / Phiund (theory notes section 3); children are intersected with
their parent's bounds (theory notes section 6), so U is nonincreasing and L nondecreasing.

Each sweep:  priority S_K = |K| (Phibar_K - Phiund_K)   (spatial boxes: |S| (sum of their gaps));
split the top `frac` of all boxes (volume and spatial pooled) by bisection of the longest side;
recompute the children's bounds in one vectorized call.
Stop: U - L <= tau L, or cells >= max_cells, or wall time >= budget, or nothing splittable.
Validity of the final U never depends on these choices (Lemma 1).
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from certified_obstacle_nn.certification.level2.grid import cell_boxes, n_cells
from certified_obstacle_nn.certification.level2.upper_sum import (FINAL_INFLATION, boundary_bounds, enclose_cells,
                                                                   midpoint_boundary, midpoint_phi, phi_bounds)

TERMS = ("R", "sig", "kw", "c", "int")


def _vol_bounds(prob, lo, hi, method, batch):
    """Per-box bounds (+ per-term upper bounds and exact midpoint terms)."""
    out = {"up": [], "dn": [], "st": [], "mid": [], **{f"u_{k}": [] for k in TERMS}, **{f"m_{k}": [] for k in TERMS}}
    for i in range(0, lo.shape[0], batch):
        l, h = lo[i:i + batch], hi[i:i + batch]
        pb = phi_bounds(prob, enclose_cells(prob, l, h, method))
        pm, pmt = midpoint_phi(prob, 0.5 * (l + h))
        out["up"].append(pb["Phi_up"])
        out["dn"].append(pb["Phi_dn"])
        out["st"].append(pb["status"])
        out["mid"].append(pm)
        for k in TERMS:
            out[f"u_{k}"].append(pb["up"][k] if torch.is_tensor(pb["up"][k]) else torch.full_like(pm, pb["up"][k]))
            out[f"m_{k}"].append(pmt[k])
    return {k: torch.cat(v) for k, v in out.items()}


def _bdry_bounds(prob, slo, shi, method, batch):
    up, dn, mid = [], [], []
    for i in range(0, slo.shape[0], batch):
        l, h = slo[i:i + batch], shi[i:i + batch]
        bb = boundary_bounds(prob, l, h, method)
        p0, pT = midpoint_boundary(prob, 0.5 * (l + h))
        up.append(bb["up0"] + bb["upT"])
        dn.append(bb["dn0"] + bb["dnT"])
        mid.append(p0 + pT)
    return torch.cat(up), torch.cat(dn), torch.cat(mid)


def _split(lo, hi):
    """Bisect every box along its longest side. Returns children lo, hi [2N, m] and parent index."""
    side = hi - lo
    j = side.argmax(1)
    midv = 0.5 * (lo.gather(1, j[:, None]) + hi.gather(1, j[:, None]))[:, 0]
    lo1, hi1 = lo.clone(), hi.clone()
    lo2, hi2 = lo.clone(), hi.clone()
    ar = torch.arange(lo.shape[0])
    hi1[ar, j] = midv
    lo2[ar, j] = midv
    parent = torch.cat([ar, ar])
    return torch.cat([lo1, lo2]), torch.cat([hi1, hi2]), parent


@torch.no_grad()
def adaptive_upper_sum(prob, method: str = "T3", init_n: int = 8, init_n_space: int | None = None,
                       tau: float = 0.1, max_cells: int = 2_000_000, max_depth: int = 40,
                       wall_time_budget: float = 7200.0, frac: float = 0.10, batch: int = 2048,
                       log=print) -> dict:
    t0 = time.time()
    m = prob.m
    box_lo, box_hi = prob.box
    counts = [init_n] * m
    ids = torch.arange(n_cells(counts))
    lo, hi = cell_boxes(ids, counts, box_lo, box_hi)
    depth = torch.zeros(lo.shape[0], dtype=torch.long)
    V = _vol_bounds(prob, lo, hi, method, batch)
    ns = init_n_space or init_n
    s_counts = [ns] * prob.d
    slo, shi = cell_boxes(torch.arange(n_cells(s_counts)), s_counts, box_lo[1:], box_hi[1:])
    sdepth = torch.zeros(slo.shape[0], dtype=torch.long)
    bup, bdn, bmid = _bdry_bounds(prob, slo, shi, method, 4 * batch)
    evaluated = lo.shape[0] + slo.shape[0]
    history = []
    sweep = 0
    while True:
        vol = (hi - lo).prod(-1)
        area = (shi - slo).prod(-1)
        U = (float((vol * V["up"]).sum()) + float((area * bup).sum())) * FINAL_INFLATION
        L = (float((vol * V["dn"]).sum()) + float((area * bdn).sum())) / FINAL_INFLATION
        rho_mid = float((vol * V["mid"]).sum()) + float((area * bmid).sum())
        unr = float(vol[V["st"] == 0].sum()) / float(vol.sum())
        rec = {"sweep": sweep, "cells": lo.shape[0], "bdry_cells": slo.shape[0], "U": U, "L": L, "rho_h_sq": rho_mid,
               "E_cert_U": math.sqrt(prob.C_eq * prob.C_rel * U), "unresolved_fraction": unr,
               "max_depth": int(depth.max()), "time": time.time() - t0, "cells_evaluated": evaluated}
        history.append(rec)
        log(rec)
        if U - L <= tau * L:
            stop = "tau"
            break
        if lo.shape[0] + slo.shape[0] >= max_cells:
            stop = "max_cells"
            break
        if time.time() - t0 >= wall_time_budget:
            stop = "wall_time"
            break
        # priorities (pooled)
        pv = vol * (V["up"] - V["dn"])
        pv[depth >= max_depth] = -1.0
        pb = area * (bup - bdn)
        pb[sdepth >= max_depth] = -1.0
        pri = torch.cat([pv, pb])
        n_split = max(1, int(frac * pri.numel()))
        n_split = min(n_split, (max_cells - lo.shape[0] - slo.shape[0]) // 1 + 1)
        top = torch.topk(pri, n_split).indices
        top = top[pri[top] > 0]
        if top.numel() == 0:
            stop = "nothing_to_split"
            break
        tv = top[top < lo.shape[0]]
        tb = top[top >= lo.shape[0]] - lo.shape[0]
        if tv.numel():
            keep = torch.ones(lo.shape[0], dtype=torch.bool)
            keep[tv] = False
            clo, chi, par = _split(lo[tv], hi[tv])
            Vc = _vol_bounds(prob, clo, chi, method, batch)
            evaluated += clo.shape[0]
            pidx = tv[par]
            use_par = Vc["up"] > V["up"][pidx]          # parent bound is tighter: keep its term split too
            for k in TERMS:
                Vc[f"u_{k}"] = torch.where(use_par, V[f"u_{k}"][pidx], Vc[f"u_{k}"])
            Vc["up"] = torch.minimum(Vc["up"], V["up"][pidx])
            Vc["dn"] = torch.maximum(Vc["dn"], V["dn"][pidx])
            # status: a child of a sign-resolved parent is resolved with the same sign
            Vc["st"] = torch.where(V["st"][pidx] != 0, V["st"][pidx], Vc["st"])
            lo = torch.cat([lo[keep], clo])
            hi = torch.cat([hi[keep], chi])
            depth = torch.cat([depth[keep], depth[pidx] + 1])
            V = {k: torch.cat([V[k][keep], Vc[k]]) for k in V}
        if tb.numel():
            keep = torch.ones(slo.shape[0], dtype=torch.bool)
            keep[tb] = False
            clo, chi, par = _split(slo[tb], shi[tb])
            cu, cd, cm = _bdry_bounds(prob, clo, chi, method, 4 * batch)
            evaluated += clo.shape[0]
            pidx = tb[par]
            cu = torch.minimum(cu, bup[pidx])
            cd = torch.maximum(cd, bdn[pidx])
            slo = torch.cat([slo[keep], clo])
            shi = torch.cat([shi[keep], chi])
            sdepth = torch.cat([sdepth[keep], sdepth[pidx] + 1])
            bup = torch.cat([bup[keep], cu])
            bdn = torch.cat([bdn[keep], cd])
            bmid = torch.cat([bmid[keep], cm])
        sweep += 1
    vol = (hi - lo).prod(-1)
    area = (shi - slo).prod(-1)
    U_terms = {k: float((vol * V[f"u_{k}"]).sum()) for k in TERMS}
    mid_terms = {k: float((vol * V[f"m_{k}"]).sum()) for k in TERMS}
    res = {"method": method, "stop": stop, "n_cells": lo.shape[0], "n_bdry_cells": slo.shape[0],
           "cells_evaluated": evaluated, "U": U, "L": L, "rho_h_sq": rho_mid, "E_cert_U": rec["E_cert_U"],
           "V_unr_frac": unr, "max_depth": int(depth.max()), "max_depth_bdry": int(sdepth.max()),
           "U_terms": U_terms, "mid_terms": mid_terms,
           "U_0": float((area * bup).sum()), "U_T": 0.0, "rho_mid_0": float((area * bmid).sum()), "rho_mid_T": 0.0,
           "U_vol": float((vol * V["up"]).sum()), "L_vol": float((vol * V["dn"]).sum()),
           "rho_mid_vol": float((vol * V["mid"]).sum()), "C_eq": prob.C_eq, "C_rel": prob.C_rel,
           "wall_s": time.time() - t0, "history": history,
           "params": {"init_n": init_n, "init_n_space": ns, "tau": tau, "max_cells": max_cells, "max_depth": max_depth,
                      "wall_time_budget": wall_time_budget, "frac": frac}}
    res["cells"] = {"lo": lo.numpy(), "hi": hi.numpy(), "depth": depth.numpy(), "status": V["st"].numpy(),
                    "up": V["up"].numpy(), "dn": V["dn"].numpy()}
    res["bdry_cells"] = {"lo": slo.numpy(), "hi": shi.numpy(), "depth": sdepth.numpy()}
    return res


_ = np
