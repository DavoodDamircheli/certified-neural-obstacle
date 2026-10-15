"""Upper-sum Level-2 certificate (docs/history/LEVEL2_LOCAL_THEORY_NOTES.md sections 1, 3, 5).

    rho_tilde^2 <= U(T) = sum_K |K| Phibar_K + sum_S |S| (Phibar0_S + PhibarT_S)
    E_cert,U = sqrt(C_eq C_rel U)

Enclosure methods for the cell fields (R, R_sig, gamma, d_t gamma, grad_x gamma, lambda):
    'T1'  interval forward mode                         (theory notes 4, m = 0)
    'T3'  affine forms with lumped remainder             (theory notes 4)
    'T2i' / 'T2a'  centered forms F(z_K) + [grad F](K) (K - z_K), gradient enclosure from an
          interval / affine jet                           (Lemma 2)
    'LL'  local Lipschitz F(z_K) +- L_{F,K} r_K, L_{F,K} from the affine jet   (Lemma 3)
    'T3+T2a'  intersection of the T3 and T2a intervals (intersection of enclosures is an enclosure)
Boundary cells (t = 0, t = T) always use value-only forms: affine unless the method is 'T1'.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from certified_obstacle_nn.certification.level2 import ops
from certified_obstacle_nn.certification.level2.aff import Aff, neg_part_range, pos_part_range, sq_range
from certified_obstacle_nn.certification.level2.fields import fields_at, fields_on_boxes
from certified_obstacle_nn.certification.level2.grid import cell_boxes, n_cells, point_densities
from certified_obstacle_nn.certification.level2.netenc import mlp_enclose

METHODS = ("T1", "T3", "T2i", "T2a", "LL", "T3+T2a", "T3z", "T3q", "T2q", "T3q+T2q", "T3+T3q", "T3q+T2a")
_MODE = {"T1": "interval", "T3": "affine", "T3z": "zono", "T3q": "tm2", "T2i": "interval", "T2a": "affine",
         "LL": "affine", "T2q": "tm2"}
FINAL_INFLATION = 1.0 + 1e-9
_NAMES = ("R", "gamma", "gamma_t", "lam", "M")


def _flat(F):
    """dict of fields -> dict name -> object, lists expanded as name_i."""
    out = {k: F[k] for k in _NAMES}
    for i, v in enumerate(F["Rsig"]):
        out[f"Rsig_{i}"] = v
    for i, v in enumerate(F["ggrad"]):
        out[f"ggrad_{i}"] = v
    return out


def _iv(a: Aff):
    return a.range()


def enclose_cells(prob, lo: torch.Tensor, hi: torch.Tensor, method: str) -> dict:
    """Returns dict name -> (lo, hi) interval tensors [B] enclosing every field over each box."""
    iv = _enclose_raw(prob, lo, hi, method)
    # lambda = M^2: intersect with the exact range of s -> s^2 over the M enclosure (lam >= 0).
    ml, mu = sq_range(*iv.pop("M"))
    iv["lam"] = (torch.maximum(iv["lam"][0], ml).clamp(min=0.0), torch.minimum(iv["lam"][1], mu))
    return iv


def _enclose_raw(prob, lo, hi, method):
    assert method in METHODS, method
    if method in ("T1", "T3", "T3z", "T3q"):
        F = _flat(fields_on_boxes(prob, lo, hi, _MODE[method]))
        iv = {k: _iv(v) for k, v in F.items()}
    elif method in ("T2i", "T2a", "LL", "T2q"):
        mid = 0.5 * (lo + hi)
        rad = 0.5 * (hi - lo)
        r_K = rad.pow(2).sum(-1).sqrt()
        Fc = _flat(fields_at(prob, mid))
        J = _flat(fields_on_boxes(prob, lo, hi, _MODE[method], grad=True))
        iv = {}
        for k, jet in J.items():
            glo, ghi = jet.grad.range()                      # [B, m]
            gmax = torch.maximum(glo.abs(), ghi.abs())
            if method == "LL":
                w = gmax.pow(2).sum(-1).sqrt() * r_K
            else:
                w = (gmax * rad).sum(-1)
            c = Fc[k]
            pad = 1e-12 * (c.abs() + w) + 1e-300
            iv[k] = (c - w - pad, c + w + pad)
    else:  # intersections 'T3+T2a', 'T3q+T2q'
        m1, m2 = method.split("+")
        a = _enclose_raw(prob, lo, hi, m1)
        b = _enclose_raw(prob, lo, hi, m2)
        iv = {k: (torch.maximum(a[k][0], b[k][0]), torch.minimum(a[k][1], b[k][1])) for k in a}
    return iv


def phi_bounds(prob, iv: dict) -> dict:
    """Per-cell upper / lower bounds of Phi and of its terms (theory notes section 3)."""
    La = prob.Lambda / prob.alpha
    d = prob.d
    Rl, Ru = sq_range(*iv["R"])
    up = {"R": La * Ru}
    dn = {"R": La * Rl}
    sl = su = 0.0
    for i in range(d):
        a, b = sq_range(*iv[f"Rsig_{i}"])
        sl, su = sl + a, su + b
    up["sig"], dn["sig"] = su, sl
    gl, gu = iv["gamma"]
    wl, wu = pos_part_range(gl, gu)
    up["kw"], dn["kw"] = prob.kappa * wu ** 2, prob.kappa * wl ** 2
    nl, nu = neg_part_range(gl, gu)
    ll, lu = iv["lam"]
    up["c"], dn["c"] = lu * nu, ll * nl
    pl = pu = 0.0
    for i in range(d):
        a, b = sq_range(*iv[f"ggrad_{i}"])
        pl, pu = pl + prob.A * a, pu + prob.A * b
    a, b = sq_range(*iv["gamma_t"])
    pl, pu = pl + La * a, pu + La * b
    up["int"] = torch.where(gu <= 0, torch.zeros_like(gu), pu)
    dn["int"] = torch.where(gl > 0, pl, torch.zeros_like(gl))
    status = torch.where(gl > 0, torch.ones_like(gl), torch.where(gu <= 0, -torch.ones_like(gl),
                                                                  torch.zeros_like(gl)))
    return {"up": up, "dn": dn, "Phi_up": sum(up.values()), "Phi_dn": sum(dn.values()), "status": status}


def enclose_boundary(prob, slo: torch.Tensor, shi: torch.Tensor, tval: float, method: str) -> dict:
    """Value-only enclosures of R_0 = u0 - u(0,.) (if tval == 0) and gamma(tval,.) on spatial boxes."""
    B = slo.shape[0]
    tcol = torch.full((B, 1), float(tval), dtype=torch.float64)
    lo = torch.cat([tcol, slo], 1)
    hi = torch.cat([tcol, shi], 1)
    if method in ("T3q", "T2q"):
        from certified_obstacle_nn.certification.level2.tm import TM
        Z = TM.from_box(lo, hi)
    else:
        Z = Aff.from_box(lo, hi, affine=(method != "T1"))
    X = [Z.sel(1, 1 + i) for i in range(prob.d)]
    N, _, _ = mlp_enclose(prob.state, Z, order=0)
    N = N.sel(1, 0)
    Bm = prob.mask(X)
    t = Z.sel(1, 0)
    if prob.name == "disk2d":
        gamma = -(Bm * (prob._h(t) + N))
    else:
        gamma = prob.g(t, X) - Bm * N
    out = {"gamma": gamma.range()}
    if tval == 0.0:
        out["R0"] = (prob.u0(X) - Bm * N).range()
    return out


def _bdry_enclose(prob, slo, shi, tval, method):
    """Value-only boundary enclosures; intersection methods intersect their components."""
    if "+" in method:
        parts = [enclose_boundary(prob, slo, shi, tval, mm) for mm in method.split("+")]
        return {k: (torch.maximum(parts[0][k][0], parts[1][k][0]), torch.minimum(parts[0][k][1], parts[1][k][1]))
                for k in parts[0]}
    return enclose_boundary(prob, slo, shi, tval, method)


def boundary_bounds(prob, slo, shi, method):
    e0 = _bdry_enclose(prob, slo, shi, 0.0, method)
    eT = _bdry_enclose(prob, slo, shi, prob.T, method)
    r0l, r0u = sq_range(*e0["R0"])
    w0l, w0u = pos_part_range(*e0["gamma"])
    wTl, wTu = pos_part_range(*eT["gamma"])
    return {"up0": r0u + w0u ** 2, "dn0": r0l + w0l ** 2, "upT": wTu ** 2, "dnT": wTl ** 2,
            "up0_R0": r0u}


_TERM_OF = {"R": "P_R", "sig": "P_sig", "kw": "P_kw", "c": "P_c", "int": "P_int"}


def midpoint_phi(prob, mid):
    """Exact Phi(z_K) and its terms (midpoint rule value; diagnostic / consistency check)."""
    D = point_densities(prob, fields_at(prob, mid))
    terms = {k: D[v] for k, v in _TERM_OF.items()}
    return sum(terms.values()), terms


def midpoint_boundary(prob, smid):
    """Exact Phi_0, Phi_T at spatial cell centers."""
    from certified_obstacle_nn.certification.level2.fields import state_u_at, u0_at
    X = [smid[:, i] for i in range(prob.d)]
    out = []
    for tv in (0.0, prob.T):
        P = torch.cat([torch.full((smid.shape[0], 1), tv), smid], 1)
        u = state_u_at(prob, P)
        w = torch.relu(prob.g(P[:, 0], X) - u)
        out.append(((u0_at(prob, smid) - u) ** 2 + w ** 2) if tv == 0.0 else w ** 2)
    return out


@torch.no_grad()
def upper_sum_uniform(prob, counts, method: str, cells_per_batch: int = 2048, store_cells: bool = False) -> dict:
    """Upper-sum certificate on the uniform grid with `counts` cells per direction (t first)."""
    t0 = time.time()
    box_lo, box_hi = prob.box
    N = n_cells(counts)
    acc_up = {}
    acc_dn = {}
    acc_mid = {}
    U = L = rho_mid = 0.0
    n_unr = n_pos = n_neg = 0
    vol_tot = 0.0
    store = {"lo": [], "hi": [], "Phi_up": [], "Phi_dn": [], "status": []} if store_cells else None
    for c0 in range(0, N, cells_per_batch):
        ids = torch.arange(c0, min(c0 + cells_per_batch, N))
        lo, hi = cell_boxes(ids, counts, box_lo, box_hi)
        vol = (hi - lo).prod(-1)
        pb = phi_bounds(prob, enclose_cells(prob, lo, hi, method))
        for k in pb["up"]:
            acc_up[k] = acc_up.get(k, 0.0) + float((vol * pb["up"][k]).sum())
            acc_dn[k] = acc_dn.get(k, 0.0) + float((vol * pb["dn"][k]).sum())
        U += float((vol * pb["Phi_up"]).sum())
        L += float((vol * pb["Phi_dn"]).sum())
        pm, pmt = midpoint_phi(prob, 0.5 * (lo + hi))
        rho_mid += float((vol * pm).sum())
        for k, v in pmt.items():
            acc_mid[k] = acc_mid.get(k, 0.0) + float((vol * v).sum())
        st = pb["status"]
        n_unr += int((st == 0).sum())
        n_pos += int((st > 0).sum())
        n_neg += int((st < 0).sum())
        vol_tot += float(vol.sum())
        if store_cells:
            for k, v in (("lo", lo), ("hi", hi), ("Phi_up", pb["Phi_up"]), ("Phi_dn", pb["Phi_dn"]), ("status", st)):
                store[k].append(v.numpy())
    # boundary
    s_counts = counts[1:]
    Ns = n_cells(s_counts)
    U0 = UT = L0 = LT = M0 = MT = 0.0
    for c0 in range(0, Ns, 4 * cells_per_batch):
        ids = torch.arange(c0, min(c0 + 4 * cells_per_batch, Ns))
        slo, shi = cell_boxes(ids, s_counts, box_lo[1:], box_hi[1:])
        area = (shi - slo).prod(-1)
        bb = boundary_bounds(prob, slo, shi, method)
        U0 += float((area * bb["up0"]).sum())
        L0 += float((area * bb["dn0"]).sum())
        UT += float((area * bb["upT"]).sum())
        LT += float((area * bb["dnT"]).sum())
        p0, pT = midpoint_boundary(prob, 0.5 * (slo + shi))
        M0 += float((area * p0).sum())
        MT += float((area * pT).sum())
    U_tot = (U + U0 + UT) * FINAL_INFLATION
    L_tot = (L + L0 + LT) / FINAL_INFLATION
    res = {"method": method, "counts": list(counts), "n_cells": N, "n_bdry_cells": Ns,
           "U": U_tot, "L": L_tot, "U_vol": U, "L_vol": L, "U_0": U0, "U_T": UT, "L_0": L0, "L_T": LT,
           "rho_mid_vol": rho_mid, "rho_mid_0": M0, "rho_mid_T": MT, "rho_h_sq": rho_mid + M0 + MT,
           "U_terms": acc_up, "L_terms": acc_dn, "mid_terms": acc_mid,
           "V_unr_frac": n_unr / N, "V_pos_frac": n_pos / N, "V_neg_frac": n_neg / N,
           "E_cert_U": math.sqrt(prob.C_eq * prob.C_rel * U_tot), "C_eq": prob.C_eq, "C_rel": prob.C_rel,
           "wall_s": time.time() - t0, "vol_check": vol_tot}
    if store_cells:
        res["cells"] = {k: np.concatenate(v) for k, v in store.items()}
    return res


_ = ops  # re-export convenience
