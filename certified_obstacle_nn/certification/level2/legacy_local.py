"""Tasks D-F: the legacy (global-constant) Level-2 certificate written as a sum of per-cell
contributions, with an opt-in LOCAL cell radius.

    E_h = sum_K E_h,K,   E_h,K = E_lip,K + E_unr,K   (+ E_0T split over the spatial cells S)

radius = 'global'  (legacy, reproduces level2_diagnostic.certify / certify_option_b):
    E_lip,K = (L_sm + L_Psi) h |K|,        sign test |gamma(z_K)| > L_gamma h,
    E_unr,K = |K| (Psi(z_K) + L_Psi h),    E_0T = (L_Phi0 + L_PhiT) h |Omega|
radius = 'local'   (global constants + local radius, Task F):
    E_lip,K = (L_sm + L_Psi) (1 + xi_q) r_K |K|      r_K = (1/2) sqrt(sum side_i^2)
    sign test |gamma(z_K)| > L_gamma r_K,  E_unr,K = |K| (Psi(z_K) + L_Psi r_K),
    E_0T = sum_S (L_Phi0 + L_PhiT) (1 + xi_q) r_S |S|
where xi_q r_K bounds the distance of every Gauss node of K from z_K. Justification
(docs/history/LEVEL2_LOCAL_THEORY_NOTES.md section 2): for a positive rule with nodes z_i in K and weights
summing to |K|,  |int_K F - sum_i w_i F(z_i)| <= int_K |F - F(z_K)| + sum_i w_i |F(z_i) - F(z_K)|
<= |K| Lip(F) (r_K + xi_q r_K);  if |gamma(z_K)| > L_gamma r_K then gamma has one sign on K;
and sup_K Psi <= Psi(z_K) + L_Psi r_K. The legacy h = 2 r_K is a valid but cruder radius.

Option B is applied identically in both modes: m_R_h = max_K |R(z_K)| + L_R r_K,
m_R_cert = min(m_R_apriori, m_R_h), T1_new = 2 (Lambda/alpha) m_R_cert L_R.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from certified_obstacle_nn.certification.level2.fields import fields_at, state_u_at, u0_at
from certified_obstacle_nn.certification.level2.grid import (cell_boxes, gauss_max_offset, n_cells,
                                                              point_densities, tensor_nodes)


@torch.no_grad()
def legacy_uniform(prob, consts: dict, counts, q: int, radius: str = "global", points_per_batch: int = 400_000,
                   store_cells: bool = False) -> dict:
    assert radius in ("global", "local")
    t0 = time.time()
    m = prob.m
    box_lo, box_hi = prob.box
    sizes = [(b - a) / n for a, b, n in zip(box_lo, box_hi, counts)]
    vol = math.prod(sizes)
    h = math.sqrt(sum(s * s for s in sizes))
    r_K = 0.5 * h
    xi = gauss_max_offset(q)
    rad = h if radius == "global" else r_K
    La = prob.Lambda / prob.alpha
    N = n_cells(counts)
    per = q ** m + 1
    cpb = max(1, points_per_batch // per)
    acc = {"rr": 0.0, "rp": 0.0, "rc": 0.0, "rr_pde": 0.0, "rr_sig": 0.0}
    n_unr, psi_unr, maxR = 0, 0.0, 0.0
    cells = {"R_mid": [], "gamma_mid": [], "Psi_mid": [], "unresolved": []} if store_cells else None
    for c0 in range(0, N, cpb):
        ids = torch.arange(c0, min(c0 + cpb, N))
        lo, hi = cell_boxes(ids, counts, box_lo, box_hi)
        mid = 0.5 * (lo + hi)
        pts, W = tensor_nodes(lo, hi, q)
        F = fields_at(prob, torch.cat([mid, pts], 0))
        D = point_densities(prob, F)
        b = ids.numel()
        acc["rr"] += float((W * (D["P_R"][b:] + D["P_sig"][b:])).sum())
        acc["rr_pde"] += float((W * D["P_R"][b:]).sum())
        acc["rr_sig"] += float((W * D["P_sig"][b:]).sum())
        acc["rp"] += float((W * (D["P_kw"][b:] + D["P_int"][b:])).sum())
        acc["rc"] += float((W * D["P_c"][b:]).sum())
        gm, Pm, Rm = F["gamma"][:b], D["Psi"][:b], F["R"][:b].abs()
        unr = gm.abs() <= consts["L_gamma"] * rad
        n_unr += int(unr.sum())
        psi_unr += float(Pm[unr].sum())
        maxR = max(maxR, float(Rm.max()))
        if store_cells:
            cells["R_mid"].append(F["R"][:b].numpy())
            cells["gamma_mid"].append(gm.numpy())
            cells["Psi_mid"].append(Pm.numpy())
            cells["unresolved"].append(unr.numpy())
    # t = 0 and t = T faces: spatial Gauss rule on the spatial grid counts[1:]
    s_counts = counts[1:]
    Ns = n_cells(s_counts)
    s_sizes = sizes[1:]
    r_S = 0.5 * math.sqrt(sum(s * s for s in s_sizes))
    R0sq = w0sq = wTsq = 0.0
    cps = max(1, points_per_batch // q ** prob.d)
    for c0 in range(0, Ns, cps):
        ids = torch.arange(c0, min(c0 + cps, Ns))
        lo, hi = cell_boxes(ids, s_counts, box_lo[1:], box_hi[1:])
        X, W = tensor_nodes(lo, hi, q)
        for tv in (0.0, prob.T):
            P = torch.cat([torch.full((X.shape[0], 1), tv), X], 1)
            u = state_u_at(prob, P)
            Xl = [X[:, i] for i in range(prob.d)]
            gam = prob.g(P[:, 0], Xl) - u
            if tv == 0.0:
                R0sq += float((W * (u0_at(prob, X) - u) ** 2).sum())
                w0sq += float((W * torch.relu(gam) ** 2).sum())
            else:
                wTsq += float((W * torch.relu(gam) ** 2).sum())
    Q_meas = math.prod(b - a for a, b in zip(box_lo, box_hi))
    Om_meas = math.prod(b - a for a, b in zip(box_lo[1:], box_hi[1:]))
    rho_r2 = acc["rr"] + R0sq
    rho_p2 = acc["rp"] + w0sq + wTsq
    rho_c2 = acc["rc"]
    rho2 = rho_r2 + rho_p2 + rho_c2
    E_unr = vol * psi_unr + n_unr * vol * consts["L_Psi"] * rad
    lip_len = h if radius == "global" else (1.0 + xi) * r_K
    lip_len_S = h if radius == "global" else (1.0 + xi) * r_S
    E_0T = (consts["L_Phi0"] + consts["L_PhiT"]) * lip_len_S * Om_meas
    m_R_h = maxR + consts["L_R"] * r_K
    m_R_cert = min(consts["m_R_apriori"], m_R_h)
    T1_old = 2 * La * consts["m_R_apriori"] * consts["L_R"]
    T1_new = 2 * La * m_R_cert * consts["L_R"]
    L_sm_new = consts["L_sm"] - T1_old + T1_new
    res = {"radius_mode": radius, "counts": list(counts), "n_cells": N, "q": q, "h": h, "r_K": r_K, "xi_q": xi,
           "rho_r": math.sqrt(rho_r2), "rho_p_tilde": math.sqrt(rho_p2), "rho_c": math.sqrt(max(rho_c2, 0.0)),
           "rho_tilde_h": math.sqrt(rho2), "rho_h_sq": rho2, "rho_r_pde_sq": acc["rr_pde"],
           "V_unr_frac": n_unr * vol / Q_meas, "E_h_unr": E_unr, "E_h_dat": 0.0, "E_h_0T": E_0T,
           "max_R_mid": maxR, "m_R_h": m_R_h, "m_R_cert": m_R_cert, "T1_old": T1_old, "T1_new": T1_new,
           "L_sm_old": consts["L_sm"], "L_sm_new": L_sm_new, "C_eq": prob.C_eq, "C_rel": prob.C_rel}
    for tag, Lsm in (("old", consts["L_sm"]), ("new", L_sm_new)):
        E_lip = (Lsm + consts["L_Psi"]) * lip_len * Q_meas
        E_h = E_lip + E_unr + E_0T
        res.update({f"E_h_lip_{tag}": E_lip, f"E_h_{tag}": E_h,
                    f"E_cert_{tag}": math.sqrt(prob.C_rel * (rho2 + E_h)),
                    f"E_cert_U_{tag}": math.sqrt(prob.C_eq * prob.C_rel * (rho2 + E_h))})
    res["wall_s"] = time.time() - t0
    if store_cells:
        res["cells"] = {k: np.concatenate(v) for k, v in cells.items()}
        res["cells"]["E_lip_K"] = np.full(N, (L_sm_new + consts["L_Psi"]) * lip_len * vol)
        res["cells"]["E_unr_K"] = np.where(res["cells"]["unresolved"],
                                           vol * (res["cells"]["Psi_mid"] + consts["L_Psi"] * rad), 0.0)
    return res
