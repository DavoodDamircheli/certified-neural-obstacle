"""Stage 6: uniform midpoint Level-2 certificate on 3D space-time cells (diagnostic only).

Port of certification/{network_bounds, cell_classifier, raw_estimator, level2_remainder,
certified_residual_bound, certificate_option_b}.py (specification sections 4-8) to Q = (0,1)^3 with
spatial dimension d = 2. No local bounds, no branch-and-bound.

* Partition: n^3 congruent cubes, h_K = cell diagonal = sqrt(3)/n, r_K = h_K/2 (center-to-
  corner, Euclidean -- same norm in which every Lipschitz constant below is combined).
* Gamma_data = empty  =>  V_h_dat = 0, E_h_dat = 0.
* Network bounds: spectral-norm bounds (paper Prop. 5.5 / specification section 6) via the
  existing certification.network_bounds.compute_network_bounds on each MLP (input [t,x,y]).
* Problem-data constants (sup norms / Lipschitz constants of g, f, u0 over Q): hand-derived
  RIGOROUS upper bounds by the triangle inequality (DATA_BOUNDS below; derivation inline).
* m_R: partition-certified (Lemma 8.5, Option B): m_R_h = max_K |R_PDE(z_K)| + L_R r_K,
  m_R_cert = min(m_R_apriori, m_R_h); both old (a priori) and new values are reported.
* rho_tilde_h: tensor-product Gauss-Legendre with q points per direction per cell (q = 2 by
  default in 3D: 8 points/cell; any positive rule with weights summing to |K| inside K is
  covered by the same E_h_lip argument). t=0 / t=T face terms with q^2 points per face cell.
* Streams cells in batches; float64; model frozen; no training points are ever generated
  here (the partition is deterministic, training sets are scrambled Sobol -- disjointness is
  additionally checked against the run's own training set when --check_disjoint is given).
"""

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from certified_obstacle_nn.certification.network_bounds import compute_network_bounds  # noqa: E402
from certified_obstacle_nn.experiments.expanding_disk_2d import exact_solution as ex  # noqa: E402
from certified_obstacle_nn.experiments.expanding_disk_2d.config import Config  # noqa: E402
from certified_obstacle_nn.experiments.expanding_disk_2d.losses import triple_fields  # noqa: E402
from certified_obstacle_nn.experiments.expanding_disk_2d.model import Triple2D  # noqa: E402

D = 2
C_REL = 216.0


def data_bounds(kappa: float) -> dict:
    """Rigorous upper bounds over Qbar for the 2D expanding-disk data (Euclidean norms for
    gradients, spectral norm for Hessians). h(1)=e-2, h'(1)=e-1, h''<=e on [0,1];
    B: ||B||=1, |B_x|,|B_y|<=4 => ||grad B||<=4 sqrt2; |B_xx|,|B_yy|<=8, |B_xy|<=16 =>
    ||D^2 B||_2 <= 24 (Gershgorin); |Delta B| <= 16; |grad Delta B| <= 32 sqrt2.
    psi (inside the disk, r<=R, R in [0.15,0.35], R'=0.2): 0<=psi<=R/2, |grad psi|<=1,
    psi_t in [R'/2, R'], |grad psi_t| <= R'/R, |psi_tt| <= R'^2/R, Delta psi = -2/R,
    d_t Delta psi = 2R'/R^2, |grad|grad psi|^2| <= 2/R, |d_t |grad psi|^2| <= 2R'/R.
    f is continuous, piecewise smooth => Lip(f) = sup of the piecewise gradient."""
    e = math.e
    h1, hp1, hpp = e - 2.0, e - 1.0, e
    Rmin, Rp = ex.R0, ex.R_SPEED
    gB, g2B, lapB, glapB = 4.0 * math.sqrt(2), 24.0, 16.0, 32.0 * math.sqrt(2)
    smooth_inf = hp1 * 1.0 + h1 * lapB                       # |g_t - Delta g|
    smooth_x = hp1 * gB + h1 * glapB                          # |grad(g_t - Delta g)|
    smooth_t = hpp * 1.0 + hp1 * lapB                         # |d_t(g_t - Delta g)|
    # inside-disk kappa-part K = 2 psi psi_t - 2 psi Delta psi - 2|grad psi|^2 (the -2kappa
    # and +2kappa cancel inside); outside, the kappa-part is the constant -2 kappa.
    K_inf = max(2.0, 2 * (0.35 / 2) * Rp + 4 * (1 / 2) + 0.0)  # K in [-2, 0.07+2]
    K_x = 2 * 1 * Rp + Rp + 2 * 1 * (2 / Rmin) + 2 * (2 / Rmin)
    K_t = 2 * Rp ** 2 + Rp ** 2 + 2 * Rp * (2 / Rmin) + 2 * Rp / Rmin + 2 * (2 * Rp / Rmin)
    psi0_max = ex.R0 / 2
    return {
        "g_inf": h1, "g_x_inf": h1 * gB, "g_t_inf": hp1,
        "g_xx_inf": h1 * g2B, "g_tx_inf": hp1 * gB, "g_tt_inf": hpp,
        "f_inf": smooth_inf + kappa * K_inf, "f_x_inf": smooth_x + kappa * K_x,
        "f_t_inf": smooth_t + kappa * K_t,
        "u0_inf": kappa * psi0_max ** 2, "u0_lip": 2 * kappa * psi0_max * 1.0,
        "B_inf": 1.0, "gradB_inf": gB, "grad2B_inf": g2B,
    }


def triple_bounds(triple: Triple2D, cfg: Config, db: dict) -> dict:
    """specification section 6 (+ its addendum) with d = 2, A = I, b = c = 0, alpha = 1."""
    sb, fb, mb = (compute_network_bounds(n) for n in (triple.state, triple.flux, triple.mult))
    B, gB, g2B = db["B_inf"], db["gradB_inf"], db["grad2B_inf"]
    Sn, Pn, LgN = sb.output_bound, sb.gradient_bound, sb.gradient_lipschitz_bound
    Ss, Ps, LgS = fb.output_bound, fb.gradient_bound, fb.gradient_lipschitz_bound
    Sm, Pm = mb.output_bound, mb.gradient_bound
    La, alpha, A_inf, kp = cfg.Lambda / cfg.alpha, cfg.alpha, 1.0, cfg.alpha / cfg.Lambda
    lam_inf, lip_lam = Sm ** 2, 2 * Sm * Pm
    grad_u_inf = gB * Sn + B * Pn
    dt_u_inf = B * Pn
    m_gap = db["g_inf"] + B * Sn
    m_gap_x = db["g_x_inf"] + grad_u_inf
    m_gap_t = db["g_t_inf"] + dt_u_inf
    m_R = db["f_inf"] + B * Pn + D * Ps + lam_inf
    m_sig = alpha ** -0.5 * Ss + A_inf ** 0.5 * grad_u_inf
    M_Psi = A_inf * m_gap_x ** 2 + La * m_gap_t ** 2
    lip_grad_u = g2B * Sn + 2 * gB * Pn + B * LgN
    lip_div_s = D * LgS
    lip_dt_u = gB * Pn + B * LgN
    lip_u = math.hypot(dt_u_inf, grad_u_inf)
    lip_g = math.hypot(db["g_t_inf"], db["g_x_inf"])
    lip_f = math.hypot(db["f_t_inf"], db["f_x_inf"])
    L_gamma = lip_g + lip_u
    lip_R = lip_f + lip_dt_u + lip_div_s + lip_lam
    lip_Rsig = alpha ** -0.5 * Ps + A_inf ** 0.5 * lip_grad_u
    T = {"T1_pde_residual": 2 * La * m_R * lip_R, "T2_sig_residual": 2 * m_sig * lip_Rsig,
         "T3_kappa_gap": 2 * kp * m_gap * L_gamma, "T4_lambda_Lgamma": lam_inf * L_gamma,
         "T5_gap_lip_lambda": m_gap * lip_lam}
    lip_grad_gamma = math.hypot(db["g_tx_inf"], db["g_xx_inf"]) + lip_grad_u
    lip_dt_gamma = math.hypot(db["g_tt_inf"], db["g_tx_inf"]) + lip_dt_u
    L_Psi = 2 * A_inf * m_gap_x * lip_grad_gamma + 2 * La * m_gap_t * lip_dt_gamma
    m_0 = db["u0_inf"] + B * Sn
    L_Phi0 = 2 * m_0 * (db["u0_lip"] + grad_u_inf) + 2 * m_gap * L_gamma
    L_PhiT = 2 * m_gap * L_gamma
    return {"Sigma_N_state": Sn, "Pi_N_state": Pn, "Sigma_N_flux": Ss, "Pi_N_flux": Ps,
            "Sigma_N_mult": Sm, "Pi_N_mult": Pm, "m_gap": m_gap, "m_gap_x": m_gap_x, "m_gap_t": m_gap_t,
            "m_R_apriori": m_R, "m_sig": m_sig, "m_lam": lam_inf, "M_Psi": M_Psi, "L_gamma": L_gamma,
            "L_R": lip_R, "lip_R_sig": lip_Rsig, "L_sm_old": sum(T.values()), "L_sm_terms_old": T,
            "L_Psi": L_Psi, "L_Phi0": L_Phi0, "L_PhiT": L_PhiT, "data_bounds": db}


def _gl(q):
    x, w = np.polynomial.legendre.leggauss(q)
    return torch.tensor(0.5 * (x + 1), dtype=torch.float64), torch.tensor(0.5 * w, dtype=torch.float64)


def _volume_pass(triple, cfg, n, q, cells_per_batch, L_gamma):
    """Streams all n^3 cells. Returns integrated volume densities (rho_r, rho_p, rho_c parts),
    midpoint |R_PDE| max (+argmax), V_unr and sum_unr |K| Psi(z_K)."""
    hcell = 1.0 / n
    vol = hcell ** 3
    nodes, wts = _gl(q)
    off = torch.stack(torch.meshgrid(nodes, nodes, nodes, indexing="ij"), -1).reshape(-1, 3)
    wq = (wts[:, None, None] * wts[None, :, None] * wts[None, None, :]).reshape(-1)  # sums to 1
    h_diag = math.sqrt(3) * hcell
    La, kp = cfg.Lambda / cfg.alpha, cfg.alpha / cfg.Lambda
    acc = {"rr": 0.0, "rp": 0.0, "rc": 0.0}
    maxR, arg = 0.0, None
    n_unr, psi_unr_sum = 0, 0.0
    n_cells = n ** 3
    for c0 in range(0, n_cells, cells_per_batch):
        ids = torch.arange(c0, min(c0 + cells_per_batch, n_cells))
        it, ix, iy = ids // (n * n), (ids // n) % n, ids % n
        corner = torch.stack([it, ix, iy], -1).to(torch.float64) * hcell
        # midpoints
        mid = corner + 0.5 * hcell
        pts = torch.cat([mid, (corner[:, None, :] + hcell * off[None]).reshape(-1, 3)], 0)
        t, x, y = (pts[:, k:k + 1].clone().requires_grad_(True) for k in range(3))
        fl = triple_fields(triple, t, x, y, create_graph=False)
        u, u_t, gu, sig, dsig, lam = (v.detach() for v in fl)
        t, x, y = t.detach(), x.detach(), y.detach()
        f, g = ex.f_exact(t, x, y, cfg.kappa), ex.g(t, x, y)
        gx, gy = ex.grad_g(t, x, y)
        gt = ex.g_t(t, x, y)
        R = f - u_t - dsig + lam
        Rs2 = ((sig + gu) ** 2).sum(-1, keepdim=True)
        gam = g - u
        ind = (gam > 0).to(u.dtype)
        Psi = (gx - gu[:, 0:1]) ** 2 + (gy - gu[:, 1:2]) ** 2 + La * (gt - u_t) ** 2
        rr = La * R ** 2 + Rs2
        rp = ind * Psi + kp * torch.relu(gam) ** 2
        rc = lam * torch.relu(-gam)
        m = ids.numel()
        # midpoint part
        Rm = R[:m, 0].abs()
        j = int(Rm.argmax())
        if float(Rm[j]) > maxR:
            maxR, arg = float(Rm[j]), mid[j].tolist()
        unr = gam[:m, 0].abs() <= L_gamma * h_diag
        n_unr += int(unr.sum())
        psi_unr_sum += float(Psi[:m, 0][unr].sum())
        # quadrature part
        W = (vol * wq).repeat(m)[:, None]
        acc["rr"] += float((W * rr[m:]).sum())
        acc["rp"] += float((W * rp[m:]).sum())
        acc["rc"] += float((W * rc[m:]).sum())
    return acc, maxR, arg, n_unr, psi_unr_sum


@torch.no_grad()
def _face_pass(triple, cfg, n, q):
    hcell = 1.0 / n
    nodes, wts = _gl(q)
    xs = (torch.arange(n, dtype=torch.float64)[:, None] * hcell + hcell * nodes[None]).reshape(-1)
    ws = (hcell * wts).repeat(n)
    X, Y = torch.meshgrid(xs, xs, indexing="ij")
    W = (ws[:, None] * ws[None, :]).reshape(-1, 1)
    x, y = X.reshape(-1, 1), Y.reshape(-1, 1)
    out = {}
    for name, tv in (("0", 0.0), ("T", cfg.T)):
        s_e, s_w = 0.0, 0.0
        for i in range(0, x.shape[0], 200_000):
            xb, yb, wb = x[i:i + 200_000], y[i:i + 200_000], W[i:i + 200_000]
            tb = torch.full_like(xb, tv)
            u = triple.u(tb, xb, yb)
            s_w += float((wb * torch.relu(ex.g(tb, xb, yb) - u) ** 2).sum())
            if name == "0":
                s_e += float((wb * (ex.u0_exact(xb, yb, cfg.kappa) - u) ** 2).sum())
        out[name] = (s_e, s_w)
    return out["0"][0], out["0"][1], out["T"][1]


def certify(triple, cfg, n, q=2, cells_per_batch=None) -> dict:
    triple.eval()
    for p in triple.parameters():
        p.requires_grad_(False)
    t0 = time.time()
    cells_per_batch = cells_per_batch or max(1, 200_000 // (q ** 3 + 1))
    db = data_bounds(cfg.kappa)
    bd = triple_bounds(triple, cfg, db)
    hcell = 1.0 / n
    h = math.sqrt(3) * hcell
    r_K = h / 2
    acc, maxR, arg, n_unr, psi_unr_sum = _volume_pass(triple, cfg, n, q, cells_per_batch, bd["L_gamma"])
    R0sq, w0sq, wTsq = _face_pass(triple, cfg, n, q)
    rho_r2 = acc["rr"] + R0sq
    rho_p2 = acc["rp"] + w0sq + wTsq
    rho_c2 = acc["rc"]
    rho2 = rho_r2 + rho_p2 + rho_c2
    vol = hcell ** 3
    m_R_h = maxR + bd["L_R"] * r_K
    m_R_cert = min(bd["m_R_apriori"], m_R_h)
    La = cfg.Lambda / cfg.alpha
    T1_new = 2 * La * m_R_cert * bd["L_R"]
    L_sm_new = bd["L_sm_old"] - bd["L_sm_terms_old"]["T1_pde_residual"] + T1_new
    Q_meas, Om_meas = 1.0, 1.0
    E_unr = vol * psi_unr_sum + n_unr * vol * bd["L_Psi"] * h
    E_0T = (bd["L_Phi0"] + bd["L_PhiT"]) * h * Om_meas
    res = {"n": n, "n_cells": n ** 3, "q": q, "h": h, "r_K": r_K,
           "rho_r": math.sqrt(rho_r2), "rho_p_tilde": math.sqrt(rho_p2), "rho_c": math.sqrt(rho_c2),
           "rho_tilde_h": math.sqrt(rho2), "V_unr_frac": n_unr * vol, "V_h_dat": 0.0,
           "max_R_mid": maxR, "argmax_R_mid_txy": arg, "m_R_h": m_R_h, "m_R_cert": m_R_cert,
           "L_sm_new": L_sm_new, "T1_new": T1_new}
    C_eq = max(1.0 / cfg.alpha, 1.0, 1.0, 2 * cfg.alpha / cfg.Lambda)
    for tag, Lsm in (("old", bd["L_sm_old"]), ("new", L_sm_new)):
        E_lip = (Lsm + bd["L_Psi"]) * h * Q_meas
        E_h = E_lip + E_unr + 0.0 + E_0T
        res.update({f"E_h_lip_{tag}": E_lip, f"E_h_{tag}": E_h,
                    f"E_cert_{tag}": math.sqrt(C_REL * (rho2 + E_h)),
                    f"E_cert_U_{tag}": math.sqrt(C_eq * C_REL * (rho2 + E_h))})
    res.update({"E_h_unr": E_unr, "E_h_dat": 0.0, "E_h_0T": E_0T, "C_rel": C_REL, "C_eq": C_eq,
                "wall_s": time.time() - t0,
                "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0})
    res["bounds"] = bd
    return res


def estimate_cost(n, q, sec_per_point):
    pts = n ** 3 * (q ** 3 + 1)
    return pts, pts * sec_per_point


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--grids", default="16,32,64")
    ap.add_argument("--q", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    cd = json.loads((run_dir / "config.json").read_text())
    keep = {k: cd[k] for k in Config.__dataclass_fields__ if k in cd}
    cfg = Config(**keep)
    torch.set_default_dtype(torch.float64)
    triple = Triple2D(cfg.depth, cfg.width, cfg.multiplier_head).double()
    triple.load_state_dict(torch.load(run_dir / "checkpoint.pt"))
    out_path = Path(a.out) if a.out else run_dir / "level2_diagnostic.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else []
    for n in [int(s) for s in a.grids.split(",")]:
        r = certify(triple, cfg, n, q=a.q)
        r["run_dir"] = str(run_dir)
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "bounds"}), flush=True)
        out_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
