"""Learned free boundary vs exact r = R(t), by radial ray marching from c0.

Criteria (positive "margin" = contact detected):
  primary   : lambda_psi >= kappa          (half the exact jump 2 kappa)   margin = lam - kappa
  secondary : u_theta - g <= eps_Gamma     (eps_Gamma = 1e-3 default)       margin = eps - (u - g)
Along each ray r in [0, 0.5] (step 1e-3) the first scan point with margin >= 0 is located and
refined by bisection (50 iterations) against the previous scan point. Fallbacks: contact
already at r = 0 -> r_theta = 0; never contact -> r_theta = 0.5 (counted in n_no_crossing).

E_FB (2D analogue of errors/free_boundary.py's ||s_theta - s||_{L2(0,T)}):
  E_FB^2 = int_0^T (1/2pi) int_0^{2pi} (r_theta(theta,t) - R(t))^2 dtheta dt,
with Gauss-Legendre (20 nodes) in t and the 360 equispaced angles, primary criterion.
"""

import numpy as np
import torch

from certified_obstacle_nn.experiments.expanding_disk_2d import exact_solution as ex

R_MAX = 0.5
STEP = 1e-3
N_ANGLES = 360
N_BISECT = 50
TIMES = (0.25, 0.5, 0.75, 1.0)


def _margin(triple, criterion, t, x, y, kappa, eps):
    if criterion == "lambda":
        return triple.lam(t, x, y) - kappa
    return eps - (triple.u(t, x, y) - ex.g(t, x, y))


@torch.no_grad()
def interface_radii(triple, t_val: float, criterion: str, kappa: float, eps: float) -> tuple[np.ndarray, int]:
    th = torch.arange(N_ANGLES, dtype=torch.float64) * (2 * np.pi / N_ANGLES)
    rs = torch.arange(0.0, R_MAX + 0.5 * STEP, STEP, dtype=torch.float64)
    c, s = torch.cos(th)[:, None], torch.sin(th)[:, None]
    X = ex.C0 + rs[None, :] * c
    Y = ex.C0 + rs[None, :] * s
    Tt = torch.full_like(X, t_val)
    m = _margin(triple, criterion, Tt.reshape(-1, 1), X.reshape(-1, 1), Y.reshape(-1, 1), kappa, eps)
    m = m.reshape(N_ANGLES, -1)
    hit = m >= 0
    any_hit = hit.any(dim=1)
    first = torch.where(any_hit, hit.float().argmax(dim=1), torch.full((N_ANGLES,), -1))
    radii = np.full(N_ANGLES, R_MAX)
    lo = torch.zeros(N_ANGLES, dtype=torch.float64)
    hi = torch.zeros(N_ANGLES, dtype=torch.float64)
    need = (first > 0)
    radii[(first == 0).numpy()] = 0.0
    idx = first.clamp(min=1)
    lo[:] = rs[idx - 1]
    hi[:] = rs[idx]
    cc, ss = torch.cos(th), torch.sin(th)
    for _ in range(N_BISECT):
        mid = 0.5 * (lo + hi)
        mm = _margin(triple, criterion, torch.full((N_ANGLES, 1), t_val, dtype=torch.float64),
                     (ex.C0 + mid * cc)[:, None], (ex.C0 + mid * ss)[:, None], kappa, eps).squeeze(1)
        inside = mm >= 0
        hi = torch.where(inside, mid, hi)
        lo = torch.where(inside, lo, mid)
    radii[need.numpy()] = (0.5 * (lo + hi)).numpy()[need.numpy()]
    return radii, int((~any_hit).sum().item())


def compute_interface_metrics(triple, kappa: float, eps: float) -> dict:
    triple.eval()
    out: dict = {}
    for crit in ("lambda", "gap"):
        means, maxs = [], []
        for t_val in TIMES:
            radii, n_miss = interface_radii(triple, t_val, crit, kappa, eps)
            err = np.abs(radii - (ex.R0 + ex.R_SPEED * t_val))
            out[f"iface_{crit}_t{t_val:g}_mean"] = float(err.mean())
            out[f"iface_{crit}_t{t_val:g}_max"] = float(err.max())
            out[f"iface_{crit}_t{t_val:g}_no_crossing"] = n_miss
            means.append(err.mean())
            maxs.append(err.max())
        out[f"iface_{crit}_mean"] = float(np.mean(means))
        out[f"iface_{crit}_max"] = float(np.mean(maxs))
    nodes, weights = np.polynomial.legendre.leggauss(20)
    tq = 0.5 * (nodes + 1.0)
    wq = 0.5 * weights
    acc = 0.0
    for tv, wv in zip(tq, wq):
        radii, _ = interface_radii(triple, float(tv), "lambda", kappa, eps)
        acc += wv * np.mean((radii - (ex.R0 + ex.R_SPEED * tv)) ** 2)
    out["E_FB"] = float(np.sqrt(acc))
    return out
