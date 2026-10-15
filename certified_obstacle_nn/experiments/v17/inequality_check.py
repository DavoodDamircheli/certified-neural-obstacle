"""v17 Stage 1 item 5 (roadmap R3): test the INEQUALITIES of Theorem 4.10 / Corollary 4.11 (manuscript v16,
eq:direct, eq:direct-weak = printed (43)) and of the primary gradient form, not only the code, on admissible
triples around the exact Stefan and expanding-disk solutions and on the stored Stefan seed-0 checkpoint.

For a triple (v, tau, mu) (v = 0 on the lateral boundary, mu >= 0), v_hat = max(v, g), e = u - v_hat,
e_sigma = sigma - tau, with A = I, alpha = 1, b = c = 0 and the EXACT Friedrichs constant c_F^2 = 1/(d pi^2):

  F = ||e_sigma||^2,  G = ||grad e||^2,  Z0 = ||e(0)||^2 = ||u0 - v_hat(0)||^2,  ZT = ||e(T)||^2,
  R2 = ||R_hat||^2 (R_hat = f - v_hat_t - div tau + mu),  S2 = ||S_hat||^2 (S_hat = tau + grad v_hat),
  C = int mu (v - g)_+,   D = S2 + Z0 + 2 C,   L = c_F^2 R2.

Checked:  (theta)    F + (1 - theta) G + ZT <= D + L/theta          theta in {1/4, 1/2, 1}      (eq:direct)
          (primary)  sqrt(G) <= sqrt(L) + sqrt(D + L)
          (weak)     G + Z0 + ZT + F + R2 <= max(1/(1-theta), 1)(D + L/theta) + Z0 + R2,  theta in {1/4,1/2,3/4}
          (graph)    R_hat = e_t + div e_sigma - (lambda - mu) pointwise (identity eq:direct-graph)
All integrals are float64 quadratures (diagnostic_float64). Each check is evaluated at two resolutions; it passes
only if LHS <= RHS at the fine resolution AND the relative gap exceeds twice the relative quadrature drift
between the two resolutions (otherwise it is reported as inconclusive, which also fails).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from certified_obstacle_nn.errors.true_error_v14 import _gl01, duffy_points

THETAS = (0.25, 0.5, 1.0)
WEAK_THETAS = (0.25, 0.5, 0.75)


def _grad(y, xs):
    return torch.autograd.grad(y.sum(), xs, create_graph=False, retain_graph=True, allow_unused=True)


# ------------------------------------------------------------------ quantities from point values
def _quantities(vol, tr0, trT):
    """vol: dict of arrays at volume points (+ 'w'); tr0/trT: dicts at trace points (+ 'w').
    Returns the integrals F, G, Z0, ZT, R2, S2, C and the max graph-identity defect."""
    w = vol["w"]
    vhat_pos = vol["v"] > vol["g"]  # v_hat = v there, = g elsewhere
    vh_t = np.where(vhat_pos, vol["v_t"], vol["g_t"])
    vh_grad = [np.where(vhat_pos, vol["v_x"][k], vol["g_x"][k]) for k in range(len(vol["v_x"]))]
    Rh = vol["f"] - vh_t - vol["div_tau"] + vol["mu"]
    Sh = [vol["tau"][k] + vh_grad[k] for k in range(len(vh_grad))]
    e_t = vol["u_t"] - vh_t
    div_es = vol["div_sigma"] - vol["div_tau"]
    graph_defect = np.abs(Rh - (e_t + div_es - (vol["lam"] - vol["mu"])))
    q = {
        "F": float(np.sum(w * sum((vol["sigma"][k] - vol["tau"][k]) ** 2 for k in range(len(Sh))))),
        "G": float(np.sum(w * sum((vol["u_x"][k] - vh_grad[k]) ** 2 for k in range(len(Sh))))),
        "R2": float(np.sum(w * Rh**2)),
        "S2": float(np.sum(w * sum(s**2 for s in Sh))),
        "C": float(np.sum(w * vol["mu"] * np.maximum(vol["v"] - vol["g"], 0.0))),
        "graph_defect_max": float(graph_defect.max()),
    }
    for nm, tr in (("Z0", tr0), ("ZT", trT)):
        vh = np.maximum(tr["v"], tr["g"])
        q[nm] = float(np.sum(tr["w"] * (tr["u"] - vh) ** 2))
    return q


def inequalities(q, cF2) -> dict:
    D = q["S2"] + q["Z0"] + 2 * q["C"]
    L = cF2 * q["R2"]
    out = {}
    for th in THETAS:
        out[f"theta_{th}"] = (q["F"] + (1 - th) * q["G"] + q["ZT"], D + L / th)
    out["primary"] = (math.sqrt(q["G"]), math.sqrt(L) + math.sqrt(D + L))
    for th in WEAK_THETAS:
        out[f"weak_{th}"] = (q["G"] + q["Z0"] + q["ZT"] + q["F"] + q["R2"],
                             max(1 / (1 - th), 1.0) * (D + L / th) + q["Z0"] + q["R2"])
    return out


def judge(ineq_coarse: dict, ineq_fine: dict) -> dict:
    res = {}
    for k, (l2, r2) in ineq_fine.items():
        l1, r1 = ineq_coarse[k]
        drift = max(abs(l2 - l1) / max(abs(l2), 1e-300), abs(r2 - r1) / max(abs(r2), 1e-300))
        gap = (r2 - l2) / r2 if r2 > 0 else (0.0 if l2 == 0 else -math.inf)
        res[k] = {"lhs": l2, "rhs": r2, "gap_rel": gap, "drift_rel": drift,
                  "holds": l2 <= r2, "conclusive": gap > 2 * drift, "ratio_rhs_lhs": r2 / l2 if l2 > 0 else math.inf}
    return res


# ------------------------------------------------------------------ Stefan (1-D space)
def stefan_exact(problem, t, x):
    """u, u_t, u_x, sigma, div sigma, lambda (closed form, problems/stefan.py; contact on t <= x)."""
    h = torch.exp(t) - t - 1
    act = t > x
    E = torch.exp(t - x)
    z = torch.zeros_like(t)
    u = problem.u_exact(t, x)
    u_t = -(torch.exp(t) - 1) * (1 - x) + torch.where(act, E - 1, z)
    u_x = h + torch.where(act, 1 - E, z)
    sig = problem.sigma_exact(t, x)
    div = torch.where(act, -E, z)
    return u, u_t, u_x, sig, div, problem.lambda_exact(t, x)


def stefan_fields(problem, trial, t, x):
    """trial(t, x) -> (v, tau, mu) torch tensors (autograd-differentiable in t, x)."""
    t = torch.as_tensor(t, dtype=torch.float64).clone().requires_grad_(True)
    x = torch.as_tensor(x, dtype=torch.float64).clone().requires_grad_(True)
    v, tau, mu = trial(t, x)
    v_t, v_x = _grad(v, (t, x))
    (tau_x,) = _grad(tau, (x,))
    with torch.no_grad():
        u, u_t, u_x, sig, div, lam = stefan_exact(problem, t, x)
        g = problem.g(t, x)
        g_t = -(torch.exp(t) - 1) * (1 - x)
        g_x = torch.exp(t) - t - 1
        f = problem.f(t, x)
    n = lambda a: a.detach().numpy()  # noqa: E731
    return {"v": n(v), "v_t": n(v_t), "v_x": [n(v_x)], "tau": [n(tau)], "div_tau": n(tau_x), "mu": n(mu),
            "u": n(u), "u_t": n(u_t), "u_x": [n(u_x)], "sigma": [n(sig)], "div_sigma": n(div), "lam": n(lam),
            "g": n(g), "g_t": n(g_t), "g_x": [n(g_x)], "f": n(f)}


def stefan_quantities(problem, trial, n: int) -> dict:
    vols = []
    for (t, x, w) in duffy_points(n):
        d = stefan_fields(problem, trial, t, x)
        d["w"] = w
        vols.append(d)
    vol = {k: (np.concatenate([d[k] for d in vols]) if not isinstance(vols[0][k], list)
               else [np.concatenate([d[k][j] for d in vols]) for j in range(len(vols[0][k]))]) for k in vols[0]}
    # traces: composite Gauss on 64 subintervals, n points each
    z, wz = _gl01(n)
    m = 64
    xs = ((np.arange(m)[:, None] + z[None, :]) / m).ravel()
    wx = np.tile(wz / m, m)
    trs = {}
    for nm, tv in (("Z0", 0.0), ("ZT", problem.T)):
        tt = torch.full((xs.size,), tv, dtype=torch.float64)
        xx = torch.as_tensor(xs)
        with torch.no_grad():
            v, _, _ = trial(tt, xx)
            u = torch.zeros_like(xx) if tv == 0.0 else problem.u_exact(tt, xx)
            g = problem.g(tt, xx)
        trs[nm] = {"v": v.numpy(), "u": u.numpy(), "g": g.numpy(), "w": wx}
    return _quantities(vol, trs["Z0"], trs["ZT"])


# ------------------------------------------------------------------ expanding disk (2-D space)
def disk_quantities(trial, n: int, q: int = 3, kappa: float = 5.0, chunk: int = 400000) -> dict:
    """Composite Gauss, n^3 cells x q^3 points on Q = (0,1)^3; traces n^2 x q^2."""
    from certified_obstacle_nn.experiments.expanding_disk_2d import exact_solution as ex

    z, wz = _gl01(q)
    p = ((np.arange(n)[:, None] + z[None, :]) / n).ravel()
    wp = np.tile(wz / n, n)
    T, X, Y = np.meshgrid(p, p, p, indexing="ij")
    W = (wp[:, None, None] * wp[None, :, None] * wp[None, None, :]).ravel()
    T, X, Y = T.ravel(), X.ravel(), Y.ravel()
    acc = {k: 0.0 for k in ("F", "G", "R2", "S2", "C")}
    gmax = 0.0
    for s in range(0, T.size, chunk):
        t = torch.as_tensor(T[s:s + chunk]).requires_grad_(True)
        x = torch.as_tensor(X[s:s + chunk]).requires_grad_(True)
        y = torch.as_tensor(Y[s:s + chunk]).requires_grad_(True)
        v, (tau1, tau2), mu = trial(t, x, y)
        v_t, v_x, v_y = _grad(v, (t, x, y))
        (d1,) = _grad(tau1, (x,))
        (d2,) = _grad(tau2, (y,))
        with torch.no_grad():
            u = ex.u_exact(t, x, y, kappa)
            u_t = ex.u_t_exact(t, x, y, kappa)
            ux, uy = ex.grad_u_exact(t, x, y, kappa)
            s1, s2 = ex.sigma_exact(t, x, y, kappa)
            divs = -ex.lap_u_exact(t, x, y, kappa)
            lam = ex.lambda_exact(t, x, y, kappa)
            g, g_t = ex.g(t, x, y), ex.g_t(t, x, y)
            gx, gy = ex.grad_g(t, x, y)
            f = ex.f_exact(t, x, y, kappa)
        nn = lambda a: a.detach().numpy()  # noqa: E731
        vol = {"v": nn(v), "v_t": nn(v_t), "v_x": [nn(v_x), nn(v_y)], "tau": [nn(tau1), nn(tau2)],
               "div_tau": nn(d1) + nn(d2), "mu": nn(mu), "u": nn(u), "u_t": nn(u_t), "u_x": [nn(ux), nn(uy)],
               "sigma": [nn(s1), nn(s2)], "div_sigma": nn(divs), "lam": nn(lam), "g": nn(g), "g_t": nn(g_t),
               "g_x": [nn(gx), nn(gy)], "f": nn(f), "w": W[s:s + chunk]}
        dummy = {"v": np.zeros(1), "u": np.zeros(1), "g": np.zeros(1), "w": np.zeros(1)}
        part = _quantities(vol, dummy, dummy)
        for k in acc:
            acc[k] += part[k]
        gmax = max(gmax, part["graph_defect_max"])
    P2X, P2Y = np.meshgrid(p, p, indexing="ij")
    W2 = (wp[:, None] * wp[None, :]).ravel()
    xx, yy = torch.as_tensor(P2X.ravel()), torch.as_tensor(P2Y.ravel())
    for nm, tv in (("Z0", 0.0), ("ZT", 1.0)):
        tt = torch.full_like(xx, tv)
        with torch.no_grad():
            v, _, _ = trial(tt, xx, yy)
            u = ex.u_exact(tt, xx, yy, kappa)
            vh = torch.maximum(v, ex.g(tt, xx, yy))
        acc[nm] = float(np.sum(W2 * (u - vh).numpy() ** 2))
    acc["graph_defect_max"] = gmax
    return acc
