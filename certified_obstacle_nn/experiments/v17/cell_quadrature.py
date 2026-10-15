"""Cellwise float64 quadrature (diagnostic_float64) of the densities whose validated upper sums v17 reports,
ON THE SAME PARTITION as the certificate: q Gauss points per axis in every cell, summed per cell. Used for the
slack U / I of every per-term upper sum, split by cell class (sign-resolved vs sign-unresolved), roadmap R3
Stages 3 and 5. Also returns the diagnostic true-error pieces of the projected output.

Densities per point (A = I, b = c = 0; v_hat = max(v, g), branch by the sign of g - v at the point):
  Rh2 = R_hat^2, Sh2 = |S_hat|^2, C = mu (v - g)_+, R02 = R^2, S02 = |S|^2,
  Eg = |grad(u - v_hat)|^2, Ef = |sigma - tau|^2, Ng = |grad u|^2.
Per cell also the max of each density over the q^D points ("sampled sup", a LOWER bound of sup_K): it splits the
slack U / I of the sup-based upper sum into the part intrinsic to the majorant sum |K| sup_K Phi (>= sum |K| sampled
sup / I) and the enclosure overestimate (<= U / sum |K| sampled sup).
Trace cells (t = 0): I0 = (u0 - v_hat(0))^2, (t = T): ZT = (u(T) - v_hat(T))^2.
Data derivatives g_t, grad g and the network derivatives are formed by autograd (first derivatives only).
"""

from __future__ import annotations

import numpy as np
import torch

from certified_obstacle_nn.errors.true_error_v14 import _gl01

VOL = ("Rh2", "Sh2", "C", "R02", "S02", "Eg", "Ef", "Ng")


def _pts(lo, hi, q):
    """Tensor-product Gauss points (q per axis) in every box; ordered cell-major. Returns (P (D, n q^D), W)."""
    z, w = _gl01(q)
    Dm = lo.shape[0]
    grids = np.meshgrid(*([np.arange(q)] * Dm), indexing="ij")
    idx = np.stack([g.ravel() for g in grids])  # (D, q^D)
    zz = z[idx]  # (D, q^D)
    ww = np.prod(w[idx], axis=0)  # (q^D,)
    P = lo[:, :, None] + (hi - lo)[:, :, None] * zz[:, None, :]  # (D, N, q^D)
    vol = np.prod(hi - lo, axis=0)
    W = vol[:, None] * ww[None, :]
    return P.reshape(Dm, -1), W.ravel()


def _grad(y, xs):
    return torch.autograd.grad(y.sum(), xs, retain_graph=True, allow_unused=True)


def _fields(kind, problem, triple, cols):
    """Point values; cols = list of 1-D tensors (t, x[, y]) with requires_grad."""
    if kind == "stefan":
        t, x = cols
        v, tau, mu = (a.reshape(-1) for a in triple(t.reshape(-1, 1), x.reshape(-1, 1)))
        taus = [tau]
        divt = _grad(tau, (x,))[0]
        g = problem.g(t, x)
        f = problem.f(t, x)
        u = problem.u_exact(t, x)
        sig = [problem.sigma_exact(t, x)]
    else:
        t, x, y = cols
        X = torch.stack([t, x, y], -1)
        v = problem.mask(x, y) * triple.state(X)[:, 0]
        S = triple.flux(X)
        taus = [S[:, 0], S[:, 1]]
        divt = _grad(S[:, 0], (x,))[0] + _grad(S[:, 1], (y,))[0]
        mu = triple.mult(X)[:, 0] ** 2
        g = problem.g(t, x, y)
        f = problem.f(t, x, y)
        u = problem.u_exact(t, x, y)
        sig = list(problem.sigma_exact(t, x, y))
    dv = _grad(v, cols)
    dg = _grad(g, cols)
    du = _grad(u, cols[1:])
    with torch.no_grad():
        pos = (v > g)
        vh_t = torch.where(pos, dv[0], dg[0])
        vh_x = [torch.where(pos, dv[k], dg[k]) for k in range(1, len(cols))]
        Rh = f - vh_t - divt + mu
        R0 = f - dv[0] - divt + mu
        out = {
            "Rh2": Rh**2,
            "Sh2": sum((taus[k] + vh_x[k]) ** 2 for k in range(len(taus))),
            "C": mu * torch.clamp(v - g, min=0),
            "R02": R0**2,
            "S02": sum((taus[k] + dv[k + 1]) ** 2 for k in range(len(taus))),
            "Eg": sum((du[k] - vh_x[k]) ** 2 for k in range(len(taus))),
            "Ef": sum((sig[k] - taus[k]) ** 2 for k in range(len(taus))),
            "Ng": sum(du[k] ** 2 for k in range(len(taus))),
        }
    return {k: v_.detach().numpy() for k, v_ in out.items()}


def cell_integrals(kind, problem, triple, lo, hi, q=2, chunk=200_000) -> dict:
    """Per-cell float64 integrals of the VOL densities on the boxes [lo, hi] (D x N, D = 2 Stefan, 3 2-D)."""
    n = lo.shape[1]
    nq = q ** lo.shape[0]
    out = {k: np.zeros(n) for k in VOL}
    out.update({"sup_" + k: np.zeros(n) for k in VOL})  # max over the q^D points: a LOWER bound of sup_K
    cells_per = max(1, chunk // nq)
    for s in range(0, n, cells_per):
        e = min(n, s + cells_per)
        P, W = _pts(lo[:, s:e], hi[:, s:e], q)
        cols = [torch.as_tensor(P[j]).clone().requires_grad_(True) for j in range(P.shape[0])]
        F = _fields(kind, problem, triple, cols)
        for k in VOL:
            out[k][s:e] = (F[k] * W).reshape(e - s, nq).sum(1)
            out["sup_" + k][s:e] = F[k].reshape(e - s, nq).max(1)
    return out


def trace_integrals(kind, problem, triple, lo, hi, tval, q=3) -> dict:
    """Per-trace-cell float64 integrals at t = tval: I0 = (u0 - v_hat(0))^2 (t = 0) or ZT = (u - v_hat)^2."""
    P, W = _pts(lo, hi, q)
    nq = q ** lo.shape[0]
    with torch.no_grad():
        xs = [torch.as_tensor(P[j]) for j in range(P.shape[0])]
        t = torch.full_like(xs[0], tval)
        if kind == "stefan":
            v = triple(t.reshape(-1, 1), xs[0].reshape(-1, 1))[0].reshape(-1)
            g = problem.g(t, xs[0])
            ref = problem.u0(xs[0]) if tval == 0.0 else problem.u_exact(t, xs[0])
        else:
            x, y = xs
            v = problem.mask(x, y) * triple.state(torch.stack([t, x, y], -1))[:, 0]
            g = problem.g(t, x, y)
            ref = problem.u0(x, y) if tval == 0.0 else problem.u_exact(t, x, y)
        d = (ref - torch.maximum(v, g)) ** 2
    return {("I0" if tval == 0.0 else "ZT"): (d.numpy() * W).reshape(lo.shape[1], nq).sum(1)}
