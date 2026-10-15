"""v14 Stage 1: component evaluator (float64 diagnostic quadrature).

For a triple (v, tau, mu), with R = f - v_t - div tau - b.grad v - c v + mu,
S = A^{-1/2} tau + A^{1/2} grad v, w = (g - v)_+, returns every term separately:

    r-terms   R_term  = (Lambda/alpha)||R||^2      S_term = ||S||^2      init_mismatch = ||v(0)-u0||^2
    p-terms   gradw   = ||A^{1/2} grad w||^2       w_term = kappa ||w||^2
              wt_term = (Lambda/alpha)||w_t||^2   w0 = ||w(0)||^2      wT = ||w(T)||^2
    c-term    compl   = int_Q mu (v - g)_+

and the projected output v_hat = max(v, g) = v + w (same tau, mu):

    R_hat = R - w_t - b.grad w - c w,   S_hat = S + A^{1/2} grad w,
    init_mismatch_hat = ||v_hat(0) - u0||^2;   complementarity unchanged; penetration 0.

Integration: composite tensor Gauss-Legendre on a uniform n x n cell grid
with q points per direction; traces use their own spatial rule (never a
space-time volume). The densities have kinks on {g = v}, so convergence is
algebraic; ``converged_components`` repeats at 2n and records the relative
change of every term. Status: ``diagnostic_float64`` (not a validated
integral). Only first derivatives of the networks are used.

Scalar-coefficient (1-D space) problems only: A, b, c constants as in
problems/base_problem.py.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import torch

from certified_obstacle_nn.autograd.derivatives import eval_derivatives_detached, eval_gap_derivatives

STATUS = "diagnostic_float64"

R_KEYS = ("R_term", "S_term", "init_mismatch")
P_KEYS = ("gradw", "w_term", "wt_term", "w0", "wT")
C_KEYS = ("compl",)
RHAT_KEYS = ("Rhat_term", "Shat_term", "init_mismatch_hat")


def _composite(n: int, q: int, lo: float, hi: float):
    z, w = np.polynomial.legendre.leggauss(q)
    edges = np.linspace(lo, hi, n + 1)
    h = (hi - lo) / n
    pts = (edges[:-1, None] + 0.5 * h * (z[None, :] + 1.0)).ravel()
    wts = np.tile(0.5 * h * w, n)
    return pts, wts


def partition_hash(kind: str, **params) -> str:
    s = kind + ":" + ",".join(f"{k}={params[k]}" for k in sorted(params))
    return hashlib.sha256(s.encode()).hexdigest()


def _weights(problem):
    La = problem.Lambda / problem.alpha
    return {
        "R_term": La, "S_term": 1.0, "init_mismatch": 1.0,
        "gradw": float(problem.A), "w_term": problem.kappa, "wt_term": La, "w0": 1.0, "wT": 1.0,
        "compl": 1.0, "Rhat_term": La, "Shat_term": 1.0, "init_mismatch_hat": 1.0,
    }


def components(problem, triple, n: int = 256, q: int = 4, chunk_rows: int = 16) -> dict:
    """All component integrals for v and v_hat on one composite rule."""
    a_, b_ = problem.omega_bounds
    T = problem.T
    A, b, c = float(problem.A), float(problem.b), float(problem.c)
    sqA = math.sqrt(A)
    La = problem.Lambda / problem.alpha
    kap = problem.kappa
    tp, tw = _composite(n, q, 0.0, T)
    xp, xw = _composite(n, q, a_, b_)
    acc = {k: 0.0 for k in R_KEYS + P_KEYS + C_KEYS + RHAT_KEYS}
    parts = {k: [] for k in acc}
    X = torch.tensor(xp, dtype=torch.float64)
    XW = torch.tensor(xw, dtype=torch.float64)
    rows = chunk_rows * q
    for s in range(0, tp.size, rows):
        tt = torch.tensor(tp[s:s + rows], dtype=torch.float64)
        twt = torch.tensor(tw[s:s + rows], dtype=torch.float64)
        Tg, Xg = torch.meshgrid(tt, X, indexing="ij")
        Wg = torch.outer(twt, XW).reshape(-1, 1)
        t = Tg.reshape(-1, 1)
        x = Xg.reshape(-1, 1)
        v, tau, mu, v_t, v_x, tau_x = eval_derivatives_detached(triple.state_net, triple.flux_net, triple.multiplier_net, t, x)
        f = problem.f(t, x)
        g = problem.g(t, x)
        gam_t, gam_x = eval_gap_derivatives(problem, t, x, v_t, v_x)
        gam = g - v
        R = f - v_t - tau_x - b * v_x - c * v + mu
        S = tau / sqA + sqA * v_x
        pos = (gam > 0).to(torch.float64)
        w = torch.clamp(gam, min=0.0)
        w_t = pos * gam_t
        w_x = pos * gam_x
        Rhat = R - w_t - b * w_x - c * w
        Shat = S + sqA * w_x
        compl = mu * torch.clamp(-gam, min=0.0)
        for key, dens in (
            ("R_term", R**2), ("S_term", S**2), ("gradw", w_x**2), ("w_term", w**2), ("wt_term", w_t**2),
            ("compl", compl), ("Rhat_term", Rhat**2), ("Shat_term", Shat**2),
        ):
            parts[key].append(float((Wg * dens).sum()))
    for key in ("R_term", "S_term", "gradw", "w_term", "wt_term", "compl", "Rhat_term", "Shat_term"):
        acc[key] = math.fsum(parts[key])
    acc["R_term"] *= La
    acc["Rhat_term"] *= La
    acc["wt_term"] *= La
    acc["w_term"] *= kap
    acc["gradw"] *= A
    # traces: spatial rule only
    with torch.no_grad():
        xx = X.reshape(-1, 1)
        for tv, key in ((0.0, "w0"), (T, "wT")):
            tcol = torch.full_like(xx, tv)
            gam = problem.g(tcol, xx) - triple.state_net(tcol, xx)
            acc[key] = math.fsum((XW * torch.clamp(gam.reshape(-1), min=0.0) ** 2).tolist())
        z0 = torch.zeros_like(xx)
        v0 = triple.state_net(z0, xx)
        u0 = problem.u0(xx)
        g0 = problem.g(z0, xx)
        acc["init_mismatch"] = math.fsum((XW * ((v0 - u0).reshape(-1)) ** 2).tolist())
        acc["init_mismatch_hat"] = math.fsum((XW * ((torch.maximum(v0, g0) - u0).reshape(-1)) ** 2).tolist())
    return _assemble(problem, acc, method=f"composite Gauss-Legendre {n}x{n} cells, q={q}",
                     phash=partition_hash("uniform_gauss", n=n, q=q))


def _assemble(problem, acc: dict, *, method: str, phash: str, status: str = STATUS) -> dict:
    out = dict(acc)
    out["r2"] = math.fsum(acc[k] for k in R_KEYS)
    out["p2"] = math.fsum(acc[k] for k in P_KEYS)
    out["c2"] = acc["compl"]
    out["r2_hat"] = math.fsum(acc[k] for k in RHAT_KEYS)
    out["weights"] = _weights(problem)
    out["method"] = method
    out["partition_hash"] = phash
    out["status"] = status
    return out


def converged_components(problem, triple, n: int = 256, q: int = 4) -> dict:
    c1 = components(problem, triple, n, q)
    c2 = components(problem, triple, 2 * n, q)
    rel = {}
    for k in R_KEYS + P_KEYS + C_KEYS + RHAT_KEYS + ("r2", "p2", "c2", "r2_hat"):
        rel[k] = abs(c2[k] - c1[k]) / max(abs(c2[k]), 1e-300)
    c2["rel_change_vs_half_resolution"] = rel
    return c2
