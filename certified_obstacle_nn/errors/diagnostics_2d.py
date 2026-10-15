"""float64 diagnostics for 2-D-space benchmarks (disk, ellipse; A = I, b = c = 0):
every component of the computable estimator and every piece of the true weak-
norm error, for the network output v and for the projected output v_hat =
max(v, g), by composite tensor Gauss-Legendre on a uniform n^3 grid (q points
per direction), traces on their own n^2 spatial grid. Repeated at 2n; the
relative change of every quantity is recorded. Status ``diagnostic_float64``
(the integrands have kinks on {g = v} and on the exact interface, so the
convergence is algebraic, not spectral).

Terms (specification section 4, roadmap section 1):
  r: R_term = (Lambda/alpha)||R||^2, S_term = ||S||^2, init_mismatch = ||v(0) - u0||^2
  p: gradw = ||grad w||^2, w_term = kappa ||w||^2, wt_term = (Lambda/alpha)||w_t||^2, w0, wT
  c: compl = int mu (v - g)_+
  projected r: Rhat_term, Shat_term, init_mismatch_hat
True error (specification section 8, computed from the exact solution, never from
the network alone):
  grad = ||grad(u - v)||^2, flux = ||sigma - tau||^2, init = ||(u - v)(0)||^2, term = ||(u - v)(T)||^2,
  graph = ||(u_t + div sigma - lambda) - (v_t + div tau - mu)||^2  (div sigma = -Delta u, exact),
  E_true = sqrt(grad + init + term + flux + graph);  mult = ||lambda - mu||^2 (not controlled).
"""

from __future__ import annotations

import math

import numpy as np
import torch

STATUS = "diagnostic_float64"


def _rule(n: int, q: int):
    z, w = np.polynomial.legendre.leggauss(q)
    e = np.arange(n) / n
    h = 1.0 / n
    p = (e[:, None] + 0.5 * h * (z[None, :] + 1.0)).ravel()
    wt = np.tile(0.5 * h * w, n)
    return p, wt


def _net_fields(problem, triple, t, x, y):
    t = t.clone().requires_grad_(True)
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    X = torch.stack([t, x, y], -1)
    N = triple.state(X)[:, 0]
    Nt, Nx, Ny = torch.autograd.grad(N.sum(), (t, x, y))
    S = triple.flux(X)
    d1 = torch.autograd.grad(S[:, 0].sum(), x, retain_graph=True)[0]
    d2 = torch.autograd.grad(S[:, 1].sum(), y)[0]
    with torch.no_grad():
        M = triple.mult(X)[:, 0]
        B = problem.mask(x, y)
        Bx, By = problem.grad_mask(x, y)
        return {"v": B * N, "v_t": B * Nt, "v_x": Bx * N + B * Nx, "v_y": By * N + B * Ny,
                "s1": S[:, 0], "s2": S[:, 1], "div": d1 + d2, "mu": M**2}


def _volume(problem, triple, n, q, chunk=262144):
    p, w = _rule(n, q)
    P = np.stack(np.meshgrid(p, p, p, indexing="ij")).reshape(3, -1)
    W = np.einsum("i,j,k->ijk", w, w, w).ravel()
    La, kap = problem.Lambda / problem.alpha, problem.kappa
    acc = {}

    def add(name, val, wt):
        acc[name] = acc.get(name, 0.0) + float((val * wt).sum())

    for s in range(0, W.size, chunk):
        t, x, y = (torch.tensor(P[j, s:s + chunk]) for j in range(3))
        wt = torch.tensor(W[s:s + chunk])
        F = _net_fields(problem, triple, t, x, y)
        with torch.no_grad():
            f = problem.f(t, x, y)
            g = problem.g(t, x, y)
            gt = problem.g_t(t, x, y)
            gx, gy = problem.grad_g(t, x, y)
            R = f - F["v_t"] - F["div"] + F["mu"]
            S1, S2 = F["s1"] + F["v_x"], F["s2"] + F["v_y"]
            gam = g - F["v"]
            pos = (gam > 0).double()
            add("R_term", La * R**2, wt)
            add("S_term", S1**2 + S2**2, wt)
            add("gradw", pos * ((gx - F["v_x"]) ** 2 + (gy - F["v_y"]) ** 2), wt)
            add("w_term", kap * torch.clamp(gam, min=0) ** 2, wt)
            add("wt_term", La * pos * (gt - F["v_t"]) ** 2, wt)
            add("compl", F["mu"] * torch.clamp(-gam, min=0), wt)
            # projected output v_hat = v + w (w_t = pos gam_t, grad w = pos grad gam)
            vt_h = F["v_t"] + pos * (gt - F["v_t"])
            vx_h = F["v_x"] + pos * (gx - F["v_x"])
            vy_h = F["v_y"] + pos * (gy - F["v_y"])
            Rh = f - vt_h - F["div"] + F["mu"]
            add("Rhat_term", La * Rh**2, wt)
            add("Shat_term", (F["s1"] + vx_h) ** 2 + (F["s2"] + vy_h) ** 2, wt)
            # true error (exact solution)
            ux, uy = problem.grad_u_exact(t, x, y)
            sx, sy = problem.sigma_exact(t, x, y)
            lam = problem.lambda_exact(t, x, y)
            exact_graph = problem.u_t_exact(t, x, y) + problem.div_sigma_exact(t, x, y) - lam
            add("E_grad", (ux - F["v_x"]) ** 2 + (uy - F["v_y"]) ** 2, wt)
            add("E_flux", (sx - F["s1"]) ** 2 + (sy - F["s2"]) ** 2, wt)
            add("E_graph", (exact_graph - (F["v_t"] + F["div"] - F["mu"])) ** 2, wt)
            add("E_mult", (lam - F["mu"]) ** 2, wt)
            add("Eh_grad", (ux - vx_h) ** 2 + (uy - vy_h) ** 2, wt)
            add("Eh_graph", (exact_graph - (vt_h + F["div"] - F["mu"])) ** 2, wt)
            add("norm_grad_u", ux**2 + uy**2, wt)
            add("norm_sigma", sx**2 + sy**2, wt)
            add("graph_minus_R", (exact_graph - (F["v_t"] + F["div"] - F["mu"]) - R) ** 2, wt)
    return acc


def _traces(problem, triple, n, q):
    p, w = _rule(n, q)
    P = np.stack(np.meshgrid(p, p, indexing="ij")).reshape(2, -1)
    W = np.outer(w, w).ravel()
    x, y = torch.tensor(P[0]), torch.tensor(P[1])
    wt = torch.tensor(W)
    acc = {}
    with torch.no_grad():
        for tval, tag in ((0.0, "0"), (problem.T, "T")):
            t = torch.full_like(x, tval)
            v = problem.mask(x, y) * triple.state(torch.stack([t, x, y], -1))[:, 0]
            g = problem.g(t, x, y)
            u = problem.u_exact(t, x, y)
            vh = torch.maximum(v, g)
            acc["w" + tag] = float((torch.clamp(g - v, min=0) ** 2 * wt).sum())
            acc["E_" + tag] = float(((u - v) ** 2 * wt).sum())
            acc["Eh_" + tag] = float(((u - vh) ** 2 * wt).sum())
            if tval == 0.0:
                u0 = problem.u0(x, y)
                acc["init_mismatch"] = float(((v - u0) ** 2 * wt).sum())
                acc["init_mismatch_hat"] = float(((vh - u0) ** 2 * wt).sum())
    return acc


def diagnostics(problem, triple, n: int = 32, q: int = 3, n_trace: int = 256) -> dict:
    a = _volume(problem, triple, n, q)
    a.update(_traces(problem, triple, n_trace, q))
    out = dict(a)
    out["r2"] = a["R_term"] + a["S_term"] + a["init_mismatch"]
    out["p2"] = a["gradw"] + a["w_term"] + a["wt_term"] + a["w0"] + a["wT"]
    out["c2"] = a["compl"]
    out["r2_hat"] = a["Rhat_term"] + a["Shat_term"] + a["init_mismatch_hat"]
    out["E_true"] = math.sqrt(a["E_grad"] + a["E_0"] + a["E_T"] + a["E_flux"] + a["E_graph"])
    out["E_true_vhat"] = math.sqrt(a["Eh_grad"] + a["Eh_0"] + a["Eh_T"] + a["E_flux"] + a["Eh_graph"])
    out["norm_grad_u"] = math.sqrt(a["norm_grad_u"])
    out["norm_sigma"] = math.sqrt(a["norm_sigma"])
    out["method"] = f"composite Gauss {n}^3 cells q={q}; traces {n_trace}^2 q={q}"
    out["status"] = STATUS
    return out


def converged_diagnostics(problem, triple, n: int = 32, q: int = 3, n_trace: int = 256) -> dict:
    d1 = diagnostics(problem, triple, n, q, n_trace)
    d2 = diagnostics(problem, triple, 2 * n, q, 2 * n_trace)
    rel = {}
    for k, v in d2.items():
        if isinstance(v, float) and isinstance(d1.get(k), float) and v != 0.0:
            rel[k] = abs(v - d1[k]) / abs(v)
    d2["rel_change"] = rel
    d2["method"] += f" (vs {n}^3: rel. change recorded)"
    return d2
