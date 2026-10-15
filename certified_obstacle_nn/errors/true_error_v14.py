"""v14 true-error components (roadmap section 1, v14 prompt Stage 1).

Returns every component of the weak-norm error SEPARATELY, for any output
state (the raw network state ``v`` or the projection ``v_hat = max(v, g)``):

    grad   = ||d_x (u - v)||^2_{L2(Q)}
    flux   = ||sigma - tau||^2_{L2(Q)}
    init   = ||(u - v)(0)||^2_{L2(Omega)}
    term   = ||(u - v)(T)||^2_{L2(Omega)}
    graph  = ||d_t(u - v) + d_x(sigma - tau) - (lambda - mu)||^2_{L2(Q)}
    weak   = grad + init + term + flux + graph          (E_true^2 of specification section 8)
    mult   = ||lambda - mu||^2_{L2(Q)}                  (diagnostic only; not controlled)

plus the exact scales ||d_x u||, ||sigma||, ||u(0)||, ||u(T)||, ||u_t + sigma_x - lambda||.

Two independent quadrature rules (status ``diagnostic_float64`` -- neither is a
validated integral):

* ``global``: tensor Gauss-Legendre on Q. The graph integrand equals R_PDE (A1)
  and is smooth; grad/flux integrands have C^0 kinks at t = x, so they converge
  algebraically.
* ``duffy``: Q split along the exact Stefan interface t = x into the triangles
  {x < t} and {t < x}, each mapped from the unit square by a collapsed (Duffy)
  map, (s, r) -> (t, x) = (s, s r) resp. (s r, s), Jacobian s. Exact pieces are
  smooth on each closed triangle, so this converges spectrally for the
  unprojected output. Only valid when the interface is t = x on (0,1)^2.

The projected output has its own kink on {g = v}, which neither rule resolves;
its numbers converge algebraically and the self-check reports the change.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch

from certified_obstacle_nn.autograd.derivatives import eval_derivatives_detached

STATUS = "diagnostic_float64"


def _gl01(n: int):
    z, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (z + 1.0), 0.5 * w


def global_points(n: int, T: float, a: float, b: float):
    zt, wt = _gl01(n)
    zx, wx = _gl01(n)
    t = zt * T
    x = a + (b - a) * zx
    Tm, Xm = np.meshgrid(t, x, indexing="ij")
    W = np.outer(wt * T, wx * (b - a))
    return Tm.ravel(), Xm.ravel(), W.ravel()


def duffy_points(n: int):
    """Two collapsed rules on {x<t} and {t<x} in (0,1)^2; returns list of (t, x, w)."""
    z, w = _gl01(n)
    S, Rr = np.meshgrid(z, z, indexing="ij")
    W = np.outer(w, w) * S
    upper = (S.ravel(), (S * Rr).ravel(), W.ravel())  # x < t (non-contact for Stefan)
    lower = ((S * Rr).ravel(), S.ravel(), W.ravel())  # t < x (contact)
    return [upper, lower]


def _eval(fns, t, x):
    tt = torch.tensor(t, dtype=torch.float64).reshape(-1, 1)
    xx = torch.tensor(x, dtype=torch.float64).reshape(-1, 1)
    return [v.reshape(-1).numpy() for v in eval_derivatives_detached(*fns, tt, xx)]


def make_projected_state(problem, state_net):
    """v_hat = max(v, g). Returns a torch callable whose autograd derivatives
    are the a.e. derivatives of v_hat (branch chosen by torch.where; on the
    null set g == v the v-branch is used)."""

    def v_hat(t, x):
        v = state_net(t, x)
        g = problem.g(t, x)
        return torch.where(g > v, g, v)

    return v_hat


def _accumulate(problem, net_fns, ref_fns, pts):
    acc = dict(grad=0.0, flux=0.0, graph=0.0, mult=0.0, scale_grad=0.0, scale_flux=0.0, scale_graph=0.0)
    for t, x, w in pts:
        v, tau, mu, v_t, v_x, tau_x = _eval(net_fns, t, x)
        u, sg, lam, u_t, u_x, sg_x = _eval(ref_fns, t, x)
        acc["grad"] += math.fsum(w * (u_x - v_x) ** 2)
        acc["flux"] += math.fsum(w * (sg - tau) ** 2)
        acc["graph"] += math.fsum(w * ((u_t - v_t) + (sg_x - tau_x) - (lam - mu)) ** 2)
        acc["mult"] += math.fsum(w * (lam - mu) ** 2)
        acc["scale_grad"] += math.fsum(w * u_x**2)
        acc["scale_flux"] += math.fsum(w * sg**2)
        acc["scale_graph"] += math.fsum(w * (u_t + sg_x - lam) ** 2)
    return acc


def _traces(problem, state_fn, u_ref, n: int, split_x: float | None = None):
    a, b = problem.omega_bounds
    T = problem.T
    zx, wx = _gl01(n)
    out = dict(init=0.0, term=0.0, scale_init=0.0, scale_term=0.0)
    with torch.no_grad():
        x = torch.tensor(a + (b - a) * zx, dtype=torch.float64).reshape(-1, 1)
        w = wx * (b - a)
        for key, tv in (("init", 0.0), ("term", T)):
            tt = torch.full_like(x, tv)
            e = (u_ref(tt, x) - state_fn(tt, x)).reshape(-1).numpy()
            out[key] = math.fsum(w * e**2)
            out["scale_" + key] = math.fsum(w * u_ref(tt, x).reshape(-1).numpy() ** 2)
    return out


def true_error_components(problem, triple, n: int = 128, rule: str = "duffy", projected: bool = False,
                          u_ref=None, sigma_ref=None, lambda_ref=None) -> dict:
    """All weak-norm components for one output (raw or projected), one rule, one n."""
    u_ref = u_ref or problem.u_exact
    sigma_ref = sigma_ref or problem.sigma_exact
    lambda_ref = lambda_ref or problem.lambda_exact
    state = make_projected_state(problem, triple.state_net) if projected else triple.state_net
    net_fns = (state, triple.flux_net, triple.multiplier_net)
    ref_fns = (u_ref, sigma_ref, lambda_ref)
    if rule == "duffy":
        a, b = problem.omega_bounds
        if not (a == 0.0 and b == 1.0 and problem.T == 1.0):
            raise ValueError("duffy rule implemented only for the Stefan interface t = x on (0,1)^2")
        pts = duffy_points(n)
    elif rule == "global":
        a, b = problem.omega_bounds
        pts = [global_points(n, problem.T, a, b)]
    else:
        raise ValueError(rule)
    vol = _accumulate(problem, net_fns, ref_fns, pts)
    tr = _traces(problem, state, u_ref, 2 * n)
    out = {**vol, **tr}
    out["weak"] = out["grad"] + out["init"] + out["term"] + out["flux"] + out["graph"]
    out["E_true"] = math.sqrt(out["weak"])
    out["norm_grad_u"] = math.sqrt(out["scale_grad"])
    out["norm_sigma"] = math.sqrt(out["scale_flux"])
    out["rule"] = rule
    out["n"] = n
    out["projected"] = projected
    out["status"] = STATUS
    return out


def converged_true_error(problem, triple, n: int = 128, rule: str = "duffy", projected: bool = False,
                         rtol: float = 1e-3, **kw) -> dict:
    """Computes at n and 2n; records the per-component relative change. Raises
    if E_true changes by more than rtol (never loosened)."""
    c1 = true_error_components(problem, triple, n, rule, projected, **kw)
    c2 = true_error_components(problem, triple, 2 * n, rule, projected, **kw)
    rel = {}
    for k in ("grad", "flux", "init", "term", "graph", "mult", "weak"):
        rel[k] = abs(c2[k] - c1[k]) / max(abs(c2[k]), 1e-300)
    rel_E = abs(c2["E_true"] - c1["E_true"]) / c2["E_true"]
    c2["rel_change"] = rel
    c2["rel_change_E_true"] = rel_E
    if rel_E > rtol:
        raise RuntimeError(f"true-error self-check failed: rel change {rel_E:.3e} > {rtol:.0e} ({rule}, n={n}->{2*n}, projected={projected})")
    return c2
