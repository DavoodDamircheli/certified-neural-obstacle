"""v14 Stage 6: the expanding-ellipse obstacle benchmark (problems/ellipse.py).

Symbolic checks (sympy) of the construction in roadmap section 6, and
consistency of the torch closed forms with the symbolic expressions on both
branches (zeta < 0 contact, zeta >= 0 non-contact)."""

import math

import numpy as np
import pytest
import sympy as sp
import torch

from certified_obstacle_nn.problems.ellipse import A0, A1, C1, C2, D0, D1, EllipseProblem

t, x1, x2 = sp.symbols("t x1 x2", real=True)
K = sp.Rational(1, 100)
a = sp.Rational(12, 100) + sp.Rational(10, 100) * t
d = sp.Rational(8, 100) + sp.Rational(7, 100) * t
c1, c2 = sp.Rational(43, 100), sp.Rational(57, 100)
Bm = 16 * x1 * (1 - x1) * x2 * (1 - x2)
UB = Bm * (sp.Rational(10, 100) + sp.Rational(5, 100) * t) * (1 + sp.Rational(2, 10) * x1 - sp.Rational(1, 10) * x2)
ZETA = (x1 - c1) ** 2 / a**2 + (x2 - c2) ** 2 / d**2 - 1


def lap(e):
    return sp.diff(e, x1, 2) + sp.diff(e, x2, 2)


GRADZ2 = sp.diff(ZETA, x1) ** 2 + sp.diff(ZETA, x2) ** 2
# branch formulas: contact (zeta < 0) and non-contact (zeta >= 0)
U_C = UB - K * ZETA**2
U_N = UB
F_C = sp.diff(UB, t) - lap(UB) - 2 * K * ZETA * (sp.diff(ZETA, t) - lap(ZETA))
F_N = sp.diff(UB, t) - lap(UB)
LAM_C = 2 * K * GRADZ2
G = UB - K * ZETA**2


def test_pde_and_complementarity_on_both_branches():
    # contact: u_t - Delta u - f - lambda == 0 and u == g
    assert sp.simplify(sp.diff(U_C, t) - lap(U_C) - F_C - LAM_C) == 0
    assert sp.simplify(U_C - G) == 0
    # non-contact: u_t - Delta u - f == 0 with lambda = 0, u - g = kappa0 zeta^2 >= 0
    assert sp.simplify(sp.diff(U_N, t) - lap(U_N) - F_N) == 0
    assert sp.simplify(U_N - G - K * ZETA**2) == 0


def test_zeta_derivative_formulas():
    zt = -2 * A1 * (x1 - c1) ** 2 / a**3 - 2 * D1 * (x2 - c2) ** 2 / d**3
    assert sp.simplify(sp.diff(ZETA, t) - sp.nsimplify(zt)) == 0
    assert sp.simplify(lap(ZETA) - (2 / a**2 + 2 / d**2)) == 0
    assert sp.simplify(sp.diff(ZETA, x1) - 2 * (x1 - c1) / a**2) == 0
    assert sp.simplify(sp.diff(ZETA, x2) - 2 * (x2 - c2) / d**2) == 0


def test_ellipse_strictly_inside_square_and_zeta_positive_on_boundary():
    # the ellipse's bounding box at time t is [c1 -+ a(t)] x [c2 -+ d(t)]; a, d increase in t
    a1, d1 = A0 + A1, D0 + D1
    assert 0 < C1 - a1 and C1 + a1 < 1 and 0 < C2 - d1 and C2 + d1 < 1
    # on each side, zeta >= (distance from the centre)^2 / a^2 - 1 > 0 (symbolic lower bound)
    for side, dist, ax in ((x1, C1, a), (x1, 1 - C1, a), (x2, C2, d), (x2, 1 - C2, d)):
        lb = sp.Rational(str(dist)) ** 2 / ax**2 - 1
        assert all(float(lb.subs(t, s)) > 0 for s in np.linspace(0, 1, 101))
        # lb is decreasing in t (ax increasing), so its minimum is at t = 1
        assert float(lb.subs(t, 1)) > 0


def test_lateral_boundary_values():
    p = EllipseProblem()
    s = torch.linspace(0, 1, 101, dtype=torch.float64)
    T, S = torch.meshgrid(s, s, indexing="ij")
    z, o = torch.zeros_like(S), torch.ones_like(S)
    for X1, X2 in ((z, S), (o, S), (S, z), (S, o)):
        assert torch.all(p.u_exact(T, X1, X2) == 0)
        assert torch.all(p.g(T, X1, X2) <= 0)
        # A7 analogue: v_hat = max(v, g) with v = 0 on the boundary (masked state) is 0 there
        assert torch.all(torch.maximum(torch.zeros_like(S), p.g(T, X1, X2)) == 0)


def _pts(n=4000, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.rand(n, generator=g, dtype=torch.float64) for _ in range(3)]


def test_gap_and_multiplier_sign():
    p = EllipseProblem()
    T, X1, X2 = _pts()
    gap = p.u_exact(T, X1, X2) - p.g(T, X1, X2)
    zp = torch.clamp(p.zeta(T, X1, X2), min=0)
    assert torch.allclose(gap, p.kappa0 * zp**2, atol=1e-15, rtol=0)
    assert torch.all(gap >= 0)
    lam = p.lambda_exact(T, X1, X2)
    assert torch.all(lam >= 0)
    assert torch.all(lam * gap == 0)  # complementarity pointwise
    # lambda = 0 at the ellipse centre (grad zeta = 0 there)
    assert p.lambda_exact(torch.tensor([0.5]), torch.tensor([C1]), torch.tensor([C2])).item() == 0.0


@pytest.mark.parametrize("branch", ["contact", "noncontact"])
def test_torch_formulas_match_sympy(branch):
    p = EllipseProblem()
    U, F, LAM = (U_C, F_C, LAM_C) if branch == "contact" else (U_N, F_N, sp.Integer(0))
    exprs = {
        "u": U, "u_t": sp.diff(U, t), "u_x1": sp.diff(U, x1), "u_x2": sp.diff(U, x2), "lap_u": lap(U),
        "f": F, "lam": LAM, "g": G, "g_t": sp.diff(G, t), "g_x1": sp.diff(G, x1), "g_x2": sp.diff(G, x2),
        "zeta_t": sp.diff(ZETA, t),
    }
    fns = {k: sp.lambdify((t, x1, x2), e, "numpy") for k, e in exprs.items()}
    T, X1, X2 = _pts(20000, 1)
    z = p.zeta(T, X1, X2)
    sel = (z < 0) if branch == "contact" else (z >= 0)
    T, X1, X2 = T[sel], X1[sel], X2[sel]
    assert T.numel() > 100
    gu = p.grad_u_exact(T, X1, X2)
    gg = p.grad_g(T, X1, X2)
    got = {
        "u": p.u_exact(T, X1, X2), "u_t": p.u_t_exact(T, X1, X2), "u_x1": gu[0], "u_x2": gu[1],
        "lap_u": p.lap_u_exact(T, X1, X2), "f": p.f(T, X1, X2), "lam": p.lambda_exact(T, X1, X2),
        "g": p.g(T, X1, X2), "g_t": p.g_t(T, X1, X2), "g_x1": gg[0], "g_x2": gg[1],
        "zeta_t": p.zeta_t(T, X1, X2),
    }
    tn, a1n, a2n = T.numpy(), X1.numpy(), X2.numpy()
    for k, fn in fns.items():
        ref = np.broadcast_to(np.asarray(fn(tn, a1n, a2n), dtype=float), tn.shape)
        assert np.allclose(got[k].numpy(), ref, rtol=1e-12, atol=1e-12), k
    # sigma = -grad u, div sigma = -Delta u
    s1, s2 = p.sigma_exact(T, X1, X2)
    assert torch.equal(s1, -gu[0]) and torch.equal(s2, -gu[1])
    assert torch.equal(p.div_sigma_exact(T, X1, X2), -p.lap_u_exact(T, X1, X2))


def test_continuity_across_interface():
    """u, grad u, u_t and f are continuous across zeta = 0 (the correction and
    its first derivatives carry a factor zeta); lambda jumps (allowed)."""
    # symbolic: the contact-minus-noncontact differences vanish on zeta = 0
    th = sp.symbols("theta", real=True)
    on_ellipse = {x1: c1 + a * sp.cos(th), x2: c2 + d * sp.sin(th)}
    assert sp.simplify(ZETA.subs(on_ellipse)) == 0
    for diff in (U_C - U_N, sp.diff(U_C - U_N, x1), sp.diff(U_C - U_N, x2), sp.diff(U_C - U_N, t), F_C - F_N):
        assert sp.simplify(diff.subs(on_ellipse)) == 0
    # numeric: points on the exact ellipse, approached from both sides
    p = EllipseProblem()
    for s in (0.0, 0.3, 1.0):
        for th in np.linspace(0, 2 * math.pi, 13):
            e1, e2 = p.ellipse_boundary(s, th)
            for eps in (1e-6,):
                tin = torch.tensor([s, s], dtype=torch.float64)
                r = torch.tensor([1 - eps, 1 + eps], dtype=torch.float64)
                X1 = C1 + (e1 - C1) * r
                X2 = C2 + (e2 - C2) * r
                z = p.zeta(tin, X1, X2)
                assert z[0] < 0 < z[1]
                for fn in (p.u_exact, p.u_t_exact, p.f):
                    v = fn(tin, X1, X2)
                    assert abs(v[0] - v[1]) < 1e-4
                gu = p.grad_u_exact(tin, X1, X2)
                assert all(abs(c[0] - c[1]) < 1e-3 for c in gu)


def test_signal_magnitudes():
    """Values recorded in PROGRESS.md (grid 201^3 and exact contact volume)."""
    p = EllipseProblem()
    s = torch.linspace(0, 1, 81, dtype=torch.float64)
    T, X1, X2 = torch.meshgrid(s, s, s, indexing="ij")
    assert abs(p.ub(T, X1, X2).max().item() - 0.158) < 2e-3
    # max lambda = 2 kappa0 (2/d(0))^2 = 12.5 (sup over the open contact set)
    assert p.lambda_exact(T, X1, X2).max().item() <= 12.5 + 1e-12
    assert abs(2 * 0.01 * (2 / D0) ** 2 - 12.5) < 1e-12
    # contact volume = int_0^1 pi a(t) d(t) dt (exact)
    vol = math.pi * (A0 * D0 + (A0 * D1 + A1 * D0) / 2 + A1 * D1 / 3)
    assert abs(vol - 0.06325) < 1e-4
    assert (p.kappa0 * torch.clamp(-p.zeta(T, X1, X2), min=0) ** 2).max().item() <= p.kappa0 + 1e-15


def test_constants():
    p = EllipseProblem()
    assert (p.alpha, p.Lambda, p.kappa) == (1.0, 2.0, 0.5)
    from certified_obstacle_nn.certification import reliability as rel

    assert rel.C_eq(p) == 1.0
    assert rel.C_rel(p) == 36


def test_interface_diagnostic_on_exact_solution():
    """Gap classifier applied to the exact solution: no missing/multiple rays and
    the declared tolerance's own bias rho* - 1 ~ sqrt(tol/kappa0)/2 (zeta ~ 2(rho-1))."""
    from certified_obstacle_nn.errors.ellipse_interface import interface_diagnostic

    p = EllipseProblem()
    for tol in (1e-4, 1e-6):
        _, agg, meta = interface_diagnostic(p, None, tol=tol, times=(0.0, 1.0), n_angles=36)
        row = agg[(agg["who"] == "exact") & (agg["t"] == "all")].iloc[0]
        assert row["missing"] == 0 and row["multiple"] == 0
        assert abs(row["mean_rho_star"] - 1 - math.sqrt(tol / p.kappa0) / 2) < 0.01
        assert meta["status"] == "diagnostic_float64"
