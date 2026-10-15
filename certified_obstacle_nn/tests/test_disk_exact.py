"""v14 Stage 5: the expanding-disk benchmark (problems/disk.py). Symbolic
checks of the PDE and complementarity on both branches, boundary values,
constants, and consistency of the torch formulas with sympy."""

import numpy as np
import pytest
import sympy as sp
import torch

from certified_obstacle_nn.certification import reliability as rel
from certified_obstacle_nn.problems.disk import DiskProblem

t, x, y = sp.symbols("t x y", real=True)
K = sp.Integer(5)
Rt = sp.Rational(15, 100) + sp.Rational(20, 100) * t
r2 = (x - sp.Rational(1, 2)) ** 2 + (y - sp.Rational(1, 2)) ** 2
PSI = (Rt**2 - r2) / (2 * Rt)
H = sp.exp(t) - t - 1
B = 16 * x * (1 - x) * y * (1 - y)
G = -H * B


def lap(e):
    return sp.diff(e, x, 2) + sp.diff(e, y, 2)


U_IN = G + K * PSI**2  # psi > 0 (non-contact)
U_OUT = G  # psi < 0 (contact)
BR = 2 * PSI * sp.diff(PSI, t) - 2 * PSI * lap(PSI) + 2 - 2 * (sp.diff(PSI, x) ** 2 + sp.diff(PSI, y) ** 2)
F_IN = sp.diff(G, t) - lap(G) - 2 * K + K * BR
F_OUT = sp.diff(G, t) - lap(G) - 2 * K


def test_pde_and_complementarity_on_both_branches():
    assert sp.simplify(sp.diff(U_IN, t) - lap(U_IN) - F_IN) == 0  # lambda = 0 inside
    assert sp.simplify(sp.diff(U_OUT, t) - lap(U_OUT) - F_OUT - 2 * K) == 0  # lambda = 2 kappa outside
    assert sp.simplify(U_OUT - G) == 0  # contact: u = g
    assert sp.simplify(U_IN - G - K * PSI**2) == 0  # non-contact: u - g = kappa psi^2 >= 0


def test_f_and_u_continuous_across_circle():
    th = sp.symbols("theta", real=True)
    on = {x: sp.Rational(1, 2) + Rt * sp.cos(th), y: sp.Rational(1, 2) + Rt * sp.sin(th)}
    assert sp.simplify(PSI.subs(on)) == 0
    for e in (F_IN - F_OUT, U_IN - U_OUT, sp.diff(U_IN - U_OUT, x), sp.diff(U_IN - U_OUT, y), sp.diff(U_IN - U_OUT, t)):
        assert sp.simplify(e.subs(on)) == 0


def test_disk_inside_square_boundary_values():
    p = DiskProblem()
    s = torch.linspace(0, 1, 101, dtype=torch.float64)
    T, S = torch.meshgrid(s, s, indexing="ij")
    z, o = torch.zeros_like(S), torch.ones_like(S)
    assert 0.5 + 0.35 < 1  # R(1) = 0.35 < 1/2
    for X, Y in ((z, S), (o, S), (S, z), (S, o)):
        assert torch.all(p.u_exact(T, X, Y) == 0)
        assert torch.all(p.g(T, X, Y) <= 0)
    # A7 analogue: g(0,.) = 0, so v_hat(0) = max(v(0), 0)
    assert torch.all(p.g(torch.zeros_like(S), S, S.T) == 0)


@pytest.mark.parametrize("inside", [True, False])
def test_torch_matches_sympy(inside):
    p = DiskProblem()
    U, F = (U_IN, F_IN) if inside else (U_OUT, F_OUT)
    lam = sp.Integer(0) if inside else 2 * K
    ex = {"u": U, "u_t": sp.diff(U, t), "u_x": sp.diff(U, x), "u_y": sp.diff(U, y), "f": F, "lam": lam,
          "g": G, "g_t": sp.diff(G, t), "g_x": sp.diff(G, x), "g_y": sp.diff(G, y)}
    fn = {k: sp.lambdify((t, x, y), e, "numpy") for k, e in ex.items()}
    g = torch.Generator().manual_seed(0)
    T, X, Y = [torch.rand(20000, generator=g, dtype=torch.float64) for _ in range(3)]
    psi = (( 0.15 + 0.2 * T) ** 2 - (X - 0.5) ** 2 - (Y - 0.5) ** 2)
    sel = psi > 1e-9 if inside else psi < -1e-9
    T, X, Y = T[sel], X[sel], Y[sel]
    gu = p.grad_u_exact(T, X, Y)
    gg = p.grad_g(T, X, Y)
    got = {"u": p.u_exact(T, X, Y), "u_t": p.u_t_exact(T, X, Y), "u_x": gu[0], "u_y": gu[1], "f": p.f(T, X, Y),
           "lam": p.lambda_exact(T, X, Y), "g": p.g(T, X, Y), "g_t": p.g_t(T, X, Y), "g_x": gg[0], "g_y": gg[1]}
    a = [v.numpy() for v in (T, X, Y)]
    for k in ex:
        ref = np.broadcast_to(np.asarray(fn[k](*a), dtype=float), a[0].shape)
        assert np.allclose(got[k].numpy(), ref, rtol=1e-11, atol=1e-11), k


def test_constants():
    p = DiskProblem()
    assert (p.alpha, p.Lambda, p.kappa) == (1.0, 2.0, 0.5)
    assert rel.C_eq(p) == 1.0  # max{1/alpha, ||A||, 1, 2 alpha/Lambda} = 1
    assert rel.C_rel(p) == 36
