"""Prompt section 6 / TODO Task W: soundness tests of the Level-2 enclosure engine.

A violation in tests 1-5 is a soundness bug. Sampling success is necessary, not proof (the proof
is docs/history/LEVEL2_LOCAL_THEORY_NOTES.md section 4).
"""

import math

import pytest
import torch

from certified_obstacle_nn.certification.level2.aff import Aff
from certified_obstacle_nn.certification.level2.fields import fields_at, fields_on_boxes
from certified_obstacle_nn.certification.level2.netenc import mlp_enclose
from certified_obstacle_nn.certification.level2.pointwise import mlp_jet
from certified_obstacle_nn.certification.level2.problems import make_problem
from certified_obstacle_nn.certification.level2.upper_sum import (METHODS, enclose_cells, phi_bounds,
                                                                   upper_sum_uniform)
from certified_obstacle_nn.models.mlp import MLP
from certified_obstacle_nn.tests.stored_data import needs

torch.set_default_dtype(torch.float64)
TOL = 1e-10  # slack allowed for float64 sampling of the exact function


def _contained(v, lo, hi, scale=1.0):
    pad = TOL * (scale + lo.abs() + hi.abs())
    return bool(((v >= lo - pad) & (v <= hi + pad)).all())


@pytest.fixture(scope="module")
def cases():
    from certified_obstacle_nn.experiments.level2.common import load_case
    out = {}
    for c in ("1d", "2d_base"):
        model, meta = load_case(c)
        out[c] = (make_problem(model, meta), meta)
    return out


# ---------------------------------------------------------------- 1. synthetic linear network
@pytest.mark.parametrize("affine", [True, False])
def test_linear_network_exact(affine):
    torch.manual_seed(0)
    net = MLP(3, 2, hidden_width=8, hidden_depth=0).double()
    W, b = net.layers[0].weight.detach(), net.layers[0].bias.detach()
    lo = torch.rand(50, 3)
    hi = lo + torch.rand(50, 3) * 0.3
    val, jac, _ = mlp_enclose(net, Aff.from_box(lo, hi, affine), order=1)
    c, r = 0.5 * (lo + hi), 0.5 * (hi - lo)
    ex_lo, ex_hi = c @ W.T + b - r @ W.abs().T, c @ W.T + b + r @ W.abs().T
    vlo, vhi = val.range()
    assert bool((vlo <= ex_lo).all() and (vhi >= ex_hi).all())
    assert float((ex_lo - vlo).abs().max()) < 1e-9 and float((vhi - ex_hi).abs().max()) < 1e-9
    jlo, jhi = jac.range()
    assert float((jlo - W).abs().max()) < 1e-9 and float((jhi - W).abs().max()) < 1e-9


# ---------------------------------------------------------------- 2. one hidden layer, 1D input
@pytest.mark.parametrize("affine", [True, False])
def test_one_hidden_layer_tanh_dense(affine):
    torch.manual_seed(1)
    net = MLP(1, 1, hidden_width=16, hidden_depth=1).double()
    for p in net.parameters():
        p.data.mul_(3.0)
    for w in (1.0, 0.1, 0.01):
        lo = torch.rand(20, 1) * 2 - 1
        hi = lo + w
        val, jac, hess = mlp_enclose(net, Aff.from_box(lo, hi, affine), order=2)
        s = lo + (hi - lo) * torch.linspace(0, 1, 100_000)[None, :]  # [20, 1e5]
        v, j, h = mlp_jet(net, s.reshape(-1, 1), order=2)
        for enc, ex in ((val, v), (jac, j), (hess, h)):
            elo, ehi = enc.range()
            ex = ex.reshape(20, 100_000, -1)
            assert _contained(ex, elo.reshape(20, 1, -1), ehi.reshape(20, 1, -1))


# ---------------------------------------------------------------- pointwise jets vs autograd
def test_pointwise_jet_matches_autograd():
    torch.manual_seed(2)
    net = MLP(3, 2, hidden_width=10, hidden_depth=3).double()
    P = torch.rand(40, 3, requires_grad=True)
    v, j, h = mlp_jet(net, P.detach(), order=2)
    out = net(P)
    assert torch.allclose(v, out.detach(), atol=1e-13)
    for o in range(2):
        (g,) = torch.autograd.grad(out[:, o].sum(), P, create_graph=True)
        assert torch.allclose(j[:, o, :], g.detach(), atol=1e-12)
        for k in range(3):
            (hk,) = torch.autograd.grad(g[:, k].sum(), P, retain_graph=True)
            assert torch.allclose(h[:, o, k, :], hk, atol=1e-11)


@needs('results/expanding_disk_2d/stage3_6x64_N20000_s0_k5')
def test_field_jets_match_autograd(cases):
    """Jet gradients of R, gamma, Rsig (incl. network Hessians) vs autograd of the exact fields."""
    for name, (prob, _) in cases.items():
        P = torch.rand(30, prob.m, generator=torch.Generator().manual_seed(3))
        J = fields_at(prob, P, grad=True)
        Pg = P.clone().requires_grad_(True)
        F = fields_at_autograd(prob, Pg)
        for k in ("R", "gamma", "gamma_t"):
            (g,) = torch.autograd.grad(F[k].sum(), Pg, retain_graph=True)
            assert torch.allclose(J[k].grad, g, atol=1e-9, rtol=1e-9), (name, k)


def fields_at_autograd(prob, P):
    """Same fields computed with torch.autograd (test-only reference, create_graph for 2nd order)."""
    def net_grad(net, out_idx):
        y = net(P)[:, out_idx]
        (g,) = torch.autograd.grad(y.sum(), P, create_graph=True)
        return y, g
    N, gN = net_grad(prob.state, 0)
    S, dS = [], []
    for i in range(prob.d):
        s, gs = net_grad(prob.flux, i)
        S.append(s)
        dS.append(gs[:, 1 + i])
    M = prob.mult(P)[:, 0]
    net = {"N": N, "N_t": gN[:, 0], "N_x": [gN[:, 1 + i] for i in range(prob.d)], "S": S, "dS": dS, "M": M}
    return prob.fields_from_net([P[:, j] for j in range(prob.m)], net)


# ---------------------------------------------------------------- 3./4. frozen checkpoints, dense sampling
def _sample_check(prob, n_cells, widths, n_samp, methods, seed):
    g = torch.Generator().manual_seed(seed)
    viol = {}
    for w in widths:
        lo = torch.rand(n_cells, prob.m, generator=g) * (1 - w)
        hi = lo + w * (0.5 + 0.5 * torch.rand(n_cells, prob.m, generator=g))
        S = lo[:, None, :] + (hi - lo)[:, None, :] * torch.rand(n_cells, n_samp, prob.m, generator=g)
        S[:, 0, :] = lo
        S[:, 1, :] = hi
        F = fields_at(prob, S.reshape(-1, prob.m))
        flat = {"R": F["R"], "gamma": F["gamma"], "gamma_t": F["gamma_t"], "lam": F["lam"]}
        for i in range(prob.d):
            flat[f"Rsig_{i}"] = F["Rsig"][i]
            flat[f"ggrad_{i}"] = F["ggrad"][i]
        for meth in methods:
            iv = enclose_cells(prob, lo, hi, meth)
            for k, v in flat.items():
                v = v.reshape(n_cells, n_samp)
                ok = _contained(v, iv[k][0][:, None], iv[k][1][:, None])
                if not ok:
                    viol[(meth, w, k)] = True
    return viol


@needs('results/expanding_disk_2d/stage3_6x64_N20000_s0_k5')
def test_derivative_enclosures_1d(cases):
    prob, _ = cases["1d"]
    viol = _sample_check(prob, 60, (0.2, 0.03, 0.004), 2000, METHODS, seed=4)
    assert not viol, viol


@needs('results/expanding_disk_2d/stage3_6x64_N20000_s0_k5')
def test_frozen_2d_checkpoint_200_cells(cases):
    """Prompt section 6 test 4: 200 random cells of three sizes, 1e4 samples per cell."""
    prob, _ = cases["2d_base"]
    viol = {}
    for w, seed in ((0.1, 5), (0.02, 6), (0.004, 7)):
        viol.update(_sample_check(prob, 67, (w,), 10_000, ("T1", "T3", "T2a", "LL", "T3q", "T3+T3q"), seed=seed))
    assert not viol, viol


@needs('results/expanding_disk_2d/stage3_6x64_N20000_s0_k5')
def test_frozen_2d_zonotope_and_tm2_jacobians(cases):
    """Tier 3z (zonotope) and the TM2 centered form on the 2D checkpoint (fewer cells: costlier tiers)."""
    prob, _ = cases["2d_base"]
    viol = _sample_check(prob, 20, (0.05, 0.01), 4000, ("T3z", "T2q", "T3q+T2a"), seed=9)
    assert not viol, viol


# ---------------------------------------------------------------- order-2 Taylor models, unit level
def test_tm2_ops_dense():
    from certified_obstacle_nn.certification.level2.tm import TM
    torch.manual_seed(11)
    lo = torch.rand(30, 3) * 2 - 1
    hi = lo + torch.rand(30, 3) * 0.5
    Z = TM.from_box(lo, hi)
    x, y, t = Z.sel(1, 1), Z.sel(1, 2), Z.sel(1, 0)
    exprs = {"tanh": lambda a, b, c: torch.tanh(3 * a - 2 * b + c),
             "prod": lambda a, b, c: (a * b + c) * (a - c),
             "exp": lambda a, b, c: torch.exp(a + b),
             "recip": lambda a, b, c: 1.0 / (a + 3.0),
             "relu": lambda a, b, c: torch.relu(a - b)}
    forms = {"tanh": ((x * 3.0) - (y * 2.0) + t).tanh(), "prod": (x * y + t) * (x - t), "exp": (x + y).exp(),
             "recip": (x + 3.0).recip_pos(), "relu": (x - y).relu()}
    S = lo[:, None, :] + (hi - lo)[:, None, :] * torch.rand(30, 20000, 3)
    for k, f in forms.items():
        v = exprs[k](S[..., 1], S[..., 2], S[..., 0])
        l, h = f.range()
        assert _contained(v, l[:, None], h[:, None]), k


def test_zonotope_mlp_dense():
    from certified_obstacle_nn.certification.level2.netenc_zono import mlp_enclose_zono
    torch.manual_seed(12)
    net = MLP(3, 2, hidden_width=12, hidden_depth=4).double()
    lo = torch.rand(15, 3)
    hi = lo + 0.2
    for K in (4, 64):
        v, j, _ = mlp_enclose_zono(net, Aff.from_box(lo, hi, True), 1, max_sym=K)
        S = lo[:, None, :] + (hi - lo)[:, None, :] * torch.rand(15, 20000, 3)
        vs, js, _ = mlp_jet(net, S.reshape(-1, 3), order=1)
        for enc, ex in ((v, vs), (j, js)):
            l, h = enc.range()
            ex = ex.reshape(15, 20000, -1)
            assert _contained(ex, l.reshape(15, 1, -1), h.reshape(15, 1, -1))


# ---------------------------------------------------------------- data enclosures (incl. the 2D indicator)
@needs('results/expanding_disk_2d/stage3_6x64_N20000_s0_k5')
def test_data_enclosures_cover_kink(cases):
    prob, _ = cases["2d_base"]
    g = torch.Generator().manual_seed(8)
    # boxes straddling the circle psi = 0 at t ~ 0.5 (R = 0.25)
    n = 100
    ang = torch.rand(n, generator=g) * 2 * math.pi
    c = torch.stack([torch.rand(n, generator=g), 0.5 + 0.25 * torch.cos(ang), 0.5 + 0.25 * torch.sin(ang)], -1)
    lo, hi = c - 0.01, c + 0.01
    for affine in (True, False):
        Z = Aff.from_box(lo, hi, affine)
        X = [Z.sel(1, 1 + i) for i in range(2)]
        f = prob.f(Z.sel(1, 0), X)
        u0 = prob.u0(X)
        S = lo[:, None, :] + (hi - lo)[:, None, :] * torch.rand(n, 5000, 3, generator=g)
        Sf = S.reshape(-1, 3)
        fs = prob.f(Sf[:, 0], [Sf[:, 1], Sf[:, 2]]).reshape(n, -1)
        us = prob.u0([Sf[:, 1], Sf[:, 2]]).reshape(n, -1)
        assert _contained(fs, *[v[:, None] for v in f.range()])
        assert _contained(us, *[v[:, None] for v in u0.range()])


# ---------------------------------------------------------------- 5. upper-sum consistency
@needs('results/expanding_disk_2d/stage3_6x64_N20000_s0_k5')
@pytest.mark.parametrize("case,n", [("1d", 16), ("1d", 32), ("2d_base", 8)])
def test_upper_sum_brackets_midpoint(cases, case, n):
    prob, _ = cases[case]
    for meth in ("T3", "T2a"):
        r = upper_sum_uniform(prob, [n] * prob.m, meth)
        assert r["L_vol"] <= r["rho_mid_vol"] <= r["U_vol"]
        assert r["L"] <= r["U"]


@needs('results/expanding_disk_2d/stage3_6x64_N20000_s0_k5')
def test_interval_children_inside_parent(cases):
    """Tier 1 is inclusion isotone: child enclosures lie inside the parent's (up to inflation)."""
    prob, _ = cases["1d"]
    lo = torch.tensor([[0.3, 0.4]])
    hi = torch.tensor([[0.34, 0.44]])
    par = enclose_cells(prob, lo, hi, "T1")
    mid = 0.5 * (lo + hi)
    clo = torch.cat([lo, torch.stack([mid[:, 0], lo[:, 1]], 1)])
    chi = torch.cat([torch.stack([mid[:, 0], hi[:, 1]], 1), hi])
    ch = enclose_cells(prob, clo, chi, "T1")
    for k in par:
        pad = 1e-9 * (par[k][1] - par[k][0]).abs() + 1e-12
        assert bool((ch[k][0] >= par[k][0] - pad).all() and (ch[k][1] <= par[k][1] + pad).all()), k
    pp = phi_bounds(prob, par)
    pc = phi_bounds(prob, ch)
    assert bool((pc["Phi_up"] <= pp["Phi_up"] * (1 + 1e-9) + 1e-12).all())


# ---------------------------------------------------------------- 6. legacy regression
@needs('results/expanding_disk_2d/stage3_6x64_N20000_s0_k5')
def test_legacy_regression_2d_16(cases):
    """Per-cell legacy 'global' mode == unchanged legacy level2_diagnostic.certify (rtol 1e-10)."""
    from certified_obstacle_nn.certification.level2.legacy_local import legacy_uniform
    from certified_obstacle_nn.experiments.expanding_disk_2d.level2_diagnostic import certify
    from certified_obstacle_nn.experiments.level2.common import legacy_constants, load_case
    from certified_obstacle_nn.certification.network_bounds import historical_svd_norms
    model, meta = load_case("2d_base")
    prob = make_problem(model, meta)
    # v14 A2: the recorded legacy value used float-SVD spectral norms; reproduce
    # it under that historical convention, and check that the proved-norm
    # version is never smaller.
    with historical_svd_norms():
        new = legacy_uniform(prob, legacy_constants(model, meta), [16] * 3, 2, radius="global")
        old = certify(model, meta["cfg"], 16, q=2)
    for k in ("rho_tilde_h", "E_h_lip_old", "E_h_lip_new", "E_h_unr", "E_h_0T", "E_cert_U_old", "E_cert_U_new",
              "V_unr_frac"):
        assert math.isclose(new[k], old[k], rel_tol=1e-10), (k, new[k], old[k])
    assert math.isclose(old["E_cert_U_new"], 774723.8225137504, rel_tol=1e-10)  # recorded legacy value
    proved = certify(model, meta["cfg"], 16, q=2)
    assert proved["E_cert_U_new"] >= old["E_cert_U_new"]


@needs('results/expanding_disk_2d/stage3_6x64_N20000_s0_k5')
def test_legacy_regression_1d_64(cases):
    from certified_obstacle_nn.certification.certificate_option_b import certify_option_b
    from certified_obstacle_nn.certification.level2.legacy_local import legacy_uniform
    from certified_obstacle_nn.certification.partition import build_partition
    from certified_obstacle_nn.experiments.level2.common import legacy_constants, load_case
    model, meta = load_case("1d")
    prob = make_problem(model, meta)
    new = legacy_uniform(prob, legacy_constants(model, meta), [64, 64], 4, radius="global")
    old = certify_option_b(model, meta["problem"], build_partition(meta["problem"], 64, 64), n_qp=4)
    pairs = (("rho_tilde_h", old.rho_tilde_h), ("E_h_lip_old", old.E_h_lip_old), ("E_h_lip_new", old.E_h_lip_new),
             ("E_h_unr", old.E_h_unr), ("E_h_0T", old.E_h_0T), ("E_cert_U_old", old.E_cert_U_old),
             ("E_cert_U_new", old.E_cert_U_new))
    for k, v in pairs:
        assert math.isclose(new[k], v, rel_tol=1e-10), (k, new[k], v)
