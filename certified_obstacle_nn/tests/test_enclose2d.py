"""v14 Stage 5 (disk): validated enclosures in 2-D space (certification/enclose/
boxes.py, engine2d.py). Containment is checked against an independent 50-digit
mpmath implementation of the network, its first derivatives and the exact
disk data, at many points of every cell; the box partitions' exact coverage
check and the status logic are checked on corrupted partitions."""

from fractions import Fraction

import mpmath as mp
import numpy as np
import pytest
import torch

from certified_obstacle_nn.certification.enclose import boxes as Bx
from certified_obstacle_nn.certification.enclose import intervals_arb as AR
from certified_obstacle_nn.certification.enclose import intervals_np as NP
from certified_obstacle_nn.certification.enclose import tm as T
from certified_obstacle_nn.certification.enclose.engine2d import DiskAdapter, TripleEncloser2D
from certified_obstacle_nn.experiments.expanding_disk_2d.model import Triple2D

mp.mp.dps = 50
KAP = 5


def _triple(seed=0, depth=2, width=6, scale=1.0):
    torch.manual_seed(seed)
    tr = Triple2D(depth, width, "square").double().eval()
    with torch.no_grad():
        for q in tr.parameters():
            q.mul_(scale)
    for q in tr.parameters():
        q.requires_grad_(False)
    return tr


def _mp_mlp(mlp, X, chans):
    """Value and d/dX_ch of a tanh MLP in mpmath (exact weights)."""
    Ws = [(l.weight.numpy().tolist(), l.bias.numpy().tolist()) for l in mlp.layers]
    a = [mp.mpf(v) for v in X]
    da = {c: [mp.mpf(1) if j == c else mp.mpf(0) for j in range(len(X))] for c in chans}
    for li, (W, b) in enumerate(Ws):
        z = [mp.fsum(mp.mpf(W[i][j]) * a[j] for j in range(len(a))) + mp.mpf(b[i]) for i in range(len(W))]
        dz = {c: [mp.fsum(mp.mpf(W[i][j]) * da[c][j] for j in range(len(a))) for i in range(len(W))] for c in chans}
        if li == len(Ws) - 1:
            return z, dz
        a = [mp.tanh(v) for v in z]
        da = {c: [mp.sech(z[i]) ** 2 * dz[c][i] for i in range(len(z))] for c in chans}


def _mp_disk(t, x, y):
    t, x, y = mp.mpf(t), mp.mpf(x), mp.mpf(y)
    h = mp.e**t - t - 1
    B = 16 * x * (1 - x) * y * (1 - y)
    Bx = 16 * (1 - 2 * x) * y * (1 - y)
    By = 16 * x * (1 - x) * (1 - 2 * y)
    lapB = -32 * (x * (1 - x) + y * (1 - y))
    R = mp.mpf(3) / 20 + t / 5
    r2 = (x - mp.mpf(1) / 2) ** 2 + (y - mp.mpf(1) / 2) ** 2
    psi = (R**2 - r2) / (2 * R)
    psi_t = (mp.mpf(1) / 5) * (R**2 + r2) / (2 * R**2)
    g = -h * B
    g_t = -(mp.e**t - 1) * B
    f = g_t + h * lapB - 2 * KAP
    if psi > 0:
        f += KAP * (2 * psi * psi_t + 4 * psi / R + 2 - 2 * r2 / R**2)
    psi0 = (mp.mpf(3) / 20) ** 2 - r2
    u0 = KAP * (max(psi0 / (2 * mp.mpf(3) / 20), 0)) ** 2
    return dict(B=B, Bx=Bx, By=By, g=g, g_t=g_t, g_x=-h * Bx, g_y=-h * By, f=f, psi=psi, u0=u0)


def _mp_fields(tr, t, x, y):
    X = (t, x, y)
    N, dN = _mp_mlp(tr.state, X, (0, 1, 2))
    S, dS = _mp_mlp(tr.flux, X, (1, 2))
    M, _ = _mp_mlp(tr.mult, X, ())
    d = _mp_disk(t, x, y)
    u = d["B"] * N[0]
    ut = d["B"] * dN[0][0]
    ux = d["Bx"] * N[0] + d["B"] * dN[1][0]
    uy = d["By"] * N[0] + d["B"] * dN[2][0]
    lam = M[0] ** 2
    div = dS[1][0] + dS[2][1]
    return {
        "R": d["f"] - ut - div + lam, "S1": S[0] + ux, "S2": S[1] + uy, "gam": d["g"] - u, "gam_t": d["g_t"] - ut,
        "gam_x": d["g_x"] - ux, "gam_y": d["g_y"] - uy, "lam": lam, "u": u,
        "Rhat_pos": d["f"] - d["g_t"] - div + lam, "Shat1_pos": S[0] + d["g_x"], "Shat2_pos": S[1] + d["g_y"],
    }


def _cells(rng, n, levels, near_interface=False):
    lev = rng.integers(levels[0], levels[1] + 1, n)
    w = np.ldexp(1.0, -lev)
    lo = np.stack([np.floor(rng.random(n) / w) * w for _ in range(3)])
    if near_interface:  # cells containing a point of the circle r = R(t)
        tt = rng.random(n)
        th = rng.random(n) * 2 * np.pi
        Rt = 0.15 + 0.2 * tt
        pts = np.stack([tt, 0.5 + Rt * np.cos(th), 0.5 + Rt * np.sin(th)])
        lo = np.floor(pts / w) * w
    return lo, lo + w


@pytest.mark.parametrize("bkname", ["numpy", "arb"])
@pytest.mark.parametrize("near", [False, True])
def test_fields_contain_mpmath(bkname, near):
    bk = NP if bkname == "numpy" else AR
    tr = _triple(1, scale=1.5)
    enc = TripleEncloser2D(tr, DiskAdapter())
    rng = np.random.default_rng(3 + near)
    lo, hi = _cells(rng, 6, (2, 6), near_interface=near)
    F = enc.fields(bk, lo, hi)
    rg = {k: F[k].range() for k in ("R", "S1", "S2", "gam", "gam_t", "gam_x", "gam_y", "lam", "u",
                                    "Rhat_pos", "Shat1_pos", "Shat2_pos")}
    n_straddle = 0
    for i in range(lo.shape[1]):
        signs = set()
        for _ in range(12):
            p = lo[:, i] + (hi[:, i] - lo[:, i]) * rng.random(3)
            vals = _mp_fields(tr, *p)
            signs.add(_mp_disk(*p)["psi"] > 0)
            for k, v in vals.items():
                assert mp.mpf(rg[k].lower()[i]) <= v <= mp.mpf(rg[k].upper()[i]), (k, i)
        # cell corners too (closed cells)
        for c in range(8):
            p = [hi[j, i] if (c >> j) & 1 else lo[j, i] for j in range(3)]
            for k, v in _mp_fields(tr, *p).items():
                assert mp.mpf(rg[k].lower()[i]) <= v <= mp.mpf(rg[k].upper()[i]), (k, i, "corner")
        n_straddle += len(signs) == 2
    if near:
        assert n_straddle >= 1  # the branch hull was exercised


def test_trace_contains_mpmath_and_u0_branch():
    tr = _triple(2)
    enc = TripleEncloser2D(tr, DiskAdapter())
    rng = np.random.default_rng(5)
    for tval in (0.0, 1.0):
        lev = 3
        w = 2.0**-lev
        lo = np.stack([rng.integers(0, 8, 10) * w for _ in range(2)])
        lo[:, 0] = [0.375, 0.5]  # touches the initial circle r = 0.15 around (1/2, 1/2)
        hi = lo + w
        d = enc.trace_densities(NP, tval, lo, hi)
        for i in range(lo.shape[1]):
            for _ in range(15):
                x, y = lo[:, i] + w * rng.random(2)
                N, _ = _mp_mlp(tr.state, (tval, x, y), ())
                dd = _mp_disk(tval, x, y)
                v = dd["B"] * N[0]
                assert max(dd["g"] - v, 0) ** 2 <= d["w2"][i]
                u0 = dd["u0"] if tval == 0.0 else None
                if tval == 0.0:
                    assert (v - u0) ** 2 <= d["r0"][i]
                    assert (max(v, dd["g"]) - u0) ** 2 <= d["r0hat"][i]


def test_rational_constants_enclose_exact_values():
    for bk in (NP, AR):
        for num, den in ((3, 20), (1, 5), (1, 10)):
            r = bk.rational(num, den, (1,))
            assert Fraction(float(r.lower()[0])) <= Fraction(num, den) <= Fraction(float(r.upper()[0]))
        r = T.tm_rational(bk, 3, 20, 3, (1,)).range()
        assert Fraction(float(r.lower()[0])) <= Fraction(3, 20) <= Fraction(float(r.upper()[0]))
        assert float(r.lower()[0]) < 0.15 or float(r.upper()[0]) > 0.15  # 0.15 (float) != 3/20


def test_recip_tm_encloses():
    for bk in (NP, AR):
        t = T.tm_variable(bk, np.array([0.25, 0.75]), np.array([0.25, 0.25]), 1, 1)
        R = T.tm_rational(bk, 3, 20, 1, (2,)) + t * bk.rational(1, 5, (2,))
        rg = T.tm_recip(R).range()
        for i, (a, b) in enumerate(((0.0, 0.5), (0.5, 1.0))):
            for s in np.linspace(a, b, 11):
                v = 1 / (mp.mpf(3) / 20 + mp.mpf(s) / 5)
                assert mp.mpf(rg.lower()[i]) <= v <= mp.mpf(rg.upper()[i])


def test_box_coverage_detects_corruption():
    p = Bx.uniform(3, 4)
    assert p.verify_coverage() == (True, "ok")
    m = np.zeros(p.n, bool)
    m[[0, 7, 40]] = True
    q = Bx.refine(p, m)
    assert q.verify_coverage()[0]
    assert not q.subset(np.arange(1, q.n)).verify_coverage()[0]  # missing cell
    dup = Bx.BoxPartition(np.concatenate([q.lo, q.lo[:, -1:]], 1), np.concatenate([q.hi, q.hi[:, -1:]], 1))
    assert not dup.verify_coverage()[0]
    nest = Bx.BoxPartition(np.concatenate([q.lo, p.lo[:, :1]], 1), np.concatenate([q.hi, p.hi[:, :1]], 1))
    assert not nest.verify_coverage()[0]
    skew = Bx.BoxPartition(q.lo.copy(), q.hi.copy())
    skew.hi[0, 0] += 2.0**-6
    assert not skew.verify_coverage()[0]


def test_status_logic_on_boxes():
    tr = _triple(0)
    enc = TripleEncloser2D(tr, DiskAdapter())
    part = Bx.uniform(3, 4)
    t0, tT = Bx.uniform(2, 4), Bx.uniform(2, 4)
    us, vol, _ = Bx.certify_boxes(enc, part, t0, tT, backend="numpy", model_frozen=True, constants_validated=True)
    assert us.status == "validated_upper_bound", us.status_reasons
    bad = part.subset(np.arange(1, part.n))
    us2, _, _ = Bx.certify_boxes(enc, bad, t0, tT, backend="numpy", model_frozen=True, constants_validated=True)
    assert us2.status != "validated_upper_bound"
    us3, _, _ = Bx.certify_boxes(enc, part, t0, Bx.uniform(2, 4).subset(np.arange(15)), backend="numpy",
                                 model_frozen=True, constants_validated=True)
    assert us3.status != "validated_upper_bound"
    us4, _, _ = Bx.certify_boxes(enc, part, t0, tT, backend="numpy", model_frozen=False, constants_validated=True)
    assert us4.status != "validated_upper_bound"
    us5, _, _ = Bx.certify_boxes(enc, part, t0, tT, backend="numpy", model_frozen=True, constants_validated=False)
    assert us5.status != "validated_upper_bound"
    # NaN injection (a non-positive interval in recip yields NaN / inf)
    from certified_obstacle_nn.certification.enclose import upper_sum as US

    vol_nan = dict(vol)
    vol_nan["r"] = vol["r"].copy()
    vol_nan["r"][3] = np.nan
    d0 = Bx.evaluate_trace(enc, t0, 0.0, "numpy")
    dT = Bx.evaluate_trace(enc, tT, 1.0, "numpy")
    us6 = US.reduce_upper_sums(part, vol_nan, t0, d0, tT, dT, backend_info=US.backend_info("numpy", 53),
                               model_frozen=True, constants_validated=True)
    assert us6.status != "validated_upper_bound"


def test_backends_agree_on_disk_cells():
    tr = _triple(4, depth=3, width=8)
    enc = TripleEncloser2D(tr, DiskAdapter())
    rng = np.random.default_rng(7)
    lo, hi = _cells(rng, 8, (3, 7))
    a = enc.densities(AR, lo, hi)
    n = enc.densities(NP, lo, hi)
    for k in ("r", "p", "c", "rhat"):
        ok = np.isclose(a[k], n[k], rtol=1e-6, atol=1e-300)
        assert ok.all(), k


def test_upper_sum_dominates_float_quadrature():
    """Diagnostic consistency: the validated U's are >= a fine float64 midpoint
    estimate of the same integrals (not a proof; the proof is containment)."""
    tr = _triple(0)
    enc = TripleEncloser2D(tr, DiskAdapter())
    part = Bx.uniform(3, 8)
    us, _, _ = Bx.certify_boxes(enc, part, Bx.uniform(2, 16), Bx.uniform(2, 16), backend="numpy",
                                model_frozen=True, constants_validated=True)
    from certified_obstacle_nn.experiments.expanding_disk_2d import exact_solution as ex

    n = 24
    s = (np.arange(n) + 0.5) / n
    tt, xx, yy = [torch.tensor(a.ravel(), requires_grad=True) for a in np.meshgrid(s, s, s, indexing="ij")]
    X = torch.stack([tt, xx, yy], -1)
    N = tr.state(X)[:, 0]
    Nt, Nx, Ny = torch.autograd.grad(N.sum(), (tt, xx, yy))
    S = tr.flux(X)
    d1 = torch.autograd.grad(S[:, 0].sum(), xx, retain_graph=True)[0]
    d2 = torch.autograd.grad(S[:, 1].sum(), yy)[0]
    with torch.no_grad():
        B = ex.B(xx, yy)
        Bx_, By_ = ex.grad_B(xx, yy)
        R = ex.f_exact(tt, xx, yy, 5.0) - B * Nt - (d1 + d2) + tr.mult(X)[:, 0] ** 2
        Sx = S[:, 0] + Bx_ * N + B * Nx
        Sy = S[:, 1] + By_ * N + B * Ny
        vol_r = float((2 * R**2 + Sx**2 + Sy**2).mean())
    assert us.parts["vol_r"] >= vol_r


def _mp_ellipse(t, x, y):
    t, x, y = mp.mpf(t), mp.mpf(x), mp.mpf(y)
    k0 = mp.mpf(1) / 100
    a = mp.mpf(12) / 100 + t / 10
    d = mp.mpf(8) / 100 + 7 * t / 100
    X1, X2 = x - mp.mpf(43) / 100, y - mp.mpf(57) / 100
    z = X1**2 / a**2 + X2**2 / d**2 - 1
    zx, zy = 2 * X1 / a**2, 2 * X2 / d**2
    lapz = 2 / a**2 + 2 / d**2
    zt = -2 * (mp.mpf(1) / 10) * X1**2 / a**3 - 2 * (mp.mpf(7) / 100) * X2**2 / d**3
    P, Qy = x * (1 - x), y * (1 - y)
    B, Bx, By, lapB = 16 * P * Qy, 16 * (1 - 2 * x) * Qy, 16 * P * (1 - 2 * y), -32 * (P + Qy)
    c = mp.mpf(1) / 10 + t / 20
    L = 1 + x / 5 - y / 10
    ub = B * c * L
    ub_t, ub_x, ub_y = B * L / 20, c * (Bx * L + B / 5), c * (By * L - B / 10)
    lap_ub = c * (lapB * L + 2 * (Bx / 5 - By / 10))
    f = ub_t - lap_ub - (2 * k0 * z * (zt - lapz) if z < 0 else 0)
    a0, d0 = mp.mpf(12) / 100, mp.mpf(8) / 100
    z0 = X1**2 / a0**2 + X2**2 / d0**2 - 1
    u0 = 16 * P * Qy * (mp.mpf(1) / 10) * L - k0 * max(-z0, 0) ** 2
    return dict(B=B, Bx=Bx, By=By, g=ub - k0 * z**2, g_t=ub_t - 2 * k0 * z * zt, g_x=ub_x - 2 * k0 * z * zx,
                g_y=ub_y - 2 * k0 * z * zy, f=f, zeta=z, u0=u0)


@pytest.mark.parametrize("bkname", ["numpy", "arb"])
def test_ellipse_adapter_contains_mpmath(bkname):
    from certified_obstacle_nn.certification.enclose.engine2d import EllipseAdapter

    bk = NP if bkname == "numpy" else AR
    ad = EllipseAdapter()
    rng = np.random.default_rng(11)
    n = 10
    lev = rng.integers(3, 7, n)
    w = np.ldexp(1.0, -lev)
    tt = rng.random(n)
    th = rng.random(n) * 2 * np.pi
    pts = np.stack([tt, 0.43 + (0.12 + 0.1 * tt) * np.cos(th), 0.57 + (0.08 + 0.07 * tt) * np.sin(th)])
    lo = np.floor(pts / w) * w  # cells containing a point of the moving ellipse
    lo[:, :3] = np.floor(rng.random((3, 3)) / w[:3]) * w[:3]  # and a few anywhere
    hi = lo + w
    c, r = 0.5 * (lo + hi), 0.5 * (hi - lo)
    tv, xv, yv = (T.tm_variable(bk, c[j], r[j], j + 1, 3) for j in range(3))
    D = ad.data(tv, xv, yv)
    rg = {k: D[k].range() for k in D}
    tr0 = T.tm_variable(bk, np.zeros(n), np.zeros(n), 1, 3)
    D0 = ad.data_trace(tr0, xv, yv)
    rg0 = {k: D0[k].range() for k in D0}
    straddle = 0
    for i in range(n):
        signs = set()
        for _ in range(25):
            p = lo[:, i] + (hi[:, i] - lo[:, i]) * rng.random(3)
            v = _mp_ellipse(*p)
            signs.add(v["zeta"] < 0)
            for k in ("f", "g", "g_t", "g_x", "g_y"):
                assert mp.mpf(rg[k].lower()[i]) <= v[k] <= mp.mpf(rg[k].upper()[i]), (k, i)
            v0 = _mp_ellipse(0, p[1], p[2])
            assert mp.mpf(rg0["u0"].lower()[i]) <= v0["u0"] <= mp.mpf(rg0["u0"].upper()[i])
            assert mp.mpf(rg0["g"].lower()[i]) <= v0["g"] <= mp.mpf(rg0["g"].upper()[i])
        straddle += len(signs) == 2
    assert straddle >= 1


def test_ellipse_problem_matches_adapter_formulas():
    """The torch problem class and the mpmath reference agree (float64 rounding)."""
    from certified_obstacle_nn.problems.ellipse import EllipseProblem

    p = EllipseProblem()
    rng = np.random.default_rng(2)
    for _ in range(200):
        q = rng.random(3)
        v = _mp_ellipse(*q)
        T_, X, Y = (torch.tensor([s], dtype=torch.float64) for s in q)
        assert abs(float(p.f(T_, X, Y)) - float(v["f"])) < 1e-12
        assert abs(float(p.g(T_, X, Y)) - float(v["g"])) < 1e-13
        assert abs(float(p.u0(X, Y)) - float(v["u0"])) < 1e-13
