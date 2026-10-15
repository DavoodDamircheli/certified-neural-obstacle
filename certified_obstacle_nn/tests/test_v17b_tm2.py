"""v17b Task 4: validated second-order Taylor models (certification/enclose/tm2.py) and the opt-in "TM2" engine
candidate. Polynomial exactness, the rules in isolation against pointwise values, 50-digit mpmath containment of every
field at corners and interior points (both backends, data- and projection-interface cells, saturating weights), and
"new range subset of TM1 range" on every cell of Stefan 32^2 and a disk smoke partition."""

from fractions import Fraction

import mpmath as mp
import numpy as np
import pytest

from certified_obstacle_nn.certification.enclose import intervals_arb as AR
from certified_obstacle_nn.certification.enclose import intervals_np as NP
from certified_obstacle_nn.certification.enclose import tm as T
from certified_obstacle_nn.certification.enclose import tm2 as T2
from certified_obstacle_nn.certification.enclose import upper_sum as US
from certified_obstacle_nn.certification.enclose.engine import StefanAdapter, TripleEncloser
from certified_obstacle_nn.certification.enclose.engine2d import DiskAdapter, TripleEncloser2D, enclose_mlp_vec
from certified_obstacle_nn.experiments.v14.common import STEFAN_SEEDS, load_frozen_stefan
from certified_obstacle_nn.tests.test_enclose2d import _cells, _mp_fields, _mp_mlp, _triple
from certified_obstacle_nn.tests.test_v17_zono import KEYS, _sampled_density_ok, _subset_densities
from certified_obstacle_nn.tests.stored_data import needs

mp.mp.dps = 50
BK = {"numpy": NP, "arb": AR}


def _vars(bk, c, r):
    return [T.tm_variable(bk, c[k], r[k], k + 1, len(c)) for k in range(len(c))]


@pytest.mark.parametrize("bkname", ["numpy", "arb"])
def test_polynomial_exactness(bkname):
    """Products of degree <= 2 are represented exactly (zero high-order part, remainder only from rounding), and the
    TM2 range of x y and x^2 is the exact range."""
    bk = BK[bkname]
    c = [np.array([0.25, -0.5]), np.array([0.5, 0.125]), np.array([0.75, 0.0])]
    r = [np.array([0.125, 0.25]), np.array([0.25, 0.5]), np.array([0.0625, 0.5])]
    x, y, z = (T2.from_tm1(v) for v in _vars(bk, c, r))
    xy = T2.mul(x, y)
    x2 = T2.sqr(x)
    for A in (xy, x2):
        assert np.all(np.abs(A.rem.upper()) < 1e-300) and np.all(np.abs(A.rem.lower()) < 1e-300)  # outward ulps of 0
    rg = xy.range()
    for i in range(2):
        a = [c[0][i] - r[0][i], c[0][i] + r[0][i]]
        b = [c[1][i] - r[1][i], c[1][i] + r[1][i]]
        prods = [p * q for p in a for q in b]
        assert rg.lower()[i] <= min(prods) and rg.upper()[i] >= max(prods)
    rg2 = x2.range()
    for i in range(2):  # x^2 over [c - r, c + r]: c^2 + 2 c r e + r^2 e^2, range exact up to the e / e^2 decoupling
        lo_ex = 0.0 if abs(c[0][i]) <= r[0][i] else min((c[0][i] - r[0][i]) ** 2, (c[0][i] + r[0][i]) ** 2)
        assert rg2.lower()[i] <= lo_ex
        assert rg2.upper()[i] >= max((c[0][i] - r[0][i]) ** 2, (c[0][i] + r[0][i]) ** 2)
    # a cubic product is not exact: x y z needs the high-order bound, still contains the values
    xyz = T2.mul(xy, z)
    rg3 = xyz.range()
    for e in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T:
        v = [Fraction(c[k][0]) + Fraction(r[k][0]) * int(e[k]) for k in range(3)]
        val = v[0] * v[1] * v[2]
        assert Fraction(rg3.lower()[0]) <= val <= Fraction(rg3.upper()[0])


def _rand_tm2(bk, rng, n, shape, scale):
    nr = 1 + n + n * (n + 1) // 2
    rows = [bk.thin(rng.normal(size=shape) * scale) for _ in range(nr)]
    rows.append(bk.symmetric(np.abs(rng.normal(size=shape)) * scale * 0.1))
    return T2.TM2(bk, bk.stack(rows), n)


def _point(A, e, rv):
    """A point of the set described by the (thin-coefficient) TM2 A at e, remainder fraction rv."""
    c = 0.5 * (A.c.lower() + A.c.upper())
    val = c[0] + sum(c[1 + j] * e[j] for j in range(A.n))
    for q, (i, j) in enumerate(T2.pairs(A.n)):
        val = val + c[1 + A.n + q] * e[i] * e[j]
    lo, hi = A.rem.lower(), A.rem.upper()
    return val + 0.5 * (lo + hi) + 0.5 * (hi - lo) * rv


def _encl(F, e):
    """Interval enclosure [lo, hi] of TM2 F evaluated at e (coefficient radii included)."""
    mid = 0.5 * (F.c.lower() + F.c.upper())
    rad = 0.5 * (F.c.upper() - F.c.lower())
    mono = [1.0] + list(e) + [e[i] * e[j] for i, j in T2.pairs(F.n)]
    v = sum(mid[k] * mono[k] for k in range(len(mono)))
    w = sum(rad[k] * abs(mono[k]) for k in range(len(mono)))
    return v - w + F.rem.lower(), v + w + F.rem.upper()


@pytest.mark.parametrize("bkname", ["numpy", "arb"])
@pytest.mark.parametrize("scale", [0.3, 3.0])
def test_rules_contain_pointwise_values(bkname, scale):
    bk = BK[bkname]
    rng = np.random.default_rng(3)
    n, shape = 3, (3, 2)
    A, B = _rand_tm2(bk, rng, n, shape, scale), _rand_tm2(bk, rng, n, shape, scale)
    P, Q = T2.mul(A, B), T2.sqr(A)
    th, sh = T2.tanh_sech2(A)
    for _ in range(40):
        e = rng.uniform(-1, 1, n)
        ra, rb = rng.uniform(-1, 1, shape), rng.uniform(-1, 1, shape)
        a, b = _point(A, e, ra), _point(B, e, rb)
        for name, F, val in (("mul", P, a * b), ("sqr", Q, a * a), ("tanh", th, np.tanh(a)),
                             ("sech2", sh, 1 / np.cosh(a) ** 2)):
            lo, hi = _encl(F, e)
            tol = 1e-12 * (1 + np.abs(val))  # float64 evaluation of the reference value only
            assert np.all(lo - tol <= val) and np.all(val <= hi + tol), name


@pytest.mark.parametrize("bkname", ["numpy", "arb"])
def test_derivative_polynomial_ranges(bkname):
    """P_3 and 2 - 5u + 3u^2 ranges contain dense samples (vertex inside / outside / at the boundary)."""
    bk = BK[bkname]
    lo = np.array([0.0, 0.5, 0.66, 0.67, 0.8, 0.9, 0.0])
    hi = np.array([1.0, 0.6, 0.67, 0.7, 0.84, 1.0, 1e-3])
    uZ = bk.from_bounds(lo, hi)
    p3 = T2._quad_range(bk, uZ, -2.0, 8.0, -6.0, 0.666, 0.667, 2, 3)
    r4 = T2._quad_range(bk, uZ, 2.0, -5.0, 3.0, 0.833, 0.834, 5, 6)
    for i in range(lo.size):
        for u in np.linspace(lo[i], hi[i], 401):
            uf = Fraction(u)
            assert Fraction(p3.lower()[i]) <= -2 + 8 * uf - 6 * uf * uf <= Fraction(p3.upper()[i])
            assert Fraction(r4.lower()[i]) <= 2 - 5 * uf + 3 * uf * uf <= Fraction(r4.upper()[i])


def _mlp_box(bk, mlp, lo, hi, chans, tm2):
    c, r = 0.5 * (lo + hi), 0.5 * (hi - lo)
    ins = _vars(bk, list(c), list(r))
    f = T2.enclose_mlp_vec_tm2 if tm2 else enclose_mlp_vec
    from certified_obstacle_nn.certification.enclose.engine import mlp_weights

    return f(mlp_weights(mlp), ins, chans)


@pytest.mark.parametrize("bkname", ["numpy", "arb"])
@pytest.mark.parametrize("scale", [1.5, 30.0])
def test_network_value_and_derivatives_contain_mpmath(bkname, scale):
    bk = BK[bkname]
    tr = _triple(4, depth=3, width=6, scale=scale)
    rng = np.random.default_rng(5)
    lo, hi = _cells(rng, 4 if bkname == "numpy" else 2, (2, 6))
    N, dN = _mlp_box(bk, tr.flux, lo, hi, (1, 2), True)
    rN = N.range()
    rd = {ch: dN[ch].range() for ch in (1, 2)}
    for i in range(lo.shape[1]):
        pts = [lo[:, i] + (hi[:, i] - lo[:, i]) * rng.random(3) for _ in range(5)]
        pts += [[hi[j, i] if (k >> j) & 1 else lo[j, i] for j in range(3)] for k in range(8)]
        for p in pts:
            V, dV = _mp_mlp(tr.flux, tuple(mp.mpf(float(q)) for q in p), (1, 2))
            for m in range(2):
                assert mp.mpf(rN.lower()[m, i]) <= V[m] <= mp.mpf(rN.upper()[m, i])
                for ch in (1, 2):
                    assert mp.mpf(rd[ch].lower()[m, i]) <= dV[ch][m] <= mp.mpf(rd[ch].upper()[m, i])


@pytest.mark.parametrize("bkname,ncell", [("numpy", 6), ("arb", 2)])
@pytest.mark.parametrize("near", [False, True])
@pytest.mark.parametrize("scale", [1.5, 30.0])
def test_tm2_fields_contain_mpmath(bkname, ncell, near, scale):
    """TM2 field enclosures (network outputs lumped to TM1, then the TM1 data / mask algebra) contain 50-digit
    values at interior points and corners, on generic and data-interface cells, moderate and saturating weights."""
    bk = BK[bkname]
    tr = _triple(1, depth=3, width=6, scale=scale)
    enc = TripleEncloser2D(tr, DiskAdapter())
    rng = np.random.default_rng(17 + near)
    lo, hi = _cells(rng, ncell, (2, 6), near_interface=near)
    F = enc.fields(bk, lo, hi, mlp=T2.enclose_mlp_vec_tm2)
    rg = {k: F[k].range() for k in KEYS}
    for i in range(lo.shape[1]):
        pts = [lo[:, i] + (hi[:, i] - lo[:, i]) * rng.random(3) for _ in range(6)]
        pts += [[hi[j, i] if (c >> j) & 1 else lo[j, i] for j in range(3)] for c in range(8)]
        for p in pts:
            for k, v in _mp_fields(tr, *p).items():
                assert mp.mpf(rg[k].lower()[i]) <= v <= mp.mpf(rg[k].upper()[i]), (k, i, scale)


@pytest.mark.parametrize("bkname", ["numpy", "arb"])
def test_tm2_densities_dominate_pointwise_on_interface_cells(bkname):
    bk = BK[bkname]
    tr = _triple(2, depth=3, width=6, scale=1.5)
    rng = np.random.default_rng(23)
    lo, hi = _cells(rng, 4 if bkname == "numpy" else 2, (3, 6), near_interface=True)
    _sampled_density_ok(TripleEncloser2D(tr, DiskAdapter(), enclosure="TM2"), bk, lo, hi, tr, rng)


def test_tm2_tighter_than_tm1_on_small_cells():
    """Report-level sanity: on fine cells the TM2 network enclosure of the flux divergence is narrower than TM1."""
    tr = _triple(3, depth=4, width=16, scale=1.5)
    rng = np.random.default_rng(29)
    lo, hi = _cells(rng, 8, (7, 7))
    _, d1 = _mlp_box(NP, tr.flux, lo, hi, (1, 2), False)
    _, d2 = _mlp_box(NP, tr.flux, lo, hi, (1, 2), True)
    w1 = (d1[1].range().upper() - d1[1].range().lower()).sum()
    w2 = (d2[1].range().upper() - d2[1].range().lower()).sum()
    assert w2 < w1


def test_new_range_subset_tm1_stefan_32_every_cell():
    problem, triple = load_frozen_stefan(STEFAN_SEEDS[0])
    part = US.uniform_2d(32, 32)
    base = TripleEncloser(triple, StefanAdapter(problem)).densities(NP, part.t0, part.t1, part.x0, part.x1)
    new = TripleEncloser(triple, StefanAdapter(problem), enclosure="TM2").densities(NP, part.t0, part.t1, part.x0,
                                                                                    part.x1)
    _subset_densities(new, base)


@needs('results/expanding_disk_2d/final/long_6x64_N20000_s0_k5')
def test_new_range_subset_tm1_disk_smoke_every_cell():
    from certified_obstacle_nn.certification.enclose import boxes as Bx
    from certified_obstacle_nn.experiments.expanding_disk_2d.final_common import load_run
    from certified_obstacle_nn.experiments.v14.common import REPO_ROOT

    triple, cfg = load_run(REPO_ROOT / "results/expanding_disk_2d/final/long_6x64_N20000_s0_k5")
    part = Bx.uniform(3, 8)
    base = TripleEncloser2D(triple, DiskAdapter()).densities(NP, part.lo, part.hi)
    new = TripleEncloser2D(triple, DiskAdapter(), enclosure="TM2").densities(NP, part.lo, part.hi)
    _subset_densities(new, base)
