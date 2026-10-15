"""Mean-value enclosure of R / R_hat (enclose/second_order.py, TripleEncloser(mean_value=True))."""

import mpmath as mp
import numpy as np
import pytest
import sympy as sp

from certified_obstacle_nn.certification.enclose import intervals_arb as AB
from certified_obstacle_nn.certification.enclose import intervals_np as NB
from certified_obstacle_nn.certification.enclose import tm as T
from certified_obstacle_nn.certification.enclose import upper_sum as US
from certified_obstacle_nn.certification.enclose.engine import StefanAdapter, TripleEncloser
from certified_obstacle_nn.certification.enclose.second_order import enclose_mlp2
from certified_obstacle_nn.problems.stefan import StefanProblem
from certified_obstacle_nn.tests.test_enclose import (
    BACKENDS, _cells, _check_fields_contain, _linear_triple, _mp_fields, _tiny_triple,
)

mp.mp.dps = 50


def _mp_mlp2(layers, t, x):
    """value, (d_t, d_x), (d_tt, d_tx, d_xx) of a tanh MLP at a point, 50 digits."""
    a = [mp.mpf(t), mp.mpf(x)]
    d1 = {0: [mp.mpf(1), mp.mpf(0)], 1: [mp.mpf(0), mp.mpf(1)]}
    pairs = ((0, 0), (0, 1), (1, 1))
    d2 = {p: [mp.mpf(0), mp.mpf(0)] for p in pairs}
    for li, (W, b) in enumerate(layers):
        Wm = [[mp.mpf(float(W[i, j])) for j in range(W.shape[1])] for i in range(W.shape[0])]
        lin = lambda v: [sum(Wm[i][j] * v[j] for j in range(W.shape[1])) for i in range(W.shape[0])]
        z = [s + mp.mpf(float(b[i])) for i, s in enumerate(lin(a))]
        z1 = {c: lin(d1[c]) for c in d1}
        z2 = {p: lin(d2[p]) for p in pairs}
        if li == len(layers) - 1:
            return z[0], (z1[0][0], z1[1][0]), {p: z2[p][0] for p in pairs}
        a = [mp.tanh(v) for v in z]
        s = [1 - mp.tanh(v) ** 2 for v in z]
        sp_ = [-2 * mp.tanh(v) * (1 - mp.tanh(v) ** 2) for v in z]
        d1 = {c: [s[i] * z1[c][i] for i in range(len(z))] for c in z1}
        d2 = {(i, j): [s[k] * z2[(i, j)][k] + sp_[k] * z1[i][k] * z1[j][k] for k in range(len(z))] for (i, j) in pairs}


def _grid(t0, t1, x0, x1, c, npts):
    s = np.linspace(0, 1, npts)
    for a in s:
        for b in s:
            yield (mp.mpf(float(t0[c])) + (mp.mpf(float(t1[c])) - mp.mpf(float(t0[c]))) * mp.mpf(a),
                   mp.mpf(float(x0[c])) + (mp.mpf(float(x1[c])) - mp.mpf(float(x0[c]))) * mp.mpf(b))


# ------------------------------------------------------------------ second-derivative channels
@pytest.mark.parametrize("bk", BACKENDS)
def test_second_derivative_channels_contain_50_digit_values(bk):
    enc = TripleEncloser(_tiny_triple(5, width=6, depth=3), StefanAdapter(StefanProblem()))
    t0, t1, x0, x1 = _cells(0, seed=5, k=10)
    tc, ht, xc, hx = 0.5 * (t0 + t1), 0.5 * (t1 - t0), 0.5 * (x0 + x1), 0.5 * (x1 - x0)
    t = T.tm_variable(bk, tc, ht, 1, 2)
    x = T.tm_variable(bk, xc, hx, 2, 2)
    for layers in (enc.Ws, enc.Wf):
        v, d1, d2 = enclose_mlp2(layers, [t, x], (0, 1), ((0, 0), (0, 1), (1, 1)))
        R = {"v": v.range(), 0: d1[0].range(), 1: d1[1].range(), **{p: d2[p].range() for p in d2}}
        for c in range(t0.size):
            for tt, xx in _grid(t0, t1, x0, x1, c, 4):
                mv, m1, m2 = _mp_mlp2(layers, tt, xx)
                ref = {"v": mv, 0: m1[0], 1: m1[1], **m2}
                for k, r in R.items():
                    assert r.lower()[c] <= ref[k] <= r.upper()[c], (k, c)


def test_second_derivatives_of_affine_networks_are_exactly_zero():
    tr = _linear_triple(state=(2.0, 0.5, -0.25), flux=(0.1, -0.3, 0.2), mult=(0.5, 0.25, -0.5))
    enc = TripleEncloser(tr, StefanAdapter(StefanProblem()))
    t0, t1, x0, x1 = _cells(0, seed=2, k=8)
    t = T.tm_variable(NB, 0.5 * (t0 + t1), 0.5 * (t1 - t0), 1, 2)
    x = T.tm_variable(NB, 0.5 * (x0 + x1), 0.5 * (x1 - x0), 2, 2)
    _, d1, d2 = enclose_mlp2(enc.Ws, [t, x], (0, 1), ((0, 0), (0, 1), (1, 1)))
    r1 = d1[0].range()  # = 0.5 up to outward rounding
    assert np.all(np.abs(r1.lower() - 0.5) <= 1e-15) and np.all(np.abs(r1.upper() - 0.5) <= 1e-15)
    for p in d2:  # zero up to the outward rounding of the range (denormals)
        r = d2[p].range()
        assert np.all(np.abs(r.lower()) <= 1e-300) and np.all(np.abs(r.upper()) <= 1e-300)


# ------------------------------------------------------------------ mean-value ranges
@pytest.mark.parametrize("bk", BACKENDS)
def test_mean_value_ranges_contain_50_digit_samples_and_corners(bk):
    enc = TripleEncloser(_tiny_triple(1), StefanAdapter(StefanProblem()), mean_value=True)
    t0, t1, x0, x1 = _cells(0, seed=1, k=12)
    mv = enc.mean_value_ranges(bk, t0, t1, x0, x1)
    for c in range(t0.size):
        for tt, xx in _grid(t0, t1, x0, x1, c, 5):
            F = _mp_fields(enc, tt, xx)
            for k in ("R", "Rhat_pos"):
                assert mv[k].lower()[c] <= F[k] <= mv[k].upper()[c], (k, c)
    # densities with the intersected ranges still dominate the 50-digit densities
    _check_fields_contain(bk, enc, t0, t1, x0, x1)


@pytest.mark.parametrize("bk", BACKENDS)
def test_mean_value_never_wider_and_densities_never_larger(bk):
    tr = _tiny_triple(3, width=8)
    old = TripleEncloser(tr, StefanAdapter(StefanProblem()))
    new = TripleEncloser(tr, StefanAdapter(StefanProblem()), mean_value=True)
    t0, t1, x0, x1 = _cells(0, seed=9, k=30)
    D0 = old.densities(bk, t0, t1, x0, x1)
    D1 = new.densities(bk, t0, t1, x0, x1)
    # arb's intersect re-rounds the endpoints outward: allow one-ulp-scale growth there
    tol = 1e-12 if bk is AB else 0.0
    assert np.all(D1["R_width"] <= D1["R_width_tm1"] * (1 + tol))
    assert np.array_equal(D1["R_width_tm1"], D0["R_width"])
    for k in ("r", "p", "c", "rhat", "R_term"):
        assert np.all(D1[k] <= D0[k] * (1 + tol)), k
    # neither form dominates in general (a smooth tiny net: TM1 wins), hence the intersection.
    # With larger weights and depth (TM1 remainders accumulate) the mean-value form must win.
    big = TripleEncloser(_tiny_triple(3, width=16, depth=3, scale=2.0), StefanAdapter(StefanProblem()),
                         mean_value=True)
    rng = np.random.default_rng(1)
    h = 2.0**-6
    s0, y0 = np.floor(rng.random(40) / h) * h, np.floor(rng.random(40) / h) * h
    D2 = big.densities(bk, s0, s0 + h, y0, y0 + h)
    assert np.all(D2["R_width_mv"] < D2["R_width_tm1"])


def test_polynomial_model_mean_value_upper_sums_from_above():
    """Affine networks (exact sympy integrals): the mean-value upper sums stay above
    the exact integrals and never exceed the TM1 upper sums."""
    tr = _linear_triple(state=(2.0, 0.5, -0.25), flux=(0.1, -0.3, 0.2), mult=(0.5, 0.25, -0.5))
    t, x = sp.symbols("t x")
    N = 2 + sp.Rational(1, 2) * t - sp.Rational(1, 4) * x
    sig = sp.Rational(1, 10) - sp.Rational(3, 10) * t + sp.Rational(1, 5) * x
    M = sp.Rational(1, 2) + sp.Rational(1, 4) * t - sp.Rational(1, 2) * x
    B = 4 * x * (1 - x)
    u = B * N
    f = -1 - (sp.exp(t) - 1) * (1 - x)
    R = f - sp.diff(u, t) - sp.diff(sig, x) + M**2
    S = sig + sp.diff(u, x)
    exact_r = float(sp.integrate(sp.integrate(sp.expand(R**2 + S**2), (x, 0, 1)), (t, 0, 1))
                    + sp.integrate(u.subs(t, 0) ** 2, (x, 0, 1)))
    for n in (4, 16):
        part, tp = US.uniform_2d(n, n), US.uniform_1d(4 * n)
        out = {}
        for mvf in (False, True):
            enc = TripleEncloser(tr, StefanAdapter(StefanProblem()), mean_value=mvf)
            us, vol, _ = US.certify_partition(enc, part, tp, tp, backend="numpy", model_frozen=True,
                                              constants_validated=True)
            assert us.status == "validated_upper_bound"
            out[mvf] = us
        assert exact_r <= out[True].U_r <= out[False].U_r


def test_fast_backend_contains_or_matches_arb_mean_value():
    enc = TripleEncloser(_tiny_triple(7, width=8, depth=3), StefanAdapter(StefanProblem()), mean_value=True)
    t0, t1, x0, x1 = _cells(0, seed=8, k=60)
    Mn = enc.mean_value_ranges(NB, t0, t1, x0, x1)
    Ma = enc.mean_value_ranges(AB, t0, t1, x0, x1)
    for k in ("R", "Rhat_pos"):
        rn, ra = Mn[k], Ma[k]
        width = rn.upper() - rn.lower()
        contain = (rn.lower() <= ra.lower()) & (rn.upper() >= ra.upper())
        close = (np.abs(rn.lower() - ra.lower()) <= width) & (np.abs(rn.upper() - ra.upper()) <= width)
        assert np.all(contain | close), k
