"""v18 Task 1a hardening (DECISIONS D34, D35): NaN endpoints are never masked, and every Taylor expansion uses the
derivative range over hull(p, range(A))."""

import numpy as np
from flint import arb

from certified_obstacle_nn.certification.enclose import intervals_arb as A
from certified_obstacle_nn.certification.enclose import intervals_np as N
from certified_obstacle_nn.certification.enclose import tm as T
from certified_obstacle_nn.certification.enclose.engine import nan_guard


def test_np_tanh_sech2_propagate_nan():
    x = N.NIA(np.array([np.nan, 1.0, -30.0, 30.0]), np.array([np.nan, 2.0, -25.0, 40.0]))
    for f in (N.tanh, N.sech2):
        y = f(x)
        assert np.isnan(y.lo[0]) and np.isnan(y.hi[0])
        assert np.all(np.isfinite(y.lo[1:])) and np.all(np.isfinite(y.hi[1:]))
    t = N.tanh(x)
    assert t.lo[2] == -1.0 and t.hi[3] == 1.0  # saturation branch unchanged


def test_nan_guard_marks_whole_cell():
    g = N.NIA(np.array([np.nan, -1.0]), np.array([np.nan, -0.5]))
    out = nan_guard({"p": np.where(g.upper() > 0, 1.0, 0.0), "sign": np.array([0, -1])}, [g])
    assert np.isnan(out["p"][0]) and out["p"][1] == 0.0 and out["sign"].dtype.kind == "i"
    same = {"p": np.array([1.0, 2.0])}
    assert nan_guard(same, [N.NIA(np.zeros(2), np.ones(2))]) is same


def test_include_point_np_bitwise_when_contained():
    z = N.NIA(np.array([0.0, 1.0]), np.array([1.0, 3.0]))
    p = N.thin(np.array([0.5, 4.0]))
    h = N.include_point(z, p)
    assert h.lo.tolist() == [0.0, 1.0] and h.hi.tolist() == [1.0, 4.0]


def test_include_point_arb_constant_rational():
    """3/20 at 64 bits: the float midpoint lies outside the ball; the Taylor range must contain it."""
    A.set_precision(64)
    sh = (2,)
    t = T.tm_variable(A, np.zeros(2), np.zeros(2), 1, 2)
    R = T.tm_rational(A, 3, 20, 2, sh) + t * A.rational(1, 5, sh)
    P = A.thin(R.const.mid())
    Z = R.range()
    assert not Z.v[0].contains(P.v[0])  # the situation D35 fixes
    H = A.include_point(Z, P)
    assert H.v[0].contains(P.v[0]) and H.v[0].contains(arb(3) / 20)
    inv = T.tm_recip(R).range()
    assert inv.v[0].contains(arb(20) / 3)
    wide = A.from_bounds(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    same = A.include_point(wide, A.thin(np.array([0.5, 0.5])))
    assert same.v[0] is wide.v[0]  # unchanged object where contained
