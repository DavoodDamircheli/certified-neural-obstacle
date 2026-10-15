"""Reference validated backend: python-flint ``arb`` balls (Arb library).

Every value is an arb ball; every arb operation returns a ball that contains
the exact result for all points of the input balls, so enclosure is automatic.
Floats (weights, biases, cell endpoints) are converted exactly (``arb(float)``
does not round: every float is a dyadic rational). Linear layers use the
C-level ``arb_mat`` product (ball arithmetic, rigorous). Squares use
``|x| * |x|`` (``x**2`` returns NaN on balls containing 0 in python-flint 0.9).
sech^2 ranges are computed from point evaluations at the endpoints (sech^2 is
unimodal with maximum 1 at 0) because ``sech`` on a wide ball is loose.

Library and precision are recorded by ``backend_info()``.
"""

from __future__ import annotations

import math

import flint
import numpy as np
from flint import arb, arb_mat, ctx

DEFAULT_PREC = 64

_arb = np.frompyfunc(arb, 1, 1)
_tanh = np.frompyfunc(lambda z: z.tanh(), 1, 1)
_exp = np.frompyfunc(lambda z: z.exp(), 1, 1)
_abs = np.frompyfunc(abs, 1, 1)
_union = np.frompyfunc(lambda a, b: a.union(b), 2, 1)
_lower = np.frompyfunc(lambda z: z.lower(), 1, 1)
_upper = np.frompyfunc(lambda z: z.upper(), 1, 1)


def set_precision(bits: int):
    ctx.prec = bits


def backend_info() -> dict:
    return {"backend": "arb", "library": "python-flint", "version": flint.__version__, "prec_bits": ctx.prec}


def _f_up(x: arb) -> float:
    if not x.is_finite():
        return math.inf
    u = x.upper()
    f = float(u.mid())
    while not (arb(f) >= u):
        f = math.nextafter(f, math.inf)
    return f


def _f_dn(x: arb) -> float:
    if not x.is_finite():
        return -math.inf
    u = x.lower()
    f = float(u.mid())
    while not (arb(f) <= u):
        f = math.nextafter(f, -math.inf)
    return f


def _interval_mul(a: arb, b: arb) -> arb:
    """Product with INTERVAL (endpoint) semantics: the ball hull of the four
    endpoint products. Midpoint-radius ball products overestimate products of
    two wide balls by up to 50% ((1+-1)(1+-1) = 1+-3 vs [0,4]); with an exact
    factor the ball product is already tight and is used directly."""
    if a.is_exact() or b.is_exact():
        return a * b
    al, ah = a.lower(), a.upper()
    bl, bh = b.lower(), b.upper()
    return (al * bl).union(al * bh).union(ah * bl).union(ah * bh)


_imul = np.frompyfunc(_interval_mul, 2, 1)

_fup = np.frompyfunc(_f_up, 1, 1)
_fdn = np.frompyfunc(_f_dn, 1, 1)
_fmid = np.frompyfunc(lambda z: float(z.mid()), 1, 1)
_isfin = np.frompyfunc(lambda z: z.is_finite(), 1, 1)


class AIA:
    __slots__ = ("v",)
    backend = "arb"

    def __init__(self, v):
        self.v = v

    @property
    def shape(self):
        return self.v.shape

    def __getitem__(self, k):
        return AIA(self.v[k])

    def __setitem__(self, k, val):
        self.v[k] = as_aia(val, self.v[k].shape).v

    def copy(self):
        return AIA(self.v.copy())

    def reshape(self, *s):
        return AIA(self.v.reshape(*s))

    def __add__(self, o):
        return AIA(self.v + as_aia(o).v)

    __radd__ = __add__

    def __neg__(self):
        return AIA(-self.v)

    def __sub__(self, o):
        return AIA(self.v - as_aia(o).v)

    def __rsub__(self, o):
        return AIA(as_aia(o).v - self.v)

    def __mul__(self, o):
        return AIA(np.asarray(_imul(self.v, as_aia(o).v), dtype=object))

    __rmul__ = __mul__

    def upper(self):
        return np.asarray(_fup(self.v), dtype=np.float64)

    def lower(self):
        return np.asarray(_fdn(self.v), dtype=np.float64)

    def mid(self):
        m = np.asarray(_fmid(self.v), dtype=np.float64)
        return np.where(np.isfinite(m), m, 0.0)

    def has_nan(self) -> bool:
        return not bool(np.all(np.asarray(_isfin(self.v), dtype=bool)))


def as_aia(x, shape=None):
    if isinstance(x, AIA):
        return x
    a = np.asarray(x, dtype=np.float64)
    if shape is not None:
        a = np.broadcast_to(a, shape)
    return AIA(np.asarray(_arb(a), dtype=object).reshape(a.shape))


def thin(x):
    return as_aia(np.asarray(x, dtype=np.float64))


def from_bounds(lo, hi):
    return AIA(np.asarray(_union(as_aia(lo).v, as_aia(hi).v), dtype=object))


def symmetric(r):
    r = np.asarray(r, dtype=np.float64)
    return from_bounds(-r, r)


def zeros(shape):
    return thin(np.zeros(shape))


def stack(items, axis=0):
    return AIA(np.stack([i.v for i in items], axis))


def concat(items, axis=0):
    return AIA(np.concatenate([i.v for i in items], axis))


def mag(x: AIA):
    return np.maximum(np.abs(x.lower()), np.abs(x.upper()))


def _sqr1(z: arb) -> arb:
    lo, hi = z.lower(), z.upper()
    a2, b2 = lo * lo, hi * hi
    if lo >= 0:
        return a2.union(b2)
    if hi <= 0:
        return b2.union(a2)
    return arb(0).union(a2).union(b2)


_sqr = np.frompyfunc(_sqr1, 1, 1)


def sqr(x: AIA) -> AIA:
    return AIA(np.asarray(_sqr(x.v), dtype=object))


def hull(a: AIA, b: AIA) -> AIA:
    return AIA(np.asarray(_union(a.v, b.v), dtype=object))


def _incl1(z: arb, p: arb) -> arb:
    return z if z.contains(p) else z.union(p)


_incl = np.frompyfunc(_incl1, 2, 1)


def include_point(z: AIA, p: AIA) -> AIA:
    """Hull of the range z and the expansion point p (v18 D35). The float midpoint of a narrow ball can lie outside
    the ball (e.g. the constant 3/20 at 64 bits); z is returned unchanged wherever it already contains p."""
    return AIA(np.asarray(_incl(z.v, p.v), dtype=object))


def intersect(a: AIA, b: AIA) -> AIA:
    lo = np.maximum(a.lower(), b.lower())
    hi = np.minimum(a.upper(), b.upper())
    if np.any(lo > hi):
        raise ArithmeticError("empty intersection of two validated enclosures (bug)")
    return from_bounds(lo, hi)


def sum_axis(x: AIA, axis: int) -> AIA:
    return AIA(np.asarray(np.sum(x.v, axis=axis), dtype=object))


_WCACHE: dict = {}


def _wmat(W: np.ndarray):
    key = (id(W), W.shape)
    m = _WCACHE.get(key)
    if m is None or m[0] is not W:
        m = (W, arb_mat([[arb(float(v)) for v in row] for row in W]))
        _WCACHE[key] = m
    return m[1]


def linmap(W: np.ndarray, X: AIA, bias: np.ndarray | None = None) -> AIA:
    """Y = W X (+ bias) with arb_mat ball products (rigorous)."""
    m, k = W.shape
    cols_shape = X.v.shape[1:]
    Xf = X.v.reshape(k, -1)
    ncol = Xf.shape[1]
    Xm = arb_mat(Xf.tolist())
    Ym = _wmat(W) * Xm
    Y = np.empty(m * ncol, dtype=object)
    Y[:] = Ym.entries()
    Y = Y.reshape((m,) + cols_shape)
    out = AIA(Y)
    if bias is not None:
        b = as_aia(bias.reshape((m,) + (1,) * len(cols_shape)))
        out = AIA(out.v + b.v)
    return out


def tanh(x: AIA) -> AIA:
    """Range over each ball: tanh is increasing, evaluate at the end points."""
    lo = AIA(np.asarray(_lower(x.v), dtype=object))
    hi = AIA(np.asarray(_upper(x.v), dtype=object))
    tl = np.asarray(_tanh(lo.v), dtype=object)
    th = np.asarray(_tanh(hi.v), dtype=object)
    return AIA(np.asarray(_union(tl, th), dtype=object))


def _sech2_point(z: arb) -> arb:
    c = z.cosh()
    return 1 / (c * c)


_sech2p = np.frompyfunc(_sech2_point, 1, 1)


def sech2(x: AIA) -> AIA:
    """Range of sech^2 (unimodal, max 1 at 0) from endpoint evaluations."""
    lo_f = x.lower()
    hi_f = x.upper()
    lo = np.asarray(_lower(x.v), dtype=object)
    hi = np.asarray(_upper(x.v), dtype=object)
    sl = np.asarray(_sech2p(lo), dtype=object)
    sh = np.asarray(_sech2p(hi), dtype=object)
    span = np.asarray(_union(sl, sh), dtype=object)
    contains0 = (lo_f <= 0) & (hi_f >= 0)
    one = arb(1)
    out = span.copy()
    idx = np.nonzero(contains0.ravel())[0]
    flat = out.reshape(-1)
    for i in idx:
        flat[i] = flat[i].union(one)
    return AIA(out)


def exp(x: AIA) -> AIA:
    lo = np.asarray(_lower(x.v), dtype=object)
    hi = np.asarray(_upper(x.v), dtype=object)
    return AIA(np.asarray(_union(_exp(lo), _exp(hi)), dtype=object))


def _recip1(z: arb) -> arb:
    lo, hi = z.lower(), z.upper()
    if not lo > 0:
        return arb("nan")
    return (1 / hi).union(1 / lo)


_recip = np.frompyfunc(_recip1, 1, 1)


def recip(x: AIA) -> AIA:
    """1/x on balls contained in (0, inf) (endpoint evaluation; 1/x is decreasing).
    A ball that is not strictly positive yields NaN (status demoted)."""
    return AIA(np.asarray(_recip(x.v), dtype=object))


def select(mask, a: AIA, b: AIA) -> AIA:
    return AIA(np.where(mask, a.v, b.v))


def rational(num: int, den: int, shape=()) -> AIA:
    q = arb(num) / arb(den)
    out = np.empty(shape, dtype=object)
    out[...] = q
    return AIA(out if shape else np.asarray(q, dtype=object))


# v20: sin(pi x), cos(pi x) over balls (arb ball functions contain the range over the ball), pi
_sinpi = np.frompyfunc(lambda z: z.sin_pi(), 1, 1)
_cospi = np.frompyfunc(lambda z: z.cos_pi(), 1, 1)


def sinpi(x: AIA) -> AIA:
    return AIA(np.asarray(_sinpi(x.v), dtype=object))


def cospi(x: AIA) -> AIA:
    return AIA(np.asarray(_cospi(x.v), dtype=object))


def pi_const(shape=()) -> AIA:
    out = np.empty(shape, dtype=object)
    out[...] = arb.pi()
    return AIA(out if shape else np.asarray(arb.pi(), dtype=object))
