"""Fast validated backend: vectorised NumPy float64 intervals, outward rounding.

Every elementary + - * on interval endpoints is followed by
``np.nextafter(lo, -inf)`` / ``np.nextafter(hi, +inf)``. IEEE 754 binary64
+ - * are correctly rounded (round to nearest), so the exact result lies
within one ulp of the computed one and the widened interval contains it.
No BLAS / ``@`` / ``matmul`` is used for interval products: linear maps loop
over the inner dimension with elementwise operations.

``tanh``/``sech^2``/``exp`` are NOT taken from ``np.tanh``/``np.exp`` (not
proved correctly rounded). Instead:

* tanh and sech^2 at a float point p, |p| <= 20, use a table of arb-computed
  outward enclosures of T = tanh, S = sech^2, S' = -2 T S (= tanh''),
  S'' = 4 S - 6 S^2 at the dyadic grid g_i = i 2^-14, and the exact
  third-order Taylor formula around the nearest grid point g:
      tanh(p) = T(g) + S(g) d + S'(g) d^2/2 + tanh'''(xi) d^3/6,  |tanh'''| <= 2
      sech2(p) = S(g) + S'(g) d + S''(g) d^2/2 + sech2'''(xi) d^3/6, |sech2'''| <= 16
  with d = p - g computed EXACTLY (|d| <= 2^-15 and ulp(p) <= 2^-14 for
  |p| <= 20, so the subtraction is exact), evaluated in outward interval
  arithmetic. Bounds: tanh''' = -2 S (S - 2 T^2) ... |tanh'''| <= 2 (max at 0);
  sech2''' = (4 - 12 S)(-2 T S), |.| <= 2 * 1 * 1 * 8 = 16.
* |p| > 20: monotonicity, tanh(p) in sign(p) [T(20)_lo, 1], sech2(p) in
  [0, S(20)_hi] (tanh increasing, sech^2 decreasing on [20, inf)).
* Interval arguments: tanh increasing -> [tanh_lo(l), tanh_hi(u)]; sech^2 is
  unimodal with maximum 1 at 0 -> [min(S_lo(l), S_lo(u)), S_hi(closest)].
* exp (problem data only, once per cell, never per neuron) is evaluated with
  arb on the exact float endpoints and rounded outward.
"""

from __future__ import annotations

import math

import numpy as np
from flint import arb, ctx

NEG = -np.inf
POS = np.inf

GRID_EXP = 14
GRID_STEP = 2.0 ** -GRID_EXP
SAT = 20.0
_N_HALF = int(SAT * 2**GRID_EXP)


def _dn(x):
    return np.nextafter(x, NEG)


def _up(x):
    return np.nextafter(x, POS)


def _arb_lo(x: arb) -> float:
    lo = x.lower()  # an arb ball, not necessarily exact
    f = float(lo.mid())
    while not (arb(f) <= lo):
        f = math.nextafter(f, -math.inf)
    return f


def _arb_hi(x: arb) -> float:
    hi = x.upper()
    f = float(hi.mid())
    while not (arb(f) >= hi):
        f = math.nextafter(f, math.inf)
    return f


class _Tables:
    """Arb-built outward enclosures of T, S, S', S'' on the dyadic grid."""

    def __init__(self):
        old = ctx.prec
        ctx.prec = 96
        try:
            n = 2 * _N_HALF + 1
            lo = {k: np.empty(n) for k in ("T", "S", "S1", "S2")}
            hi = {k: np.empty(n) for k in ("T", "S", "S1", "S2")}
            for j in range(n):
                g = arb(j - _N_HALF) / arb(2**GRID_EXP)  # exact dyadic
                T = g.tanh()
                S = 1 / g.cosh() ** 2
                S1 = -2 * T * S
                S2 = 4 * S - 6 * S * S
                for k, v in (("T", T), ("S", S), ("S1", S1), ("S2", S2)):
                    lo[k][j] = _arb_lo(v)
                    hi[k][j] = _arb_hi(v)
            self.lo, self.hi = lo, hi
            sat = arb(SAT)
            self.T_sat_lo = _arb_lo(sat.tanh())
            self.S_sat_hi = _arb_hi(1 / sat.cosh() ** 2)
        finally:
            ctx.prec = old


_TABLES: _Tables | None = None


def tables() -> _Tables:
    global _TABLES
    if _TABLES is None:
        _TABLES = _Tables()
    return _TABLES


# ------------------------------------------------------------------ interval
class NIA:
    """Interval array [lo, hi] with outward-rounded arithmetic."""

    __slots__ = ("lo", "hi")
    backend = "numpy_outward_float64"

    def __init__(self, lo, hi=None):
        lo = np.asarray(lo, dtype=np.float64)
        self.lo = lo
        self.hi = lo if hi is None else np.asarray(hi, dtype=np.float64)

    # --- structure
    @property
    def shape(self):
        return self.lo.shape

    def __getitem__(self, k):
        return NIA(self.lo[k], self.hi[k])

    def __setitem__(self, k, v):
        v = as_nia(v, self.lo[k].shape)
        self.lo[k] = v.lo
        self.hi[k] = v.hi

    def copy(self):
        return NIA(self.lo.copy(), self.hi.copy())

    def reshape(self, *s):
        return NIA(self.lo.reshape(*s), self.hi.reshape(*s))

    # --- arithmetic
    def __add__(self, o):
        o = as_nia(o)
        return NIA(_dn(self.lo + o.lo), _up(self.hi + o.hi))

    __radd__ = __add__

    def __neg__(self):
        return NIA(-self.hi, -self.lo)

    def __sub__(self, o):
        o = as_nia(o)
        return NIA(_dn(self.lo - o.hi), _up(self.hi - o.lo))

    def __rsub__(self, o):
        return as_nia(o) - self

    def __mul__(self, o):
        o = as_nia(o)
        p1 = self.lo * o.lo
        p2 = self.lo * o.hi
        p3 = self.hi * o.lo
        p4 = self.hi * o.hi
        lo = np.minimum(np.minimum(p1, p2), np.minimum(p3, p4))
        hi = np.maximum(np.maximum(p1, p2), np.maximum(p3, p4))
        return NIA(_dn(lo), _up(hi))

    __rmul__ = __mul__

    # --- queries
    def upper(self):
        return self.hi

    def lower(self):
        return self.lo

    def mid(self):
        m = 0.5 * self.lo + 0.5 * self.hi
        return np.where(np.isfinite(m), m, 0.0)

    def has_nan(self) -> bool:
        return bool(np.isnan(self.lo).any() or np.isnan(self.hi).any())


def as_nia(x, shape=None):
    if isinstance(x, NIA):
        return x
    a = np.asarray(x, dtype=np.float64)
    if shape is not None:
        a = np.broadcast_to(a, shape)
    return NIA(a, a)


# ------------------------------------------------------------------ helpers
def thin(x):
    return as_nia(np.asarray(x, dtype=np.float64))


def from_bounds(lo, hi):
    return NIA(np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64))


def symmetric(r):
    r = np.asarray(r, dtype=np.float64)
    return NIA(-r, r)


def zeros(shape):
    z = np.zeros(shape)
    return NIA(z, z.copy())


def stack(items, axis=0):
    return NIA(np.stack([i.lo for i in items], axis), np.stack([i.hi for i in items], axis))


def concat(items, axis=0):
    return NIA(np.concatenate([i.lo for i in items], axis), np.concatenate([i.hi for i in items], axis))


def mag(x: NIA):
    """Upper bound of |x| (float array)."""
    return np.maximum(np.abs(x.lo), np.abs(x.hi))


def sqr(x: NIA) -> NIA:
    a2 = x.lo * x.lo
    b2 = x.hi * x.hi
    lo = np.where(x.lo >= 0, a2, np.where(x.hi <= 0, b2, 0.0))
    hi = np.maximum(a2, b2)
    lo = np.where(lo > 0, _dn(lo), lo)
    return NIA(lo, _up(hi))


def hull(a: NIA, b: NIA) -> NIA:
    return NIA(np.minimum(a.lo, b.lo), np.maximum(a.hi, b.hi))


def include_point(z: NIA, p: NIA) -> NIA:
    """Hull of the range z and the expansion point p (v18 D35): a Taylor expansion at p needs the derivative
    enclosure over hull(p, range). min/max only, so z is returned bit for bit when p already lies in z."""
    return NIA(np.minimum(z.lo, p.lo), np.maximum(z.hi, p.hi))


def intersect(a: NIA, b: NIA) -> NIA:
    lo = np.maximum(a.lo, b.lo)
    hi = np.minimum(a.hi, b.hi)
    if np.any(lo > hi):
        raise ArithmeticError("empty intersection of two validated enclosures (bug)")
    return NIA(lo, hi)


def sum_axis(x: NIA, axis: int) -> NIA:
    """Outward sum along an axis, one rounding-widened addition at a time."""
    lo = np.moveaxis(x.lo, axis, 0)
    hi = np.moveaxis(x.hi, axis, 0)
    alo = lo[0].copy()
    ahi = hi[0].copy()
    for k in range(1, lo.shape[0]):
        alo = _dn(alo + lo[k])
        ahi = _up(ahi + hi[k])
    return NIA(alo, ahi)


def linmap(W: np.ndarray, X: NIA, bias: np.ndarray | None = None) -> NIA:
    """Y = W X (+ bias[:, None]) for exact float W (m, k) and interval X (k, cols).
    Loops over the inner dimension with elementwise outward operations."""
    m, k = W.shape
    cols = X.lo.shape[1:]
    if bias is None:
        alo = np.zeros((m,) + cols)
        ahi = np.zeros((m,) + cols)
    else:
        alo = np.broadcast_to(bias.reshape((m,) + (1,) * len(cols)), (m,) + cols).copy()
        ahi = alo.copy()
    ex = (slice(None),) + (None,) * len(cols)
    for j in range(k):
        w = W[:, j][ex]
        xl = X.lo[j][None]
        xh = X.hi[j][None]
        p1 = w * xl
        p2 = w * xh
        plo = _dn(np.minimum(p1, p2))
        phi = _up(np.maximum(p1, p2))
        alo = _dn(alo + plo)
        ahi = _up(ahi + phi)
    return NIA(alo, ahi)


# ------------------------------------------------------------------ elementary
def _poly_enclose(p, kind):
    """Enclosure of tanh(p) or sech2(p) at float points p (array)."""
    tb = tables()
    p = np.asarray(p, dtype=np.float64)
    inside = np.abs(p) <= SAT
    pc = np.where(inside, p, 0.0)
    i = np.rint(pc * 2**GRID_EXP).astype(np.int64)
    g = i.astype(np.float64) * GRID_STEP
    d = pc - g  # exact (module docstring)
    j = i + _N_HALF
    D = NIA(d, d)
    D2 = sqr(D) * 0.5
    ad3 = _up(_up(np.abs(d) * np.abs(d)) * np.abs(d))
    if kind == "tanh":
        c0 = NIA(tb.lo["T"][j], tb.hi["T"][j])
        c1 = NIA(tb.lo["S"][j], tb.hi["S"][j])
        c2 = NIA(tb.lo["S1"][j], tb.hi["S1"][j])
        rem = _up(ad3 / 3.0)  # (2/6)|d|^3; division by 3 rounded up
        rem = _up(rem)
    else:
        c0 = NIA(tb.lo["S"][j], tb.hi["S"][j])
        c1 = NIA(tb.lo["S1"][j], tb.hi["S1"][j])
        c2 = NIA(tb.lo["S2"][j], tb.hi["S2"][j])
        rem = _up(_up(ad3 * 16.0) / 6.0)
        rem = _up(rem)
    val = c0 + c1 * D + c2 * D2 + symmetric(rem)
    if kind == "tanh":
        sat_lo = np.where(p > 0, tb.T_sat_lo, -1.0)
        sat_hi = np.where(p > 0, 1.0, -tb.T_sat_lo)
        lo = np.where(inside, np.maximum(val.lo, -1.0), sat_lo)
        hi = np.where(inside, np.minimum(val.hi, 1.0), sat_hi)
    else:
        lo = np.where(inside, np.maximum(val.lo, 0.0), 0.0)
        hi = np.where(inside, np.minimum(val.hi, 1.0), tb.S_sat_hi)
    # v18 D34: a NaN argument fails |p| <= SAT and would otherwise get the finite saturation enclosure; propagate
    # it so that the cell density is non-finite and the status is demoted (upper_sum.reduce_upper_sums)
    nan = np.isnan(p)
    return np.where(nan, np.nan, lo), np.where(nan, np.nan, hi)


def tanh(x: NIA) -> NIA:
    lo, _ = _poly_enclose(x.lo, "tanh")
    _, hi = _poly_enclose(x.hi, "tanh")
    return NIA(lo, hi)


def sech2(x: NIA) -> NIA:
    """Range of sech^2 over each interval (unimodal, max 1 at 0)."""
    l_lo, l_hi = _poly_enclose(x.lo, "sech2")
    u_lo, u_hi = _poly_enclose(x.hi, "sech2")
    contains0 = (x.lo <= 0) & (x.hi >= 0)
    closest_hi = np.where(x.lo > 0, l_hi, u_hi)  # closest endpoint to 0
    hi = np.where(contains0, 1.0, closest_hi)
    lo = np.minimum(l_lo, u_lo)
    nan = np.isnan(x.lo) | np.isnan(x.hi)  # v18 D34: never mask a NaN endpoint
    return NIA(np.where(nan, np.nan, lo), np.where(nan, np.nan, hi))


def exp(x: NIA) -> NIA:
    """Monotone; endpoints by arb (data only -- one call per cell, not per neuron)."""
    lo = np.empty_like(x.lo)
    hi = np.empty_like(x.hi)
    flo = x.lo.ravel()
    fhi = x.hi.ravel()
    olo = lo.ravel()
    ohi = hi.ravel()
    cache = {}
    for n in range(flo.size):
        a = float(flo[n])
        b = float(fhi[n])
        if a not in cache:
            cache[a] = arb(a).exp()
        if b not in cache:
            cache[b] = arb(b).exp()
        olo[n] = _arb_lo(cache[a])
        ohi[n] = _arb_hi(cache[b])
    return NIA(lo.reshape(x.lo.shape), hi.reshape(x.hi.shape))


def recip(x: NIA) -> NIA:
    """1/x on intervals with x.lo > 0 (IEEE division is correctly rounded, so one
    outward step encloses the exact quotient). Non-positive lower ends give
    [-inf, inf] (and hence a non-finite density -> status demoted)."""
    pos = x.lo > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        lo = np.where(pos, _dn(1.0 / np.where(pos, x.hi, 1.0)), -np.inf)
        hi = np.where(pos, _up(1.0 / np.where(pos, x.lo, 1.0)), np.inf)
    return NIA(lo, hi)


def select(mask, a: NIA, b: NIA) -> NIA:
    """Elementwise a where mask else b (no arithmetic, exact)."""
    return NIA(np.where(mask, a.lo, b.lo), np.where(mask, a.hi, b.hi))


def rational(num: int, den: int, shape=()) -> NIA:
    """Enclosure of the exact rational num/den: the correctly rounded quotient
    widened by one ulp on each side (exact when the quotient is a float)."""
    from fractions import Fraction

    q = num / den
    lo, hi = (q, q) if Fraction(q) == Fraction(num, den) else (_dn(q), _up(q))
    return NIA(np.full(shape, lo), np.full(shape, hi))


# ------------------------------------------------------------------ v20: sin(pi x), cos(pi x), pi (problem data only)
_TRIG_PREC = 128


def _arb_range_fn(x: NIA, method: str) -> NIA:
    """Range of z -> sin(pi z) / cos(pi z) over each [lo, hi]: arb ball evaluation (sin_pi / cos_pi) on the ball
    hull of the exact float endpoints (a ball that contains [lo, hi]), endpoints rounded outward to floats. Arb's
    ball functions return a ball containing f(z) for every z in the input ball, so the result contains the range.
    Data only (once per cell, cached by endpoints); NaN endpoints give NaN (v18 D34)."""
    lo = np.empty_like(x.lo)
    hi = np.empty_like(x.hi)
    flo, fhi = x.lo.ravel(), x.hi.ravel()
    olo, ohi = lo.ravel(), hi.ravel()
    cache = {}
    old = ctx.prec
    ctx.prec = _TRIG_PREC
    try:
        for n in range(flo.size):
            a, b = float(flo[n]), float(fhi[n])
            if not (math.isfinite(a) and math.isfinite(b)):
                olo[n] = ohi[n] = np.nan
                continue
            key = (a, b)
            if key not in cache:
                ball = arb(a) if a == b else arb(a).union(arb(b))
                v = getattr(ball, method)()
                cache[key] = (_arb_lo(v), _arb_hi(v))
            olo[n], ohi[n] = cache[key]
    finally:
        ctx.prec = old
    return NIA(lo.reshape(x.lo.shape), hi.reshape(x.hi.shape))


def sinpi(x: NIA) -> NIA:
    return _arb_range_fn(x, "sin_pi")


def cospi(x: NIA) -> NIA:
    return _arb_range_fn(x, "cos_pi")


def pi_const(shape=()) -> NIA:
    """Outward float enclosure of pi."""
    old = ctx.prec
    ctx.prec = _TRIG_PREC
    try:
        p = arb.pi()
        lo, hi = _arb_lo(p), _arb_hi(p)
    finally:
        ctx.prec = old
    return NIA(np.full(shape, lo), np.full(shape, hi))
