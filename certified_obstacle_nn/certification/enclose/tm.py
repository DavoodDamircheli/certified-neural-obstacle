"""First-order Taylor models with interval coefficients (backend-generic).

A TM over a box K = prod_j [m_j - r_j, m_j + r_j] in local variables
e in [-1, 1]^n is

    q(e) in c_0 + sum_j c_j e_j + I        for every e in [-1,1]^n,

with c_0..c_n and the remainder I all intervals (validated backend objects).
Storage: one interval array ``c`` of shape (n + 2, *batch): c[0] constant,
c[1..n] linear coefficients, c[n+1] remainder.

Soundness of each operation (all interval operations are outward):

* sum / difference / multiplication by an interval scalar: termwise.
* linear map with an exact float matrix: termwise (each component is mapped;
  the bias is added to the constant).
* product A B: c0 = a0 b0, c_j = a0 b_j + b0 a_j, and
  I = [-LA LB, LA LB] + P_A I_B + P_B I_A + I_A I_B, with LA = sum_j |a_j|
  (rounded up), P_A = a0 + [-LA, LA]. Exact expansion of
  (a0 + La + I_A)(b0 + Lb + I_B) with |La Lb| <= LA LB.
* square A^2: same with La^2 in [0, LA^2].
* phi(A) for phi in {tanh, sech^2, exp}: Taylor at the float p = mid(a0):
  phi(z) = phi(p) + phi'(p)(z - p) + phi''(xi)(z - p)^2 / 2, xi in Z = range(A);
  z - p is the TM D = A - p, so
  phi(A) in phi(p) + phi'(p) D + [phi''(Z)] * sqr(range(D)) / 2,
  with phi''(Z) an interval enclosure over Z = hull(p, range(A)) (xi lies between p and z; v18 D35):
    tanh''  = -2 tanh sech^2,  (sech^2)'' = 4 S - 6 S^2,  exp'' = exp.
"""

from __future__ import annotations

import numpy as np


def _add_up(a, b):
    return np.nextafter(a + b, np.inf)


def _mul_up(a, b):
    return np.nextafter(a * b, np.inf)


class TM:
    __slots__ = ("bk", "c")

    def __init__(self, bk, c):
        self.bk = bk
        self.c = c

    # ----------------------------------------------------------- structure
    @property
    def n(self) -> int:
        return self.c.shape[0] - 2

    @property
    def const(self):
        return self.c[0]

    @property
    def rem(self):
        return self.c[self.n + 1]

    def lin_mag(self):
        """Upper bound of sum_j |c_j| (float array, rounded up)."""
        bk = self.bk
        acc = np.zeros(self.c.shape[1:])
        for j in range(1, self.n + 1):
            acc = _add_up(acc, bk.mag(self.c[j]))
        return acc

    def poly_range(self):
        return self.const + self.bk.symmetric(self.lin_mag())

    def range(self):
        return self.poly_range() + self.rem

    def __getitem__(self, k):
        if not isinstance(k, tuple):
            k = (k,)
        return TM(self.bk, self.c[(slice(None),) + k])

    # ----------------------------------------------------------- arithmetic
    def __add__(self, o):
        if isinstance(o, TM):
            return TM(self.bk, self.c + o.c)
        c = self.c.copy()
        c[0] = self.c[0] + o
        return TM(self.bk, c)

    __radd__ = __add__

    def __neg__(self):
        return TM(self.bk, -self.c)

    def __sub__(self, o):
        if isinstance(o, TM):
            return TM(self.bk, self.c - o.c)
        c = self.c.copy()
        c[0] = self.c[0] - o
        return TM(self.bk, c)

    def __rsub__(self, o):
        return (-self) + o

    def scale(self, s):
        """Multiply by an interval (or float) scalar of batch shape."""
        bk = self.bk
        s = s if not isinstance(s, (int, float, np.ndarray)) else bk.thin(np.broadcast_to(np.asarray(s, float), self.c.shape[1:]))
        return TM(bk, bk.stack([self.c[i] * s for i in range(self.n + 2)]))

    def __mul__(self, o):
        if not isinstance(o, TM):
            return self.scale(o)
        return tm_mul(self, o)

    __rmul__ = __mul__


def tm_mul(A: TM, B: TM) -> TM:
    bk = A.bk
    n = A.n
    a0, b0 = A.const, B.const
    LA, LB = A.lin_mag(), B.lin_mag()
    PA = a0 + bk.symmetric(LA)
    PB = b0 + bk.symmetric(LB)
    comps = [a0 * b0]
    for j in range(1, n + 1):
        comps.append(a0 * B.c[j] + b0 * A.c[j])
    ra, rb = A.rem, B.rem
    rem = bk.symmetric(_mul_up(LA, LB)) + PA * rb + PB * ra + ra * rb
    comps.append(rem)
    return TM(bk, bk.stack(comps))


def tm_sqr(A: TM) -> TM:
    bk = A.bk
    n = A.n
    a0 = A.const
    LA = A.lin_mag()
    PA = a0 + bk.symmetric(LA)
    comps = [bk.sqr(a0)]
    for j in range(1, n + 1):
        comps.append(a0 * A.c[j] * 2.0)
    ra = A.rem
    rem = bk.from_bounds(np.zeros_like(LA), _mul_up(LA, LA)) + PA * ra * 2.0 + bk.sqr(ra)
    comps.append(rem)
    return TM(bk, bk.stack(comps))


def _taylor1(A: TM, f_p, df_p, d2f_Z) -> TM:
    """phi(A) = phi(p) + phi'(p) D + phi''(Z) sqr(range D)/2 (module docstring)."""
    bk = A.bk
    n = A.n
    D = A - bk.thin(A.const.mid())
    DR = D.range()
    comps = [f_p + df_p * D.const]
    for j in range(1, n + 1):
        comps.append(df_p * D.c[j])
    comps.append(df_p * D.rem + d2f_Z * bk.sqr(DR) * 0.5)
    return TM(bk, bk.stack(comps))


def tm_tanh_sech2(A: TM) -> tuple[TM, TM]:
    """(tanh(A), sech^2(A)) sharing the expansion point and the range."""
    bk = A.bk
    P = bk.thin(A.const.mid())
    Z = bk.include_point(A.range(), P)  # xi lies between p and z (v18 D35)
    T_p = bk.tanh(P)
    S_p = bk.sech2(P)
    T_Z = bk.tanh(Z)
    S_Z = bk.sech2(Z)
    d2tanh_Z = T_Z * S_Z * -2.0
    dS_p = T_p * S_p * -2.0
    d2S_Z = S_Z * 4.0 - bk.sqr(S_Z) * 6.0
    return _taylor1(A, T_p, S_p, d2tanh_Z), _taylor1(A, S_p, dS_p, d2S_Z)


def tm_exp(A: TM) -> TM:
    bk = A.bk
    P = bk.thin(A.const.mid())
    E_p = bk.exp(P)
    E_Z = bk.exp(bk.include_point(A.range(), P))  # v18 D35
    return _taylor1(A, E_p, E_p, E_Z)


def tm_sinpi(A: TM) -> TM:
    """sin(pi A) (v20 data only): phi = sin(pi z), phi' = pi cos(pi z), phi'' = -pi^2 sin(pi z) enclosed over
    Z = hull(p, range(A)); point and range values from the backend's arb-evaluated sinpi / cospi, pi as an interval."""
    bk = A.bk
    P = bk.thin(A.const.mid())
    Z = bk.include_point(A.range(), P)
    pi = bk.pi_const(P.shape)
    return _taylor1(A, bk.sinpi(P), pi * bk.cospi(P), -(bk.sqr(pi) * bk.sinpi(Z)))


def tm_const(bk, value, n: int, batch_shape) -> TM:
    """Constant TM (exact float value broadcast to batch)."""
    v = np.broadcast_to(np.asarray(value, dtype=np.float64), batch_shape)
    comps = [bk.thin(v)] + [bk.zeros(batch_shape) for _ in range(n + 1)]
    return TM(bk, bk.stack(comps))


def tm_variable(bk, center, halfwidth, j: int, n: int) -> TM:
    """The coordinate center + halfwidth * e_j (exact floats)."""
    center = np.asarray(center, dtype=np.float64)
    halfwidth = np.asarray(halfwidth, dtype=np.float64)
    comps = [bk.thin(center)]
    for k in range(1, n + 1):
        comps.append(bk.thin(halfwidth) if k == j else bk.zeros(center.shape))
    comps.append(bk.zeros(center.shape))
    return TM(bk, bk.stack(comps))


def tm_stack(items: list[TM], axis: int = 0) -> TM:
    """Stack TMs along a new batch axis (axis counts batch dims)."""
    bk = items[0].bk
    return TM(bk, bk.stack([t.c for t in items], axis=axis + 1))


def tm_linmap(W: np.ndarray, A: TM, bias: np.ndarray | None = None) -> TM:
    """W (m,k) exact applied along batch axis 0 of A (shape (n+2, k, B))."""
    bk = A.bk
    n2, k, B = A.c.shape
    X = _transpose_to_rows(bk, A.c)  # (k, n2, B)
    Y = bk.linmap(W, X)  # (m, n2, B)
    c = _transpose_to_rows(bk, Y)  # (n2, m, B)
    if bias is not None:
        m = W.shape[0]
        c0 = c[0] + bk.thin(np.broadcast_to(bias.reshape(m, 1), (m, B)))
        c[0] = c0
    return TM(bk, c)


def _transpose_to_rows(bk, x):
    """Swap the first two axes of an interval array (backend-agnostic)."""
    if hasattr(x, "lo"):
        return type(x)(np.swapaxes(x.lo, 0, 1).copy(), np.swapaxes(x.hi, 0, 1).copy())
    return type(x)(np.swapaxes(x.v, 0, 1).copy())


def tm_recip(A: TM) -> TM:
    """1/A for TMs whose range is strictly positive: phi = 1/z, phi' = -1/z^2,
    phi'' = 2/z^3 enclosed over the whole range Z (backend ``recip``)."""
    bk = A.bk
    P = bk.thin(A.const.mid())
    iP = bk.recip(P)
    iZ = bk.recip(bk.include_point(A.range(), P))  # v18 D35
    return _taylor1(A, iP, -bk.sqr(iP), iZ * bk.sqr(iZ) * 2.0)


def tm_add_interval(A: TM, I) -> TM:
    """A + I with I an interval of batch shape, absorbed into the remainder."""
    c = A.c.copy()
    c[A.n + 1] = A.rem + I
    return TM(A.bk, c)


def tm_select(mask, A: TM, B: TM) -> TM:
    """Per-batch-element choice between two TMs (exact; no arithmetic)."""
    bk = A.bk
    m = np.broadcast_to(np.asarray(mask, dtype=bool), A.c.shape[1:])
    return TM(bk, bk.stack([bk.select(m, A.c[i], B.c[i]) for i in range(A.n + 2)]))


def tm_rational(bk, num: int, den: int, n: int, batch_shape) -> TM:
    """Constant TM enclosing the exact rational num/den."""
    comps = [bk.rational(num, den, batch_shape)] + [bk.zeros(batch_shape) for _ in range(n + 1)]
    return TM(bk, bk.stack(comps))
