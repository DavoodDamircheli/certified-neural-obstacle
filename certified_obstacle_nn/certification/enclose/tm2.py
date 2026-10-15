"""v17b Task 4: validated second-order Taylor models (TM2) for a tanh MLP and its first input derivatives
(backend-generic: intervals_np outward float64 or intervals_arb). Opt-in only (engine2d.parse_enclosure "TM2");
the default TM1 path is untouched. Derivations: docs/v17_enclosure_derivations.md, section "TM2".

A TM2 over a cell with local coordinates e in [-1, 1]^n is
    q(e) in c_0 + sum_j c_j e_j + sum_{i <= j} c_ij e_i e_j + I,
with interval coefficients. Storage: interval array c of shape (1 + n + n(n+1)/2 + 1, *batch):
c[0] constant, c[1..n] linear, then the quadratic coefficients in the order of `pairs(n)`, last the remainder.
With LA = sum_j |c_j|, QA = sum_{i<=j} |c_ij| (rounded up), the polynomial part ranges over
P_A = c_0 + sum_j c_j [-1, 1] + sum_i c_ii [0, 1] + sum_{i<j} c_ij [-1, 1].

Rules (all interval operations outward):
* sum / difference / exact linear map: termwise (coefficient rows mapped; bias into c_0).
* product A B = pA pB + pA I_B + pB I_A + I_A I_B with pA = a_0 + LinA + QuadA:
    c_0 = a_0 b_0;  c_j = a_0 b_j + b_0 a_j;
    c_ii = a_0 b_ii + b_0 a_ii + a_i b_i;  c_ij = a_0 b_ij + b_0 a_ij + a_i b_j + a_j b_i  (i < j);
    I = [-H, H] + P_A I_B + P_B I_A + I_A I_B,  H = LA QB + QA LB + QA QB  (|LinA QuadB + QuadA LinB + QuadA QuadB|).
  Square: c_ii = 2 a_0 a_ii + a_i^2, c_ij = 2 a_0 a_ij + 2 a_i a_j, H = 2 LA QA + QA^2, I = [-H, H] + 2 P_A I_A + sqr(I_A).
* phi(A), phi in {tanh, sech^2}: Taylor of order 2 at the float p = mid(a_0) with the Lagrange remainder of order 3,
    phi(z) = phi(p) + phi'(p) (z - p) + phi''(p) (z - p)^2 / 2 + phi'''(xi) (z - p)^3 / 6,  xi in Z = range(A);
  D = A - p (TM2), D^2 by the square rule, (z - p)^3 in range(D)^3 (interval cube), so
    phi(A) in phi(p) + phi'(p) D + (phi''(p)/2) D^2 + phi'''(Z) range(D)^3 / 6.
  With T = tanh, S = sech^2 = 1 - T^2 and u = T^2 (the P_q polynomials, tanh^(q) = P_q(T)):
    T' = S, T'' = -2 T S, T''' = P_3 = -2 + 8u - 6u^2,
    S' = -2 T S, S'' = 4 S - 6 S^2 (= P_3), S''' = T'''' = P_4 = 8 T (1 - u)(2 - 3u) = 8 T (2 - 5u + 3u^2).
  phi(p), phi'(p), phi''(p) from the backend's validated tanh / sech2 at the point p (thin intervals);
  phi'''(Z) from T_Z = tanh(Z) (saturation handled by the backend: |z| > SAT gives the tabulated saturated enclosure)
  and u_Z = sqr(T_Z) subset [0, 1]: P_3 is a concave quadratic in u (vertex u = 2/3), its range on [u_lo, u_hi] is
  the hull of the endpoint values and, when the vertex may lie inside, the vertex value (hulling more values only
  enlarges the enclosure); 2 - 5u + 3u^2 is convex in u (vertex 5/6), handled the same way, then multiplied by 8 T_Z.
* lumping to TM1 (network outputs): c_ii e_i^2 = c_ii / 2 + c_ii (e_i^2 - 1/2) with e_i^2 - 1/2 in [-1/2, 1/2]:
  constant += c_ii / 2, remainder += c_ii [-1/2, 1/2]; c_ij e_i e_j (i < j): remainder += c_ij [-1, 1].
"""

from __future__ import annotations

import numpy as np

from certified_obstacle_nn.certification.enclose import tm as T


def _up(x):
    return np.nextafter(x, np.inf)


def pairs(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i, n)]


class TM2:
    __slots__ = ("bk", "c", "n")

    def __init__(self, bk, c, n: int):
        self.bk, self.c, self.n = bk, c, n

    @property
    def nq(self) -> int:
        return self.n * (self.n + 1) // 2

    @property
    def const(self):
        return self.c[0]

    @property
    def rem(self):
        return self.c[1 + self.n + self.nq]

    def lin(self, j):
        return self.c[1 + j]

    def quad(self, q):
        return self.c[1 + self.n + q]

    def _mag(self, rows):
        acc = np.zeros(self.c.shape[1:])
        for i in rows:
            acc = _up(acc + self.bk.mag(self.c[i]))
        return acc

    def lin_mag(self):
        return self._mag(range(1, 1 + self.n))

    def quad_mag(self):
        return self._mag(range(1 + self.n, 1 + self.n + self.nq))

    def poly_range(self):
        bk = self.bk
        shp = self.c.shape[1:]
        u01 = bk.from_bounds(np.zeros(shp), np.ones(shp))
        r = self.const + bk.symmetric(self.lin_mag())
        offm = np.zeros(shp)
        for q, (i, j) in enumerate(pairs(self.n)):
            if i == j:
                r = r + self.quad(q) * u01
            else:
                offm = _up(offm + bk.mag(self.quad(q)))
        return r + bk.symmetric(offm)

    def range(self):
        return self.poly_range() + self.rem

    def __add__(self, o):
        if isinstance(o, TM2):
            return TM2(self.bk, self.c + o.c, self.n)
        c = self.c.copy()
        c[0] = self.c[0] + o
        return TM2(self.bk, c, self.n)

    __radd__ = __add__

    def __neg__(self):
        return TM2(self.bk, -self.c, self.n)

    def __sub__(self, o):
        if isinstance(o, TM2):
            return TM2(self.bk, self.c - o.c, self.n)
        c = self.c.copy()
        c[0] = self.c[0] - o
        return TM2(self.bk, c, self.n)

    def scale(self, s):
        """Multiply every coefficient by an interval of batch shape."""
        return TM2(self.bk, self.bk.stack([self.c[i] * s for i in range(self.c.shape[0])]), self.n)


def from_tm1(A: T.TM) -> TM2:
    bk, n = A.bk, A.n
    z = bk.zeros(A.c.shape[1:])
    rows = [A.c[i] for i in range(n + 1)] + [z] * (n * (n + 1) // 2) + [A.rem]
    return TM2(bk, bk.stack(rows), n)


def to_tm1(A: TM2) -> T.TM:
    """Lump the quadratic part into constant and remainder (module docstring)."""
    bk, n = A.bk, A.n
    shp = A.c.shape[1:]
    half = bk.from_bounds(np.full(shp, -0.5), np.full(shp, 0.5))
    c0, rem = A.const, A.rem
    offm = np.zeros(shp)
    for q, (i, j) in enumerate(pairs(n)):
        cq = A.quad(q)
        if i == j:
            c0 = c0 + cq * 0.5
            rem = rem + cq * half
        else:
            offm = _up(offm + bk.mag(cq))
    rem = rem + bk.symmetric(offm)
    return T.TM(bk, bk.stack([c0] + [A.lin(j) for j in range(n)] + [rem]))


def mul(A: TM2, B: TM2) -> TM2:
    bk, n = A.bk, A.n
    a0, b0 = A.const, B.const
    rows = [a0 * b0] + [a0 * B.lin(j) + b0 * A.lin(j) for j in range(n)]
    for q, (i, j) in enumerate(pairs(n)):
        cross = A.lin(i) * B.lin(i) if i == j else A.lin(i) * B.lin(j) + A.lin(j) * B.lin(i)
        rows.append(a0 * B.quad(q) + b0 * A.quad(q) + cross)
    LA, LB, QA, QB = A.lin_mag(), B.lin_mag(), A.quad_mag(), B.quad_mag()
    H = _up(_up(_up(LA * QB) + _up(QA * LB)) + _up(QA * QB))
    ra, rb = A.rem, B.rem
    rows.append(bk.symmetric(H) + A.poly_range() * rb + B.poly_range() * ra + ra * rb)
    return TM2(bk, bk.stack(rows), n)


def sqr(A: TM2) -> TM2:
    bk, n = A.bk, A.n
    a0 = A.const
    rows = [bk.sqr(a0)] + [a0 * A.lin(j) * 2.0 for j in range(n)]
    for q, (i, j) in enumerate(pairs(n)):
        cross = bk.sqr(A.lin(i)) if i == j else A.lin(i) * A.lin(j) * 2.0
        rows.append(a0 * A.quad(q) * 2.0 + cross)
    LA, QA = A.lin_mag(), A.quad_mag()
    H = _up(_up(_up(LA * QA) * 2.0) + _up(QA * QA))
    ra = A.rem
    rows.append(bk.symmetric(H) + A.poly_range() * ra * 2.0 + bk.sqr(ra))
    return TM2(bk, bk.stack(rows), n)


def _quad_range(bk, uZ, a, b, c, vertex_lo, vertex_hi, vertex_num, vertex_den):
    """Range of a + b u + c u^2 over the intervals uZ (exact float coefficients). Endpoint values always; the value
    at the vertex (exact rational vertex_num/vertex_den, enclosed) is hulled in wherever [u_lo, u_hi] meets the
    float bracket [vertex_lo, vertex_hi] of the vertex (a superset of the cells containing it)."""
    shp = uZ.shape

    def q(u):
        return (u * c + b) * u + a

    lo, hi = uZ.lower(), uZ.upper()
    r = bk.hull(q(bk.thin(lo)), q(bk.thin(hi)))
    v = q(bk.rational(vertex_num, vertex_den, shp))
    inside = (lo <= vertex_hi) & (hi >= vertex_lo)
    return bk.select(inside, bk.hull(r, v), r)


def _taylor2(A: TM2, f_p, d1_p, d2_p, d3_Z) -> TM2:
    bk = A.bk
    D = A - bk.thin(A.const.mid())
    D2 = sqr(D)
    DR = D.range()
    cube = DR * DR * DR
    h2 = d2_p * 0.5
    rows = [f_p + d1_p * D.const + h2 * D2.const]
    for i in range(1, D.c.shape[0] - 1):
        rows.append(d1_p * D.c[i] + h2 * D2.c[i])
    sixth = bk.rational(1, 6, DR.shape)
    rows.append(d1_p * D.rem + h2 * D2.rem + d3_Z * cube * sixth)
    return TM2(bk, bk.stack(rows), A.n)


def tanh_sech2(A: TM2) -> tuple[TM2, TM2]:
    bk = A.bk
    P = bk.thin(A.const.mid())
    T_p, S_p = bk.tanh(P), bk.sech2(P)
    TS2 = T_p * S_p * -2.0  # T''(p) = S'(p)
    S2_p = S_p * 4.0 - bk.sqr(S_p) * 6.0  # S''(p)
    T_Z = bk.tanh(bk.include_point(A.range(), P))  # v18 D35
    uZ = bk.sqr(T_Z)
    P3 = _quad_range(bk, uZ, -2.0, 8.0, -6.0, 0.666, 0.667, 2, 3)  # T'''(Z)
    r4 = _quad_range(bk, uZ, 2.0, -5.0, 3.0, 0.833, 0.834, 5, 6)
    P4 = T_Z * r4 * 8.0  # S'''(Z)
    return _taylor2(A, T_p, S_p, TS2, P3), _taylor2(A, S_p, TS2, S2_p, P4)


def linmap(W: np.ndarray, A: TM2, bias: np.ndarray | None = None) -> TM2:
    """W (m, k) exact along batch axis 0 of A (rows, k, B): every coefficient row mapped (tm.tm_linmap)."""
    out = T.tm_linmap(W, T.TM(A.bk, A.c), bias)
    return TM2(A.bk, out.c, A.n)


def const(bk, value, n: int, batch_shape) -> TM2:
    return from_tm1(T.tm_const(bk, value, n, batch_shape))


def enclose_mlp_vec_tm2(layers, inputs: list[T.TM], channels: tuple[int, ...]):
    """Same contract as engine2d.enclose_mlp_vec (TM1 inputs, TM1 outputs (m, B) and {channel: TM1 (m, B)}), with
    every layer propagated as TM2 and the outputs lumped to TM1 (to_tm1)."""
    bk = inputs[0].bk
    n = inputs[0].n
    B = inputs[0].c.shape[1]
    a = from_tm1(T.tm_stack(inputs, axis=0))
    W0, b0 = layers[0]
    z = linmap(W0, a, b0)
    dz = {}
    for ch in channels:
        col = np.broadcast_to(W0[:, ch].reshape(-1, 1), (W0.shape[0], B))
        dz[ch] = const(bk, col, n, col.shape)
    for W, b in layers[1:]:
        act, s = tanh_sech2(z)
        da = {ch: mul(s, dz[ch]) for ch in channels}
        z = linmap(W, act, b)
        dz = {ch: linmap(W, da[ch]) for ch in channels}
    return to_tm1(z), {ch: to_tm1(dz[ch]) for ch in channels}


def enclose_mlp_tm2(layers, inputs: list[T.TM], channels: tuple[int, ...]):
    """Scalar-output contract of engine.enclose_mlp (1-D Stefan engine)."""
    z, dz = enclose_mlp_vec_tm2(layers, inputs, channels)
    return z[0], {ch: dz[ch][0] for ch in channels}
