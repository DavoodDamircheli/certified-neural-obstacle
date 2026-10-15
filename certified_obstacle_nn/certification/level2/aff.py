"""Batched first-order affine arithmetic with a lumped (per-entry) remainder.

An `Aff` represents, for every entry of a tensor of shape S, the set
        { c + sum_j G[..., j] eps_j + E  :  eps in [-1,1]^m,  |E| <= e }
where eps_1..eps_m are the noise symbols of the input box (one per space-time coordinate,
eps_j = (z_j - mid_j)/rad_j, SHARED by every entry of the same batch row) and E is an
entry-wise remainder (not shared: remainders of different entries are treated as independent,
which is always sound). Shapes: c: S, G: S + (m,), e: S (e >= 0).

m = 0 gives plain midpoint-radius interval arithmetic ("Tier 1"); m = 1+d gives the affine
("Tier 3") forms. Every operation below returns a sound enclosure: for every eps and every
admissible E of the operands, the exact result of the real operation lies in the returned set.
The soundness argument of each rule is written next to it and collected in
docs/history/LEVEL2_LOCAL_THEORY_NOTES.md section 4.

Floating point: after every operation the remainder is inflated by
    rel_eps * (|c| + rad) + abs_eps        (linear maps: rel_eps * (|W| (|c| + rad)) + abs_eps)
("mathematically rigorous enclosures evaluated in float64 with conservative outward
inflation (not verified interval arithmetic)").
"""

from __future__ import annotations

import math

import torch

REL_EPS = 1e-12
ABS_EPS = 1e-300


def _infl(c, rad):
    return REL_EPS * (c.abs() + rad) + ABS_EPS


def _pad(G, M):
    """Zero generator columns for symbols a form does not depend on (symbols are append-only)."""
    k = G.shape[-1]
    if k == M:
        return G
    return torch.cat([G, G.new_zeros(G.shape[:-1] + (M - k,))], -1)


def _align(a, b):
    M = max(a.G.shape[-1], b.G.shape[-1])
    return _pad(a.G, M), _pad(b.G, M)


class Aff:
    __slots__ = ("c", "G", "e", "_r")

    def __init__(self, c, G, e):
        self.c, self.G, self.e = c, G, e
        self._r = None

    # ------------------------------------------------------------------ construction
    @staticmethod
    def from_box(lo: torch.Tensor, hi: torch.Tensor, affine: bool = True) -> "Aff":
        """lo, hi [B, m]. Returns Aff of shape [B, m]: coordinate j = mid_j + rad_j eps_j."""
        mid = 0.5 * (lo + hi)
        rad = 0.5 * (hi - lo)
        # rad is exact up to rounding; inflate so that [mid-rad, mid+rad] contains [lo, hi]
        slack = REL_EPS * (mid.abs() + rad) + ABS_EPS
        if affine:
            G = torch.diag_embed(rad)
            e = slack
        else:
            G = lo.new_zeros(lo.shape + (0,))
            e = rad + slack
        return Aff(mid, G, e)

    @staticmethod
    def const(x: torch.Tensor, m: int) -> "Aff":
        return Aff(x, x.new_zeros(x.shape + (m,)), torch.zeros_like(x))

    @property
    def m(self) -> int:
        return self.G.shape[-1]

    @property
    def shape(self):
        return self.c.shape

    def rad(self) -> torch.Tensor:
        if self._r is None:
            self._r = self.G.abs().sum(-1) + self.e
        return self._r

    def lo(self) -> torch.Tensor:
        return self.c - self.rad()

    def hi(self) -> torch.Tensor:
        return self.c + self.rad()

    def range(self):
        r = self.rad()
        return self.c - r, self.c + r

    # ------------------------------------------------------------------ shape helpers
    def __getitem__(self, idx) -> "Aff":
        if not isinstance(idx, tuple):
            idx = (idx,)
        return Aff(self.c[idx], self.G[idx], self.e[idx])

    def sel(self, dim: int, i: int) -> "Aff":
        """Select index i along dimension dim (dim refers to S, non-negative)."""
        return Aff(self.c.select(dim, i), self.G.select(dim, i), self.e.select(dim, i))

    def unsqueeze(self, dim: int) -> "Aff":
        """dim refers to S (negative values count from the end of S, as for torch on c)."""
        if dim < 0:
            dim = dim + self.c.dim() + 1
        return Aff(self.c.unsqueeze(dim), self.G.unsqueeze(dim), self.e.unsqueeze(dim))

    def expand_m(self, m: int) -> "Aff":
        """Constant/interval Aff with m=0 generators -> same set with m zero generators."""
        if self.m == m:
            return self
        assert self.m == 0
        return Aff(self.c, self.c.new_zeros(self.c.shape + (m,)), self.e)

    @staticmethod
    def where(mask: torch.Tensor, a: "Aff", b: "Aff") -> "Aff":
        """Entry-wise choice (mask broadcast over S): the result encloses a where mask, b elsewhere."""
        c = torch.where(mask, a.c, b.c)
        Ga, Gb = _align(a, b)
        G = torch.where(mask.unsqueeze(-1), Ga, Gb)
        e = torch.where(mask, a.e, b.e)
        return Aff(c, G, e)

    def zeros_like(self) -> "Aff":
        return Aff(torch.zeros_like(self.c), torch.zeros_like(self.G), torch.zeros_like(self.e))

    def hull0(self) -> "Aff":
        """Interval enclosing both the range of self and 0 (no generators kept)."""
        lo, hi = self.range()
        return Aff.from_interval(torch.minimum(lo, torch.zeros_like(lo)), torch.maximum(hi, torch.zeros_like(hi)),
                                 self.m)

    def sum(self, dim: int) -> "Aff":
        """Exact sum along an S-dimension (generators add, remainders add)."""
        c = self.c.sum(dim)
        G = self.G.sum(dim)
        e = self.e.sum(dim)
        return Aff(c, G, e + _infl(self.c.abs().sum(dim), self.rad().sum(dim)))

    @staticmethod
    def stack(items, dim: int) -> "Aff":
        M = max(a.G.shape[-1] for a in items)
        return Aff(torch.stack([a.c for a in items], dim), torch.stack([_pad(a.G, M) for a in items], dim),
                   torch.stack([a.e for a in items], dim))

    # ------------------------------------------------------------------ linear ops (exact up to rounding)
    def __neg__(self) -> "Aff":
        return Aff(-self.c, -self.G, self.e)

    def __add__(self, o) -> "Aff":
        if isinstance(o, Aff):
            c = self.c + o.c
            G1, G2 = _align(self, o)
            G = G1 + G2
            e = self.e + o.e
            scale = self.c.abs() + o.c.abs() + self.rad() + o.rad()
        else:  # exact constant (float or tensor broadcastable to S)
            c = self.c + o
            G = self.G.expand(c.shape + (self.m,)) if torch.is_tensor(o) and o.dim() > 0 else self.G
            e = self.e.expand(c.shape) if torch.is_tensor(o) and o.dim() > 0 else self.e
            scale = c.abs() + self.c.abs() + self.rad()
        return Aff(c, G, e + REL_EPS * scale + ABS_EPS)

    __radd__ = __add__

    def __sub__(self, o) -> "Aff":
        return self + (-o)

    def __rsub__(self, o) -> "Aff":
        return (-self) + o

    def scale(self, s) -> "Aff":
        """Multiplication by an exact constant (python float or tensor broadcastable to S)."""
        if torch.is_tensor(s):
            c = self.c * s
            G = self.G * s.unsqueeze(-1)
            e = self.e * s.abs()
            scale = (self.c.abs() + self.rad()) * s.abs()
        else:
            c, G, e = self.c * s, self.G * s, self.e * abs(s)
            scale = (self.c.abs() + self.rad()) * abs(s)
        return Aff(c, G, e + REL_EPS * scale + ABS_EPS)

    def linear(self, W: torch.Tensor, b: torch.Tensor | None = None, dim: int = -1) -> "Aff":
        """y = W x (+ b) along S-dimension `dim` (x indexed by W's columns). Exact for the
        generators; the remainder maps through |W| (triangle inequality)."""
        nd = self.c.dim()
        d = dim % nd
        c = torch.movedim(self.c, d, -1) @ W.T
        e = torch.movedim(self.e, d, -1) @ W.abs().T
        G = torch.movedim(self.G, d, -1) @ W.T                     # [..., m, n']  (G's S-dim d moved last)
        absW = W.abs()
        cscale = torch.movedim(self.c.abs() + self.rad(), d, -1) @ absW.T
        if b is not None:
            c = c + b
        e = e + REL_EPS * (cscale + c.abs()) + ABS_EPS
        c = torch.movedim(c, -1, d)
        e = torch.movedim(e, -1, d)
        G = torch.movedim(torch.movedim(G, -1, d), -1, -1)          # put n' back at d, m stays last
        return Aff(c, G, e)

    # ------------------------------------------------------------------ products
    def __mul__(self, o) -> "Aff":
        """x*y = c1c2 + c1 G2 eps + c2 G1 eps + c1 E2 + c2 E1 + (G1 eps + E1)(G2 eps + E2);
        |last| <= rad1 rad2. Broadcasting over S as in torch."""
        if not isinstance(o, Aff):
            return self.scale(o)
        c = self.c * o.c
        G1, G2 = _align(self, o)
        G = self.c.unsqueeze(-1) * G2 + o.c.unsqueeze(-1) * G1
        r1, r2 = self.rad(), o.rad()
        a1, a2 = self.c.abs(), o.c.abs()
        e = a1 * o.e + a2 * self.e + r1 * r2
        return Aff(c, G, e + REL_EPS * (a1 * a2 + a1 * r2 + a2 * r1 + r1 * r2) * 2.0 + ABS_EPS)

    __rmul__ = __mul__

    def square(self) -> "Aff":
        """x^2 with delta = G eps + E, |delta| <= r:  x^2 = c^2 + 2c G eps + 2c E + delta^2,
        delta^2 in [0, r^2]  =>  center c^2 + r^2/2, generators 2cG, remainder 2|c|e + r^2/2."""
        r = self.rad()
        a = self.c.abs()
        c = self.c * self.c + 0.5 * r * r
        G = 2.0 * self.c.unsqueeze(-1) * self.G
        e = 2.0 * a * self.e + 0.5 * r * r
        return Aff(c, G, e + REL_EPS * 2.0 * (a + r) ** 2 + ABS_EPS)

    # ------------------------------------------------------------------ elementwise nonlinearities
    def _apply_affine_relax(self, alpha, beta, mu) -> "Aff":
        """f(x) in alpha*x + beta +- mu for every x in [lo, hi]  =>  sound result form."""
        c = alpha * self.c + beta
        G = alpha.unsqueeze(-1) * self.G
        aa = alpha.abs()
        e = aa * self.e + mu
        return Aff(c, G, e + REL_EPS * (aa * (self.c.abs() + self.rad()) + beta.abs() + mu) + ABS_EPS)

    def tanh(self) -> "Aff":
        """Interval mode (m=0): alpha = 0, exact range [tanh lo, tanh hi] (tanh increasing).
        Affine mode: alpha = secant slope on [lo,hi] (tanh'(mid) if degenerate); then
        phi(x) = tanh(x) - alpha x has its extrema on [lo,hi] at lo, hi or at the interior
        points where tanh'(x) = alpha, i.e. x = +-atanh(sqrt(1-alpha)) (0 < alpha <= 1).
        beta = (max phi + min phi)/2, mu = (max phi - min phi)/2."""
        lo, hi = self.range()
        tl, th = torch.tanh(lo), torch.tanh(hi)
        if self.m == 0:
            alpha = torch.zeros_like(lo)
            beta = 0.5 * (tl + th)
            mu = 0.5 * (th - tl) + REL_EPS * (tl.abs() + th.abs()) + 4e-16
            return self._apply_affine_relax(alpha, beta, mu)
        w = hi - lo
        mid = 0.5 * (lo + hi)
        tm = torch.tanh(mid)
        sec = torch.where(w > 1e-9, (th - tl) / torch.where(w > 1e-9, w, torch.ones_like(w)), 1.0 - tm * tm)
        alpha = sec.clamp(min=0.0, max=1.0)
        # candidate points
        xs = torch.atanh(torch.sqrt((1.0 - alpha).clamp(min=0.0, max=1.0 - 1e-16)))
        cands = [lo, hi]
        vals = [tl - alpha * lo, th - alpha * hi]
        for x in (xs, -xs):
            inside = (x > lo) & (x < hi)
            v = torch.tanh(x) - alpha * x
            # points outside [lo,hi] are replaced by lo (already a candidate)
            vals.append(torch.where(inside, v, vals[0]))
        V = torch.stack(vals, 0)
        vmax, vmin = V.max(0).values, V.min(0).values
        beta = 0.5 * (vmax + vmin)
        mu = 0.5 * (vmax - vmin) + REL_EPS * (V.abs().max(0).values + alpha * (lo.abs() + hi.abs())) + 4e-16
        return self._apply_affine_relax(alpha, beta, mu)

    def convex_tangent(self, f, df, d2f_max) -> "Aff":
        """Convex f on [lo,hi]: f(x) = f(m) + f'(m)(x-m) + rho, 0 <= rho <= d2f_max (x-m)^2/2
        <= d2f_max r^2/2 (Taylor, f'' >= 0). d2f_max(lo, hi) must bound f'' on [lo,hi]."""
        lo, hi = self.range()
        mid, r = 0.5 * (lo + hi), 0.5 * (hi - lo)
        fm, dfm = f(mid), df(mid)
        q = 0.5 * d2f_max(lo, hi) * r * r
        alpha = dfm
        beta = fm - dfm * mid + 0.5 * q
        mu = 0.5 * q + REL_EPS * (fm.abs() + dfm.abs() * (mid.abs() + r)) + 4e-16
        return self._apply_affine_relax(alpha, beta, mu)

    def exp(self) -> "Aff":
        return self.convex_tangent(torch.exp, torch.exp, lambda lo, hi: torch.exp(hi))

    def recip_pos(self) -> "Aff":
        """1/x for x >= lo > 0 (convex, f'' = 2/x^3 <= 2/lo^3)."""
        lo, _ = self.range()
        assert bool((lo > 0).all()), "recip_pos needs a positive enclosure"
        return self.convex_tangent(lambda x: 1.0 / x, lambda x: -1.0 / (x * x), lambda lo, hi: 2.0 / lo ** 3)

    def relu(self) -> "Aff":
        """lo>=0: identity; hi<=0: 0; else relu(x) in alpha x + [0, -alpha lo], alpha = hi/(hi-lo)
        (the chord from (lo,0) to (hi,hi) lies above relu, the line alpha x lies below)."""
        lo, hi = self.range()
        pos, neg = lo >= 0, hi <= 0
        mix = ~(pos | neg)
        den = torch.where(mix, hi - lo, torch.ones_like(lo))
        alpha = torch.where(pos, torch.ones_like(lo), torch.where(neg, torch.zeros_like(lo), hi / den))
        off = torch.where(mix, -alpha * lo, torch.zeros_like(lo))
        return self._apply_affine_relax(alpha, 0.5 * off, 0.5 * off + REL_EPS * hi.abs() + 4e-16)

    def add_interval(self, lo: torch.Tensor, hi: torch.Tensor) -> "Aff":
        """x + v with v an independent unknown in [lo, hi]."""
        c = self.c + 0.5 * (lo + hi)
        e = self.e + 0.5 * (hi - lo)
        return Aff(c, self.G, e + REL_EPS * (self.c.abs() + self.rad() + lo.abs() + hi.abs()) + ABS_EPS)

    @staticmethod
    def from_interval(lo: torch.Tensor, hi: torch.Tensor, m: int) -> "Aff":
        c = 0.5 * (lo + hi)
        e = 0.5 * (hi - lo)
        return Aff(c, c.new_zeros(c.shape + (m,)), e + _infl(c, e))


# ---------------------------------------------------------------------- range helpers (exact on intervals)
def sq_range(lo, hi):
    """Exact range of x^2 over [lo, hi]."""
    a, b = lo.abs(), hi.abs()
    up = torch.maximum(a, b) ** 2
    dn = torch.where((lo <= 0) & (hi >= 0), torch.zeros_like(lo), torch.minimum(a, b) ** 2)
    return dn, up


def pos_part_range(lo, hi):
    return lo.clamp(min=0.0), hi.clamp(min=0.0)


def neg_part_range(lo, hi):
    """Range of (-x)^+ = max(-x, 0)."""
    return (-hi).clamp(min=0.0), (-lo).clamp(min=0.0)


def up(x):
    """Round a nonnegative float64 result upward by a relative margin (outward inflation)."""
    return x * (1.0 + 4 * REL_EPS) + ABS_EPS


def dn(x):
    return x * (1.0 - 4 * REL_EPS)


_ = math  # keep import (used by doc tooling)
