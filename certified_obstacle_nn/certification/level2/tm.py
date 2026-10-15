"""Order-2 Taylor models ("Tier 3q") over the m input noise symbols of a box.

A TM represents, for every entry of a tensor of shape S, the set
    { c + g . eps + eps^T H eps + E  :  eps in [-1,1]^m,  |E| <= e }
with c: S, g: S+(m,), H: S+(m,m) (not necessarily symmetric), e: S. Polynomial parts are
propagated exactly through linear maps (no wrapping); everything of total degree >= 3 in eps, and
everything multiplied by a remainder, is bounded into e. Remainders are therefore O(rad^3) for
smooth operations (theory notes A10), against O(rad^2) for first-order affine forms.

Bounds used below, for p(eps) = g.eps + eps^T H eps (no constant):
    |g.eps| <= |g|_1 =: n1(g),  |eps^T H eps| <= sum |H_jk| =: n1(H)  (|eps_j eps_k| <= 1),
    so |p| <= P := n1(g) + n1(H) and |X - c| <= P + e.
Exact range enclosure (used for final ranges and as nonlinearity input): diagonal terms
H_jj eps_j^2 in [min(0,H_jj), max(0,H_jj)], off-diagonal +-|H_jk|.

Floating point: every operation adds REL_EPS * (scale) + ABS_EPS to e (as in aff.py).
"""

from __future__ import annotations

import torch

from certified_obstacle_nn.certification.level2.aff import ABS_EPS, REL_EPS


def _n1g(g):
    return g.abs().sum(-1)


def _n1H(H):
    return H.abs().sum((-1, -2))


class TM:
    __slots__ = ("c", "g", "H", "e", "_P")

    def __init__(self, c, g, H, e):
        self.c, self.g, self.H, self.e = c, g, H, e
        self._P = None

    # ------------------------------------------------------------------ construction / info
    @staticmethod
    def from_box(lo, hi, affine=True):
        mid = 0.5 * (lo + hi)
        rad = 0.5 * (hi - lo)
        m = lo.shape[1]
        g = torch.diag_embed(rad)
        H = lo.new_zeros(lo.shape + (m, m))
        return TM(mid, g, H, REL_EPS * (mid.abs() + rad) + ABS_EPS)

    @staticmethod
    def const(x, m):
        return TM(x, x.new_zeros(x.shape + (m,)), x.new_zeros(x.shape + (m, m)), torch.zeros_like(x))

    @property
    def m(self):
        return self.g.shape[-1]

    def P(self):
        if self._P is None:
            self._P = _n1g(self.g) + _n1H(self.H)
        return self._P

    def rad(self):
        """Upper bound of |X - c| (cheap)."""
        return self.P() + self.e

    def range(self):
        """Tighter enclosure of the range using eps_j^2 in [0,1]."""
        Hs = 0.5 * (self.H + self.H.transpose(-1, -2))
        diag = torch.diagonal(Hs, dim1=-2, dim2=-1)
        off = Hs.abs().sum((-1, -2)) - diag.abs().sum(-1)
        lo = self.c - _n1g(self.g) - off + diag.clamp(max=0).sum(-1) - self.e
        hi = self.c + _n1g(self.g) + off + diag.clamp(min=0).sum(-1) + self.e
        return lo, hi

    def lo(self):
        return self.range()[0]

    def hi(self):
        return self.range()[1]

    # ------------------------------------------------------------------ shape helpers
    def sel(self, dim, i):
        return TM(self.c.select(dim, i), self.g.select(dim, i), self.H.select(dim, i), self.e.select(dim, i))

    def unsqueeze(self, dim):
        if dim < 0:
            dim = dim + self.c.dim() + 1
        return TM(self.c.unsqueeze(dim), self.g.unsqueeze(dim), self.H.unsqueeze(dim), self.e.unsqueeze(dim))

    @staticmethod
    def where(mask, a, b):
        return TM(torch.where(mask, a.c, b.c), torch.where(mask[..., None], a.g, b.g),
                  torch.where(mask[..., None, None], a.H, b.H), torch.where(mask, a.e, b.e))

    def zeros_like(self):
        return TM(torch.zeros_like(self.c), torch.zeros_like(self.g), torch.zeros_like(self.H),
                  torch.zeros_like(self.e))

    @staticmethod
    def from_interval(lo, hi, m):
        c = 0.5 * (lo + hi)
        e = 0.5 * (hi - lo)
        return TM(c, c.new_zeros(c.shape + (m,)), c.new_zeros(c.shape + (m, m)),
                  e + REL_EPS * (c.abs() + e) + ABS_EPS)

    def hull0(self):
        lo, hi = self.range()
        return TM.from_interval(torch.minimum(lo, torch.zeros_like(lo)), torch.maximum(hi, torch.zeros_like(hi)),
                                self.m)

    # ------------------------------------------------------------------ linear
    def __neg__(self):
        return TM(-self.c, -self.g, -self.H, self.e)

    def __add__(self, o):
        if isinstance(o, TM):
            sc = self.c.abs() + o.c.abs() + self.rad() + o.rad()
            return TM(self.c + o.c, self.g + o.g, self.H + o.H, self.e + o.e + REL_EPS * sc + ABS_EPS)
        c = self.c + o
        sc = c.abs() + self.c.abs() + self.rad()
        if torch.is_tensor(o) and o.dim() > 0:
            return TM(c, self.g.expand(c.shape + (self.m,)), self.H.expand(c.shape + (self.m, self.m)),
                      self.e.expand(c.shape) + REL_EPS * sc + ABS_EPS)
        return TM(c, self.g, self.H, self.e + REL_EPS * sc + ABS_EPS)

    __radd__ = __add__

    def __sub__(self, o):
        return self + (-o)

    def __rsub__(self, o):
        return (-self) + o

    def scale(self, s):
        if torch.is_tensor(s):
            sa = s.abs()
            return TM(self.c * s, self.g * s[..., None], self.H * s[..., None, None],
                      self.e * sa + REL_EPS * (self.c.abs() + self.rad()) * sa + ABS_EPS)
        return TM(self.c * s, self.g * s, self.H * s, self.e * abs(s) + REL_EPS * (self.c.abs() + self.rad()) * abs(s)
                  + ABS_EPS)

    def linear(self, W, b=None, dim=-1):
        nd = self.c.dim()
        d = dim % nd
        c = torch.movedim(self.c, d, -1) @ W.T
        e = torch.movedim(self.e, d, -1) @ W.abs().T
        sc = torch.movedim(self.c.abs() + self.rad(), d, -1) @ W.abs().T
        if b is not None:
            c = c + b
        e = e + REL_EPS * (sc + c.abs()) + ABS_EPS
        g = torch.movedim(torch.movedim(self.g, d, -1) @ W.T, -1, d)                 # S-dim d last, then back
        Hm = torch.movedim(self.H, d, -1)
        H = torch.movedim(Hm @ W.T, -1, d)
        return TM(torch.movedim(c, -1, d), g, H, torch.movedim(e, -1, d))

    # ------------------------------------------------------------------ products
    def __mul__(self, o):
        """(c1 + p1 + E1)(c2 + p2 + E2): exact degree <= 2 part kept, the rest bounded:
        |g1.eps * eps^T H2 eps| <= n1(g1) n1(H2), |eps^T H1 eps * g2.eps| <= n1(H1) n1(g2),
        |eps^T H1 eps eps^T H2 eps| <= n1(H1) n1(H2), |(c1+p1) E2| <= (|c1|+P1) e2, etc."""
        if not isinstance(o, TM):
            return self.scale(o)
        c1, c2 = self.c, o.c
        g1, g2 = self.g, o.g
        c = c1 * c2
        g = c1[..., None] * g2 + c2[..., None] * g1
        H = c1[..., None, None] * o.H + c2[..., None, None] * self.H + g1[..., :, None] * g2[..., None, :]
        nH1, nH2 = _n1H(self.H), _n1H(o.H)
        ng1, ng2 = _n1g(g1), _n1g(g2)
        hi_ord = ng1 * nH2 + nH1 * ng2 + nH1 * nH2
        a1, a2 = c1.abs() + self.P(), c2.abs() + o.P()
        e = hi_ord + a1 * o.e + a2 * self.e + self.e * o.e
        sc = (c1.abs() + self.rad()) * (c2.abs() + o.rad())
        return TM(c, g, H, e + 4 * REL_EPS * sc + ABS_EPS)

    __rmul__ = __mul__

    def square(self):
        return self * self

    # ------------------------------------------------------------------ smooth univariate functions
    def _taylor2(self, f0, f1, f2half, d3bound):
        """F(c + d) = f0 + f1 d + f2half d^2 + R3, |R3| <= d3bound * |d|^3 / 6, d = X - c."""
        dlt = TM(torch.zeros_like(self.c), self.g, self.H, self.e)
        d2 = dlt * dlt
        r = dlt.rad()
        out = dlt.scale(f1) + d2.scale(f2half)
        rem = d3bound * r ** 3 / 6.0
        sc = f0.abs() + REL_EPS
        return TM(out.c + f0, out.g, out.H, out.e + rem + REL_EPS * (sc + out.c.abs() + out.rad()) + 4e-16)

    def tanh(self):
        lo, hi = self.range()
        t = torch.tanh(self.c)
        s = 1 - t * t
        # sup |tanh'''| on [lo,hi] = sup |q(u)|, q(u) = -6u^2 + 8u - 2, u = tanh^2 in [ulo, uhi]
        tl, th = torch.tanh(lo), torch.tanh(hi)
        straddle = (lo <= 0) & (hi >= 0)
        ulo = torch.where(straddle, torch.zeros_like(lo), torch.minimum(tl * tl, th * th))
        uhi = torch.maximum(tl * tl, th * th)
        q = lambda u: -6 * u * u + 8 * u - 2  # noqa: E731
        cand = torch.stack([q(ulo).abs(), q(uhi).abs(),
                            torch.where((ulo <= 2 / 3) & (uhi >= 2 / 3), torch.full_like(lo, 2 / 3), torch.zeros_like(lo))])
        d3 = cand.max(0).values * (1 + 1e-12) + 1e-15
        return self._taylor2(t, s, -s * t, d3)

    def exp(self):
        lo, hi = self.range()
        ec = torch.exp(self.c)
        return self._taylor2(ec, ec, 0.5 * ec, torch.exp(hi) * (1 + 1e-12))

    def recip_pos(self):
        lo, _ = self.range()
        assert bool((lo > 0).all()), "recip_pos needs a positive enclosure"
        ic = 1.0 / self.c
        return self._taylor2(ic, -ic * ic, ic ** 3, 6.0 / lo ** 4 * (1 + 1e-12))

    def relu(self):
        """lo >= 0: identity; hi <= 0: 0; else linear relaxation alpha X + [0, -alpha lo] (A6)."""
        lo, hi = self.range()
        pos, neg = lo >= 0, hi <= 0
        mix = ~(pos | neg)
        den = torch.where(mix, hi - lo, torch.ones_like(lo))
        alpha = torch.where(pos, torch.ones_like(lo), torch.where(neg, torch.zeros_like(lo), hi / den))
        off = torch.where(mix, -alpha * lo, torch.zeros_like(lo))
        out = self.scale(alpha)
        return TM(out.c + 0.5 * off, out.g, out.H, out.e + 0.5 * off + REL_EPS * hi.abs() + 4e-16)

    def add_interval(self, lo, hi):
        return TM(self.c + 0.5 * (lo + hi), self.g, self.H,
                  self.e + 0.5 * (hi - lo) + REL_EPS * (self.c.abs() + self.rad() + lo.abs() + hi.abs()) + ABS_EPS)
