"""Type-generic elementary operations and a first-order forward-mode `Jet`.

The data formulas (problems.py) and the field assembly (fields.py) are written ONCE with the
functions below and then evaluated on three kinds of objects:
  * torch.Tensor           -- exact pointwise values (cell centers, diagnostics, quadrature);
  * Aff                    -- sound enclosures over a box (aff.py);
  * Jet(val, grad)         -- value + space-time gradient, where val/grad are tensors or Aff;
                              Jet over Aff gives sound enclosures of the GRADIENT of a field on a box
                              (Tier 2 centered forms, local Lipschitz constants).
Jet.grad carries a trailing axis of length k (number of differentiation directions).
"""

from __future__ import annotations

import torch

from certified_obstacle_nn.certification.level2.aff import Aff


def _t(v):
    return torch.is_tensor(v)


def _mp(v):
    """mpmath scalars (high-precision reference evaluation, experiments/level2/mpmath_spotcheck.py)."""
    return type(v).__module__.startswith("mpmath")


def exp(v):
    if _mp(v):
        import mpmath
        return mpmath.exp(v)
    return torch.exp(v) if _t(v) else v.exp()


def recip_pos(v):
    return 1.0 / v if (_t(v) or _mp(v)) else v.recip_pos()


def square(v):
    return v * v if (_t(v) or _mp(v)) else v.square()


def relu(v):
    if _mp(v):
        return v if v > 0 else 0 * v
    return torch.relu(v) if _t(v) else v.relu()


def relu_sq(v):
    """(v^+)^2 -- C^1, derivative 2 v^+ v'."""
    if _mp(v):
        return relu(v) ** 2
    if _t(v):
        return torch.relu(v) ** 2
    if isinstance(v, Jet):
        return v.relu_sq()
    return v.relu().square()


def ind_mul(psi, br):
    """1{psi > 0} * br, where br is continuous and vanishes on {psi = 0} wherever this is used
    (so the product is continuous). Aff: psi > 0 on the box -> br; psi <= 0 on the box -> 0;
    otherwise hull(0, range br) (both branches enclosed)."""
    if _mp(psi):
        return br if psi > 0 else 0 * br
    if _t(psi):
        return torch.where(psi > 0, br, torch.zeros_like(br))
    if isinstance(psi, Jet):
        return Jet(ind_mul(psi.val, br.val), ind_mul(_unsq(psi.val), br.grad))
    lo, hi = psi.range()
    pos, neg = lo > 0, hi <= 0
    zero = br.zeros_like()
    mixed = br.hull0()
    pos_b = pos.expand(br.c.shape) if pos.shape != br.c.shape else pos
    neg_b = neg.expand(br.c.shape) if neg.shape != br.c.shape else neg
    W = type(br).where
    return W(pos_b, br, W(neg_b, zero, mixed))


def _unsq(v):
    return v.unsqueeze(-1)


class Jet:
    """(val, grad): val of shape S, grad of shape S + (k,). Product/chain rules."""
    __slots__ = ("val", "grad")

    def __init__(self, val, grad):
        self.val, self.grad = val, grad

    def __neg__(self):
        return Jet(-self.val, -self.grad)

    def __add__(self, o):
        if isinstance(o, Jet):
            return Jet(self.val + o.val, self.grad + o.grad)
        return Jet(self.val + o, self.grad)

    __radd__ = __add__

    def __sub__(self, o):
        return self + (-o)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if isinstance(o, Jet):
            return Jet(self.val * o.val, _unsq(self.val) * o.grad + _unsq(o.val) * self.grad)
        if isinstance(o, (int, float)):
            return Jet(self.val * o, self.grad * o)
        raise TypeError("Jet * non-Jet: wrap as constant Jet")

    __rmul__ = __mul__

    def square(self):
        return Jet(square(self.val), (_unsq(self.val) * self.grad) * 2.0)

    def exp(self):
        ev = exp(self.val)
        return Jet(ev, _unsq(ev) * self.grad)

    def recip_pos(self):
        r = recip_pos(self.val)
        return Jet(r, (_unsq(square(r)) * self.grad) * -1.0)

    def relu_sq(self):
        rv = relu(self.val)
        return Jet(square(rv) if not _t(rv) else rv * rv, (_unsq(rv) * self.grad) * 2.0)
