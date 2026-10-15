"""Sound enclosures of a tanh MLP and of its first/second input derivatives over a batch of
boxes (forward mode, Aff arithmetic). Same recursion as pointwise.mlp_jet:

    a^1 = W_1 z + b_1,             da^1 = W_1[:, dirs] (exact),   d2a^1 = 0
    z^k = tanh(a^k),  s^k = 1 - (z^k)^2,  s'^k = -2 z^k s^k
    dz^k  = s^k da^k
    d2z^k = s^k d2a^k + s'^k (da^k (x) da^k)
    a^{k+1} = W_{k+1} z^k + b,  da^{k+1} = W_{k+1} dz^k,  d2a^{k+1} = W_{k+1} d2z^k

Every arrow is an Aff operation (aff.py), so the outputs enclose the exact value / Jacobian /
Hessian of the network at every point of the box. In interval mode (m = 0) z and s are
replaced by their EXACT ranges on the pre-activation interval (tanh monotone; sech^2 even and
decreasing in |a|), which is the classical interval forward mode ("Tier 1").
"""

import torch

from certified_obstacle_nn.certification.level2.aff import Aff, REL_EPS
from certified_obstacle_nn.certification.level2.pointwise import mlp_layers


def _sech2_range(lo, hi):
    """Exact range of sech^2 = 1 - tanh^2 on [lo, hi] (max at the point closest to 0)."""
    tl, th = torch.tanh(lo), torch.tanh(hi)
    sl, sh = 1 - tl * tl, 1 - th * th
    smin = torch.minimum(sl, sh)
    straddle = (lo <= 0) & (hi >= 0)
    near = torch.where(straddle, torch.zeros_like(lo), torch.minimum(lo.abs(), hi.abs()))
    tn = torch.tanh(near)
    smax = 1 - tn * tn
    pad = 4 * REL_EPS + 1e-15
    return (smin - pad).clamp(min=0.0), (smax + pad).clamp(max=1.0)


def mlp_enclose(mlp, Z: Aff, dirs=None, order: int = 1):
    """Z: Aff [B, m_in] (input box). Returns (val [B,out], jac [B,out,k] | None, hess [B,out,k,k] | None)."""
    layers = mlp_layers(mlp)
    m_in = Z.c.shape[1]
    dirs = list(range(m_in)) if dirs is None else list(dirs)
    interval = Z.m == 0
    W, b = layers[0]
    a = Z.linear(W, b, dim=1)                       # [B, n]
    da_const = W[:, dirs]                           # [n, k] exact
    da = None                                       # Aff once nonconstant
    d2a = None
    for li, (W, b) in enumerate(layers[1:]):
        if interval:
            lo, hi = a.range()
            tl, th = torch.tanh(lo), torch.tanh(hi)
            pad = REL_EPS * (tl.abs() + th.abs()) + 4e-16
            z = Aff.from_interval(tl - pad, th + pad, 0)
            slo, shi = _sech2_range(lo, hi)
            s = Aff.from_interval(slo, shi, 0)
        else:
            z = a.tanh()
            s = 1.0 - z.square()
        if order >= 1:
            s1 = s.unsqueeze(-1)                    # [B, n, 1]
            dz = s1.scale(da_const) if da is None else s1 * da
            if order >= 2:
                sp = (z * s).scale(-2.0).unsqueeze(-1).unsqueeze(-1)  # [B, n, 1, 1]
                if da is None:
                    outer_c = da_const[:, :, None] * da_const[:, None, :]
                    d2z = sp.scale(outer_c)
                else:
                    d2z = sp * (da.unsqueeze(-1) * da.unsqueeze(-2))
                if d2a is not None:
                    d2z = d2z + s1.unsqueeze(-1) * d2a
            da = dz.linear(W, None, dim=1)
            if order >= 2:
                d2a = d2z.linear(W, None, dim=1)
        a = z.linear(W, b, dim=1)
    if order >= 1 and da is None:  # single linear layer network
        da = Aff.const(da_const.expand(Z.c.shape[0], -1, -1).clone(), Z.m)
    return a, (da if order >= 1 else None), (d2a if order >= 2 else None)
