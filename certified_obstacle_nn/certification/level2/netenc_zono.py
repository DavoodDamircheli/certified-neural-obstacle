"""Tier 3z: zonotope (DeepZ-style) forward mode for a tanh MLP, its value and Jacobian.

Same recursion as netenc.mlp_enclose (theory notes A8) but every remainder produced by a
nonlinear step (tanh relaxation of z, square remainder in s = 1 - z^2, product remainder in
dz = s * da) is turned into a FRESH noise symbol (one per entry) instead of being lumped into the
per-entry remainder e. Symbols of one cell are shared by all forms of that cell (value and
Jacobian), so later linear layers can cancel them (W eps instead of |W| e).

Soundness (theory notes A9): a fresh symbol eps' in [-1,1] with coefficient e on entry i and 0
elsewhere represents exactly the same set as the remainder |E_i| <= e_i for that entry, and the
added freedom (independent symbols per entry) is the same independence assumption the lumped
form already makes. Order reduction (Girard): any generator column may be removed from a cell's
forms by adding |G[..., j]| to every entry's remainder e -- the represented set only grows.
After each layer at most `max_sym` non-input symbols are kept per cell (largest total |G| mass).
"""

from __future__ import annotations

import torch

from certified_obstacle_nn.certification.level2.aff import Aff, _pad
from certified_obstacle_nn.certification.level2.pointwise import mlp_layers


class SymCtx:
    """Per-computation symbol counter (columns 0..m_in-1 are the input symbols)."""

    def __init__(self, m_in: int):
        self.m_in = m_in
        self.M = m_in

    def promote(self, a: Aff) -> Aff:
        """Move the remainder e of every entry into a fresh symbol."""
        B = a.c.shape[0]
        P = a.e[0].numel()
        G = _pad(a.G, self.M)
        new = torch.diag_embed(a.e.reshape(B, P)).reshape(a.c.shape + (P,))
        self.M += P
        return Aff(a.c, torch.cat([G, new], -1), torch.zeros_like(a.e))

    def reduce(self, forms, max_sym: int):
        """Keep the max_sym heaviest non-input symbols per cell, box the rest into e."""
        n_extra = self.M - self.m_in
        if n_extra <= max_sym:
            return [Aff(f.c, _pad(f.G, self.M), f.e) for f in forms]
        Gs = [_pad(f.G, self.M) for f in forms]
        B = forms[0].c.shape[0]
        mass = sum(G[..., self.m_in:].abs().reshape(B, -1, n_extra).sum(1) for G in Gs)   # [B, n_extra]
        keep = torch.topk(mass, max_sym, dim=1).indices                                    # [B, K]
        drop_mask = torch.ones(B, n_extra, dtype=torch.bool)
        drop_mask.scatter_(1, keep, False)
        out = []
        for f, G in zip(forms, Gs):
            ext = G[..., self.m_in:]
            shp = ext.shape
            flat = ext.reshape(B, -1, n_extra)
            kept = torch.gather(flat, 2, keep[:, None, :].expand(-1, flat.shape[1], -1))
            lumped = (flat.abs() * drop_mask[:, None, :]).sum(-1)
            e = f.e + lumped.reshape(f.e.shape)
            G2 = torch.cat([G[..., :self.m_in], kept.reshape(shp[:-1] + (max_sym,))], -1)
            out.append(Aff(f.c, G2, e))
        self.M = self.m_in + max_sym
        return out


def mlp_enclose_zono(mlp, Z: Aff, order: int = 1, max_sym: int = 128, promote_jac: bool = True):
    """Z: Aff [B, m] with m input symbols (affine mode). Returns (val [B,out], jac [B,out,m] | None, M)."""
    assert Z.m == Z.c.shape[1], "zonotope mode needs the affine input box"
    layers = mlp_layers(mlp)
    m = Z.m
    ctx = SymCtx(m)
    W, b = layers[0]
    a = Z.linear(W, b, dim=1)
    da_const = W                                    # [n, m] (all directions)
    da = None
    for W, b in layers[1:]:
        z = ctx.promote(a.tanh())
        if order >= 1:
            s = 1.0 - z.square()
            s = ctx.promote(s)
            s1 = s.unsqueeze(-1)
            dz = s1.scale(da_const) if da is None else s1 * da
            if promote_jac:
                dz = ctx.promote(dz)
            da = dz.linear(W, None, dim=1)
        a = z.linear(W, b, dim=1)
        if order >= 1:
            a, da = ctx.reduce([a, da], max_sym)
        else:
            (a,) = ctx.reduce([a], max_sym)
    if order >= 1 and da is None:
        da = Aff.const(da_const.expand(Z.c.shape[0], -1, -1).clone(), m)
    return a, (da if order >= 1 else None), ctx.M


def relayout(a: Aff, m: int, offset: int, total: int) -> Aff:
    """Place a network's extra symbols (columns m..) at [offset, offset + extra) of a global list."""
    G = a.G
    extra = G.shape[-1] - m
    parts = [G[..., :m], G.new_zeros(G.shape[:-1] + (offset - m,)), G[..., m:],
             G.new_zeros(G.shape[:-1] + (total - offset - extra,))]
    return Aff(a.c, torch.cat(parts, -1), a.e)
