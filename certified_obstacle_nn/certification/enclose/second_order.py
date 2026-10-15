"""TM1 enclosures of second derivatives of a frozen tanh MLP over a cell
(used ONLY by the mean-value enclosure of the residual in enclose/engine.py).

specification section 0 ("never compute u_xx") governs the residuals, the training
loss and the estimator: those stay first-order. Inside certification/enclose
this module encloses second derivatives of the networks over a closed cell, to
bound the cell variation of the first-order residual R (mean-value form,
DECISIONS D16). Nothing here is used for training or for any float64 quantity.

Forward mode with value, first and second channels, per hidden layer
(a = tanh z, s = sech^2 z, s' = d/dz sech^2 z = -2 tanh(z) sech^2(z)):

    a_i  = s z_i
    a_ij = s z_ij + s' z_i z_j

followed by the exact linear map (z_next = W a + b, z_next_i = W a_i,
z_next_ij = W a_ij). The first layer has z_i = W0[:, i] (exact constants) and
z_ij = 0. Every product is a TM1 product (enclose/tm.py), so each channel is a
validated TM1 enclosure of the exact derivative on the whole cell: pointwise,
the chain rule above is an identity, and act / s enclose tanh(z(e)) /
sech^2(z(e)) for every e in the cell.
"""

from __future__ import annotations

import numpy as np

from certified_obstacle_nn.certification.enclose import tm as T


def enclose_mlp2(layers, inputs: list[T.TM], first: tuple[int, ...], second: tuple[tuple[int, int], ...]):
    """TM1 enclosure of a tanh MLP's value, the first derivatives in `first`
    and the second derivatives in `second` (pairs (i, j), i <= j, w.r.t. input
    indices). Returns (value TM, {i: TM}, {(i, j): TM}) of the scalar output."""
    bk = inputs[0].bk
    n = inputs[0].n
    B = inputs[0].c.shape[1]
    second = tuple(tuple(sorted(p)) for p in second)
    chans = sorted(set(first) | {i for p in second for i in p})
    a = T.tm_stack(inputs, axis=0)
    W0, b0 = layers[0]
    z = T.tm_linmap(W0, a, b0)
    dz = {}
    for ch in chans:
        col = np.broadcast_to(W0[:, ch].reshape(-1, 1), (W0.shape[0], B))
        dz[ch] = T.tm_const(bk, col, n, col.shape)
    d2z = {p: None for p in second}  # None = exactly zero (first layer is affine)
    for W, b in layers[1:]:
        act, s = T.tm_tanh_sech2(z)
        sp = T.tm_mul(act, s) * (-2.0) if second else None
        da = {ch: T.tm_mul(s, dz[ch]) for ch in chans}
        d2a = {}
        for (i, j) in second:
            term = T.tm_mul(sp, T.tm_mul(dz[i], dz[j]))
            d2a[(i, j)] = term if d2z[(i, j)] is None else T.tm_mul(s, d2z[(i, j)]) + term
        z = T.tm_linmap(W, act, b)
        dz = {ch: T.tm_linmap(W, da[ch]) for ch in chans}
        d2z = {p: T.tm_linmap(W, d2a[p]) for p in second}
    out2 = {}
    for p in second:
        if d2z[p] is None:  # depth-0 network: affine, second derivatives vanish
            out2[p] = T.tm_const(bk, 0.0, n, (B,))
        else:
            out2[p] = d2z[p][0]
    return z[0], {ch: dz[ch][0] for ch in first}, out2
