"""v17 Stage 4 (roadmap R3, = cleanup C7a): validated affine (zonotope) forward mode for a tanh MLP and its first
input derivatives -- the rules of certification/level2/netenc_zono.py + aff.py ported onto the validated
backends (intervals_np outward float64, intervals_arb), as Taylor models with EXTRA noise symbols.

Representation: a TM (enclose/tm.py) whose linear part has n = n_in + n_extra symbols,
    q(e) in c_0 + sum_{j <= n_in} c_j e_j + sum_{j > n_in} c_j eps_j + I,   e in [-1,1]^n_in, eps in [-1,1]^n_extra,
with interval coefficients. Symbols 1..n_in are the cell coordinates (shared by every form of the cell, as in TM1).
Extra symbols eps_j are created by PROMOTION: for an entry with remainder I (an interval), choose the float m = mid(I)
and r >= |I - m| (rounded up); then I subset m + r [-1, 1] and the entry is rewritten with constant c_0 + m, a fresh
symbol eps with coefficient r (zero on every other entry) and remainder 0. This represents a superset of the same
set (each eps ranges over [-1,1] independently, exactly the independence the lumped remainder already assumes;
netenc_zono.py docstring / docs/history/LEVEL2_LOCAL_THEORY_NOTES.md A9). The symbols of one network evaluation are
shared by its value and derivative channels, so later linear layers combine them with signed weights
(W eps) instead of |W| r -- this removes the wrapping measured in Stage 3.

Every other operation is the TM1 rule of tm.py applied to all n symbols (sums, exact linear maps, products with
remainder LA LB + P_A I_B + P_B I_A + I_A I_B, tanh / sech^2 by a second-order Taylor expansion at the float midpoint
of the constant with phi'' enclosed over the whole range): they never use the meaning of a symbol, only that each
ranges over [-1, 1], so they are sound for extra symbols unchanged.

Order reduction (Girard): any extra symbol may be removed from the forms of a cell by adding sum |c_j| over the
entries it touches to their remainders (the set only grows). After every layer at most `max_sym` extra symbols are
kept per cell (largest total coefficient magnitude over all forms of the network); at the network output ALL extra
symbols are lumped, so the result is an ordinary TM1 over the n_in cell coordinates and the rest of the engine is
unchanged. Summations of nonnegative magnitudes in float64 are bounded by s (1 + (n + 1) 2^-52), then one ulp up
(recursive/pairwise summation error <= (n - 1) u sum, u = 2^-53).

Linear layers: the arb backend keeps its arb_mat product; for the numpy outward backend a midpoint-radius product
(BLAS) with an a-priori error bound is used (Rump, BIT 39 (1999) 534-554; DECISIONS D21):
    W X subset W m (+-) [ |W| r + g ( |W| |m| + |W| r ) + k eta ],  g = (k + 2) u / (1 - (k + 2) u),
with m, r the float midpoint and upward radius of X, eta = 2^-1074, every product computed in float64 and the
final bounds rounded outward (the bound is applied with 3 k eta to also cover underflow in the two magnitude products
|W| r and |W| |m|, and g also covers their own rounding).
"""

from __future__ import annotations

import numpy as np

from certified_obstacle_nn.certification.enclose import tm as T

U = 2.0**-53
ETA = 2.0**-1074


def _up(x):
    return np.nextafter(x, np.inf)


def _sum_up(m: np.ndarray, axis=0) -> np.ndarray:
    """Upper bound of the exact sum of NONNEGATIVE floats along `axis`."""
    n = m.shape[axis]
    s = m.sum(axis)
    return _up(s * (1.0 + (n + 1) * 2.0**-52))


def _is_np(x) -> bool:
    return hasattr(x, "lo")


def _arr_take(x, idx):
    """Fancy-index an interval array (backend-agnostic)."""
    if _is_np(x):
        return type(x)(x.lo[idx], x.hi[idx])
    return type(x)(x.v[idx])


def _cat(bk, items, axis=0):
    return bk.concat(items, axis)


def lin_mag(A: T.TM) -> np.ndarray:
    n = A.n
    if n == 0:
        return np.zeros(A.c.shape[1:])
    return _sum_up(A.bk.mag(A.c[1:n + 1]), 0)


def rng(A: T.TM):
    return A.const + A.bk.symmetric(lin_mag(A)) + A.rem


def _mk(bk, c0, lin, rem):
    """Assemble a TM from constant (*S), linear rows (n, *S) and remainder (*S)."""
    return T.TM(bk, _cat(bk, [c0[None], lin, rem[None]], 0))


def _lin(A: T.TM):
    return A.c[1:A.n + 1]


def zmul(A: T.TM, B: T.TM) -> T.TM:
    """tm.tm_mul with every symbol, vectorised over the symbol axis."""
    bk = A.bk
    a0, b0 = A.const, B.const
    LA, LB = lin_mag(A), lin_mag(B)
    PA = a0 + bk.symmetric(LA)
    PB = b0 + bk.symmetric(LB)
    lin = a0[None] * _lin(B) + b0[None] * _lin(A)
    ra, rb = A.rem, B.rem
    rem = bk.symmetric(_up(LA * LB)) + PA * rb + PB * ra + ra * rb
    return _mk(bk, a0 * b0, lin, rem)


def _taylor(A: T.TM, f_p, df_p, d2f_Z) -> T.TM:
    """tm._taylor1 with every symbol: phi(A) = phi(p) + phi'(p) D + phi''(Z) sqr(range D)/2."""
    bk = A.bk
    p = bk.thin(A.const.mid())
    D0 = A.const - p
    DR = D0 + bk.symmetric(lin_mag(A)) + A.rem
    c0 = f_p + df_p * D0
    lin = df_p[None] * _lin(A)
    rem = df_p * A.rem + d2f_Z * bk.sqr(DR) * 0.5
    return _mk(bk, c0, lin, rem)


def ztanh_sech2(A: T.TM):
    bk = A.bk
    P = bk.thin(A.const.mid())
    Z = bk.include_point(rng(A), P)  # v18 D35
    T_p, S_p = bk.tanh(P), bk.sech2(P)
    T_Z, S_Z = bk.tanh(Z), bk.sech2(Z)
    return (_taylor(A, T_p, S_p, T_Z * S_Z * -2.0),
            _taylor(A, S_p, T_p * S_p * -2.0, S_Z * 4.0 - bk.sqr(S_Z) * 6.0))


def _linmap_np(W: np.ndarray, X):
    """Midpoint-radius outward product W X for the numpy backend (module docstring, D21). X: NIA (k, *cols)."""
    from certified_obstacle_nn.certification.enclose.intervals_np import NIA

    k = W.shape[1]
    m = 0.5 * X.lo + 0.5 * X.hi
    r = _up(np.maximum(_up(X.hi - m), _up(m - X.lo)))
    if not (np.all(np.isfinite(m)) and np.all(np.isfinite(r))):
        return NIA(np.full((W.shape[0],) + X.lo.shape[1:], -np.inf), np.full((W.shape[0],) + X.lo.shape[1:], np.inf))
    absW = np.abs(W)
    sh = X.lo.shape[1:]
    m2, r2 = m.reshape(k, -1), r.reshape(k, -1)
    Y = W @ m2
    Rb = absW @ r2
    Mb = absW @ np.abs(m2)
    g = (k + 2) * U / (1 - (k + 2) * U)
    rad = _up(_up(Rb + _up(_up(g * _up(Mb + Rb)) + 3 * k * ETA)) * (1 + 4 * 2.0**-52))
    lo = np.nextafter(Y - rad, -np.inf)
    hi = _up(Y + rad)
    return NIA(lo.reshape((W.shape[0],) + sh), hi.reshape((W.shape[0],) + sh))


def zlinmap(W: np.ndarray, A: T.TM, bias=None) -> T.TM:
    """W (m, k) applied along entry axis 0 of A (shape (n+2, k, B)); bias added to the constant."""
    bk = A.bk
    n2, k, B = A.c.shape
    if _is_np(A.c):
        X = type(A.c)(np.swapaxes(A.c.lo, 0, 1), np.swapaxes(A.c.hi, 0, 1))  # (k, n2, B)
        Y = _linmap_np(W, X)
        c = type(A.c)(np.ascontiguousarray(np.swapaxes(Y.lo, 0, 1)), np.ascontiguousarray(np.swapaxes(Y.hi, 0, 1)))
    else:
        X = type(A.c)(np.swapaxes(A.c.v, 0, 1).copy())
        Y = bk.linmap(W, X)
        c = type(A.c)(np.swapaxes(Y.v, 0, 1).copy())
    if bias is not None:
        m = W.shape[0]
        c0 = c[0] + bk.thin(np.broadcast_to(bias.reshape(m, 1), (m, B)))
        c[0] = c0
    return T.TM(bk, c)


class SymCtx:
    """Extra-symbol bookkeeping for one network evaluation on a batch of cells (forms shape (n+2, k, B))."""

    def __init__(self, n_in: int):
        self.n_in = n_in
        self.M = n_in

    def pad(self, A: T.TM) -> T.TM:
        """Zero rows for symbols created after A (symbols are append-only)."""
        if A.n == self.M:
            return A
        bk = A.bk
        z = bk.zeros((self.M - A.n,) + A.c.shape[1:])
        return _mk(bk, A.const, _cat(bk, [_lin(A), z], 0), A.rem)

    def promote(self, forms):
        """Move every entry's remainder into a fresh symbol (one per entry per form)."""
        forms = [self.pad(f) for f in forms]
        out = []
        new_rows = []
        offsets = []
        for f in forms:
            bk = f.bk
            k = f.c.shape[1]
            m = f.rem.mid()
            r = _up(np.maximum(_up(f.rem.upper() - m), _up(m - f.rem.lower())))
            offsets.append(k)
            new_rows.append((f, m, r))
        total_new = sum(offsets)
        start = 0
        for (f, m, r) in new_rows:
            bk = f.bk
            k, B = f.c.shape[1], f.c.shape[2]
            rows = np.zeros((total_new, k, B))
            rows[start + np.arange(k), np.arange(k), :] = r
            start += k
            lin = _cat(bk, [_lin(f), bk.thin(rows)], 0)
            out.append(_mk(bk, f.const + bk.thin(m), lin, bk.zeros(f.rem.shape)))
        self.M += total_new
        return out

    def reduce(self, forms, max_sym: int):
        """Girard reduction per cell: keep the max_sym heaviest extra symbols, lump the rest into remainders."""
        forms = [self.pad(f) for f in forms]
        n_extra = self.M - self.n_in
        if n_extra <= max_sym:
            return forms
        B = forms[0].c.shape[2]
        mags = [f.bk.mag(_lin(f)[self.n_in:]) for f in forms]  # (n_extra, k, B) floats
        mass = sum(mg.sum(1) for mg in mags)  # (n_extra, B), selection only
        keep = np.argsort(-mass, axis=0)[:max_sym]  # (max_sym, B)
        drop = np.ones((n_extra, B), dtype=bool)
        drop[keep, np.arange(B)[None, :]] = False
        out = []
        for f, mg in zip(forms, mags):
            bk = f.bk
            k = f.c.shape[1]
            lumped = _sum_up(mg * drop[:, None, :], 0)  # (k, B)
            ext = _lin(f)[self.n_in:]
            idx = (keep[:, None, :], np.arange(k)[None, :, None], np.arange(B)[None, None, :])
            kept = _arr_take(ext, idx)
            lin = _cat(bk, [_lin(f)[:self.n_in], kept], 0)
            out.append(_mk(bk, f.const, lin, f.rem + bk.symmetric(lumped)))
        self.M = self.n_in + max_sym
        return out

    def lump(self, A: T.TM) -> T.TM:
        """All extra symbols into the remainder -> a TM over the n_in input symbols (TM1)."""
        A = self.pad(A)
        bk = A.bk
        ext = bk.mag(_lin(A)[self.n_in:])
        lumped = _sum_up(ext, 0) if ext.shape[0] else np.zeros(A.c.shape[1:])
        return _mk(bk, A.const, _lin(A)[:self.n_in], A.rem + bk.symmetric(lumped))


def enclose_mlp_vec_zono(layers, inputs: list[T.TM], channels: tuple[int, ...], max_sym: int = 64):
    """Drop-in for engine2d.enclose_mlp_vec: (TM (m, B), {channel: TM (m, B)}) over the input symbols, computed
    with fresh symbols for every remainder of act, sech^2 and the derivative channels (shared within the network)."""
    bk = inputs[0].bk
    n = inputs[0].n
    Bn = inputs[0].c.shape[1]
    ctx = SymCtx(n)
    a = T.tm_stack(inputs, axis=0)
    W0, b0 = layers[0]
    z = zlinmap(W0, a, b0)
    dz = {}
    for ch in channels:
        col = np.broadcast_to(W0[:, ch].reshape(-1, 1), (W0.shape[0], Bn))
        dz[ch] = T.tm_const(bk, col, n, col.shape)
    for W, b in layers[1:]:
        act, s = ztanh_sech2(ctx.pad(z))
        act, s = ctx.promote([act, s])  # tanh / sech^2 remainders -> symbols shared by all channels
        da = ctx.promote([zmul(s, ctx.pad(dz[ch])) for ch in channels]) if channels else []
        red = ctx.reduce([act] + da, max_sym)  # s is no longer needed
        act, da = red[0], dict(zip(channels, red[1:]))
        z = zlinmap(W, act, b)
        dz = {ch: zlinmap(W, da[ch]) for ch in channels}
    return ctx.lump(z), {ch: ctx.lump(dz[ch]) for ch in channels}


def enclose_mlp_zono(layers, inputs, channels, max_sym: int = 64):
    """Scalar-output variant (engine.enclose_mlp interface)."""
    z, dz = enclose_mlp_vec_zono(layers, inputs, channels, max_sym)
    return z[0], {ch: dz[ch][0] for ch in channels}
