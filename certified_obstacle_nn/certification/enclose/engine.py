"""Per-cell validated enclosures of the v14 component densities (Stefan class:
1-D space, constant A > 0, b = c = 0, box mask, square multiplier head).

Pipeline per batch of closed cells K = [t0,t1] x [x0,x1] (dyadic endpoints):

1. t = tc + ht e1, x = xc + hx e2 as exact first-order Taylor models (TM1).
2. Forward-mode TM1 propagation through each frozen tanh MLP: value and the
   requested first-derivative channels, d a = sech^2(z) d z (enclose/tm.py).
3. Data (f, g, g_t, g_x, u0) and mask B, B' as TM1s (problem adapter).
4. Fields  u = B N, u_t = B N_t, u_x = B' N + B N_x, lambda = M^2,
           R = f - u_t - sigma_x + lambda,   S = A^{-1/2} sigma + A^{1/2} u_x,
           gamma = g - u, gamma_t, gamma_x,
           projected branch (gamma > 0):  R_hat = f - g_t - sigma_x + lambda,
                                          S_hat = A^{-1/2} sigma + A^{1/2} g_x.
5. Per-cell density suprema from TM ranges (outward float upper bounds):
     r_K    = (Lambda/alpha) sup R^2 + sup S^2
     p_K    = [sup gamma > 0] ( A sup gamma_x^2 + kappa (sup gamma^+)^2 + (Lambda/alpha) sup gamma_t^2 )
     c_K    = sup lambda * sup (-gamma)^+
     rhat_K = r_K if sup gamma <= 0;  (Lambda/alpha) sup R_hat^2 + sup S_hat^2 if inf gamma > 0;
              max of the two termwise otherwise (hull of both branches).
   (grad w = 1_{gamma>0} grad gamma a.e., so sup |grad w|^2 <= sup |grad gamma|^2 on any
   cell where gamma can be positive and grad w = 0 a.e. where gamma <= 0.)
6. Trace cells S = [x0,x1] at t = 0 and t = T (separate spatial partitions, ht = 0):
     r0_S = sup (v(0) - u0)^2,   p0_S = (sup (g - v)(0)^+)^2,   pT_S = (sup (g - v)(T)^+)^2,
     r0hat_S = sup (max(v,g)(0) - u0)^2 via [max(lo_v, lo_g), max(hi_v, hi_g)].

Upper sums U = sum |K| * density_K (+ sum |S| * trace_S) are reduced in arb
(certification/enclose/upper_sum.py).
"""

from __future__ import annotations

import numpy as np

from certified_obstacle_nn.certification.enclose import tm as T
from certified_obstacle_nn.certification.enclose.second_order import enclose_mlp2


def _up(x):
    return np.nextafter(x, np.inf)


def _sq_up(m):
    return _up(m * m)


def nan_guard(out: dict, ranges) -> dict:
    """v18 D34: a NaN endpoint in any field range makes every float density of that cell NaN. Branch selections such
    as np.where(sup gamma > 0, p, 0) would otherwise map a NaN gap to a finite density. Bit for bit no-op when every
    range is NaN-free."""
    bad = None
    for r in ranges:
        lo, hi = (r.lower(), r.upper()) if hasattr(r, "lower") else (r, r)
        b = np.isnan(lo) | np.isnan(hi)
        bad = b if bad is None else bad | b
    if bad is None or not bad.any():
        return out
    return {k: (np.where(bad, np.nan, v) if np.issubdtype(np.asarray(v).dtype, np.floating) else v)
            for k, v in out.items()}


def split_terms(g_lo, g_hi, supR2, supS2, supRh2, supSh2, supw2, supgt2) -> dict:
    """v17 per-term (UNWEIGHTED) cell densities, roadmap R3 Stage 1 items 2-3. Each is the sup
    of ONE term on the cell, enclosed separately (termwise majorant, never a sup of a sum):
      R0_term = sup R^2,  (S_term = sup |S|^2 is stored already)          unprojected r
      Rh_term = sup R_hat^2,  Sh_term = sup |S_hat|^2                       projected r_hat
                by the sign of gamma on the cell: gamma <= 0 -> v_hat = v (R, S);
                gamma > 0 -> v_hat = g (projected branch); sign-unresolved -> termwise
                hull max(.,.) of both branches (R_hat, S_hat equal one of the two a.e.)
      w2_term = [sup gamma > 0] (sup gamma^+)^2,  wt2_term = [sup gamma > 0] sup gamma_t^2.
    The aggregates satisfy rhat = up(La Rh_term + Sh_term) etc. cell by cell."""
    hull = (g_hi > 0) & ~(g_lo > 0)
    pos = g_lo > 0
    Rh = np.where(pos, supRh2, np.where(hull, np.maximum(supR2, supRh2), supR2))
    Sh = np.where(pos, supSh2, np.where(hull, np.maximum(supS2, supSh2), supS2))
    return {"R0_term": supR2, "Rh_term": Rh, "Sh_term": Sh,
            "w2_term": np.where(g_hi > 0, supw2, 0.0), "wt2_term": np.where(g_hi > 0, supgt2, 0.0)}


def mlp_weights(mlp) -> list[tuple[np.ndarray, np.ndarray]]:
    """Exact float64 copies of the stored weights (float32 -> float64 is exact)."""
    out = []
    for layer in mlp.layers:
        W = layer.weight.detach().double().numpy().copy()
        b = layer.bias.detach().double().numpy().copy()
        out.append((W, b))
    return out


def enclose_mlp(layers, inputs: list[T.TM], channels: tuple[int, ...]):
    """TM1 enclosure of a tanh MLP and its derivatives w.r.t. input indices
    in `channels`. Returns (value TM, {channel: derivative TM}) of the scalar
    output, batch shape (B,)."""
    bk = inputs[0].bk
    n = inputs[0].n
    B = inputs[0].c.shape[1]
    a = T.tm_stack(inputs, axis=0)  # (n+2, d_in, B)
    W0, b0 = layers[0]
    z = T.tm_linmap(W0, a, b0)
    dz = {}
    for ch in channels:
        col = np.broadcast_to(W0[:, ch].reshape(-1, 1), (W0.shape[0], B))
        dz[ch] = T.tm_const(bk, col, n, col.shape)
    for W, b in layers[1:]:
        act, s = T.tm_tanh_sech2(z)
        da = {ch: T.tm_mul(s, dz[ch]) for ch in channels}
        z = T.tm_linmap(W, act, b)
        dz = {ch: T.tm_linmap(W, da[ch]) for ch in channels}
    return z[0], {ch: dz[ch][0] for ch in channels}


class StefanAdapter:
    """Exact Stefan data as TM1s: h = e^t - t - 1, g = -h (1-x),
    g_t = -(e^t - 1)(1-x), g_x = h, f = -1 + g_t, u0 = 0, B = 4x(1-x) on (0,1)."""

    name = "stefan"

    def __init__(self, problem):
        self.p = problem
        a, b = problem.omega_bounds
        assert (a, b) == (0.0, 1.0), "adapter written for Omega = (0,1)"
        assert float(problem.b) == 0.0 and float(problem.c) == 0.0
        self.A = float(problem.A)
        self.La = problem.Lambda / problem.alpha
        self.kappa = problem.kappa
        # constants used in densities must be exact floats here (Stefan: 1, 1, 1)
        for v in (self.A, self.La, self.kappa):
            assert v == 1.0, "validated Stefan adapter assumes A = Lambda/alpha = kappa = 1 exactly"

    def data(self, t: T.TM, x: T.TM) -> dict:
        E = T.tm_exp(t)
        h = E - t - 1.0
        omx = 1.0 - x
        g = -T.tm_mul(h, omx)
        g_t = -T.tm_mul(E - 1.0, omx)
        g_x = h
        f = g_t - 1.0
        return {"f": f, "g": g, "g_t": g_t, "g_x": g_x}

    def data_grad(self, t: T.TM, x: T.TM) -> dict:
        """Cell enclosures of the data derivatives used by the mean-value form:
        f_t = -e^t (1-x), f_x = e^t - 1, and grad (f - g_t) = 0 exactly
        (f = g_t - 1 identically for Stefan)."""
        E = T.tm_exp(t)
        zero = T.tm_const(x.bk, 0.0, x.n, x.c.shape[1:])
        return {"f_t": -T.tm_mul(E, 1.0 - x), "f_x": E - 1.0, "fmg_t": zero, "fmg_x": zero}

    def data_trace(self, t: T.TM, x: T.TM) -> dict:
        E = T.tm_exp(t)
        h = E - t - 1.0
        g = -T.tm_mul(h, 1.0 - x)
        u0 = T.tm_const(x.bk, 0.0, x.n, x.c.shape[1:])
        return {"g": g, "u0": u0}

    @staticmethod
    def mask(x: T.TM):
        B = T.tm_mul(x * 4.0, 1.0 - x)
        dB = 4.0 - x * 8.0
        return B, dB


class Rising1DAdapter:
    """v20 rising1d data as TM1s (problems/rising1d.py): f = -8 sin(pi x) (tm_sinpi, arb ball evaluation),
    g = -3/5 + 3/5 t - 2 (x - 1/2)^2, g_t = 3/5, g_x = -4 (x - 1/2) (rationals enclosed by bk.rational), u0 = 0,
    B = 4x(1-x) on (0,1). Requires A = Lambda/alpha = kappa = 1 exactly (as the Stefan adapter)."""

    name = "rising1d"

    def __init__(self, problem):
        self.p = problem
        a, b = problem.omega_bounds
        assert (a, b) == (0.0, 1.0), "adapter written for Omega = (0,1)"
        assert float(problem.b) == 0.0 and float(problem.c) == 0.0
        self.A = float(problem.A)
        self.La = problem.Lambda / problem.alpha
        self.kappa = problem.kappa
        for v in (self.A, self.La, self.kappa):
            assert v == 1.0, "validated rising1d adapter assumes A = Lambda/alpha = kappa = 1 exactly"

    @staticmethod
    def _g(t: T.TM, x: T.TM):
        bk, n, shape = x.bk, x.n, x.c.shape[1:]
        three_fifths = bk.rational(3, 5, shape)
        return T.tm_rational(bk, -3, 5, n, shape) + t.scale(three_fifths) - T.tm_sqr(x - 0.5) * 2.0

    def data(self, t: T.TM, x: T.TM) -> dict:
        bk, n, shape = x.bk, x.n, x.c.shape[1:]
        f = T.tm_sinpi(x) * -8.0
        g_t = T.tm_rational(bk, 3, 5, n, shape)
        g_x = (x - 0.5) * -4.0
        return {"f": f, "g": self._g(t, x), "g_t": g_t, "g_x": g_x}

    def data_grad(self, t: T.TM, x: T.TM) -> dict:
        raise NotImplementedError("mean-value form not used for rising1d (protocol: TM1)")

    def data_trace(self, t: T.TM, x: T.TM) -> dict:
        u0 = T.tm_const(x.bk, 0.0, x.n, x.c.shape[1:])
        return {"g": self._g(t, x), "u0": u0}

    @staticmethod
    def mask(x: T.TM):
        B = T.tm_mul(x * 4.0, 1.0 - x)
        dB = 4.0 - x * 8.0
        return B, dB


def sq_lower(I) -> np.ndarray:
    """v20: lower endpoint of {z^2 : z in [a, b]} rounded downward: 0 if a <= 0 <= b, else min(a^2, b^2).
    The float square is correctly rounded, so one step toward -inf is <= the exact square. NaN propagates."""
    a, b = I.lower(), I.upper()
    m = np.minimum(np.abs(a), np.abs(b))
    lo = np.nextafter(m * m, -np.inf)
    lo = np.maximum(lo, 0.0)  # squares are >= 0 (nextafter(0) = -tiny)
    out = np.where((a <= 0) & (b >= 0), 0.0, lo)
    return np.where(np.isnan(a) | np.isnan(b), np.nan, out)


def lower_terms(g_lo, g_hi, tau_rng, ux_rng, gx_rng) -> dict:
    """v20 Task 2 per-cell lower endpoints: tau2_lo <= inf_K tau^2, gvh2_lo <= inf_K (d_x v_hat)^2 with
    d_x v_hat = d_x v where gamma = g - v <= 0 on K (g_hi <= 0), = g_x where gamma > 0 on K (g_lo > 0), and on
    sign-unresolved cells the minimum of both branch lower endpoints (d_x v_hat equals one of the two a.e.)."""
    vx = sq_lower(ux_rng)
    gx = sq_lower(gx_rng)
    gvh = np.where(g_hi <= 0, vx, np.where(g_lo > 0, gx, np.minimum(vx, gx)))
    return {"tau2_lo": sq_lower(tau_rng), "gvh2_lo": gvh}


class TripleEncloser:
    """`mean_value=True` additionally encloses R and R_hat by the mean-value form
        R(K) in R(z_K) + [-ht, ht] dR/dt(K) + [-hx, hx] dR/dx(K)
    (R(z_K) from a degenerate point cell; dR/dt, dR/dx from TM1 enclosures of the
    networks' second derivatives over K, enclose/second_order.py) and uses the
    intersection with the TM1 range: both are validated enclosures of the same
    set, so their intersection is one too and is never wider (DECISIONS D16)."""

    def __init__(self, triple, adapter, mean_value: bool = False, enclosure: str = "TM1", lower_sums: bool = False):
        from certified_obstacle_nn.certification.enclose.engine2d import parse_enclosure

        self.mean_value = bool(mean_value)
        self.lower_sums = bool(lower_sums)  # v20: also return tau2_lo, gvh2_lo (lower_terms); off = unchanged
        self.enclosure = parse_enclosure(enclosure)
        if triple.multiplier_net.head != "square":
            raise NotImplementedError("validated path implemented for the square multiplier head")
        for net in (triple.state_net, triple.flux_net, triple.multiplier_net):
            if getattr(net, "x_rescale", None) is not None:
                raise NotImplementedError("x_rescale not supported in the validated path")
        self.Ws = mlp_weights(triple.state_net.net)
        self.Wf = mlp_weights(triple.flux_net.net)
        self.Wm = mlp_weights(triple.multiplier_net.net)
        self.ad = adapter

    # ------------------------------------------------------------------ volume
    def fields(self, bk, t0, t1, x0, x1, mlp=None) -> dict:
        mlp = enclose_mlp if mlp is None else mlp
        tc = 0.5 * (t0 + t1)
        ht = 0.5 * (t1 - t0)
        xc = 0.5 * (x0 + x1)
        hx = 0.5 * (x1 - x0)
        t = T.tm_variable(bk, tc, ht, 1, 2)
        x = T.tm_variable(bk, xc, hx, 2, 2)
        N, dN = mlp(self.Ws, [t, x], (0, 1))
        sig, dsig = mlp(self.Wf, [t, x], (1,))
        M, _ = mlp(self.Wm, [t, x], ())
        d = self.ad.data(t, x)
        Bm, dB = self.ad.mask(x)
        u = T.tm_mul(Bm, N)
        u_t = T.tm_mul(Bm, dN[0])
        u_x = T.tm_mul(dB, N) + T.tm_mul(Bm, dN[1])
        lam = T.tm_sqr(M)
        sig_x = dsig[1]
        sA = np.sqrt(self.ad.A)
        assert sA == 1.0
        R = d["f"] - u_t - sig_x + lam
        S = sig + u_x
        gam = d["g"] - u
        gam_t = d["g_t"] - u_t
        gam_x = d["g_x"] - u_x
        Rhat = d["f"] - d["g_t"] - sig_x + lam
        Shat = sig + d["g_x"]
        return {"R": R, "S": S, "gam": gam, "gam_t": gam_t, "gam_x": gam_x, "lam": lam, "M": M,
                "Rhat_pos": Rhat, "Shat_pos": Shat, "u": u, "sig": sig, "u_x": u_x, "g_x": d["g_x"]}

    def mean_value_ranges(self, bk, t0, t1, x0, x1) -> dict:
        """Mean-value enclosures of R and R_hat on the cells (class docstring)."""
        tc = 0.5 * (t0 + t1)
        ht = 0.5 * (t1 - t0)
        xc = 0.5 * (x0 + x1)
        hx = 0.5 * (x1 - x0)
        Fz = self.fields(bk, tc, tc, xc, xc)  # dyadic midpoints are exact: a point cell
        t = T.tm_variable(bk, tc, ht, 1, 2)
        x = T.tm_variable(bk, xc, hx, 2, 2)
        _, dN, d2N = enclose_mlp2(self.Ws, [t, x], (0,), ((0, 0), (0, 1)))
        _, _, d2s = enclose_mlp2(self.Wf, [t, x], (), ((0, 1), (1, 1)))
        M, dM = enclose_mlp(self.Wm, [t, x], (0, 1))
        dd = self.ad.data_grad(t, x)
        Bm, dB = self.ad.mask(x)
        lam_t = T.tm_mul(M, dM[0]) * 2.0
        lam_x = T.tm_mul(M, dM[1]) * 2.0
        sig_xt, sig_xx = d2s[(0, 1)], d2s[(1, 1)]
        R_t = dd["f_t"] - T.tm_mul(Bm, d2N[(0, 0)]) - sig_xt + lam_t
        R_x = dd["f_x"] - T.tm_mul(dB, dN[0]) - T.tm_mul(Bm, d2N[(0, 1)]) - sig_xx + lam_x
        Rh_t = dd["fmg_t"] - sig_xt + lam_t
        Rh_x = dd["fmg_x"] - sig_xx + lam_x
        Ht, Hx = bk.symmetric(ht), bk.symmetric(hx)
        return {
            "R": Fz["R"].range() + Ht * R_t.range() + Hx * R_x.range(),
            "Rhat_pos": Fz["Rhat_pos"].range() + Ht * Rh_t.range() + Hx * Rh_x.range(),
        }

    def densities(self, bk, t0, t1, x0, x1) -> dict:
        F = self.fields(bk, t0, t1, x0, x1)
        rng = {k: F[k].range() for k in ("R", "S", "gam", "gam_t", "gam_x", "Rhat_pos", "Shat_pos")}
        extra = {}
        prods = [F]
        alts = []
        if self.enclosure["aff"] is not None:  # v17 opt-in, see engine2d.parse_enclosure
            from functools import partial

            from certified_obstacle_nn.certification.enclose.zono import enclose_mlp_zono

            alts.append(partial(enclose_mlp_zono, max_sym=self.enclosure["aff"]))
        if self.enclosure["tm2"]:  # v17b opt-in
            from certified_obstacle_nn.certification.enclose.tm2 import enclose_mlp_tm2

            alts.append(enclose_mlp_tm2)
        for mlp in alts:
            G = self.fields(bk, t0, t1, x0, x1, mlp=mlp)
            for k in rng:
                g = G[k].range()
                better = (g.lower() > rng[k].lower()) | (g.upper() < rng[k].upper())
                rng[k] = bk.select(better, bk.intersect(rng[k], g), rng[k])
            prods.append(G)
        if self.mean_value:
            mv = self.mean_value_ranges(bk, t0, t1, x0, x1)
            extra["R_width_tm1"] = rng["R"].upper() - rng["R"].lower()
            extra["R_width_mv"] = mv["R"].upper() - mv["R"].lower()
            for k in ("R", "Rhat_pos"):
                # keep the TM1 interval untouched where the mean-value form does not
                # improve an endpoint (arb's intersect re-rounds endpoints outward, D8)
                better = (mv[k].lower() > rng[k].lower()) | (mv[k].upper() < rng[k].upper())
                rng[k] = bk.select(better, bk.intersect(rng[k], mv[k]), rng[k])
        lam_r = bk.sqr(F["M"].range())  # lambda = M^2 >= 0: range of the square of M's range
        lam_tm = F["lam"].range()
        lam_hi = np.minimum(lam_r.upper(), lam_tm.upper())  # min of two valid upper bounds
        for Gx in prods[1:]:
            lam_hi = np.minimum(lam_hi, np.minimum(bk.sqr(Gx["M"].range()).upper(), Gx["lam"].range().upper()))
        La, A, kap = self.ad.La, self.ad.A, self.ad.kappa
        supR2 = _sq_up(bk.mag(rng["R"]))
        supS2 = _sq_up(bk.mag(rng["S"]))
        g_lo = rng["gam"].lower()
        g_hi = rng["gam"].upper()
        gpos = np.maximum(g_hi, 0.0)
        supgx2 = _sq_up(bk.mag(rng["gam_x"]))
        supgt2 = _sq_up(bk.mag(rng["gam_t"]))
        r_K = _up(_up(La * supR2) + supS2)
        p_core = _up(_up(_up(A * supgx2) + _up(kap * _sq_up(gpos))) + _up(La * supgt2))
        p_K = np.where(g_hi > 0, p_core, 0.0)
        c_K = _up(np.maximum(lam_hi, 0.0) * np.maximum(-g_lo, 0.0))
        if self.enclosure["prod"]:
            for Fx in prods:
                c_K = np.minimum(c_K, np.maximum(T.tm_mul(Fx["lam"], -Fx["gam"]).range().upper(), 0.0))
        supRh2 = _sq_up(bk.mag(rng["Rhat_pos"]))
        supSh2 = _sq_up(bk.mag(rng["Shat_pos"]))
        rpos = _up(_up(La * supRh2) + supSh2)
        rhull = _up(_up(La * np.maximum(supR2, supRh2)) + np.maximum(supS2, supSh2))
        rhat_K = np.where(g_hi <= 0, r_K, np.where(g_lo > 0, rpos, rhull))
        guard = list(rng.values()) + [lam_hi]
        if self.lower_sums:
            low_rng = [F[k].range() for k in ("sig", "u_x", "g_x")]
            extra.update(lower_terms(g_lo, g_hi, *low_rng))
            guard += low_rng
        # split of r_K / p_K into their terms (for reporting)
        return nan_guard({
            "r": r_K, "p": p_K, "c": c_K, "rhat": rhat_K,
            "R_term": _up(La * supR2), "S_term": supS2,
            "gradw": np.where(g_hi > 0, _up(A * supgx2), 0.0),
            "w_term": np.where(g_hi > 0, _up(kap * _sq_up(gpos)), 0.0),
            "wt_term": np.where(g_hi > 0, _up(La * supgt2), 0.0),
            "sign": np.where(g_hi <= 0, -1, np.where(g_lo > 0, 1, 0)),
            "R_width": rng["R"].upper() - rng["R"].lower(),
            **split_terms(g_lo, g_hi, supR2, supS2, supRh2, supSh2, _sq_up(gpos), supgt2),
            **extra,
        }, guard)

    # ------------------------------------------------------------------ traces
    def trace_densities(self, bk, tval: float, x0, x1) -> dict:
        xc = 0.5 * (x0 + x1)
        hx = 0.5 * (x1 - x0)
        n = 2
        tt = T.tm_variable(bk, np.full_like(xc, tval), np.zeros_like(xc), 1, n)
        x = T.tm_variable(bk, xc, hx, 2, n)
        N, _ = enclose_mlp(self.Ws, [tt, x], ())
        Bm, _ = self.ad.mask(x)
        v = T.tm_mul(Bm, N)
        d = self.ad.data_trace(tt, x)
        gam = (d["g"] - v).range()
        vm = (v - d["u0"]).range()
        gm = (d["g"] - d["u0"]).range()
        w2 = _sq_up(np.maximum(gam.upper(), 0.0))
        r0 = _sq_up(bk.mag(vm))
        hat_lo = np.maximum(vm.lower(), gm.lower())
        hat_hi = np.maximum(vm.upper(), gm.upper())
        r0hat = _sq_up(np.maximum(np.abs(hat_lo), np.abs(hat_hi)))
        return nan_guard({"w2": w2, "r0": r0, "r0hat": r0hat}, [gam, vm, gm])
