"""Per-cell validated enclosures of the v14 component densities for 2-D space
(space-time boxes K = [t0,t1] x [x0,x1] x [y0,y1], dyadic), A = I, b = c = 0,
box mask B = 16 x(1-x) y(1-y), square multiplier head. Same construction as
the 1-D engine (engine.py) with three Taylor-model variables (t, x, y):

  u = B N,  u_t = B N_t,  grad u = grad B N + B grad N,  lambda = M^2,
  R = f - u_t - div sigma + lambda,  S = sigma + grad u,
  gamma = g - u,  gamma_t,  grad gamma,
  projected branch (gamma > 0): R_hat = f - g_t - div sigma + lambda,  S_hat = sigma + grad g.

Densities per cell (outward float upper bounds of the suprema):
  r_K    = (Lambda/alpha) sup R^2 + sup S_1^2 + sup S_2^2
  p_K    = [sup gamma > 0] ( sup gamma_x^2 + sup gamma_y^2 + kappa (sup gamma^+)^2
                             + (Lambda/alpha) sup gamma_t^2 )
  c_K    = sup lambda * sup (-gamma)^+
  rhat_K = r_K | projected branch | termwise hull, by the sign of gamma on K.
Traces on spatial squares at t = 0 and t = T (D - 1 = 2 spatial variables):
  r0 = sup (v(0) - u0)^2,  w2 = (sup (g - v)^+)^2,  r0hat = sup (max(v, g)(0) - u0)^2.

Problem data come from an adapter. Branch-defined data (indicator of the
contact set in f, positive part in u0) are enclosed branch-aware: on a cell
where the branch is fixed the smooth branch's TM is used, on a straddling cell
the smooth part plus the hull of {0} and the range of the switched term (the
switched term vanishes on the interface, so this hull is tight near it).
Non-dyadic decimal constants (0.15, 0.2, ...) are enclosed as the exact
rationals they denote (``tm_rational``), never as their float roundings.
"""

from __future__ import annotations

import numpy as np

from certified_obstacle_nn.certification.enclose import tm as T
from certified_obstacle_nn.certification.enclose.engine import _sq_up, _up, mlp_weights, nan_guard, split_terms


def enclose_mlp_vec(layers, inputs: list[T.TM], channels: tuple[int, ...]):
    """TM1 enclosure of a tanh MLP with m outputs and of its derivatives with
    respect to the input indices in `channels`. Returns (TM of shape (m, B),
    {channel: TM (m, B)})."""
    bk = inputs[0].bk
    n = inputs[0].n
    B = inputs[0].c.shape[1]
    a = T.tm_stack(inputs, axis=0)
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
    return z, dz


def _branch(lo_pos_mask, hi_nonpos_mask, smooth: T.TM, switched: T.TM) -> T.TM:
    """smooth + 1{branch} * switched with the branch decided per cell:
    always on -> smooth + switched; always off -> smooth; undecided ->
    smooth + hull({0}, range(switched))."""
    bk = smooth.bk
    on = smooth + switched
    rg = switched.range()
    hullz = bk.from_bounds(np.minimum(rg.lower(), 0.0), np.maximum(rg.upper(), 0.0))
    mixed = T.tm_add_interval(smooth, hullz)
    return T.tm_select(lo_pos_mask, on, T.tm_select(hi_nonpos_mask, smooth, mixed))


class DiskAdapter:
    """Expanding disk (experiments/expanding_disk_2d/exact_solution.py), constant-
    multiplier variant, multiplier constant kap (= 5 in every stored run):
      h = e^t - t - 1,  B = 16 x(1-x) y(1-y),  g = -h B,  R(t) = 3/20 + t/5,
      r^2 = (x-1/2)^2 + (y-1/2)^2,  psi = R/2 - r^2/(2R),
      psi_t = (1/10)(1 + r^2/R^2),  Delta psi = -2/R,  |grad psi|^2 = r^2/R^2,
      f = g_t - Delta g - 2 kap + kap 1{psi>0} [2 psi psi_t - 2 psi Delta psi + 2 - 2 r^2/R^2],
      Delta g = -h Delta B,  Delta B = -32 (x(1-x) + y(1-y)),  u0 = kap (psi(0,.)^+)^2.
    (psi_t = R'(R^2 + r^2)/(2R^2) with R' = 1/5.) A = I, alpha = 1, Lambda = 2, kappa = 1/2."""

    name = "disk2d"

    def __init__(self, kap: float = 5.0, Lambda: float = 2.0, alpha: float = 1.0):
        assert kap == 5.0 and Lambda == 2.0 and alpha == 1.0, "constants checked for the stored runs only"
        self.kap = kap
        self.La = Lambda / alpha  # 2, exact
        self.kappa = alpha / Lambda  # 1/2, exact (b = c = 0)
        self.A = 1.0

    @staticmethod
    def mask(x: T.TM, y: T.TM):
        px = T.tm_mul(x, 1.0 - x)
        py = T.tm_mul(y, 1.0 - y)
        B = T.tm_mul(px, py) * 16.0
        Bx = T.tm_mul(1.0 - x * 2.0, py) * 16.0
        By = T.tm_mul(px, 1.0 - y * 2.0) * 16.0
        lapB = (px + py) * -32.0
        return B, Bx, By, lapB

    def _psi(self, t: T.TM, x: T.TM, y: T.TM):
        bk, n, sh = t.bk, t.n, t.c.shape[1:]
        R = T.tm_rational(bk, 3, 20, n, sh) + t * bk.rational(1, 5, sh)
        iR = T.tm_recip(R)
        xm, ym = x - 0.5, y - 0.5
        r2 = T.tm_sqr(xm) + T.tm_sqr(ym)
        r2iR = T.tm_mul(r2, iR)
        psi = R * 0.5 - r2iR * 0.5
        return psi, iR, r2, r2iR

    def data(self, t: T.TM, x: T.TM, y: T.TM) -> dict:
        bk, n, sh = t.bk, t.n, t.c.shape[1:]
        E = T.tm_exp(t)
        h = E - t - 1.0
        B, Bx, By, lapB = self.mask(x, y)
        g = -T.tm_mul(h, B)
        g_t = -T.tm_mul(E - 1.0, B)
        g_x = -T.tm_mul(h, Bx)
        g_y = -T.tm_mul(h, By)
        lap_g = -T.tm_mul(h, lapB)
        psi, iR, r2, r2iR = self._psi(t, x, y)
        gp2 = T.tm_mul(r2iR, iR)  # r^2 / R^2
        psi_t = (gp2 + 1.0) * bk.rational(1, 10, sh)
        bracket = T.tm_mul(psi, psi_t) * 2.0 + T.tm_mul(psi, iR) * 4.0 + 2.0 - gp2 * 2.0
        f0 = g_t - lap_g - 2.0 * self.kap
        pr = psi.range()
        f = _branch(pr.lower() > 0, pr.upper() <= 0, f0, bracket * self.kap)
        return {"f": f, "g": g, "g_t": g_t, "g_x": g_x, "g_y": g_y}

    def data_trace(self, t: T.TM, x: T.TM, y: T.TM) -> dict:
        """t is the constant TM t = 0 or t = T."""
        E = T.tm_exp(t)
        h = E - t - 1.0
        B, _, _, _ = self.mask(x, y)
        g = -T.tm_mul(h, B)
        psi, _, _, _ = self._psi(t, x, y)
        zero = psi * 0.0
        pr = psi.range()
        u0 = _branch(pr.lower() > 0, pr.upper() <= 0, zero, T.tm_sqr(psi) * self.kap)
        return {"g": g, "u0": u0}


class EllipseAdapter:
    """Expanding-ellipse obstacle (problems/ellipse.py, roadmap v14 section 6), every
    decimal constant enclosed as the exact rational it denotes:
      a = 3/25 + t/10,  d = 2/25 + 7t/100,  X1 = x - 43/100,  X2 = y - 57/100,
      zeta = X1^2/a^2 + X2^2/d^2 - 1,  grad zeta = (2 X1/a^2, 2 X2/d^2),
      Delta zeta = 2/a^2 + 2/d^2,  zeta_t = -2 (1/10) X1^2/a^3 - 2 (7/100) X2^2/d^3,
      B = 16 P Q (P = x(1-x), Q = y(1-y)),  c = 1/10 + t/20,  L = 1 + x/5 - y/10,
      u_b = B c L,  g = u_b - k0 zeta^2,  k0 = 1/100,
      f = (d_t - Delta) u_b - 2 k0 zeta (zeta_t - Delta zeta) 1{zeta < 0},
      u0 = u_b(0) - k0 (zeta(0)_-)^2.
    A = I, alpha = 1, Lambda = 2, kappa = 1/2."""

    name = "ellipse"

    def __init__(self, k0_num: int = 1, k0_den: int = 100):
        self.k0 = (k0_num, k0_den)
        self.La = 2.0
        self.kappa = 0.5
        self.A = 1.0

    mask = staticmethod(DiskAdapter.mask)

    def _zeta(self, t: T.TM, x: T.TM, y: T.TM):
        bk, n, sh = t.bk, t.n, t.c.shape[1:]
        Q = lambda p, q: bk.rational(p, q, sh)  # noqa: E731
        a = T.tm_rational(bk, 3, 25, n, sh) + t * Q(1, 10)
        d = T.tm_rational(bk, 2, 25, n, sh) + t * Q(7, 100)
        ia, id_ = T.tm_recip(a), T.tm_recip(d)
        ia2, id2 = T.tm_sqr(ia), T.tm_sqr(id_)
        X1 = x + (-Q(43, 100))
        X2 = y + (-Q(57, 100))
        X1s, X2s = T.tm_sqr(X1), T.tm_sqr(X2)
        zeta = T.tm_mul(X1s, ia2) + T.tm_mul(X2s, id2) - 1.0
        zx = T.tm_mul(X1, ia2) * 2.0
        zy = T.tm_mul(X2, id2) * 2.0
        lapz = (ia2 + id2) * 2.0
        zt = -(T.tm_mul(T.tm_mul(X1s, ia2), ia) * Q(1, 5) + T.tm_mul(T.tm_mul(X2s, id2), id_) * Q(7, 50))
        return zeta, zx, zy, lapz, zt

    def _ub(self, t: T.TM, x: T.TM, y: T.TM):
        bk, sh = t.bk, t.c.shape[1:]
        Q = lambda p, q: bk.rational(p, q, sh)  # noqa: E731
        B, Bx, By, lapB = self.mask(x, y)
        c = t * Q(1, 20) + Q(1, 10)
        L = x * Q(1, 5) - y * Q(1, 10) + 1.0
        BL = T.tm_mul(B, L)
        ub = T.tm_mul(BL, c)
        ub_t = BL * Q(1, 20)
        ub_x = T.tm_mul(T.tm_mul(Bx, L) + B * Q(1, 5), c)
        ub_y = T.tm_mul(T.tm_mul(By, L) - B * Q(1, 10), c)
        lap_ub = T.tm_mul(T.tm_mul(lapB, L) + Bx * Q(2, 5) - By * Q(1, 5), c)
        return ub, ub_t, ub_x, ub_y, lap_ub

    def data(self, t: T.TM, x: T.TM, y: T.TM) -> dict:
        bk, sh = t.bk, t.c.shape[1:]
        k0 = bk.rational(*self.k0, sh)
        ub, ub_t, ub_x, ub_y, lap_ub = self._ub(t, x, y)
        zeta, zx, zy, lapz, zt = self._zeta(t, x, y)
        k02z = zeta * (k0 * 2.0)
        g = ub - T.tm_mul(T.tm_sqr(zeta), T.tm_const(bk, 0.0, t.n, sh) + k0)
        g_t = ub_t - T.tm_mul(k02z, zt)
        g_x = ub_x - T.tm_mul(k02z, zx)
        g_y = ub_y - T.tm_mul(k02z, zy)
        zr = zeta.range()
        switched = -T.tm_mul(k02z, zt - lapz)
        f = _branch(zr.upper() < 0, zr.lower() >= 0, ub_t - lap_ub, switched)
        return {"f": f, "g": g, "g_t": g_t, "g_x": g_x, "g_y": g_y}

    def data_trace(self, t: T.TM, x: T.TM, y: T.TM) -> dict:
        bk, sh = t.bk, t.c.shape[1:]
        k0 = bk.rational(*self.k0, sh)
        ub = self._ub(t, x, y)[0]
        zeta = self._zeta(t, x, y)[0]
        kz2 = T.tm_mul(T.tm_sqr(zeta), T.tm_const(bk, 0.0, t.n, sh) + k0)
        g = ub - kz2
        zr = zeta.range()
        u0 = _branch(zr.upper() < 0, zr.lower() >= 0, ub, -kz2)
        return {"g": g, "u0": u0}


def parse_enclosure(spec: str) -> dict:
    """v17 (roadmap R3 Stage 4) opt-in enclosure candidates, recorded verbatim in config.json:
      "TM1"            -- default, the v14 engine (unchanged);
      "TM1+AFF<m>"     -- network enclosures also by validated affine forms with fresh per-neuron symbols
                          (enclose/zono.py, at most m extra symbols per cell), every field range intersected with
                          the TM1 range (two validated enclosures of the same set; never wider than TM1);
      "+PROD"          -- additionally c_K <= max(sup TM(mu (v - g)), 0) (mu >= 0, so mu (v-g)_+ = (mu (v-g))_+),
                          the minimum with the TM1 product of suprema;
      "TM2" (= "TM1+TM2") -- v17b: network enclosures also by second-order Taylor models (enclose/tm2.py, lumped to
                          TM1 at the network outputs), intersected with TM1 in the same way as AFF<m>."""
    parts = ["TM1", "TM2"] if spec == "TM2" else spec.split("+")
    if parts[0] != "TM1":
        raise ValueError(f"enclosure spec must start with TM1: {spec}")
    out = {"aff": None, "prod": False, "tm2": False, "spec": spec}
    for p in parts[1:]:
        if p.startswith("AFF"):
            out["aff"] = int(p[3:])
        elif p == "TM2":
            out["tm2"] = True
        elif p == "PROD":
            out["prod"] = True
        else:
            raise ValueError(f"unknown enclosure component {p}")
    return out


class TripleEncloser2D:
    def __init__(self, triple, adapter, enclosure: str = "TM1"):
        if triple.head != "square":
            raise NotImplementedError("validated path implemented for the square multiplier head")
        self.Ws = mlp_weights(triple.state)
        self.Wf = mlp_weights(triple.flux)
        self.Wm = mlp_weights(triple.mult)
        self.ad = adapter
        self.enclosure = parse_enclosure(enclosure)

    def fields(self, bk, lo, hi, mlp=None) -> dict:
        mlp = enclose_mlp_vec if mlp is None else mlp
        c = 0.5 * (lo + hi)
        r = 0.5 * (hi - lo)
        t = T.tm_variable(bk, c[0], r[0], 1, 3)
        x = T.tm_variable(bk, c[1], r[1], 2, 3)
        y = T.tm_variable(bk, c[2], r[2], 3, 3)
        N, dN = mlp(self.Ws, [t, x, y], (0, 1, 2))
        N, Nt, Nx, Ny = N[0], dN[0][0], dN[1][0], dN[2][0]
        sig, dsig = mlp(self.Wf, [t, x, y], (1, 2))
        s1, s2 = sig[0], sig[1]
        div = dsig[1][0] + dsig[2][1]
        M, _ = mlp(self.Wm, [t, x, y], ())
        M = M[0]
        d = self.ad.data(t, x, y)
        Bm, Bx, By, _ = self.ad.mask(x, y)
        u = T.tm_mul(Bm, N)
        u_t = T.tm_mul(Bm, Nt)
        u_x = T.tm_mul(Bx, N) + T.tm_mul(Bm, Nx)
        u_y = T.tm_mul(By, N) + T.tm_mul(Bm, Ny)
        lam = T.tm_sqr(M)
        R = d["f"] - u_t - div + lam
        return {
            "R": R, "S1": s1 + u_x, "S2": s2 + u_y, "gam": d["g"] - u, "gam_t": d["g_t"] - u_t,
            "gam_x": d["g_x"] - u_x, "gam_y": d["g_y"] - u_y, "lam": lam, "M": M,
            "Rhat_pos": d["f"] - d["g_t"] - div + lam, "Shat1_pos": s1 + d["g_x"], "Shat2_pos": s2 + d["g_y"], "u": u,
        }

    def densities(self, bk, lo, hi) -> dict:
        F = self.fields(bk, lo, hi)
        keys = ("R", "S1", "S2", "gam", "gam_t", "gam_x", "gam_y", "Rhat_pos", "Shat1_pos", "Shat2_pos")
        rng = {k: F[k].range() for k in keys}
        lam_hi = np.minimum(bk.sqr(F["M"].range()).upper(), F["lam"].range().upper())
        enc = self.enclosure
        prods = [F]
        alts = []
        if enc["aff"] is not None:
            from functools import partial

            from certified_obstacle_nn.certification.enclose.zono import enclose_mlp_vec_zono

            alts.append(partial(enclose_mlp_vec_zono, max_sym=enc["aff"]))
        if enc["tm2"]:
            from certified_obstacle_nn.certification.enclose.tm2 import enclose_mlp_vec_tm2

            alts.append(enclose_mlp_vec_tm2)
        for mlp in alts:
            G = self.fields(bk, lo, hi, mlp=mlp)
            for k in keys:
                g = G[k].range()
                # keep the TM1 interval bit for bit where the affine range improves neither endpoint (D16, D8)
                better = (g.lower() > rng[k].lower()) | (g.upper() < rng[k].upper())
                rng[k] = bk.select(better, bk.intersect(rng[k], g), rng[k])
            lam_hi = np.minimum(lam_hi, np.minimum(bk.sqr(G["M"].range()).upper(), G["lam"].range().upper()))
            prods.append(G)
        La, A, kap = self.ad.La, self.ad.A, self.ad.kappa
        sq = {k: _sq_up(bk.mag(rng[k])) for k in keys if k != "gam"}
        g_lo, g_hi = rng["gam"].lower(), rng["gam"].upper()
        supS2 = _up(sq["S1"] + sq["S2"])
        supSh2 = _up(sq["Shat1_pos"] + sq["Shat2_pos"])
        supgrad = _up(sq["gam_x"] + sq["gam_y"])
        r_K = _up(_up(La * sq["R"]) + supS2)
        wterm = _up(kap * _sq_up(np.maximum(g_hi, 0.0)))
        p_core = _up(_up(_up(A * supgrad) + wterm) + _up(La * sq["gam_t"]))
        p_K = np.where(g_hi > 0, p_core, 0.0)
        c_K = _up(np.maximum(lam_hi, 0.0) * np.maximum(-g_lo, 0.0))
        if enc["prod"]:
            for Fx in prods:  # sup of (mu (v - g))_+ from the TM product, each a valid upper bound of sup c
                c_K = np.minimum(c_K, np.maximum(T.tm_mul(Fx["lam"], -Fx["gam"]).range().upper(), 0.0))
        rpos = _up(_up(La * sq["Rhat_pos"]) + supSh2)
        rhull = _up(_up(La * np.maximum(sq["R"], sq["Rhat_pos"])) + np.maximum(supS2, supSh2))
        rhat_K = np.where(g_hi <= 0, r_K, np.where(g_lo > 0, rpos, rhull))
        return nan_guard({
            "r": r_K, "p": p_K, "c": c_K, "rhat": rhat_K,
            "R_term": _up(La * sq["R"]), "S_term": supS2,
            "gradw": np.where(g_hi > 0, _up(A * supgrad), 0.0),
            "w_term": np.where(g_hi > 0, wterm, 0.0),
            "wt_term": np.where(g_hi > 0, _up(La * sq["gam_t"]), 0.0),
            "sign": np.where(g_hi <= 0, -1, np.where(g_lo > 0, 1, 0)),
            "R_width": rng["R"].upper() - rng["R"].lower(),
            **split_terms(g_lo, g_hi, sq["R"], supS2, sq["Rhat_pos"], supSh2, _sq_up(np.maximum(g_hi, 0.0)),
                          sq["gam_t"]),
        }, list(rng.values()) + [lam_hi])

    def trace_densities(self, bk, tval: float, lo, hi) -> dict:
        c = 0.5 * (lo + hi)
        r = 0.5 * (hi - lo)
        tt = T.tm_variable(bk, np.full_like(c[0], tval), np.zeros_like(c[0]), 1, 3)
        x = T.tm_variable(bk, c[0], r[0], 2, 3)
        y = T.tm_variable(bk, c[1], r[1], 3, 3)
        N, _ = enclose_mlp_vec(self.Ws, [tt, x, y], ())
        Bm, _, _, _ = self.ad.mask(x, y)
        v = T.tm_mul(Bm, N[0])
        d = self.ad.data_trace(tt, x, y)
        gam = (d["g"] - v).range()
        vm = (v - d["u0"]).range()
        gm = (d["g"] - d["u0"]).range()
        w2 = _sq_up(np.maximum(gam.upper(), 0.0))
        r0 = _sq_up(bk.mag(vm))
        hat_lo = np.maximum(vm.lower(), gm.lower())
        hat_hi = np.maximum(vm.upper(), gm.upper())
        r0hat = _sq_up(np.maximum(np.abs(hat_lo), np.abs(hat_hi)))
        return nan_guard({"w2": w2, "r0": r0, "r0hat": r0hat}, [gam, vm, gm])
