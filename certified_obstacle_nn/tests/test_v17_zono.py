"""v17 Stage 4 (roadmap R3): validated affine/zonotope forward mode (certification/enclose/zono.py) and the opt-in
enclosure candidates of the engines. Containment against 50-digit mpmath (network value, first derivatives, disk
data) at corners and interior points on both backends, saturation, data-interface and projection-interface cells,
the arithmetic rules in isolation, and "new range subset of TM1 range" on every cell of Stefan 32^2 and a disk smoke
partition."""

from fractions import Fraction
from functools import partial

import mpmath as mp
import numpy as np
import pytest

from certified_obstacle_nn.certification.enclose import intervals_arb as AR
from certified_obstacle_nn.certification.enclose import intervals_np as NP
from certified_obstacle_nn.certification.enclose import tm as T
from certified_obstacle_nn.certification.enclose import upper_sum as US
from certified_obstacle_nn.certification.enclose import zono as Z
from certified_obstacle_nn.certification.enclose.engine import StefanAdapter, TripleEncloser
from certified_obstacle_nn.certification.enclose.engine2d import DiskAdapter, TripleEncloser2D, parse_enclosure
from certified_obstacle_nn.experiments.v14.common import STEFAN_SEEDS, load_frozen_stefan
from certified_obstacle_nn.tests.test_enclose2d import _cells, _mp_disk, _mp_fields, _triple
from certified_obstacle_nn.tests.stored_data import needs

mp.mp.dps = 50
KEYS = ("R", "S1", "S2", "gam", "gam_t", "gam_x", "gam_y", "lam", "u", "Rhat_pos", "Shat1_pos", "Shat2_pos")


def test_parse_enclosure():
    assert parse_enclosure("TM1") == {"aff": None, "prod": False, "tm2": False, "spec": "TM1"}
    assert parse_enclosure("TM1+AFF64+PROD")["aff"] == 64
    with pytest.raises(ValueError):
        parse_enclosure("AFF64")
    # v17b: TM2 is an opt-in component ("TM2" alone = "TM1+TM2"); unknown components still raise
    assert parse_enclosure("TM1+TM2")["tm2"] and parse_enclosure("TM2")["tm2"]
    with pytest.raises(ValueError):
        parse_enclosure("TM1+TM3")


def test_linmap_np_midrad_contains_exact_products():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(7, 9)) * 10.0 ** rng.integers(-3, 4, size=(7, 9))
    lo = rng.normal(size=(9, 5))
    hi = lo + rng.random((9, 5)) * 1e-3
    Y = Z._linmap_np(W, NP.NIA(lo, hi))
    for _ in range(20):
        x = lo + (hi - lo) * rng.random(lo.shape)
        for i in range(7):
            for c in range(5):
                exact = sum(Fraction(float(W[i, j])) * Fraction(float(x[j, c])) for j in range(9))
                assert Fraction(float(Y.lo[i, c])) <= exact <= Fraction(float(Y.hi[i, c]))
    # thin input, extreme magnitudes incl. underflow
    Wt = np.array([[1e-300, -1e-300, 3.0], [1e300, 1e-310, -2.0]])
    xt = np.array([[1e-20], [3e-15], [1.0 / 3.0]])
    Yt = Z._linmap_np(Wt, NP.NIA(xt, xt))
    for i in range(2):
        exact = sum(Fraction(float(Wt[i, j])) * Fraction(float(xt[j, 0])) for j in range(3))
        assert Fraction(float(Yt.lo[i, 0])) <= exact <= Fraction(float(Yt.hi[i, 0]))


def _rand_tm(bk, rng, n, shape, scale):
    rows = [rng.normal(size=shape) * scale for _ in range(n + 1)]
    rad = np.abs(rng.normal(size=shape)) * scale * 0.1
    comps = [bk.thin(r) for r in rows] + [bk.symmetric(rad)]
    return T.TM(bk, bk.stack(comps))


def _eval(A, e, rv):
    """Point of the set described by TM A at symbol values e (n,) and remainder fraction rv in [-1, 1]."""
    lo = A.c.lower() if hasattr(A.c, "lower") else None
    c = 0.5 * (A.c.lower() + A.c.upper())
    val = c[0] + sum(c[j + 1] * e[j] for j in range(A.n))
    rem_lo, rem_hi = A.rem.lower(), A.rem.upper()
    return val + 0.5 * (rem_lo + rem_hi) + 0.5 * (rem_hi - rem_lo) * rv, lo


@pytest.mark.parametrize("bkname", ["numpy", "arb"])
def test_zmul_and_taylor_contain_pointwise_values(bkname):
    bk = NP if bkname == "numpy" else AR
    rng = np.random.default_rng(1)
    n, shape = 5, (3, 2)
    A, B = _rand_tm(bk, rng, n, shape, 0.3), _rand_tm(bk, rng, n, shape, 0.3)
    P = Z.zmul(A, B)
    th, sh = Z.ztanh_sech2(A)
    for _ in range(30):
        e = rng.uniform(-1, 1, n)
        ra, rb = rng.uniform(-1, 1, shape), rng.uniform(-1, 1, shape)
        a, _ = _eval(A, e, ra)
        b, _ = _eval(B, e, rb)
        for name, F, val in (("mul", P, a * b), ("tanh", th, np.tanh(a)), ("sech2", sh, 1 / np.cosh(a) ** 2)):
            mid = 0.5 * (F.c.lower() + F.c.upper())
            lin = mid[0] + sum(mid[j + 1] * e[j] for j in range(F.n))
            # the TM at e encloses [lin - coefficient widths - rem]: use the full interval evaluation
            rad_coef = 0.5 * (F.c.upper() - F.c.lower())
            w = rad_coef[0] + sum(rad_coef[j + 1] * abs(e[j]) for j in range(F.n))
            lo = lin - w + F.rem.lower()
            hi = lin + w + F.rem.upper()
            tol = 1e-12 * (1 + np.abs(val))  # float64 evaluation of the reference value only
            assert np.all(lo - tol <= val) and np.all(val <= hi + tol), name


def _zono_fields(enc, bk, lo, hi, ms):
    return enc.fields(bk, lo, hi, mlp=partial(Z.enclose_mlp_vec_zono, max_sym=ms))


@pytest.mark.parametrize("bkname,ms,ncell", [("numpy", 8, 6), ("numpy", 64, 6), ("arb", 8, 3)])
@pytest.mark.parametrize("near", [False, True])
@pytest.mark.parametrize("scale", [1.5, 30.0])
def test_zono_fields_contain_mpmath(bkname, ms, ncell, near, scale):
    """Affine-form enclosures of every field contain 50-digit values at interior points and corners, on generic
    and data-interface cells, for moderate and saturating (x30) weights."""
    bk = NP if bkname == "numpy" else AR
    tr = _triple(1, depth=3, width=6, scale=scale)
    enc = TripleEncloser2D(tr, DiskAdapter())
    rng = np.random.default_rng(7 + near)
    lo, hi = _cells(rng, ncell, (2, 6), near_interface=near)
    F = _zono_fields(enc, bk, lo, hi, ms)
    rg = {k: F[k].range() for k in KEYS}
    for i in range(lo.shape[1]):
        pts = [lo[:, i] + (hi[:, i] - lo[:, i]) * rng.random(3) for _ in range(6)]
        pts += [[hi[j, i] if (c >> j) & 1 else lo[j, i] for j in range(3)] for c in range(8)]
        for p in pts:
            for k, v in _mp_fields(tr, *p).items():
                assert mp.mpf(rg[k].lower()[i]) <= v <= mp.mpf(rg[k].upper()[i]), (k, i, scale)


def _sampled_density_ok(enc, bk, lo, hi, tr, rng):
    """Every cell density of the candidate dominates the 50-digit pointwise densities (projection interface:
    branch of v_hat chosen by the sign of gamma at the point)."""
    d = enc.densities(bk, lo, hi)
    for i in range(lo.shape[1]):
        for _ in range(8):
            p = lo[:, i] + (hi[:, i] - lo[:, i]) * rng.random(3)
            v = _mp_fields(tr, *p)
            pos = v["gam"] > 0
            Rh = v["Rhat_pos"] if pos else v["R"]
            Sh2 = (v["Shat1_pos"] ** 2 + v["Shat2_pos"] ** 2) if pos else (v["S1"] ** 2 + v["S2"] ** 2)
            c = v["lam"] * max(-v["gam"], 0)
            assert Rh**2 <= mp.mpf(d["Rh_term"][i])
            assert Sh2 <= mp.mpf(d["Sh_term"][i])
            assert c <= mp.mpf(d["c"][i])
            assert v["R"] ** 2 <= mp.mpf(d["R0_term"][i])
    return d


@pytest.mark.parametrize("bkname", ["numpy", "arb"])
def test_candidate_densities_dominate_pointwise_on_interface_cells(bkname):
    bk = NP if bkname == "numpy" else AR
    tr = _triple(2, depth=3, width=6, scale=1.5)
    rng = np.random.default_rng(11)
    lo, hi = _cells(rng, 4 if bkname == "numpy" else 2, (3, 6), near_interface=True)
    for spec in ("TM1+PROD", "TM1+AFF8+PROD"):
        _sampled_density_ok(TripleEncloser2D(tr, DiskAdapter(), enclosure=spec), bk, lo, hi, tr, rng)


def _subset_densities(d_new, d_tm1):
    for k in ("r", "p", "c", "rhat", "R0_term", "Rh_term", "Sh_term", "S_term", "gradw", "w2_term", "wt2_term"):
        assert np.all(d_new[k] <= d_tm1[k]), k
    # sign of gamma: resolved by TM1 -> resolved identically by the candidate
    s0, s1 = d_tm1["sign"], d_new["sign"]
    assert np.all(s1[s0 != 0] == s0[s0 != 0])


def test_new_range_subset_tm1_stefan_32_every_cell():
    problem, triple = load_frozen_stefan(STEFAN_SEEDS[0])
    part = US.uniform_2d(32, 32)
    base = TripleEncloser(triple, StefanAdapter(problem)).densities(NP, part.t0, part.t1, part.x0, part.x1)
    for spec in ("TM1+PROD", "TM1+AFF16+PROD"):
        enc = TripleEncloser(triple, StefanAdapter(problem), enclosure=spec)
        new = enc.densities(NP, part.t0, part.t1, part.x0, part.x1)
        _subset_densities(new, base)
        F, G = enc.fields(NP, part.t0, part.t1, part.x0, part.x1), _stefan_zono_fields(enc, part)
        for k in ("R", "S", "gam"):  # the affine forms alone are valid too; report-only improvement check below
            assert np.all(np.isfinite(G[k].range().upper()))
    assert np.sum(new["rhat"]) < np.sum(base["rhat"])


def _stefan_zono_fields(enc, part):
    return enc.fields(NP, part.t0, part.t1, part.x0, part.x1, mlp=partial(Z.enclose_mlp_zono, max_sym=16))


@needs('results/expanding_disk_2d/final/long_6x64_N20000_s0_k5')
def test_new_range_subset_tm1_disk_smoke_every_cell():
    from certified_obstacle_nn.certification.enclose import boxes as Bx
    from certified_obstacle_nn.experiments.expanding_disk_2d.final_common import load_run
    from certified_obstacle_nn.experiments.v14.common import REPO_ROOT

    triple, cfg = load_run(REPO_ROOT / "results/expanding_disk_2d/final/long_6x64_N20000_s0_k5")
    part = Bx.uniform(3, 8)
    base = TripleEncloser2D(triple, DiskAdapter()).densities(NP, part.lo, part.hi)
    new = TripleEncloser2D(triple, DiskAdapter(), enclosure="TM1+AFF32+PROD").densities(NP, part.lo, part.hi)
    _subset_densities(new, base)
