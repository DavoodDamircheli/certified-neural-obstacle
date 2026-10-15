"""v14 Stage 4 required tests for certification/enclose (both backends)."""

import math

import mpmath as mp
import numpy as np
import pytest
import sympy as sp
import torch

from certified_obstacle_nn.certification.enclose import intervals_arb as AB
from certified_obstacle_nn.certification.enclose import intervals_np as NB
from certified_obstacle_nn.certification.enclose import tm as T
from certified_obstacle_nn.certification.enclose import upper_sum as US
from certified_obstacle_nn.certification.enclose.engine import StefanAdapter, TripleEncloser, enclose_mlp, mlp_weights
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.stefan import StefanProblem

BACKENDS = [pytest.param(NB, id="numpy"), pytest.param(AB, id="arb")]
mp.mp.dps = 50


def _frozen(tr):
    tr = tr.double().eval()
    for q in tr.parameters():
        q.requires_grad_(False)
    return tr


def _tiny_triple(seed=0, width=4, depth=2, scale=1.0):
    torch.manual_seed(seed)
    tr = NeuralTriple(StefanProblem(), width=width, depth=depth)
    with torch.no_grad():
        for q in tr.parameters():
            q.mul_(scale)
    return _frozen(tr)


def _linear_triple(state, flux, mult):
    """depth-0 'networks' = affine maps (a, b, c): a + b t + c x."""
    tr = NeuralTriple(StefanProblem(), width=1, depth=0)
    with torch.no_grad():
        for net, (a0, bt, cx) in ((tr.state_net.net, state), (tr.flux_net.net, flux), (tr.multiplier_net.net, mult)):
            net.layers[0].weight.copy_(torch.tensor([[bt, cx]]))
            net.layers[0].bias.copy_(torch.tensor([a0]))
    return _frozen(tr)


# ------------------------------------------------------------------ mpmath reference
def _mp_mlp(layers, t, x):
    """value, d/dt, d/dx of a tanh MLP at a point, 50 digits."""
    a = [mp.mpf(t), mp.mpf(x)]
    da = {0: [mp.mpf(1), mp.mpf(0)], 1: [mp.mpf(0), mp.mpf(1)]}
    for li, (W, b) in enumerate(layers):
        z = [sum(mp.mpf(float(W[i, j])) * a[j] for j in range(W.shape[1])) + mp.mpf(float(b[i])) for i in range(W.shape[0])]
        dz = {c: [sum(mp.mpf(float(W[i, j])) * da[c][j] for j in range(W.shape[1])) for i in range(W.shape[0])] for c in da}
        if li == len(layers) - 1:
            return z[0], dz[0][0], dz[1][0]
        a = [mp.tanh(v) for v in z]
        s = [1 - mp.tanh(v) ** 2 for v in z]
        da = {c: [s[i] * dz[c][i] for i in range(len(z))] for c in dz}


def _mp_fields(enc, t, x):
    t = mp.mpf(t)
    x = mp.mpf(x)
    N, Nt, Nx = _mp_mlp(enc.Ws, t, x)
    sg, _, sgx = _mp_mlp(enc.Wf, t, x)
    M, _, _ = _mp_mlp(enc.Wm, t, x)
    E = mp.e**t
    h = E - t - 1
    g = -h * (1 - x)
    g_t = -(E - 1) * (1 - x)
    g_x = h
    f = -1 - (E - 1) * (1 - x)
    B = 4 * x * (1 - x)
    dB = 4 - 8 * x
    u = B * N
    u_t = B * Nt
    u_x = dB * N + B * Nx
    lam = M * M
    return {
        "R": f - u_t - sgx + lam, "S": sg + u_x, "gam": g - u, "gam_t": g_t - u_t, "gam_x": g_x - u_x,
        "lam": lam, "Rhat_pos": f - g_t - sgx + lam, "Shat_pos": sg + g_x,
    }


def _mp_densities(F):
    gam = F["gam"]
    r = F["R"] ** 2 + F["S"] ** 2
    p = (F["gam_x"] ** 2 + gam**2 + F["gam_t"] ** 2) if gam > 0 else mp.mpf(0)
    c = F["lam"] * max(-gam, 0)
    rhat = (F["Rhat_pos"] ** 2 + F["Shat_pos"] ** 2) if gam > 0 else r
    return {"r": r, "p": p, "c": c, "rhat": rhat}


def _cells(n, seed=0, k=40):
    rng = np.random.default_rng(seed)
    lev = rng.integers(2, 7, size=k)
    h = 2.0 ** -lev
    i = np.floor(rng.random(k) / h)
    j = np.floor(rng.random(k) / h)
    t0, x0 = i * h, j * h
    return t0, t0 + h, x0, x0 + h


def _check_fields_contain(bk, enc, t0, t1, x0, x1, npts=5):
    F = enc.fields(bk, t0, t1, x0, x1)
    R = {k: F[k].range() for k in ("R", "S", "gam", "gam_t", "gam_x", "lam", "Rhat_pos", "Shat_pos")}
    lo = {k: R[k].lower() for k in R}
    hi = {k: R[k].upper() for k in R}
    D = enc.densities(bk, t0, t1, x0, x1)
    s = np.linspace(0, 1, npts)
    for c in range(t0.size):
        for a in s:
            for b in s:
                tt = mp.mpf(float(t0[c])) + (mp.mpf(float(t1[c])) - mp.mpf(float(t0[c]))) * mp.mpf(a)
                xx = mp.mpf(float(x0[c])) + (mp.mpf(float(x1[c])) - mp.mpf(float(x0[c]))) * mp.mpf(b)
                F = _mp_fields(enc, tt, xx)
                for k in lo:
                    assert lo[k][c] <= F[k] <= hi[k][c], (k, c, float(F[k]), lo[k][c], hi[k][c])
                dn = _mp_densities(F)
                for k in dn:
                    assert dn[k] <= D[k][c], (k, c, float(dn[k]), D[k][c])
    return D


# ------------------------------------------------------------------ 1. polynomial model
@pytest.mark.parametrize("bk", BACKENDS)
def test_polynomial_model_upper_sums_converge_from_above(bk):
    tr = _linear_triple(state=(2.0, 0.5, -0.25), flux=(0.1, -0.3, 0.2), mult=(0.5, 0.25, -0.5))
    enc = TripleEncloser(tr, StefanAdapter(StefanProblem()))
    t, x = sp.symbols("t x")
    N = 2 + sp.Rational(1, 2) * t - sp.Rational(1, 4) * x
    sig = sp.Rational(1, 10) - sp.Rational(3, 10) * t + sp.Rational(1, 5) * x
    M = sp.Rational(1, 2) + sp.Rational(1, 4) * t - sp.Rational(1, 2) * x
    B = 4 * x * (1 - x)
    u = B * N
    h = sp.exp(t) - t - 1
    g = -h * (1 - x)
    f = -1 - (sp.exp(t) - 1) * (1 - x)
    R = f - sp.diff(u, t) - sp.diff(sig, x) + M**2
    S = sig + sp.diff(u, x)
    exact_r = sp.integrate(sp.integrate(sp.expand(R**2 + S**2), (x, 0, 1)), (t, 0, 1)) + sp.integrate(u.subs(t, 0) ** 2, (x, 0, 1))
    exact_c = sp.integrate(sp.integrate(sp.expand(M**2 * (u - g)), (x, 0, 1)), (t, 0, 1))
    exact_r, exact_c = float(exact_r), float(exact_c)
    errs = []
    for n in (4, 8, 16):
        part, tp = US.uniform_2d(n, n), US.uniform_1d(4 * n)
        name = "arb" if bk is AB else "numpy"
        us, vol, _ = US.certify_partition(enc, part, tp, tp, backend=name, model_frozen=True, constants_validated=True)
        # u >= g everywhere, so p = 0 exactly; gamma = 0 on x = 1, so boundary
        # cells cannot be sign-resolved and U_p is a small positive upper bound
        assert us.U_r >= exact_r and us.U_c >= exact_c and us.U_p >= 0.0
        errs.append((us.U_r - exact_r, us.U_c - exact_c, us.U_p))
        assert us.status == "validated_upper_bound"
    assert errs[2][0] < errs[1][0] < errs[0][0] and errs[2][1] < errs[1][1] < errs[0][1]
    assert errs[2][2] < errs[1][2] < errs[0][2]
    assert errs[2][0] / exact_r < 0.6 and errs[2][1] / exact_c < 0.6  # O(h) convergence


# ------------------------------------------------------------------ 2. tiny tanh vs mpmath
@pytest.mark.parametrize("bk", BACKENDS)
def test_tiny_tanh_network_contains_50_digit_samples(bk):
    enc = TripleEncloser(_tiny_triple(1), StefanAdapter(StefanProblem()))
    t0, t1, x0, x1 = _cells(0, seed=1, k=12)
    _check_fields_contain(bk, enc, t0, t1, x0, x1)


# ------------------------------------------------------------------ 3. straddling gamma = 0
@pytest.mark.parametrize("bk", BACKENDS)
def test_cells_straddling_gap_sign(bk):
    # N = -1 + t: u = -B(1 - t) ... gamma = g + B (1-t) changes sign inside Q
    tr = _linear_triple(state=(-1.0, 1.0, 0.0), flux=(0.0, 0.0, 0.0), mult=(0.3, 0.0, 0.0))
    enc = TripleEncloser(tr, StefanAdapter(StefanProblem()))
    part = US.uniform_2d(8, 8)
    D = enc.densities(bk, part.t0, part.t1, part.x0, part.x1)
    assert (D["sign"] == 0).sum() > 0 and (D["sign"] == 1).sum() > 0
    idx = np.nonzero(D["sign"] == 0)[0][:6]
    _check_fields_contain(bk, enc, part.t0[idx], part.t1[idx], part.x0[idx], part.x1[idx], npts=7)


# ------------------------------------------------------------------ 4. saturation
@pytest.mark.parametrize("bk", BACKENDS)
def test_saturated_activations(bk):
    tr = _tiny_triple(2, scale=150.0)
    enc = TripleEncloser(tr, StefanAdapter(StefanProblem()))
    t = np.array([0.5])
    z = enc.Ws[0][0] @ np.array([0.5, 0.5]) + enc.Ws[0][1]
    assert np.max(np.abs(z)) > 20
    t0, t1, x0, x1 = _cells(0, seed=4, k=6)
    D = _check_fields_contain(bk, enc, t0, t1, x0, x1, npts=3)
    for k in ("r", "p", "c", "rhat"):
        assert np.all(np.isfinite(D[k]))


# ------------------------------------------------------------------ 5. cancellation
@pytest.mark.parametrize("bk", BACKENDS)
def test_nearly_cancelling_quantities_stay_tight(bk):
    enc = TripleEncloser(_tiny_triple(3, width=8), StefanAdapter(StefanProblem()))
    t0, t1, x0, x1 = _cells(0, seed=5, k=10)
    tc, ht = 0.5 * (t0 + t1), 0.5 * (t1 - t0)
    xc, hx = 0.5 * (x0 + x1), 0.5 * (x1 - x0)
    tt = T.tm_variable(bk, tc, ht, 1, 2)
    xx = T.tm_variable(bk, xc, hx, 2, 2)
    a, _ = enclose_mlp(enc.Wm, [tt, xx], ())
    b, _ = enclose_mlp(enc.Wm, [tt, xx], ())
    d = (a - b).range()
    whole = a.range()
    width_d = d.upper() - d.lower()
    width_a = whole.upper() - whole.lower()
    # the linear parts cancel exactly in the TM; only remainders survive
    assert np.all(width_d <= 2.5 * (a.rem.upper() - a.rem.lower()) + 1e-12)
    assert np.all(d.lower() <= 0) and np.all(d.upper() >= 0)
    assert np.all(width_d < width_a)


# ------------------------------------------------------------------ 6. traces
@pytest.mark.parametrize("bk", BACKENDS)
def test_both_temporal_traces(bk):
    # N = -1: v = -B. v(0) - u0 = -B: int B^2 = 8/15. w(0) = (0 + B)^+ = B: 8/15.
    # w(T) = (g(T) + B)^+ = (1-x)(4x - (e-2))^+.
    tr = _linear_triple(state=(-1.0, 0.0, 0.0), flux=(0.0, 0.0, 0.0), mult=(0.0, 0.0, 0.0))
    enc = TripleEncloser(tr, StefanAdapter(StefanProblem()))
    x = sp.symbols("x")
    xs = (sp.E - 2) / 4
    wT_exact = float(sp.integrate(((1 - x) * (4 * x - (sp.E - 2))) ** 2, (x, xs, 1)))
    prev = None
    name = "arb" if bk is AB else "numpy"
    for n in (16, 64, 256):
        tp = US.uniform_1d(n)
        tr0 = US.evaluate_trace(enc, tp, 0.0, name)
        trT = US.evaluate_trace(enc, tp, 1.0, name)
        L = tp.length()
        r0 = float(np.sum(L * tr0["r0"]))
        w0 = float(np.sum(L * tr0["w2"]))
        wT = float(np.sum(L * trT["w2"]))
        assert r0 >= 8 / 15 and w0 >= 8 / 15 and wT >= wT_exact
        err = (r0 - 8 / 15, w0 - 8 / 15, wT - wT_exact)
        if prev:
            assert all(e < p_ for e, p_ in zip(err, prev))
        prev = err
    assert prev[2] / wT_exact < 0.05


# ------------------------------------------------------------------ 7. interface cells
@pytest.mark.parametrize("bk", BACKENDS)
def test_cells_touching_exact_interface(bk):
    enc = TripleEncloser(_tiny_triple(5), StefanAdapter(StefanProblem()))
    h = 1 / 16
    k = np.arange(0, 16, 3)
    t0 = np.concatenate([k * h, k * h])
    x0 = np.concatenate([k * h, (k + 1) * h])  # on the diagonal / touching it at a corner
    _check_fields_contain(bk, enc, t0, t0 + h, x0, x0 + h, npts=5)


# ------------------------------------------------------------------ 8. status logic
def _small_cert(**kw):
    enc = TripleEncloser(_tiny_triple(6), StefanAdapter(StefanProblem()))
    part, tp = US.uniform_2d(4, 4), US.uniform_1d(8)
    vol = US.evaluate_volume(enc, part, "numpy")
    tr0 = US.evaluate_trace(enc, tp, 0.0, "numpy")
    trT = US.evaluate_trace(enc, tp, 1.0, "numpy")
    return enc, part, tp, vol, tr0, trT


def test_status_logic():
    enc, part, tp, vol, tr0, trT = _small_cert()
    bi = US.backend_info("numpy", 64)
    ok = US.reduce_upper_sums(part, vol, tp, tr0, tp, trT, backend_info=bi, model_frozen=True, constants_validated=True)
    assert ok.status == "validated_upper_bound"
    bad = {k: v.copy() for k, v in vol.items()}
    bad["r"][3] = np.nan
    r = US.reduce_upper_sums(part, bad, tp, tr0, tp, trT, backend_info=bi, model_frozen=True, constants_validated=True)
    assert r.status != "validated_upper_bound"
    # missing trace cell
    tp_missing = US.Partition1D(tp.x0[1:], tp.x1[1:])
    tr0m = {k: v[1:] for k, v in tr0.items()}
    r = US.reduce_upper_sums(part, vol, tp_missing, tr0m, tp, trT, backend_info=bi, model_frozen=True, constants_validated=True)
    assert r.status != "validated_upper_bound" and any("trace0" in s for s in r.status_reasons)
    # missing volume cell
    sub = part.subset(np.arange(1, part.n))
    volm = {k: v[1:] for k, v in vol.items()}
    r = US.reduce_upper_sums(sub, volm, tp, tr0, tp, trT, backend_info=bi, model_frozen=True, constants_validated=True)
    assert r.status != "validated_upper_bound"
    # unvalidated constant / backend / unfrozen model
    for kw in ({"constants_validated": False, "model_frozen": True}, {"constants_validated": True, "model_frozen": False}):
        r = US.reduce_upper_sums(part, vol, tp, tr0, tp, trT, backend_info=bi, **kw)
        assert r.status != "validated_upper_bound"
    r = US.reduce_upper_sums(part, vol, tp, tr0, tp, trT, backend_info={"backend": "float64"}, model_frozen=True, constants_validated=True)
    assert r.status != "validated_upper_bound"


def test_adapter_rejects_inexact_constants():
    class P(StefanProblem):
        Lambda = 2.0
    with pytest.raises(AssertionError):
        StefanAdapter(P())


# ------------------------------------------------------------------ backend agreement / coverage
def test_fast_backend_contains_or_matches_arb_on_random_cells():
    enc = TripleEncloser(_tiny_triple(7, width=8, depth=3), StefanAdapter(StefanProblem()))
    t0, t1, x0, x1 = _cells(0, seed=8, k=60)
    Fn = enc.fields(NB, t0, t1, x0, x1)
    Fa = enc.fields(AB, t0, t1, x0, x1)
    for k in ("R", "S", "gam", "gam_t", "gam_x", "lam"):
        rn, ra = Fn[k].range(), Fa[k].range()
        width = rn.upper() - rn.lower()
        contain = (rn.lower() <= ra.lower()) & (rn.upper() >= ra.upper())
        close = (np.abs(rn.lower() - ra.lower()) <= width) & (np.abs(rn.upper() - ra.upper()) <= width)
        assert np.all(contain | close), k


def test_coverage_checks_and_refinement():
    part = US.uniform_2d(4, 4)
    mark = np.zeros(part.n, bool)
    mark[[0, 5]] = True
    ref = US.refine_2d(part, mark)
    assert US.verify_coverage_2d(ref)[0]
    assert not US.verify_coverage_2d(ref.subset(np.arange(1, ref.n)))[0]
    dup = US.Partition2D(np.r_[ref.t0, ref.t0[:1]], np.r_[ref.t1, ref.t1[:1]], np.r_[ref.x0, ref.x0[:1]], np.r_[ref.x1, ref.x1[:1]])
    assert not US.verify_coverage_2d(dup)[0]
    assert ref.hash() != part.hash()


def test_numpy_tanh_table_encloses_mpmath():
    rng = np.random.default_rng(0)
    z = np.concatenate([rng.normal(0, 3, 400), rng.uniform(-25, 25, 100), [0.0, 20.0, -20.0, 1e-300, 19.99999]])
    I = NB.from_bounds(z, z)
    th = NB.tanh(I)
    s2 = NB.sech2(I)
    for i, v in enumerate(z):
        tv = mp.tanh(mp.mpf(float(v)))
        sv = 1 / mp.cosh(mp.mpf(float(v))) ** 2
        assert th.lo[i] <= tv <= th.hi[i]
        assert s2.lo[i] <= sv <= s2.hi[i]
        if abs(v) < 19:
            assert th.hi[i] - th.lo[i] < 1e-13
