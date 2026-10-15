"""v20 rising1d: problem facts (exact / arb) and the read-only training monitor."""

from fractions import Fraction

import torch
from flint import arb

from certified_obstacle_nn.problems.rising1d import Rising1DProblem


def _a(q):
    q = Fraction(q)
    return arb(q.numerator) / q.denominator


def _ball(lo, hi):
    return _a(lo).union(_a(hi))


def _g(t, x):
    return -arb(3) / 5 + arb(3) / 5 * t - 2 * (x - arb(1) / 2) ** 2


def test_boundary_and_initial_compatibility():
    # exact rationals: g is affine in t with g_t = 3/5 > 0, so max_t g(t, xb) = g(1, xb) = -1/2 at xb in {0, 1}
    gq = lambda t, x: Fraction(-3, 5) + Fraction(3, 5) * t - 2 * (x - Fraction(1, 2)) ** 2  # noqa: E731
    for xb in (0, 1):
        assert gq(1, xb) == Fraction(-1, 2) and gq(0, xb) == Fraction(-11, 10)
    # g(0, x) = -3/5 - 2 (x - 1/2)^2 <= -3/5 < 0 = u0 on [0,1] (subdivide for a sharp ball bound)
    for k in range(64):
        assert _g(arb(0), _ball(Fraction(k, 64), Fraction(k + 1, 64))) < 0


def test_psi_positive_and_contact_multiplier():
    # psi = g_t - g_xx - f = 3/5 + 4 + 8 sin(pi x) >= 23/5 > 0 on [0,1]; values of the contact multiplier
    for k in range(64):
        X = _ball(Fraction(k, 64), Fraction(k + 1, 64))
        psi = arb(3) / 5 + 4 + 8 * X.sin_pi()
        assert psi > 4
    lam = lambda x: arb(23) / 5 + 8 * _a(x).sin_pi()  # noqa: E731
    assert 11.4 < lam(Fraction(1, 3)) < 11.6 and 12.5 < lam(Fraction(1, 2)) < 12.7


def test_problem_torch_data_matches_definition():
    p = Rising1DProblem()
    t = torch.tensor([0.0, 0.3, 1.0])
    x = torch.tensor([0.0, 0.25, 0.5])
    assert torch.allclose(p.g(t, x), torch.tensor([-1.1, -0.42 - 0.125, 0.0]))
    assert torch.allclose(p.f(t, x), -8 * torch.sin(torch.pi * x))
    assert p.g_inf == 1.1 and p.g_x_inf == 2.0 and p.g_xx_inf == 4.0 and p.kappa == 1.0


def test_monitor_is_read_only():
    from certified_obstacle_nn.experiments.v20.train_rising1d import train_one

    torch.set_num_threads(1)
    a, ha, oa, rows, _ = train_one(0, n_adam=12, n_lbfgs=2, monitor_on=True)
    b, hb, ob, _, _ = train_one(0, n_adam=12, n_lbfgs=2, monitor_on=False)
    assert rows and rows[0]["step"] == 0 and rows[-1]["phase"] == "lbfgs_end"
    assert oa == ob and ha.equals(hb)
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)


# ---------------------------------------------------------------- Task 2 primitives
import numpy as np  # noqa: E402

from certified_obstacle_nn.certification.enclose import intervals_np as NB  # noqa: E402
from certified_obstacle_nn.certification.enclose.engine import lower_terms, sq_lower  # noqa: E402


def _sq_lower_exact(a, b):
    A, B = Fraction(a), Fraction(b)
    if A <= 0 <= B:
        return Fraction(0)
    return min(A * A, B * B)


def test_sq_lower_adversarial_against_fraction():
    tiny = 5e-324
    cases = [(-1.0, 2.0), (-3.0, -0.0), (0.0, 0.0), (-0.0, 1e-300), (0.0, 5.0), (-5.0, 0.0), (1e-200, 2e-200),
             (tiny, 1.0), (-1.0, -tiny), (0.1, 0.3), (-0.3, -0.1), (1.0 + 2**-52, 3.0), (-7.5, -7.5), (3.0, 3.0),
             (-1e154, -1e153), (1e-160, 1e-150), (0.7, 0.7000000000000001), (-2.0, -1e-320)]
    rng = np.random.default_rng(20261014)
    for _ in range(2000):
        a, b = np.sort(rng.normal(0, 10, 2) * 10.0 ** rng.integers(-30, 30, 2))
        cases.append((float(a), float(b)))
    lo = np.array([c[0] for c in cases])
    hi = np.array([c[1] for c in cases])
    got = sq_lower(NB.from_bounds(lo, hi))
    for (a, b), g in zip(cases, got):
        ex = _sq_lower_exact(a, b)
        assert Fraction(float(g)) <= ex, (a, b, g)
        assert g >= 0
        if ex == 0:
            assert g == 0.0
        else:  # at most a few ulps below the exact square
            assert float(ex) - float(g) <= 4 * np.spacing(float(ex)) + 5e-324, (a, b, g)


def test_sq_lower_nan_propagates():
    v = sq_lower(NB.from_bounds(np.array([np.nan, 1.0]), np.array([1.0, np.nan])))
    assert np.isnan(v).all()


def test_lower_terms_branches():
    I = lambda lo, hi: NB.from_bounds(np.array(lo), np.array(hi))  # noqa: E731
    g_lo, g_hi = np.array([-2.0, 0.5, -0.1]), np.array([-1.0, 1.0, 0.2])  # v-branch, g-branch, unresolved
    out = lower_terms(g_lo, g_hi, I([1.0, -1.0, 2.0], [2.0, 1.0, 3.0]), I([2.0, 2.0, 3.0], [3.0, 3.0, 4.0]),
                      I([4.0, 4.0, 0.5], [5.0, 5.0, 1.0]))
    assert out["tau2_lo"][0] <= 1.0 and out["tau2_lo"][1] == 0.0
    assert 3.99 < out["gvh2_lo"][0] <= 4.0 and 15.9 < out["gvh2_lo"][1] <= 16.0 and 0.24 < out["gvh2_lo"][2] <= 0.25


def test_sinpi_enclosures_contain_samples():
    rng = np.random.default_rng(1)
    lo = np.sort(rng.random((2, 500)), axis=0)
    X = NB.from_bounds(lo[0], lo[1])
    s, c = NB.sinpi(X), NB.cospi(X)
    for k in range(500):
        z = np.linspace(lo[0, k], lo[1, k], 17)
        assert np.all(s.lo[k] <= np.sin(np.pi * z) + 1e-15) and np.all(np.sin(np.pi * z) - 1e-15 <= s.hi[k])
        assert np.all(c.lo[k] <= np.cos(np.pi * z) + 1e-15) and np.all(np.cos(np.pi * z) - 1e-15 <= c.hi[k])
    p = NB.pi_const(())
    assert p.lo < np.pi + 1e-15 and Fraction(float(p.lo)) < Fraction(355, 113) and p.hi > 3.14159265358979


def test_rising1d_adapter_data_contain_exact_values():
    from certified_obstacle_nn.certification.enclose import tm as T
    from certified_obstacle_nn.certification.enclose.engine import Rising1DAdapter

    ad = Rising1DAdapter(Rising1DProblem())
    rng = np.random.default_rng(2)
    n = 300
    k = rng.integers(0, 64, (2, n))
    t0, x0 = k[0] / 64, k[1] / 64
    h = 1 / 64
    for bkname in ("np", "arb"):
        if bkname == "np":
            bk = NB
        else:
            from certified_obstacle_nn.certification.enclose import intervals_arb as bk
            bk.set_precision(64)
        m = 40 if bkname == "arb" else n
        t = T.tm_variable(bk, t0[:m] + h / 2, np.full(m, h / 2), 1, 2)
        x = T.tm_variable(bk, x0[:m] + h / 2, np.full(m, h / 2), 2, 2)
        d = ad.data(t, x)
        rngs = {kk: (v.range().lower(), v.range().upper()) for kk, v in d.items()}
        for s_ in np.linspace(0, 1, 5):
            for r_ in np.linspace(0, 1, 5):
                tt, xx = t0[:m] + s_ * h, x0[:m] + r_ * h
                ex = {"f": -8 * np.sin(np.pi * xx), "g": -0.6 + 0.6 * tt - 2 * (xx - 0.5) ** 2,
                      "g_t": np.full(m, 0.6), "g_x": -4 * (xx - 0.5)}
                for kk, v in ex.items():
                    lo_, hi_ = rngs[kk]
                    assert np.all(lo_ <= v + 1e-13) and np.all(v - 1e-13 <= hi_), (bkname, kk)
