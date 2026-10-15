"""v17: closed form of the ellipse reference norm N = ||grad_x u||_{L2(Q)} (= ||sigma||, sigma = -grad u)
for the expanding-ellipse benchmark (problems/ellipse.py, kappa0 = 1/100), derived with sympy in exact rationals.

    python3 -m certified_obstacle_nn.experiments.v17.ellipse_norm

u = u_b - k0 (zeta_-)^2  =>  grad u = grad u_b - 2 k0 zeta grad zeta 1{zeta < 0}, hence
  |grad u|^2 = |grad u_b|^2 + 1_E(t) [ -4 k0 zeta (grad u_b . grad zeta) + 4 k0^2 zeta^2 |grad zeta|^2 ],
E(t) = {zeta(t, .) < 0} = ellipse with centre (43/100, 57/100), semi-axes a(t) = 3/25 + t/10, d(t) = 2/25 + 7t/100
(inside (0,1)^2 for t in [0,1]: x1 in [0.21, 0.65], x2 in [0.42, 0.72] at t = 1). The first term is a polynomial on
the unit square; the second is integrated over E(t) in the affine-polar coordinates x1 = C1 + a rho cos phi,
x2 = C2 + d rho sin phi (Jacobian a d rho, zeta = rho^2 - 1), where the integrand is a polynomial in rho, cos, sin;
the remaining t-integral of a rational function of a(t), d(t) is done by sympy exactly (logs appear).
Validation: two independent Gauss-Legendre quadratures (mpmath, 40 digits): (A) Gauss in t x Gauss in rho x
trapezoid in phi on E plus tensor Gauss on the square; (B) Gauss in t x Cartesian Gauss on E with x1 = C1 + a sin s
(chord parametrisation) plus tensor Gauss on the square; and the stored numerical N of the v14 runs.
Evaluation of the closed form in arb (192 bits); the LOWER endpoint is the validated denominator.
Writes results/ellipse_norm/<timestamp>/ (derivation output, srepr, LaTeX, validation, arb bracket).
"""

from __future__ import annotations

import json
import time

import mpmath as mp
import sympy as sp
from flint import arb, ctx

from certified_obstacle_nn.experiments import results_io

R = sp.Rational
k0 = R(1, 100)
C1, C2 = R(43, 100), R(57, 100)
t, x1, x2, rho, phi = sp.symbols("t x1 x2 rho phi", real=True)
a = R(3, 25) + t / 10
d = R(2, 25) + R(7, 100) * t


def u_b_expr(X1, X2, T):
    B = 16 * X1 * (1 - X1) * X2 * (1 - X2)
    return B * (R(1, 10) + T / 20) * (1 + X1 / 5 - X2 / 10)


def derive():
    ub = u_b_expr(x1, x2, t)
    gub = [sp.diff(ub, x1), sp.diff(ub, x2)]
    # square part: polynomial
    I1 = sp.integrate(sp.expand(gub[0] ** 2 + gub[1] ** 2), (x1, 0, 1), (x2, 0, 1))
    I1 = sp.expand(I1)
    # ellipse part in affine-polar coordinates
    zeta = (x1 - C1) ** 2 / a**2 + (x2 - C2) ** 2 / d**2 - 1
    gz = [sp.diff(zeta, x1), sp.diff(zeta, x2)]
    dens = -4 * k0 * zeta * (gub[0] * gz[0] + gub[1] * gz[1]) + 4 * k0**2 * zeta**2 * (gz[0] ** 2 + gz[1] ** 2)
    sub = {x1: C1 + a * rho * sp.cos(phi), x2: C2 + d * rho * sp.sin(phi)}
    integrand = sp.expand(dens.subs(sub) * a * d * rho)
    inner = sp.integrate(integrand, (phi, 0, 2 * sp.pi))
    I2 = sp.integrate(sp.expand(inner), (rho, 0, 1))
    I2 = sp.simplify(I2)
    N2 = sp.integrate(I1, (t, 0, 1)) + sp.integrate(sp.apart(sp.together(I2), t), (t, 0, 1))
    N2 = sp.expand_log(sp.simplify(N2), force=True)
    return I1, I2, N2


def to_arb(e):
    """Exact sympy expression -> arb ball (ctx precision)."""
    if e.is_Integer:
        return arb(int(e))
    if e.is_Rational:
        return arb(int(e.p)) / int(e.q)
    if e is sp.pi:
        return arb.pi()
    if e is sp.E:
        return arb(1).exp()
    if e.is_Add:
        s = arb(0)
        for x in e.args:
            s += to_arb(x)
        return s
    if e.is_Mul:
        s = arb(1)
        for x in e.args:
            s *= to_arb(x)
        return s
    if e.is_Pow:
        b, ex = e.args
        if ex.is_Integer:
            return to_arb(b) ** int(ex)
        if ex == R(1, 2):
            return to_arb(b).sqrt()
        return (to_arb(ex) * to_arb(b).log()).exp()
    if isinstance(e, sp.log):
        return to_arb(e.args[0]).log()
    if isinstance(e, sp.atan):
        return to_arb(e.args[0]).atan()
    raise NotImplementedError(f"to_arb: {type(e)} {e}")


# ------------------------------------------------------------------ independent quadratures (mpmath)
def _ub_grad_mp(X1, X2, T):
    c = mp.mpf(1) / 10 + T / 20
    L = 1 + X1 / 5 - X2 / 10
    p, q = X1 * (1 - X1), X2 * (1 - X2)
    B = 16 * p * q
    Bx1, Bx2 = 16 * (1 - 2 * X1) * q, 16 * p * (1 - 2 * X2)
    return c * (Bx1 * L + B / 5), c * (Bx2 * L - B / 10)


def _ell_dens_mp(X1, X2, T):
    A = mp.mpf(3) / 25 + T / 10
    Dd = mp.mpf(2) / 25 + mp.mpf(7) * T / 100
    c1, c2 = mp.mpf(43) / 100, mp.mpf(57) / 100
    K = mp.mpf(1) / 100
    z = (X1 - c1) ** 2 / A**2 + (X2 - c2) ** 2 / Dd**2 - 1
    z1, z2 = 2 * (X1 - c1) / A**2, 2 * (X2 - c2) / Dd**2
    u1, u2 = _ub_grad_mp(X1, X2, T)
    return -4 * K * z * (u1 * z1 + u2 * z2) + 4 * K**2 * z**2 * (z1**2 + z2**2), A, Dd, c1, c2


def _gl(n):
    """Gauss-Legendre nodes/weights on [0, 1] at the current mpmath precision (mpmath's own GaussLegendre rule,
    3 * 2^(m-1) nodes for degree m; n must be of that form)."""
    m = {3: 1, 6: 2, 12: 3, 24: 4, 48: 5, 96: 6}[n]
    nodes = mp.calculus.quadrature.GaussLegendre(mp.mp).calc_nodes(m, mp.mp.prec)
    return [(x + 1) / 2 for x, _ in nodes], [w / 2 for _, w in nodes]


def _quad(xt, wt, nr, nphi, nsq, polar):
    xr, wr = _gl(nr)
    xs, ws = _gl(nsq)
    total = mp.mpf(0)
    for T, WT in zip(xt, wt):
        sq = mp.mpf(0)
        for X1, W1 in zip(xs, ws):
            for X2, W2 in zip(xs, ws):
                u1, u2 = _ub_grad_mp(X1, X2, T)
                sq += W1 * W2 * (u1**2 + u2**2)
        el = mp.mpf(0)
        A = mp.mpf(3) / 25 + T / 10
        Dd = mp.mpf(2) / 25 + mp.mpf(7) * T / 100
        c1, c2 = mp.mpf(43) / 100, mp.mpf(57) / 100
        if polar:
            for j in range(nphi):
                ph = 2 * mp.pi * j / nphi
                for r_, wr_ in zip(xr, wr):
                    X1 = c1 + A * r_ * mp.cos(ph)
                    X2 = c2 + Dd * r_ * mp.sin(ph)
                    el += (2 * mp.pi / nphi) * wr_ * A * Dd * r_ * _ell_dens_mp(X1, X2, T)[0]
        else:
            # x1 = c1 + A sin s, s in (-pi/2, pi/2); chord x2 in c2 +- Dd cos s; Jacobian A cos s * 2 Dd cos s
            for s_, ws_ in zip(xr, wr):
                s = mp.pi * (s_ - mp.mpf(1) / 2)
                X1 = c1 + A * mp.sin(s)
                h = Dd * mp.cos(s)
                for y_, wy_ in zip(xr, wr):
                    X2 = c2 - h + 2 * h * y_
                    el += mp.pi * ws_ * wy_ * A * mp.cos(s) * 2 * h * _ell_dens_mp(X1, X2, T)[0]
        total += WT * (sq + el)
    return total


def _stored_numerical_norm():
    """norm_grad_u of the v16 ellipse input records (v14 diagnostic quadrature), read from the data, not typed."""
    import csv

    from certified_obstacle_nn.experiments.v17.common import V16_REF

    vals = {r["norm_grad_u"] for r in csv.DictReader((V16_REF / "data/certificates_input.csv").open())
            if r["problem"] == "ellipse"}
    return {"values": sorted(vals), "source": "docs/v16_reference/v16/data/certificates_input.csv norm_grad_u"}


def main():
    t0 = time.time()
    mp.mp.dps = 40
    I1, I2, N2 = derive()
    ctx.prec = 192
    N2_arb = to_arb(N2)
    N_arb = N2_arb.sqrt()
    from certified_obstacle_nn.certification.reliability import down, up

    xt, wt = _gl(24)
    qA = _quad(xt, wt, 24, 48, 6, polar=True)
    xt2, wt2 = _gl(48)
    qB = _quad(xt2, wt2, 24, 0, 6, polar=False)
    N2f = float(N2_arb.mid())
    N2mp = mp.mpf(str(sp.N(N2, 45)))
    out = {
        "N2_closed_form_latex": sp.latex(N2), "N2_srepr": sp.srepr(N2), "N2_str": str(N2),
        "I1_t": str(I1), "I2_t": str(I2),
        "N2_arb": str(N2_arb), "N_lower": down(N_arb), "N_upper": up(N_arb), "N2_lower": down(N2_arb),
        "precision_bits": 192,
        "quad_A_N2": mp.nstr(qA, 25), "quad_B_N2": mp.nstr(qB, 25),
        "rel_diff_A": float(abs(qA - N2mp) / N2mp), "rel_diff_B": float(abs(qB - N2mp) / N2mp),
        "N2_closed_form_45_digits": mp.nstr(N2mp, 40),
        "quad_A": "Gauss-Legendre 24 in t x Gauss 24 in rho x trapezoid 48 in phi on E(t) + tensor Gauss 6x6 square",
        "quad_B": "Gauss-Legendre 48 in t x chord-parametrised Gauss 24x24 on E(t) (x1 = C1 + a sin s) + Gauss 6x6",
        "stored_numerical_N_v14": _stored_numerical_norm(),
        "status": "validated_upper_bound (arb evaluation of an exact closed form)", "seconds": None,
    }
    out["agree_14_digits"] = out["rel_diff_A"] < 1e-14 and out["rel_diff_B"] < 1e-14
    out["seconds"] = time.time() - t0
    d = results_io.make_run_dir("ellipse_norm")
    (d / "result.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("N2_str", "N_lower", "N_upper", "quad_A_N2", "quad_B_N2", "rel_diff_A",
                                           "rel_diff_B", "agree_14_digits", "seconds")}, indent=1))
    print("wrote", d)


if __name__ == "__main__":
    main()
