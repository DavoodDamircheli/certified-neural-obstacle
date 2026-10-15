"""v14 reliability formulas (roadmap sections 2-3, v14 prompt A3 and Stage 2-3).

Every bound here is a function of the three component upper sums

    U_r >= r^2   = (Lambda/alpha)||R||^2 + ||S||^2 + ||v(0)-u0||^2
    U_p >= p^2   = ||A^{1/2} grad w||^2 + kappa||w||^2 + (Lambda/alpha)||w_t||^2 + ||w(0)||^2 + ||w(T)||^2
    U_c >= c_*^2 = int_Q mu (v-g)_+

and of the problem constants (alpha, Lambda, ||A||_inf, ||b||_inf, ||c||_inf).
All final operations (sums, constant multiples, division, sqrt) are carried
out in arb ball arithmetic on the exact binary values of the inputs and the
UPPER endpoint is returned (rounded upward to float64), so the arithmetic of
this module never decreases a bound. Whether a result may carry the status
``validated_upper_bound`` depends on how the U's themselves were obtained;
that status is passed in and propagated, never upgraded here.

Constants:
    C_eq  = max{1/alpha + 4(||b||^2 + Lambda||c||^2)/alpha, ||A||_inf, 1, 2 alpha/Lambda}
    C_rel = 36 (diffusion only: b = c = 0 exactly), 144 (general coefficients)
    216   = historical comparison convention only
"""

from __future__ import annotations

import math
from fractions import Fraction

from flint import arb, ctx

HISTORICAL_C_REL = 216
STATUSES = ("diagnostic_float64", "unvalidated_enclosure", "validated_upper_bound")
# v20: status vocabulary of LOWER bounds (separate ordering; a lower bound is never mixed into STATUSES)
LOWER_STATUSES = ("diagnostic_float64", "unvalidated_enclosure", "validated_lower_bound")

ctx.prec = max(ctx.prec, 128)


# ---------------------------------------------------------------- arb helpers
def a(x) -> arb:
    """Exact arb from a float/int/Fraction (floats are dyadic rationals)."""
    if isinstance(x, arb):
        return x
    if isinstance(x, Fraction):
        return arb(x.numerator) / arb(x.denominator)
    if isinstance(x, int):
        return arb(x)
    xf = float(x)
    if not math.isfinite(xf):
        return arb("nan") if math.isnan(xf) else arb.pos_inf()
    return arb(xf)  # exact: arb(float) does not round


def up(x: arb) -> float:
    """float64 >= every point of the ball x (inf if x is unbounded/NaN)."""
    if not x.is_finite():
        return math.inf
    u = x.upper()  # an arb ball itself (not necessarily exact)
    f = float(u.mid())
    while not (arb(f) >= u):
        f = math.nextafter(f, math.inf)
    return f


def down(x: arb) -> float:
    if not x.is_finite():
        return -math.inf
    lo = x.lower()
    f = float(lo.mid())
    while not (arb(f) <= lo):
        f = math.nextafter(f, -math.inf)
    return f


# ---------------------------------------------------------------- constants
def is_diffusion_only(problem) -> bool:
    return float(problem.b) == 0.0 and float(problem.c) == 0.0


def C_rel(problem) -> int:
    """36 for diffusion only (b = c = 0 exactly), 144 otherwise (v14 A3)."""
    return 36 if is_diffusion_only(problem) else 144


def p_weight(problem) -> int:
    """Weight of U_p in the component-weighted bound: 36 (heat) or 144."""
    return 36 if is_diffusion_only(problem) else 144


def C_eq_arb(alpha, Lambda, A_inf, b_inf, c_inf) -> arb:
    al, La = a(alpha), a(Lambda)
    t1 = 1 / al + 4 * (a(b_inf) ** 2 + La * a(c_inf) ** 2) / al
    vals = [t1, a(A_inf), arb(1), 2 * al / La]
    m = vals[0]
    for v in vals[1:]:
        m = m.max(v)
    return m


def C_eq(problem) -> float:
    """C_eq computed from the formula (upper endpoint), never looked up."""
    return up(C_eq_arb(problem.alpha, problem.Lambda, problem.A_inf, abs(problem.b), abs(problem.c)))


def kappa(problem) -> float:
    al, La = a(problem.alpha), a(problem.Lambda)
    return up(al / La + (a(problem.b) ** 2 + La * a(problem.c) ** 2) / al)


def problem_constants_record(problem) -> dict:
    """Everything config.json must carry (v14 A8)."""
    return {
        "alpha": float(problem.alpha),
        "Lambda": float(problem.Lambda),
        "A_norm_upper": float(problem.A_inf),
        "b_norm_upper": abs(float(problem.b)),
        "c_norm_upper": abs(float(problem.c)),
        "kappa": kappa(problem),
        "C_eq": C_eq(problem),
        "C_rel": C_rel(problem),
        "diffusion_only": is_diffusion_only(problem),
        "reliability_formula": (
            "weighted B = 18 U_r + 36 U_p + 12 U_c (diffusion only); uniform C_rel = 36"
            if is_diffusion_only(problem)
            else "weighted B = 18 U_r + 144 U_p + 12 U_c (general); uniform C_rel = 144"
        ),
        "Lambda_justification": getattr(problem, "Lambda_justification", "UNRECORDED"),
    }


# ---------------------------------------------------------------- delta family
def delta_interval(diffusion_only: bool) -> tuple[Fraction, Fraction]:
    return (Fraction(0), Fraction(1, 3)) if diffusion_only else (Fraction(0), Fraction(1, 6))


def B_delta_arb(Ur: arb, Up: arb, Uc: arb, delta: Fraction, diffusion_only: bool) -> arb:
    lo, hi = delta_interval(diffusion_only)
    if not (lo < delta < hi):
        return arb.pos_inf()
    d = a(delta)
    if diffusion_only:
        return (9 * Ur + 6 * Uc + 3 * Up / d) / (1 - 3 * d)
    return (9 * Ur + 6 * Uc + 6 * Up / d) / (1 - 6 * d)


def delta_star(Ur: float, Up: float, Uc: float, diffusion_only: bool) -> float:
    """Closed-form minimiser of B_delta (float; only a CHOICE of delta)."""
    aa = 9 * Ur + 6 * Uc
    if not all(math.isfinite(v) for v in (Ur, Up, Uc)):
        return math.nan
    if diffusion_only:
        c = 3 * Up
        if aa <= 0:
            return 1.0 / 3.0 * (1 - 1e-9)
        return (-3 * c + math.sqrt(9 * c * c + 3 * aa * c)) / (3 * aa)
    c = 6 * Up
    if aa <= 0:
        return 1.0 / 6.0 * (1 - 1e-9)
    return (-6 * c + math.sqrt(36 * c * c + 6 * aa * c)) / (6 * aa)


def rational_inside(x: float, diffusion_only: bool, max_den: int = 10**6) -> Fraction:
    lo, hi = delta_interval(diffusion_only)
    if not math.isfinite(x):
        return (lo + hi) / 2
    q = Fraction(x).limit_denominator(max_den)
    eps = Fraction(1, max_den)
    if q <= lo:
        q = lo + eps
    if q >= hi:
        q = hi - eps
    return q


# ---------------------------------------------------------------- all bounds
def all_bounds(problem, U_r: float, U_p: float, U_c: float, *, status: str,
               U_r_hat: float | None = None, norm_grad_u: float | None = None,
               norm_sigma: float | None = None) -> dict:
    """Every bound variant of the v14 prompt Stage 2-3 from one set of U's.

    Returns floats that are upper endpoints of arb evaluations. `status`
    labels the U's (and is inherited by every derived bound). The historical
    216 uniform bound is labelled as a comparison convention.
    """
    if status not in STATUSES:
        raise ValueError(status)
    for name, v in (("U_r", U_r), ("U_p", U_p), ("U_c", U_c)):
        if v is None or not math.isfinite(v) or v < 0:
            status = "diagnostic_float64" if status == "validated_upper_bound" else status
            status_reason = f"{name} invalid ({v})"
            break
    else:
        status_reason = ""
    heat = is_diffusion_only(problem)
    Ur, Up, Uc = a(U_r), a(U_p), a(U_c)
    Ceq = C_eq_arb(problem.alpha, problem.Lambda, problem.A_inf, abs(problem.b), abs(problem.c))
    total = Ur + Up + Uc
    crel = C_rel(problem)
    B = 18 * Ur + p_weight(problem) * Up + 12 * Uc

    out = {
        "status": status,
        "status_reason": status_reason,
        "diffusion_only": heat,
        "C_eq": up(Ceq),
        "C_rel": crel,
        "U_r": U_r, "U_p": U_p, "U_c": U_c,
        "U_total": up(total),
        "E_U_uniform_216_comparison_convention": up((Ceq * HISTORICAL_C_REL * total).sqrt()),
        "E_U_uniform": up((Ceq * crel * total).sqrt()),
        "B_weighted": up(B),
        "E_U_weighted": up((Ceq * B).sqrt()),
        "grad_bound": up((B / a(problem.alpha)).sqrt()),
        "flux_bound": up((a(problem.A_inf) * B).sqrt()),
    }
    # delta families at fixed deltas and at the rounded closed-form minimiser
    for label, dl in (("1_12", Fraction(1, 12)), ("1_6", Fraction(1, 6))):
        Bd = B_delta_arb(Ur, Up, Uc, dl, heat)
        out[f"B_delta_{label}"] = up(Bd)
        out[f"E_U_delta_{label}"] = up((Ceq * Bd).sqrt()) if Bd.is_finite() else math.inf
    ds = delta_star(U_r, U_p, U_c, heat)
    dq = rational_inside(ds, heat)
    Bds = B_delta_arb(Ur, Up, Uc, dq, heat)
    out["delta_star_float"] = ds
    out["delta_star_rational"] = f"{dq.numerator}/{dq.denominator}"
    out["B_delta_star"] = up(Bds)
    out["E_U_delta_star"] = up((Ceq * Bds).sqrt())
    out["E_U_best"] = min(out["E_U_weighted"], out["E_U_delta_star"], out["E_U_delta_1_12"],
                          out["E_U_delta_1_6"], out["E_U_uniform"])
    Bbest = min(out["B_weighted"], out["B_delta_star"], out["B_delta_1_12"], out["B_delta_1_6"])
    out["grad_bound_best"] = up((a(Bbest) / a(problem.alpha)).sqrt())
    out["flux_bound_best"] = up((a(problem.A_inf) * a(Bbest)).sqrt())
    if norm_grad_u:
        out["grad_bound_rel"] = out["grad_bound"] / norm_grad_u
        out["grad_bound_best_rel"] = out["grad_bound_best"] / norm_grad_u
    if norm_sigma:
        out["flux_bound_rel"] = out["flux_bound"] / norm_sigma
        out["flux_bound_best_rel"] = out["flux_bound_best"] / norm_sigma

    # projection (Stage 3)
    if U_r_hat is not None:
        Urh = a(U_r_hat)
        out["U_r_hat"] = U_r_hat
        out["E_U_projected"] = up((6 * Ceq * (Urh + Uc)).sqrt())
        if heat:
            out["E_U_original_via_projection"] = up(Ceq.sqrt() * ((6 * (Urh + Uc)).sqrt() + Up.sqrt()))
        else:
            out["E_U_original_via_projection"] = "NOT_IMPLEMENTED"
    return out


# ---------------------------------------------------------------- v17: direct certificates (roadmap R3 section 1)
# Manuscript v16 (docs/v16_reference/v16/main.tex, read-only): Theorem 4.10 (thm:direct, eq:direct),
# Corollary 4.11 (cor:direct-weak, eq:direct-weak = printed eq. (43)), eq:direct-finite, eq:direct-nosplit,
# printed eq. (65) (E_dir = sqrt(G_dir^2 + U_r_hat)). All arithmetic in 192-bit arb on the exact binary values of
# the inputs; upper endpoints rounded upward to float64 (lower endpoints downward for denominators).
DIRECT_PREC = 192
THETA_SCAN = tuple(Fraction(k, 32) for k in range(1, 32))


class _prec:
    def __init__(self, bits):
        self.bits = bits

    def __enter__(self):
        self.old = ctx.prec
        ctx.prec = self.bits

    def __exit__(self, *exc):
        ctx.prec = self.old


def cF2_upper(d: int) -> dict:
    """Upward-rounded c_F^2 = 1/(d pi^2) on the unit box (0,1)^d (exact Friedrichs constant of the
    Dirichlet Laplacian), from a 192-bit arb ball; returns the float, the expression and the bracket."""
    with _prec(DIRECT_PREC):
        v = arb(1) / (d * arb.pi() ** 2)
        return {"cF2_up": up(v), "expression": f"1/({d} pi^2)", "bracket": [down(v), up(v)],
                "precision_bits": DIRECT_PREC}


REFERENCE_NORMS = {
    # N = ||grad_x u||_{L2(Q)} = ||sigma||_{L2(Q)}; closed forms of v16 Appendix C / roadmap R3 section 1.
    "stefan": ("23/12 - e^2/4", lambda: arb(23) / 12 - arb(2).exp() / 4),
    "disk2d": ("1408/135 + 128 e^2/45 - 512 e/45 + 17878909 pi/80640000 - 833879 e pi/9600000",
               lambda: (arb(1408) / 135 + 128 * arb(2).exp() / 45 - 512 * arb(1).exp() / 45
                        + arb(17878909) * arb.pi() / 80640000 - arb(833879) * arb(1).exp() * arb.pi() / 9600000)),
    # v17: derived with sympy (experiments/v17/ellipse_norm.py; results/ellipse_norm/), validated against two
    # independent Gauss-Legendre quadratures to >= 26 digits.
    "ellipse": ("2356/23625 - 341875988641 pi/2625000000000000 - pi log 2/61250 - pi log 11/375000"
                " + pi log 6/375000 + pi log 15/183750",
                lambda: (arb(2356) / 23625 - arb(341875988641) * arb.pi() / 2625000000000000
                         + arb.pi() * (-arb(2).log() / 61250 - arb(11).log() / 375000 + arb(6).log() / 375000
                                       + arb(15).log() / 183750))),
}


def reference_norm(problem_name: str) -> dict:
    """Validated bracket of the analytic reference norm N (its LOWER endpoint is the denominator of every
    validated relative bound). Problems without a closed form raise KeyError (ratio stays diagnostic)."""
    formula, fn = REFERENCE_NORMS[problem_name]
    with _prec(DIRECT_PREC):
        N = fn().sqrt()
        return {"N2_formula": formula, "norm_lower": down(N), "norm_upper": up(N), "precision_bits": DIRECT_PREC,
                "status": "validated_upper_bound"}


def _weakest(*statuses) -> str:
    return min(statuses, key=STATUSES.index)


def direct_projected_bounds(*, U_R: float, U_S: float, U_I: float, U_C: float, cF2_up: float, status: str,
                            alpha: float = 1.0, A_inf: float = 1.0, U_r_hat: float | None = None,
                            Lambda: float | None = None, U_gradw: float | None = None,
                            reference_norm_lower: float | None = None,
                            reference_status: str | None = None) -> dict:
    """Direct bounds for the feasible output v_hat = max(v, g) (Theorem 4.10), from per-term upper sums
      U_R >= ||R_hat||^2, U_S >= ||S_hat||^2, U_I >= ||u0 - v_hat(0)||^2, U_C >= int mu (v - g)_+.
    With D = U_S + U_I + 2 U_C and L = cF2_up U_R / alpha (eq:direct-finite):
      F_dir = sqrt(D + L)                 >= sqrt(||A^{-1/2}(sigma - tau)||^2 + ||e(T)||^2)   (eq:direct, theta = 1)
      G_dir = sqrt(L) + sqrt(D + L)       >= ||A^{1/2} grad(u - v_hat)||   (inf over theta of (D + L/theta)/(1 - theta))
      G_check = sqrt(2 D + 4 L)           (theta = 1/2, eq:direct-grad); G_dir <= G_check is asserted
      graph = sqrt(U_R)                   (b = c = 0: the graph term equals ||R_hat||)
      E_dir = min over theta in {1/32..31/32} of sqrt(max{1/(alpha(1-theta)), ||A||, 1}(D + L/theta) + U_I + U_R)
                                          (eq:direct-weak, printed (43); b = c = 0)
    No-split fallbacks (eq:direct-nosplit and printed (65); need U_r_hat, Lambda >= alpha, cF2_up <= Lambda):
      F_ns = sqrt(U_r_hat + 2 U_C), G_ns = sqrt((U_r_hat + 2 U_C)/(1 - cF2_up/Lambda)), E_ns = sqrt(G_ns^2 + U_r_hat).
    Original (unprojected) state, optional: grad bound G_dir + sqrt(||A|| U_gradw) (Remark 4.12).
    Relative bounds: absolute upper endpoint / reference LOWER endpoint, validated only if both are.
    The status of every output is the weakest input status (never upgraded)."""
    if status not in STATUSES:
        raise ValueError(status)
    vals = {"U_R": U_R, "U_S": U_S, "U_I": U_I, "U_C": U_C, "cF2_up": cF2_up}
    reasons = [f"{k} invalid ({v})" for k, v in vals.items() if v is None or not math.isfinite(v) or v < 0]
    if reasons:
        status = "diagnostic_float64" if status == "validated_upper_bound" else status
    out = {"status": status, "status_reason": ";".join(reasons), "cF2_up": cF2_up, "alpha": alpha, "A_inf": A_inf,
           "U_R": U_R, "U_S": U_S, "U_I": U_I, "U_C": U_C}
    with _prec(DIRECT_PREC):
        R, S, I, C, cf = (a(v) for v in (U_R, U_S, U_I, U_C, cF2_up))
        al, An = a(alpha), a(A_inf)
        D = S + I + 2 * C
        L = cf * R / al
        Gd = L.sqrt() + (D + L).sqrt()
        Gc = (2 * D + 4 * L).sqrt()
        out.update({"D": up(D), "L": up(L), "sqrt_L": up(L.sqrt()), "sqrt_D_plus_L": up((D + L).sqrt()),
                    "flux_bound_direct_split": up((D + L).sqrt()), "grad_bound_direct_split": up(Gd),
                    "grad_bound_direct_check_theta_half": up(Gc), "graph_bound_direct": up(R.sqrt())})
        if not out["grad_bound_direct_split"] <= out["grad_bound_direct_check_theta_half"]:
            raise AssertionError("G_dir > sqrt(2D + 4L): impossible by Cauchy-Schwarz, arithmetic error")
        best = None
        for th in THETA_SCAN:
            t = a(th)
            fac = (1 / (al * (1 - t))).max(An).max(arb(1))
            E2 = fac * (D + L / t) + I + R
            e = up(E2.sqrt())
            if best is None or e < best[0]:
                best = (e, th)
        out["E_dir_split"] = best[0]
        out["E_dir_split_theta"] = f"{best[1].numerator}/{best[1].denominator}"
        if U_gradw is not None:
            out["grad_bound_direct_original"] = up(Gd + (An * a(U_gradw)).sqrt())
    if U_r_hat is not None:
        out.update(direct_nosplit_bounds(U_r_hat=U_r_hat, U_C=U_C, cF2_up=cF2_up, Lambda=Lambda, alpha=alpha))
    with _prec(DIRECT_PREC):
        if reference_norm_lower is not None:
            Nlo = a(reference_norm_lower)
            if not (Nlo > 0):
                raise ValueError("reference norm lower endpoint must be positive")
            rst = _weakest(status, reference_status or "diagnostic_float64")
            out["reference_norm_lower"] = reference_norm_lower
            out["ratio_status"] = rst
            for k in ("flux_bound_direct_split", "grad_bound_direct_split", "flux_bound_direct", "grad_bound_direct",
                      "grad_bound_direct_original"):
                if k in out:
                    out[k + "_rel"] = up(a(out[k]) / Nlo)
    return out


def direct_nosplit_bounds(*, U_r_hat: float, U_C: float, cF2_up: float, Lambda: float, alpha: float = 1.0) -> dict:
    """eq:direct-nosplit and printed (65), for runs that stored only U_r_hat: theta = cF2_up/Lambda,
    F_ns^2 = U_r_hat + 2 U_C, G_ns^2 = F_ns^2/(1 - theta), E_ns^2 = G_ns^2 + U_r_hat (valid for Lambda >= alpha,
    cF2_up <= Lambda). Same operation order and precision as the v16 build_assets.py (bit-for-bit gate)."""
    with _prec(DIRECT_PREC):
        al, cf = a(alpha), a(cF2_up)
        if Lambda is None or not (a(Lambda) >= al) or not (cf <= a(Lambda)):
            raise ValueError("no-split fallback needs Lambda >= alpha and cF2_up <= Lambda")
        Urh, La = a(U_r_hat), a(Lambda)
        theta = cf / La
        Fd2 = Urh + 2 * a(U_C)
        Gd2 = Fd2 / (1 - theta)
        return {"U_r_hat": U_r_hat, "Lambda": Lambda, "theta_direct": up(theta),
                "flux_bound_direct": up(Fd2.sqrt()), "grad_bound_direct": up(Gd2.sqrt()),
                "E_U_direct": up((Gd2 + Urh).sqrt())}


def C_eq_heat(alpha, A_inf, Lambda) -> float:
    """eq:Ceq-heat of manuscript v16 (b = c = 0): C_eq = max{1/alpha, ||A||_inf, 1, alpha/Lambda} (= 1 for every
    benchmark here). Used by the v17 reporting only; C_eq() above keeps the v14/v15 convention for old tables."""
    with _prec(DIRECT_PREC):
        al = a(alpha)
        return up((1 / al).max(a(A_inf)).max(arb(1)).max(al / a(Lambda)))


def projected_estimator_bounds(*, U_r_hat: float, U_c: float, alpha: float = 1.0, A_inf: float = 1.0,
                               C_eq: float = 1.0) -> dict:
    """Estimator route for the projected output (eq:projected-direct / printed (64)), as in the v16 assets:
    B = 6 (U_r_hat + U_c), G_U = sqrt(B/alpha), F_U = sqrt(||A|| B), E_U_projected = sqrt(C_eq B)."""
    with _prec(DIRECT_PREC):
        B = 6 * (a(U_r_hat) + a(U_c))
        return {"B_projected": up(B), "grad_bound_projected": up((B / a(alpha)).sqrt()),
                "flux_bound_projected": up((a(A_inf) * B).sqrt()), "E_U_projected": up((a(C_eq) * B).sqrt())}


def weighted_bound_heat(*, U_r: float, U_p: float, U_c: float, C_eq: float = 1.0) -> float:
    """E_U_weighted = sqrt(C_eq (18 U_r + 36 U_p + 12 U_c)) (b = c = 0), at DIRECT_PREC."""
    with _prec(DIRECT_PREC):
        return up((a(C_eq) * (18 * a(U_r) + 36 * a(U_p) + 12 * a(U_c))).sqrt())


def relative_upper(bound_upper: float, norm_lower: float) -> float:
    """Upper endpoint of bound / N at DIRECT_PREC (N > 0 a lower endpoint)."""
    with _prec(DIRECT_PREC):
        return up(a(bound_upper) / a(norm_lower))


# ---------------------------------------------------------------- v20 Task 2: validated lower bound of N
def grad_bound_AR(*, U_R: float, U_S: float, U_I: float, U_C: float, cF2_up: float) -> float:
    """G_AR = (a + sqrt(a^2 + 2 (U_I + 2 U_C)))/2, a = sqrt(U_S) + sqrt(cF2_up U_R) (alpha = 1, A = I, b = c = 0;
    manuscript Cor. 4.12), 192-bit arb, upper endpoint. Same operations as scripts/make_v19_grad_ar.g_ar."""
    with _prec(DIRECT_PREC):
        R, S, I, C, c = (a(v) for v in (U_R, U_S, U_I, U_C, cF2_up))
        av = S.sqrt() + (c * R).sqrt()
        return up((av + (av * av + 2 * (I + 2 * C)).sqrt()) / 2)


def solution_scale_lower(*, L_tau: float, L_gradvhat: float, F_dir: float, G_AR: float, upper_status: str,
                         lower_status: str) -> dict:
    """N = ||d_x u|| = ||sigma|| (A = I) >= N_lo = max{sqrt(L_tau) - F_dir, sqrt(L_gradvhat) - G_AR}:
    ||sigma|| >= ||tau|| - ||sigma - tau|| with ||tau|| >= sqrt(L_tau) and ||sigma - tau|| <= F_dir;
    ||d_x u|| >= ||d_x v_hat|| - ||d_x(u - v_hat)|| with ||d_x v_hat|| >= sqrt(L_gradvhat), G_AR the upper bound.
    192-bit arb on the exact binary inputs (lower sums are lower endpoints, F_dir and G_AR upper endpoints), lower
    endpoint rounded downward. Status validated_lower_bound only if the lower sums are validated_lower_bound and
    F_dir / G_AR are validated_upper_bound."""
    st = ("validated_lower_bound" if (upper_status == "validated_upper_bound"
                                       and lower_status == "validated_lower_bound") else
          ("diagnostic_float64" if "diagnostic_float64" in (upper_status, lower_status) else "unvalidated_enclosure"))
    with _prec(DIRECT_PREC):
        n1 = a(L_tau).sqrt() - a(F_dir)
        n2 = a(L_gradvhat).sqrt() - a(G_AR)
        return {"N_lo_tau": down(n1), "N_lo_grad": down(n2), "N_lo": down(n1.max(n2)), "N_lo_status": st,
                "sqrt_L_tau_lo": down(a(L_tau).sqrt()), "sqrt_L_gradvhat_lo": down(a(L_gradvhat).sqrt())}


def ratio_over_lower(bound_upper: float, N_lo: float):
    """Upper endpoint of bound / N_lo (192 bits), or 'not positive' when N_lo <= 0."""
    if not N_lo > 0:
        return "not positive"
    return relative_upper(bound_upper, N_lo)
