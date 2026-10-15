"""Network- and triple-level a priori bounds from trained weights, specification
section 6. No network evaluation happens anywhere in this module -- every
bound here is a closed-form function of the trained W_k, b_k matrices (via
models/mlp.py's `weights`/`biases`) and the problem's own data constants.

Several of the requested quantities (L_gamma, L_sm, L_Psi, L_Phi0, L_PhiT)
reference sub-formulas specification never spells out explicitly. Per an explicit
decision (2026-09-09, recorded in specification as an addendum to section 6):

  - A function's separate space/time first-derivative sup-bounds (M_x, M_t)
    are combined into one space-time Lipschitz constant via the Euclidean
    norm sqrt(M_t^2 + M_x^2) -- consistent with h_K = sqrt(dt^2+dx^2)
    elsewhere. Used for Lip(u_theta), Lip(g), Lip(f), Lip(grad_x gamma),
    Lip(d_t gamma).
  - Lip(d_t u_theta): since u_theta = B(x)*N(t,x) with B depending on x only,
    d_t u_theta = B(x)*N_t(t,x) is a plain product (no cross term from d/dt B,
    since there is none) -- same derivation pattern as the already-given
    Lip(grad_x u_theta), simplified: Lip(d_t u_theta) <=
    ||grad B||_inf * Pi_N + ||B||_inf * Lip(grad N).
  - Lip(R_PDE), Lip(R_sig): derived via the standard sum/product rule from
    R_PDE = f - u_t - sigma_x - b*u_x - c*u + lambda and
    R_sig = A^{-1/2}*sigma_phi + A^{1/2}*grad_x u_theta, using only
    already-defined pieces (including the two points above).
  - L_Phi0, L_PhiT: built the same way L_sm is (the d/dx[c*h^2] <=
    2*c*|h|*Lip(h) rule, applied per section-5 term of Phi_0/Phi_T), with
    Lip(u_theta(0,.)) taken to be the global (hence valid, if not maximally
    tight) spatial bound ||grad_x u_theta||_inf, and a new m_0 bound
    (||u0||_inf + ||B||_inf*Sigma_N) playing the role m_gap plays for gamma.

d (spatial dimension) is hardcoded to 1: both benchmark problems (Stefan,
American put) are 1D in x, and BaseProblem.omega_bounds is a single interval.

Multiplier-head-specific bounds (experiments/multiplier_ablation.py,
2026-09-11), specification section 6's own "Alternative heads" paragraph:
branches on triple.multiplier_net.head (models/multiplier_net.py already
exposes this) to compute lambda_inf and lip_lambda -- the only two
quantities that depend on the multiplier head's own nonlinearity, since
every downstream use (m_R's lambda term, m_lam, L_sm's T4/T5 terms,
lip_R_PDE) is already built purely from these two values, never from
Sigma_Mps/Pi_Mps directly:

  'square' (M^2), 'relu_square' ((relu M)^2): lambda_inf = Sigma_Mps^2,
  lip_lambda = 2*Sigma_Mps*Pi_Mps -- specification section 6 states relu_square
  has "the same bounds as M^2" explicitly (exact: |relu(z)^2 - relu(z')^2| =
  |relu(z)+relu(z')|*|relu(z)-relu(z')| <= 2*Sigma_Mps*|z-z'| since
  0<=relu(z)<=|z|<=Sigma_Mps, matching square's own 2*Sigma*Pi derivation).

  'softplus': lambda_inf = Sigma_Mps + log(2), lip_lambda = Pi_Mps --
  specification section 6's own given formulas. Exact: softplus(z) =
  max(z,0) + log(1+e^-|z|) <= max(z,0) + log(2) <= |z| + log(2), so
  sup|softplus(M)| <= Sigma_Mps + log(2); softplus'(z) = sigmoid(z) in (0,1)
  always, so softplus is 1-Lipschitz in its own argument, giving
  Lip(softplus(M)) <= 1 * Lip(M) = Pi_Mps (no factor of Sigma, unlike the
  squaring heads -- softplus never squares its argument).

This is also the reason experiments/multiplier_ablation.py can certify all
three heads through the identical certify_option_b pipeline: before this,
only 'square' had a correct a priori lambda_inf/lip_lambda, so a rigorous
Level-2 certificate could only ever have been computed for that one head --
specification's own instruction ("never report a rigorous Level-2 certificate
for one head next to a raw estimator for another") is what this addition
exists to satisfy, not a preference.
"""

import contextlib
import math
from fractions import Fraction
from dataclasses import dataclass, replace

import torch

from certified_obstacle_nn.models.mlp import MLP
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.base_problem import BaseProblem

C2 = 4.0 / (3.0 * math.sqrt(3.0))
SPATIAL_DIMENSION = 1


@dataclass(frozen=True)
class NetworkBounds:
    output_bound: float  # Sigma_N: ||N||_inf <= Sigma_N
    gradient_bound: float  # Pi_N: ||grad N||_inf <= Pi_N, also Lip(N) <= Pi_N
    gradient_lipschitz_bound: float  # Lip(grad N) <= c2 * Pi_N * sum_{k<L} Pi_k


def _float_up(q: Fraction) -> float:
    """Smallest-ish float >= the exact rational q (round, then step up if needed)."""
    f = float(q)
    while Fraction(f) < q:
        f = math.nextafter(f, math.inf)
    return f


def _sqrt_up(q: Fraction) -> float:
    """A float s with s >= sqrt(q), checked exactly (s*s >= q in rationals)."""
    s = math.sqrt(float(q))
    while Fraction(s) * Fraction(s) < q:
        s = math.nextafter(s, math.inf)
    return s


def guaranteed_spectral_norm_upper(w: torch.Tensor) -> float:
    """A PROVED upper bound on ||W||_2 (v14 A2): min(||W||_F, sqrt(||W||_1 ||W||_inf)),
    each evaluated in exact rational arithmetic on the stored binary floats and
    rounded upward. Both are classical upper bounds of the spectral norm."""
    entries = [[Fraction(float(v)) for v in row] for row in w.detach().to(torch.float64).tolist()]
    frob_sq = sum((v * v for row in entries for v in row), Fraction(0))
    row_sums = [sum((abs(v) for v in row), Fraction(0)) for row in entries]
    col_sums = [sum((abs(row[j]) for row in entries), Fraction(0)) for j in range(len(entries[0]))]
    norm_inf = max(row_sums)
    norm_1 = max(col_sums)
    return min(_sqrt_up(frob_sq), _sqrt_up(norm_1 * norm_inf))


def spectral_norms_svd(mlp: MLP) -> list[float]:
    """||W_k||_2 via a float64 SVD (torch.linalg.matrix_norm(ord=2)). A float
    APPROXIMATION, not a proved bound -- kept only as a diagnostic (v14 A2)."""
    with torch.no_grad():
        return [torch.linalg.matrix_norm(w.to(torch.float64), ord=2).item() for w in mlp.weights]


_HISTORICAL_SVD = False


@contextlib.contextmanager
def historical_svd_norms():
    """Reproduce HISTORICAL (pre-v14) numbers: inside this context
    spectral_norms returns the float SVD values again. For regression tests
    against recorded legacy values only -- never for a reported bound."""
    global _HISTORICAL_SVD
    old = _HISTORICAL_SVD
    _HISTORICAL_SVD = True
    try:
        yield
    finally:
        _HISTORICAL_SVD = old


def spectral_norms(mlp: MLP) -> list[float]:
    """Proved upper bounds on ||W_k||_2 for every layer (v14 A2, 2026-10-06).

    Previously the float SVD value (spectral_norms_svd), which is an
    approximation and can fall below the true norm by rounding; replaced by
    guaranteed_spectral_norm_upper.
    """
    if _HISTORICAL_SVD:
        return spectral_norms_svd(mlp)
    with torch.no_grad():
        return [guaranteed_spectral_norm_upper(w) for w in mlp.weights]


def compute_network_bounds(mlp: MLP) -> NetworkBounds:
    norms = spectral_norms(mlp)  # ||W_1||_2, ..., ||W_L||_2

    pis = []
    running = 1.0
    for norm in norms:
        running *= norm
        pis.append(running)  # Pi_1, ..., Pi_L
    pi_N = pis[-1]

    with torch.no_grad():
        last_weight = mlp.weights[-1]
        last_bias = mlp.biases[-1]
        n_prev = last_weight.shape[1]
        sigma_N = norms[-1] * math.sqrt(n_prev) + torch.linalg.vector_norm(last_bias).item()

    lip_grad_N = C2 * pi_N * sum(pis[:-1])  # sum_{k=1}^{L-1} Pi_k

    return NetworkBounds(output_bound=sigma_N, gradient_bound=pi_N, gradient_lipschitz_bound=lip_grad_N)


def mask_bounds(omega_bounds: tuple[float, float]) -> tuple[float, float, float]:
    """(||B||_inf, ||grad B||_inf, ||grad^2 B||_inf) for the box mask
    B(x) = 4(x-a)(b-x)/(b-a)^2, specification section 1.
    """
    a, b = omega_bounds
    width = b - a
    return 1.0, 4.0 / width, 8.0 / width**2


@dataclass(frozen=True)
class TripleBounds:
    state_bounds: NetworkBounds
    flux_bounds: NetworkBounds
    multiplier_bounds: NetworkBounds
    multiplier_head: str  # 'square' / 'softplus' / 'relu_square' -- provenance for which lambda_inf/lip_lambda formula applied (experiments/multiplier_ablation.py, 2026-09-11)

    # Triple-level sup-norms, specification section 6.
    u_inf: float
    grad_x_u_inf: float
    d_t_u_inf: float
    sigma_inf: float
    div_x_sigma_inf: float
    lambda_inf: float

    # Gap and residual sup-norms ("m_*").
    m_gap: float
    m_gap_x: float
    m_gap_t: float
    m_R: float
    m_sig: float
    m_lam: float
    M_Psi: float

    # Lipschitz constants.
    lip_grad_x_u: float
    lip_div_x_sigma: float
    lip_lambda: float
    lip_u: float
    lip_d_t_u: float
    lip_g: float
    lip_f: float
    lip_R_PDE: float
    lip_R_sig: float
    L_gamma: float
    L_sm: float
    L_Psi: float
    m_0: float
    L_Phi0: float
    L_PhiT: float


_K1_FIELDS = (  # scale linearly in K: sup-norms/Lipschitz constants of a SINGLE
    # quantity (u, sigma, lambda, gamma=g-u, R_PDE, R_sig, u0, problem data
    # g/f) that itself scales as K^1 when converting normalized -> physical.
    "u_inf", "grad_x_u_inf", "d_t_u_inf", "sigma_inf", "div_x_sigma_inf", "lambda_inf",
    "m_gap", "m_gap_x", "m_gap_t", "m_R", "m_sig", "m_lam",
    "lip_grad_x_u", "lip_div_x_sigma", "lip_lambda", "lip_u", "lip_d_t_u",
    "lip_g", "lip_f", "lip_R_PDE", "lip_R_sig", "L_gamma", "m_0",
)
_K2_FIELDS = (  # scale as K^2: every one of these is a SUM of PRODUCTS of two
    # K^1 quantities (specification section 6's own d/dx[c*h^2]<=2c|h|Lip(h)
    # sum/product-rule pattern -- see the section 6 addendum -- means L_sm's
    # five terms, L_Psi's two terms, M_Psi's two terms, L_Phi0's two terms,
    # and L_PhiT's one term are ALL literally "(K^1 quantity) * (K^1
    # quantity)" or "(K^1 quantity)^2", hence K^2 overall.
    "M_Psi", "L_sm", "L_Psi", "L_Phi0", "L_PhiT",
)


def physical_units_bounds(bounds_normalized: TripleBounds, K: float) -> TripleBounds:
    """Converts a TripleBounds computed on a NORMALIZED-output triple (e.g.
    compute_triple_bounds(NormalizedAmericanPutProblem(), base_triple),
    specification section 10's ubar=u/K, sigmabar=sigma/K, lambdabar=lambda/K
    training convention, confirmed with the user 2026-09-12) into the
    corresponding PHYSICAL-units TripleBounds -- WITHOUT needing the
    physical-unit triple to have real, weight-introspectable sub-networks at
    all (models/unit_conversion.py's PhysicalUnitsTriple only wraps
    __call__, it has no .net attribute compute_network_bounds could use).

    Every field of TripleBounds is either (see module-level _K1_FIELDS/
    _K2_FIELDS for the itemized classification and reasoning):
      K^1 -- a sup-norm or Lipschitz constant of a single quantity (u, sigma,
             lambda, gamma=g-u, R_PDE, R_sig, u0, or the problem data g/f
             themselves) that is LINEAR in the solution scale, since dividing
             the whole PDE by K (specification section 0) scales u, sigma=-A*grad(u),
             and lambda=lambda_bar*K identically and additively -- true for
             ANY of these built from sums of such terms.
      K^2 -- built as a SUM of PRODUCTS of two K^1 quantities (L_sm's five
             terms, L_Psi's two, M_Psi's two, L_Phi0's two, L_PhiT's one --
             every one of specification section 6's own product-rule
             constructions), hence scales quadratically.

    The nested state_bounds/flux_bounds/multiplier_bounds (NetworkBounds) and
    multiplier_head are left UNCHANGED (still describe the actual trained,
    normalized-scale sub-networks' own weights -- not used standalone by
    certify_option_b beyond what is already captured in the K1/K2 fields
    above, which are recomputed from them via compute_triple_bounds BEFORE
    this function is ever called).
    """
    updates = {name: getattr(bounds_normalized, name) * K for name in _K1_FIELDS}
    updates.update({name: getattr(bounds_normalized, name) * K ** 2 for name in _K2_FIELDS})
    return replace(bounds_normalized, **updates)


def compute_triple_bounds(problem: BaseProblem, triple: NeuralTriple) -> TripleBounds:
    state_bounds = compute_network_bounds(triple.state_net.net)
    flux_bounds = compute_network_bounds(triple.flux_net.net)
    multiplier_bounds = compute_network_bounds(triple.multiplier_net.net)

    B_inf, gradB_inf, grad2B_inf = mask_bounds(problem.omega_bounds)

    Sigma_Nth, Pi_Nth = state_bounds.output_bound, state_bounds.gradient_bound
    Sigma_Sph, Pi_Sph = flux_bounds.output_bound, flux_bounds.gradient_bound
    Sigma_Mps, Pi_Mps = multiplier_bounds.output_bound, multiplier_bounds.gradient_bound

    d = SPATIAL_DIMENSION

    multiplier_head = triple.multiplier_net.head
    if multiplier_head == "softplus":
        lambda_inf = Sigma_Mps + math.log(2.0)
        lip_lambda = Pi_Mps
    else:  # "square" or "relu_square" -- identical bounds, see module docstring
        lambda_inf = Sigma_Mps**2
        lip_lambda = 2.0 * Sigma_Mps * Pi_Mps

    u_inf = B_inf * Sigma_Nth
    grad_x_u_inf = gradB_inf * Sigma_Nth + B_inf * Pi_Nth
    d_t_u_inf = B_inf * Pi_Nth
    sigma_inf = Sigma_Sph
    div_x_sigma_inf = d * Pi_Sph

    m_gap = problem.g_inf + B_inf * Sigma_Nth
    m_gap_x = problem.g_x_inf + gradB_inf * Sigma_Nth + B_inf * Pi_Nth
    m_gap_t = problem.g_t_inf + B_inf * Pi_Nth
    m_R = (
        problem.f_inf
        + B_inf * Pi_Nth
        + d * Pi_Sph
        + abs(problem.b) * (gradB_inf * Sigma_Nth + B_inf * Pi_Nth)
        + abs(problem.c) * B_inf * Sigma_Nth
        + lambda_inf
    )
    m_sig = problem.alpha**-0.5 * Sigma_Sph + problem.A_inf**0.5 * (gradB_inf * Sigma_Nth + B_inf * Pi_Nth)
    m_lam = lambda_inf
    M_Psi = problem.A_inf * m_gap_x**2 + (problem.Lambda / problem.alpha) * m_gap_t**2

    lip_grad_x_u = grad2B_inf * Sigma_Nth + 2.0 * gradB_inf * Pi_Nth + B_inf * state_bounds.gradient_lipschitz_bound
    lip_div_x_sigma = d * flux_bounds.gradient_lipschitz_bound

    # Lip(d_t u_theta) -- derived, see module docstring.
    lip_d_t_u = gradB_inf * Pi_Nth + B_inf * state_bounds.gradient_lipschitz_bound

    # Lip(u_theta), Lip(g), Lip(f) -- Euclidean combination, see module docstring.
    lip_u = math.hypot(d_t_u_inf, grad_x_u_inf)
    lip_g = math.hypot(problem.g_t_inf, problem.g_x_inf)
    lip_f = math.hypot(problem.f_t_inf, problem.f_x_inf)
    L_gamma = lip_g + lip_u

    # Lip(R_PDE), Lip(R_sig) -- derived, see module docstring.
    lip_R_PDE = lip_f + lip_d_t_u + lip_div_x_sigma + abs(problem.b) * lip_grad_x_u + abs(problem.c) * lip_u + lip_lambda
    lip_R_sig = problem.alpha**-0.5 * flux_bounds.gradient_bound + problem.A_inf**0.5 * lip_grad_x_u

    L_sm = (
        (2.0 * problem.Lambda / problem.alpha) * m_R * lip_R_PDE
        + 2.0 * m_sig * lip_R_sig
        + 2.0 * problem.kappa * m_gap * L_gamma
        + m_lam * L_gamma
        + m_gap * lip_lambda
    )

    lip_grad_x_g = math.hypot(problem.g_tx_inf, problem.g_xx_inf)
    lip_d_t_g = math.hypot(problem.g_tt_inf, problem.g_tx_inf)
    lip_grad_x_gamma = lip_grad_x_g + lip_grad_x_u
    lip_d_t_gamma = lip_d_t_g + lip_d_t_u

    lip_A = 0.0  # both benchmark problems have constant A
    L_Psi = (
        2.0
        * problem.A_inf**0.5
        * m_gap_x
        * (lip_A / (2.0 * math.sqrt(problem.alpha)) * m_gap_x + problem.A_inf**0.5 * lip_grad_x_gamma)
        + (2.0 * problem.Lambda / problem.alpha) * m_gap_t * lip_d_t_gamma
    )

    # L_Phi0, L_PhiT -- derived, see module docstring.
    m_0 = problem.u0_inf + B_inf * Sigma_Nth
    lip_u_theta_at_0 = grad_x_u_inf
    L_Phi0 = 2.0 * m_0 * (problem.u0_lip + lip_u_theta_at_0) + 2.0 * m_gap * L_gamma
    L_PhiT = 2.0 * m_gap * L_gamma

    return TripleBounds(
        state_bounds=state_bounds,
        flux_bounds=flux_bounds,
        multiplier_bounds=multiplier_bounds,
        multiplier_head=multiplier_head,
        u_inf=u_inf,
        grad_x_u_inf=grad_x_u_inf,
        d_t_u_inf=d_t_u_inf,
        sigma_inf=sigma_inf,
        div_x_sigma_inf=div_x_sigma_inf,
        lambda_inf=lambda_inf,
        m_gap=m_gap,
        m_gap_x=m_gap_x,
        m_gap_t=m_gap_t,
        m_R=m_R,
        m_sig=m_sig,
        m_lam=m_lam,
        M_Psi=M_Psi,
        lip_grad_x_u=lip_grad_x_u,
        lip_div_x_sigma=lip_div_x_sigma,
        lip_lambda=lip_lambda,
        lip_u=lip_u,
        lip_d_t_u=lip_d_t_u,
        lip_g=lip_g,
        lip_f=lip_f,
        lip_R_PDE=lip_R_PDE,
        lip_R_sig=lip_R_sig,
        L_gamma=L_gamma,
        L_sm=L_sm,
        L_Psi=L_Psi,
        m_0=m_0,
        L_Phi0=L_Phi0,
        L_PhiT=L_PhiT,
    )


def decompose_L_sm_and_L_Psi(problem: BaseProblem, bounds: TripleBounds) -> dict:
    """Additive decomposition of L_sm (5 terms) and L_Psi (2 terms), specification
    section 6's own formulas -- a pure REPORTING decomposition, not a new or
    modified bound: every term below is computed only from fields
    compute_triple_bounds already put on `bounds`, plus pure problem-data
    constants (g's own second-derivative sup-norms), so this cannot drift
    from or change what compute_triple_bounds itself computes. Verified to
    sum back to bounds.L_sm / bounds.L_Psi exactly in
    tests/test_network_bounds.py.

    Exists to answer "which single additive term dominates L_sm (or L_Psi)",
    e.g. when E_h_lip = (L_sm + L_Psi) * h * Q_measure is found to dominate
    E_h across a whole certification-mesh sweep (experiments/
    stefan_bound_decomposition.py) -- compute_triple_bounds itself only
    returns the two final sums, not which of their five/two addends is
    responsible.
    """
    Lambda_over_alpha = problem.Lambda / problem.alpha

    L_sm_terms = {
        "T1_pde_residual": Lambda_over_alpha * 2.0 * bounds.m_R * bounds.lip_R_PDE,
        "T2_sig_residual": 2.0 * bounds.m_sig * bounds.lip_R_sig,
        "T3_kappa_gap": 2.0 * problem.kappa * bounds.m_gap * bounds.L_gamma,
        "T4_lambda_Lgamma": bounds.m_lam * bounds.L_gamma,
        "T5_gap_lip_lambda": bounds.m_gap * bounds.lip_lambda,
    }

    # lip_grad_x_gamma / lip_d_t_gamma: same derivation as compute_triple_bounds
    # (lip_A=0 for both benchmark problems, constant A), re-derived here from
    # bounds.lip_grad_x_u / bounds.lip_d_t_u (already on `bounds`) plus g's own
    # problem-data second derivatives -- not a new computation, the identical
    # one compute_triple_bounds performs internally but does not itself expose.
    lip_grad_x_g = math.hypot(problem.g_tx_inf, problem.g_xx_inf)
    lip_d_t_g = math.hypot(problem.g_tt_inf, problem.g_tx_inf)
    lip_grad_x_gamma = lip_grad_x_g + bounds.lip_grad_x_u
    lip_d_t_gamma = lip_d_t_g + bounds.lip_d_t_u

    L_Psi_terms = {
        "S1_spatial": 2.0 * problem.A_inf * bounds.m_gap_x * lip_grad_x_gamma,
        "S2_temporal": Lambda_over_alpha * 2.0 * bounds.m_gap_t * lip_d_t_gamma,
    }

    return {
        "L_sm_terms": L_sm_terms,
        "L_sm_recomputed": sum(L_sm_terms.values()),
        "L_Psi_terms": L_Psi_terms,
        "L_Psi_recomputed": sum(L_Psi_terms.values()),
    }
