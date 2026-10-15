"""Tests for certification/network_bounds.py.

Every analytic bound in TripleBounds is checked against densely sampled
network outputs/AD derivatives. Passing does not prove the bound (finite
sampling can miss the true supremum); FAILING proves the implementation is
wrong (a bound that is genuinely valid can never be violated by any sample, no
matter how the sample was chosen). Every check also logs sampled_max/bound so
a formula bug that produces a technically-true-but-meaningless bound (e.g. off
by a large constant factor) is visible even when the pass/fail assertion alone
would not catch it.

Sup-norm bounds are tested by direct sampling. Lipschitz bounds are tested via
the *exact* two-point secant property |f(p2)-f(p1)| <= Lip(f)*dist(p1,p2),
which holds for any finite separation (not just infinitesimal) -- so this is
a direct, exact test of the Lipschitz claim, not an approximation. Only first
derivatives are ever computed (specification section 0): Lip(grad N) is tested via
the difference of two ordinary (first-order autograd) gradient samples, never
a second derivative.
"""

import math

import pytest
import torch

from certified_obstacle_nn.autograd.derivatives import compute_derivatives
from certified_obstacle_nn.certification.densities import compute_boundary_densities, phi_smooth, psi_majorant
from certified_obstacle_nn.certification.network_bounds import (
    compute_network_bounds,
    compute_triple_bounds,
    decompose_L_sm_and_L_Psi,
)
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.stefan import StefanProblem
from certified_obstacle_nn.residuals.residuals import compute_residuals

pytestmark = pytest.mark.legacy

N_SUP_SAMPLES = 4000
N_LIP_PAIRS = 1000
MARGIN = 1e-3  # keep perturbed points inside (0,1)^2
UPPER_SLACK = 1.0 + 1e-6  # tiny float/finite-difference slack on the analytic upper bound
LOOSE_RATIO_FLOOR = 1e-8  # catches a wildly-wrong (e.g. unit-error) bound, not tightness


def _sample_domain(n, seed, lo=MARGIN, hi=1.0 - MARGIN):
    generator = torch.Generator().manual_seed(seed)
    t = torch.rand(n, 1, generator=generator, dtype=torch.float64) * (hi - lo) + lo
    x = torch.rand(n, 1, generator=generator, dtype=torch.float64) * (hi - lo) + lo
    return t, x


def _random_directions(n, eps, seed):
    generator = torch.Generator().manual_seed(seed)
    theta = torch.rand(n, 1, generator=generator, dtype=torch.float64) * 2 * math.pi
    dt = eps * torch.cos(theta)
    dx = eps * torch.sin(theta)
    return dt, dx


def _check_and_log(name, sampled_max, bound):
    ratio = sampled_max / bound if bound > 0 else float("inf")
    print(f"{name:28s} sampled_max={sampled_max:.6e}  bound={bound:.6e}  ratio={ratio:.4f}")
    assert sampled_max <= bound * UPPER_SLACK, f"{name} VIOLATED: sampled {sampled_max:.6e} > bound {bound:.6e}"
    assert ratio > LOOSE_RATIO_FLOOR, f"{name} bound is suspiciously loose (ratio={ratio:.3e}) -- check for a formula bug"


def _mlp_value_and_grad(mlp, t, x):
    t = t.clone().requires_grad_(True)
    x = x.clone().requires_grad_(True)
    out = mlp(torch.cat([t, x], dim=-1))
    grad_t, grad_x = torch.autograd.grad(out, (t, x), grad_outputs=torch.ones_like(out))
    return out.detach(), grad_t.detach(), grad_x.detach()


def _empirical_scalar_lipschitz(fn, t, x, eps, seed):
    """max |fn(p2) - fn(p1)| / dist(p1,p2) over random-direction point pairs.

    No blanket no_grad() here: several `fn`s need their own internal
    autograd.grad calls (they each set requires_grad_(True) on their own
    local clones), and no_grad() would suppress that globally regardless.
    Each such fn detaches its own return value.
    """
    dt, dx = _random_directions(t.shape[0], eps, seed)
    f1 = fn(t, x)
    f2 = fn(t + dt, x + dx)
    dist = torch.sqrt(dt**2 + dx**2)
    return ((f2 - f1).abs() / dist).max().item()


def _empirical_vector_lipschitz(fn, t, x, eps, seed):
    """fn(t,x) -> (comp1, comp2); max ||grad-like vector difference|| / dist."""
    dt, dx = _random_directions(t.shape[0], eps, seed)
    a1, b1 = fn(t, x)
    a2, b2 = fn(t + dt, x + dx)
    dist = torch.sqrt(dt**2 + dx**2)
    diff_norm = torch.sqrt((a2 - a1) ** 2 + (b2 - b1) ** 2)
    return (diff_norm / dist).max().item()


@pytest.fixture(scope="module")
def problem():
    return StefanProblem()


@pytest.fixture(scope="module")
def triple(problem):
    # A briefly-trained (not just randomly initialized) triple: a fresh random
    # network can legitimately have zero obstacle violation everywhere in the
    # sampled domain (u_theta happens to stay below g nowhere), which would
    # make L_Phi0/L_PhiT's "not suspiciously loose" check meaningless rather
    # than a real signal. A few hundred epochs is enough to produce genuine
    # nonzero gap/penetration dynamics without needing full convergence.
    from certified_obstacle_nn.training.sampler import TrainingSampler
    from certified_obstacle_nn.training.trainer import train

    torch.manual_seed(0)
    net_triple = NeuralTriple(problem, width=24, depth=3)
    sampler = TrainingSampler(problem, n_interior=1000, n_initial=100, seed=0)
    t, x, x0 = sampler.sample()
    train(problem, net_triple, t, x, x0, n_epochs=300, lr=1e-3)
    net_triple.eval()
    return net_triple


@pytest.fixture(scope="module")
def bounds(problem, triple):
    return compute_triple_bounds(problem, triple)


# ---------------------------------------------------------------------------
# per-network bounds: ||N||_inf <= Sigma_N, ||grad N||_inf <= Pi_N,
# Lip(grad N) <= gradient_lipschitz_bound
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("net_name", ["state_net", "flux_net", "multiplier_net"])
def test_per_network_bounds(problem, triple, net_name):
    net = getattr(triple, net_name)
    mlp = net.net
    nb = compute_network_bounds(mlp)

    t, x = _sample_domain(N_SUP_SAMPLES, seed=1)
    out, grad_t, grad_x = _mlp_value_and_grad(mlp, t, x)

    sampled_output_max = out.abs().max().item()
    _check_and_log(f"{net_name}.output", sampled_output_max, nb.output_bound)

    sampled_grad_norm_max = torch.sqrt(grad_t**2 + grad_x**2).max().item()
    _check_and_log(f"{net_name}.gradient", sampled_grad_norm_max, nb.gradient_bound)

    t2, x2 = _sample_domain(N_LIP_PAIRS, seed=2)

    def grad_fn(tt, xx):
        _, gt, gx = _mlp_value_and_grad(mlp, tt, xx)
        return gt, gx

    for eps, seed in [(1e-4, 3), (5e-2, 4)]:
        sampled = _empirical_vector_lipschitz(grad_fn, t2, x2, eps, seed)
        _check_and_log(f"{net_name}.grad_lipschitz(eps={eps})", sampled, nb.gradient_lipschitz_bound)


# ---------------------------------------------------------------------------
# triple-level sup-norms
# ---------------------------------------------------------------------------


def test_triple_level_sup_norms(problem, triple, bounds):
    t, x = _sample_domain(N_SUP_SAMPLES, seed=10)
    t.requires_grad_(True)
    x.requires_grad_(True)
    u, sigma, lam, u_t, u_x, sigma_x = compute_derivatives(triple.state_net, triple.flux_net, triple.multiplier_net, t, x)

    _check_and_log("u_theta", u.abs().max().item(), bounds.u_inf)
    _check_and_log("grad_x u_theta", u_x.abs().max().item(), bounds.grad_x_u_inf)
    _check_and_log("d_t u_theta", u_t.abs().max().item(), bounds.d_t_u_inf)
    _check_and_log("sigma_phi", sigma.abs().max().item(), bounds.sigma_inf)
    _check_and_log("div_x sigma_phi", sigma_x.abs().max().item(), bounds.div_x_sigma_inf)
    _check_and_log("lambda_psi", lam.abs().max().item(), bounds.lambda_inf)


# ---------------------------------------------------------------------------
# gap / residual sup-norms
# ---------------------------------------------------------------------------


def test_gap_and_residual_sup_norms(problem, triple, bounds):
    t, x = _sample_domain(N_SUP_SAMPLES, seed=20)
    t.requires_grad_(True)
    x.requires_grad_(True)
    x0 = _sample_domain(N_SUP_SAMPLES, seed=21)[1]

    residuals = compute_residuals(problem, triple.state_net, triple.flux_net, triple.multiplier_net, t, x, x0)
    (gamma_t,) = torch.autograd.grad(residuals.gamma, t, grad_outputs=torch.ones_like(residuals.gamma), retain_graph=True)
    (gamma_x,) = torch.autograd.grad(residuals.gamma, x, grad_outputs=torch.ones_like(residuals.gamma))

    _check_and_log("gamma", residuals.gamma.abs().max().item(), bounds.m_gap)
    _check_and_log("grad_x gamma", gamma_x.abs().max().item(), bounds.m_gap_x)
    _check_and_log("d_t gamma", gamma_t.abs().max().item(), bounds.m_gap_t)
    _check_and_log("R_PDE", residuals.R_PDE.abs().max().item(), bounds.m_R)
    _check_and_log("R_sig", residuals.R_sig.abs().max().item(), bounds.m_sig)

    lam = triple.multiplier_net(t, x)
    _check_and_log("lambda_psi (m_lam)", lam.abs().max().item(), bounds.m_lam)

    psi = psi_majorant(problem, triple, t, x)
    _check_and_log("Psi (M_Psi)", psi.max().item(), bounds.M_Psi)


# ---------------------------------------------------------------------------
# Lipschitz bounds, via the exact two-point secant property
# ---------------------------------------------------------------------------


def test_lip_grad_x_u_and_lip_lambda_and_lip_div_x_sigma(problem, triple, bounds):
    t, x = _sample_domain(N_LIP_PAIRS, seed=30)

    def grad_x_u_fn(tt, xx):
        tt = tt.clone().requires_grad_(True)
        xx = xx.clone().requires_grad_(True)
        _, _, _, _, u_x, _ = compute_derivatives(triple.state_net, triple.flux_net, triple.multiplier_net, tt, xx)
        return u_x.detach()

    def lambda_fn(tt, xx):
        return triple.multiplier_net(tt, xx)

    def sigma_x_fn(tt, xx):
        tt = tt.clone().requires_grad_(True)
        xx = xx.clone().requires_grad_(True)
        _, _, _, _, _, sigma_x = compute_derivatives(triple.state_net, triple.flux_net, triple.multiplier_net, tt, xx)
        return sigma_x.detach()

    for eps, seed in [(1e-4, 31), (5e-2, 32)]:
        _check_and_log(
            f"Lip(grad_x u_theta)(eps={eps})", _empirical_scalar_lipschitz(grad_x_u_fn, t, x, eps, seed), bounds.lip_grad_x_u
        )
        _check_and_log(
            f"Lip(lambda_psi)(eps={eps})", _empirical_scalar_lipschitz(lambda_fn, t, x, eps, seed), bounds.lip_lambda
        )
        _check_and_log(
            f"Lip(div_x sigma_phi)(eps={eps})", _empirical_scalar_lipschitz(sigma_x_fn, t, x, eps, seed), bounds.lip_div_x_sigma
        )


def test_L_gamma(problem, triple, bounds):
    t, x = _sample_domain(N_LIP_PAIRS, seed=40)

    def gamma_fn(tt, xx):
        tt = tt.clone().requires_grad_(True)
        xx = xx.clone().requires_grad_(True)
        residuals = compute_residuals(problem, triple.state_net, triple.flux_net, triple.multiplier_net, tt, xx, xx)
        return residuals.gamma.detach()

    for eps, seed in [(1e-4, 41), (5e-2, 42)]:
        _check_and_log(f"L_gamma(eps={eps})", _empirical_scalar_lipschitz(gamma_fn, t, x, eps, seed), bounds.L_gamma)


def test_L_sm(problem, triple, bounds):
    t, x = _sample_domain(N_LIP_PAIRS, seed=50)

    def phi_sm_fn(tt, xx):
        tt = tt.clone().requires_grad_(True)
        xx = xx.clone().requires_grad_(True)
        return phi_smooth(problem, triple, tt, xx).detach()

    for eps, seed in [(1e-4, 51), (5e-2, 52)]:
        _check_and_log(f"L_sm(eps={eps})", _empirical_scalar_lipschitz(phi_sm_fn, t, x, eps, seed), bounds.L_sm)


def test_L_Psi(problem, triple, bounds):
    t, x = _sample_domain(N_LIP_PAIRS, seed=60)

    def psi_fn(tt, xx):
        tt = tt.clone().requires_grad_(True)
        xx = xx.clone().requires_grad_(True)
        return psi_majorant(problem, triple, tt, xx).detach()

    for eps, seed in [(1e-4, 61), (5e-2, 62)]:
        _check_and_log(f"L_Psi(eps={eps})", _empirical_scalar_lipschitz(psi_fn, t, x, eps, seed), bounds.L_Psi)


def test_L_Phi0_and_L_PhiT(problem, triple, bounds):
    generator = torch.Generator().manual_seed(70)
    n = N_LIP_PAIRS
    x = torch.rand(n, 1, generator=generator, dtype=torch.float64) * (1.0 - 2 * MARGIN) + MARGIN

    for eps, seed in [(1e-4, 71), (5e-2, 72)]:
        gen = torch.Generator().manual_seed(seed)
        dx = eps * (torch.rand(n, 1, generator=gen, dtype=torch.float64) * 2 - 1)
        x2 = x + dx
        dist = dx.abs()

        Phi_0_1, Phi_T_1 = compute_boundary_densities(problem, triple, x)
        Phi_0_2, Phi_T_2 = compute_boundary_densities(problem, triple, x2)

        sampled_phi0 = ((Phi_0_2 - Phi_0_1).abs() / dist).max().item()
        sampled_phiT = ((Phi_T_2 - Phi_T_1).abs() / dist).max().item()

        _check_and_log(f"L_Phi0(eps={eps})", sampled_phi0, bounds.L_Phi0)
        _check_and_log(f"L_PhiT(eps={eps})", sampled_phiT, bounds.L_PhiT)


def test_decompose_L_sm_and_L_Psi_sums_back_exactly(problem, bounds):
    """decompose_L_sm_and_L_Psi is a pure reporting decomposition -- its
    terms must sum back to compute_triple_bounds's own L_sm/L_Psi exactly
    (to float rounding), or it has silently drifted from the real formula."""
    decomposition = decompose_L_sm_and_L_Psi(problem, bounds)

    assert decomposition["L_sm_recomputed"] == pytest.approx(bounds.L_sm, rel=1e-12)
    assert decomposition["L_Psi_recomputed"] == pytest.approx(bounds.L_Psi, rel=1e-12)
    assert sum(decomposition["L_sm_terms"].values()) == pytest.approx(bounds.L_sm, rel=1e-12)
    assert sum(decomposition["L_Psi_terms"].values()) == pytest.approx(bounds.L_Psi, rel=1e-12)


# ---------------------------------------------------------------------------
# multiplier-head-specific bounds (experiments/multiplier_ablation.py,
# 2026-09-11): lambda_inf and lip_lambda must be correct for softplus and
# relu_square, not just the default square head every test above exercises.
# ---------------------------------------------------------------------------


def _make_triple_with_head(problem, head, seed=0):
    from certified_obstacle_nn.training.sampler import TrainingSampler
    from certified_obstacle_nn.training.trainer import train

    torch.manual_seed(seed)
    net_triple = NeuralTriple(problem, width=24, depth=3, multiplier_head=head)
    sampler = TrainingSampler(problem, n_interior=1000, n_initial=100, seed=seed)
    t, x, x0 = sampler.sample()
    train(problem, net_triple, t, x, x0, n_epochs=300, lr=1e-3)
    net_triple.eval()
    return net_triple


@pytest.mark.parametrize("head", ["square", "softplus", "relu_square"])
def test_lambda_inf_and_lip_lambda_per_head(problem, head):
    """Direct sampling check of lambda_inf/lip_lambda for EVERY head, not
    just the default 'square' the module-scoped `bounds` fixture uses --
    this is what makes certifying softplus/relu_square through the same
    certify_option_b pipeline as square legitimate (module docstring)."""
    triple = _make_triple_with_head(problem, head)
    bounds = compute_triple_bounds(problem, triple)
    assert bounds.multiplier_head == head

    t, x = _sample_domain(N_SUP_SAMPLES, seed=50)
    lam = triple.multiplier_net(t, x)
    _check_and_log(f"lambda_psi[{head}]", lam.abs().max().item(), bounds.lambda_inf)
    assert (lam >= -1e-12).all()  # nonnegativity holds for every head, structurally

    t2, x2 = _sample_domain(N_LIP_PAIRS, seed=51)

    def lambda_fn(tt, xx):
        return triple.multiplier_net(tt, xx)

    for eps, seed in [(1e-4, 52), (5e-2, 53)]:
        sampled = _empirical_scalar_lipschitz(lambda_fn, t2, x2, eps, seed)
        _check_and_log(f"Lip(lambda_psi)[{head}](eps={eps})", sampled, bounds.lip_lambda)


def test_softplus_lambda_inf_uses_additive_log2_not_square():
    """Confirms the two head-family formulas are actually DIFFERENT in code
    (not e.g. both silently falling through to the square formula): for the
    SAME Sigma_Mps/Pi_Mps, softplus's lambda_inf/lip_lambda must differ from
    square's by exactly the documented algebraic relationship."""
    problem = StefanProblem()
    square_triple = _make_triple_with_head(problem, "square", seed=1)
    softplus_triple = _make_triple_with_head(problem, "softplus", seed=1)  # SAME seed/arch/training data

    # These are different networks (different heads change training dynamics
    # too), so Sigma_Mps itself differs -- verify the FORMULA relationship
    # directly instead, using compute_network_bounds on each head's own net.
    square_bounds = compute_triple_bounds(problem, square_triple)
    softplus_bounds = compute_triple_bounds(problem, softplus_triple)

    sq_mb = square_bounds.multiplier_bounds
    sp_mb = softplus_bounds.multiplier_bounds

    assert square_bounds.lambda_inf == pytest.approx(sq_mb.output_bound ** 2)
    assert square_bounds.lip_lambda == pytest.approx(2.0 * sq_mb.output_bound * sq_mb.gradient_bound)

    assert softplus_bounds.lambda_inf == pytest.approx(sp_mb.output_bound + math.log(2.0))
    assert softplus_bounds.lip_lambda == pytest.approx(sp_mb.gradient_bound)


def test_relu_square_matches_square_formula_exactly():
    problem = StefanProblem()
    triple = _make_triple_with_head(problem, "relu_square", seed=2)
    bounds = compute_triple_bounds(problem, triple)
    mb = bounds.multiplier_bounds
    assert bounds.lambda_inf == pytest.approx(mb.output_bound ** 2)
    assert bounds.lip_lambda == pytest.approx(2.0 * mb.output_bound * mb.gradient_bound)
