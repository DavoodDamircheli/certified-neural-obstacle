"""Post-hoc network corruptions for stress-testing the certificate
(experiments/stefan_stress.py S2/S3, 2026-09-11).

Both corruptions ADD a simple, closed-form, exactly-boundable function to
one sub-network's OUTPUT, after training -- never touching the trained
weights themselves. Since the a priori certification bounds
(certification/network_bounds.py) are computed purely from sub-network
WEIGHTS, a post-hoc output addition is invisible to them unless corrected
by hand: this module provides that correction, using ONLY elementary,
EXACT (non-approximate) real-analysis facts --

    sup(f+g)_inf <= sup(f)_inf + sup(g)_inf         (sup-norm subadditivity)
    Lip(f+g)     <= Lip(f) + Lip(g)                  (Lipschitz subadditivity)

-- plus, for the flux corruption specifically, the EXACT (not bounded, not
approximated) fact that d_x a(t) = 0 when a depends on t alone, so
div_x(sigma+a) = div_x(sigma) IDENTICALLY: no correction is needed for
div_x_sigma_inf, Pi_Sph, or any quantity built only from Pi_Sph (m_R's
d*Pi_Sph term, lip_R_sig, lip_R_PDE's lip_div_x_sigma term). This is why the
stress test's qualitative claim ("since d_x a=0 the PDE balance barely
moves") is also mathematically forced at the level of the a priori bounds
themselves, not just an empirical observation about R_PDE's sampled values.

WHY THE CORRECTION MATTERS (not just pedantry). certify_option_b's Option B
mode (min_apriori_and_partition, certification/certified_residual_bound.py)
takes m_R_cert = min(m_R_apriori, m_R_h). The a posteriori m_R_h is computed
by literally EVALUATING the (wrapped, corrupted) triple at sample points, so
it is automatically correct regardless of this module. But m_R_apriori comes
from compute_triple_bounds(triple) -- called on the UNCORRUPTED base
triple's weights, the only thing it can see, it would silently UNDER-
estimate m_R for the corrupted network. If that under-estimate happened to
be smaller than the true (corrected) m_R_h, Option B's min() would pick the
WRONG, invalid a priori value, and E_cert_U could fail to bound E_true --
defeating the entire point of the stress test ("the certificate must remain
valid on defective networks"). The corrections here are what make
m_R_apriori (and L_sm, hence E_h, hence E_cert_U) valid for the ACTUAL
corrupted network being certified, not just for the network whose weights
happen to be on disk.

Recomputing L_sm: rather than hand-duplicating specification section 6's 5-term
L_sm formula (risking a transcription error), each corrected-bounds function
below patches ONLY the constituent fields the corruption actually touches,
then calls the already-tested decompose_L_sm_and_L_Psi(problem, bounds) and
takes its L_sm_recomputed -- the SAME function certify_option_b itself calls
internally, so the two are guaranteed consistent by construction (and
certify_option_b's own internal assert, T1_old+other_terms_sum==bounds.L_sm,
will hold automatically for a bounds object built this way).
"""

import math
from dataclasses import replace

import torch

from certified_obstacle_nn.certification.network_bounds import TripleBounds, decompose_L_sm_and_L_Psi
from certified_obstacle_nn.problems.base_problem import BaseProblem


class _AddLinearInT:
    """Wraps a flux_net-shaped callable: output -> output + a_coefficient*t.
    d_x(a_coefficient*t) = 0 exactly -- see module docstring."""

    def __init__(self, base_flux_net, a_coefficient: float):
        self.base_flux_net = base_flux_net
        self.a_coefficient = a_coefficient

    def __call__(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.base_flux_net(t, x) + self.a_coefficient * t


class _AddReluBump:
    """Wraps a multiplier_net-shaped callable: output -> output +
    bump_height*relu(t-x). relu(t-x) >= 0 everywhere, so lambda_psi's
    nonnegativity is preserved even though this post-hoc addition breaks the
    exact M_psi^2 structure -- only nonnegativity is needed for the bound
    machinery below, not the specific squared-head construction."""

    def __init__(self, base_multiplier_net, bump_height: float):
        self.base_multiplier_net = base_multiplier_net
        self.bump_height = bump_height

    def __call__(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.base_multiplier_net(t, x) + self.bump_height * torch.clamp(t - x, min=0.0)


class CorruptedTriple:
    """Duck-typed triple: state_net untouched, exactly ONE of flux_net /
    multiplier_net wrapped with a post-hoc addition. .eval()/.parameters()
    delegate to the base triple -- the corruption itself has no learnable
    parameters (it is a FIXED, non-trained perturbation, consistent with
    "post-hoc"), so there is nothing extra to freeze or assert frozen."""

    def __init__(self, base_triple, flux_net=None, multiplier_net=None):
        self.base_triple = base_triple
        self.state_net = base_triple.state_net
        self.flux_net = flux_net if flux_net is not None else base_triple.flux_net
        self.multiplier_net = multiplier_net if multiplier_net is not None else base_triple.multiplier_net

    def eval(self):
        self.base_triple.eval()

    def parameters(self):
        return self.base_triple.parameters()


def make_flux_add_linear_t_triple(base_triple, a_coefficient: float) -> CorruptedTriple:
    return CorruptedTriple(base_triple, flux_net=_AddLinearInT(base_triple.flux_net, a_coefficient))


def make_lambda_add_relu_bump_triple(base_triple, bump_height: float) -> CorruptedTriple:
    return CorruptedTriple(base_triple, multiplier_net=_AddReluBump(base_triple.multiplier_net, bump_height))


def corrected_bounds_flux_add_linear_t(base_bounds: TripleBounds, problem: BaseProblem, a_coefficient: float) -> TripleBounds:
    """a(t) = a_coefficient * t on [0, T]: ||a||_inf = |a_coefficient|*T
    (exact, sup attained at t=T); Lip(a) = |a_coefficient| (exact -- a is
    linear in t alone, d_x a = 0, Euclidean (t,x)-combination convention
    matches specification section 6's addendum)."""
    T = problem.T
    a_inf = abs(a_coefficient) * T

    intermediate = replace(
        base_bounds,
        sigma_inf=base_bounds.sigma_inf + a_inf,
        m_sig=base_bounds.m_sig + problem.alpha ** -0.5 * a_inf,
        # div_x_sigma_inf, lip_R_sig, lip_div_x_sigma, lip_R_PDE, m_R: UNCHANGED
        # (d_x a = 0 exactly -- see module docstring).
    )
    L_sm_new = decompose_L_sm_and_L_Psi(problem, intermediate)["L_sm_recomputed"]
    return replace(intermediate, L_sm=L_sm_new)


def corrected_bounds_lambda_add_relu_bump(base_bounds: TripleBounds, problem: BaseProblem, bump_height: float) -> TripleBounds:
    """bump(t,x) = bump_height * relu(t-x) on [0,1]^2: ||bump||_inf =
    bump_height (exact, sup attained at the corner t=1,x=0); Lip(bump) =
    bump_height*sqrt(2) (exact -- grad(relu(t-x)) = (1,-1) where t>x,
    Euclidean norm sqrt(2); 0 elsewhere)."""
    bump_inf = bump_height
    bump_lip = bump_height * math.sqrt(2.0)

    intermediate = replace(
        base_bounds,
        lambda_inf=base_bounds.lambda_inf + bump_inf,
        m_R=base_bounds.m_R + bump_inf,
        m_lam=base_bounds.m_lam + bump_inf,
        lip_lambda=base_bounds.lip_lambda + bump_lip,
        lip_R_PDE=base_bounds.lip_R_PDE + bump_lip,
        # L_Psi, M_Psi: UNCHANGED (neither depends on lambda_psi at all,
        # only on gamma's own spatial/temporal Lipschitz constants and A_inf).
    )
    L_sm_new = decompose_L_sm_and_L_Psi(problem, intermediate)["L_sm_recomputed"]
    return replace(intermediate, L_sm=L_sm_new)
