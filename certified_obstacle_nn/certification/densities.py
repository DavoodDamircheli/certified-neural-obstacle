"""Level-2 densities, specification section 5.

phi_smooth, phi_interface, psi_majorant are three SEPARATE functions, each
independently callable and each recomputing whatever residual/derivative
pieces it needs from scratch -- deliberately not inlined into one combined
computation, so any one of them can be debugged or tested in isolation.

Facts used elsewhere but not re-derived here: grad_x w = 1_{gamma>0} grad_x
gamma, d_t w = 1_{gamma>0} d_t gamma, hence phi_interface = 1_{gamma>0} *
psi_majorant pointwise, so 0 <= phi_interface <= psi_majorant everywhere. A
violation of that inequality (checked in tests/test_densities.py) means
grad w / d_t w is being computed wrongly, not just imprecisely.

phi_smooth is Lipschitz on all of Qbar, including across the free boundary and
data interfaces; only phi_interface loses regularity there. psi_majorant is
the majorant used on sign-unresolved cells (specification section 7).

Two layers per density, so a chunked caller (raw_estimator.py,
level2_remainder.py) that has ALREADY paid for one eval_triple_derivatives /
eval_gap_derivatives pass per chunk can reuse those pieces instead of paying
for a second, redundant one:

  *_from_residuals(...)     -- the formula core, taking an already-computed
                                Residuals (and, for phi_interface/psi_majorant,
                                already-computed gamma_t/gamma_x). No network
                                evaluation, no autograd, here.
  phi_smooth/phi_interface/
  psi_majorant(problem, triple, t, x)
                                -- the standalone, independently-callable public
                                API (what tests/test_densities.py exercises):
                                runs eval_triple_derivatives (+
                                eval_gap_derivatives where needed) itself, then
                                delegates to the _from_residuals core. Safe to
                                call on an unfrozen triple -- freezing happens
                                inside _eval_residuals_and_derivatives, mirroring
                                the same redundant-freeze pattern already used
                                by cell_classifier.classify_cells and
                                raw_estimator.compute_raw_estimator.

Certification-only throughout: derivatives come from eval_triple_derivatives /
eval_gap_derivatives (create_graph=False), never compute_derivatives
(create_graph=True) -- that create_graph=True path is training-only
(docs/history/MEMORY_OOM_DEBUGGING_AND_SAFE_RUN_POLICY.md section 1), and computing gamma's
derivatives by backpropagating a second time through w/gamma (as an earlier
version of this module did) is not just wasteful but WRONG once u itself is
detached (eval_triple_derivatives' whole point) -- there would be no path left
for autograd to find du/dt through. eval_gap_derivatives sidesteps this
entirely: gamma_t = g_t - u_t, gamma_x = g_x - u_x, using already-known u_t/u_x
and a separate, tiny backward pass through the closed-form g alone.
"""

import torch

from certified_obstacle_nn.autograd.derivatives import eval_gap_derivatives, eval_triple_derivatives
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.base_problem import BaseProblem
from certified_obstacle_nn.residuals.residuals import Residuals, assemble_residuals


def phi_smooth_from_residuals(problem: BaseProblem, residuals: Residuals) -> torch.Tensor:
    """Phi_sm = (Lambda/alpha) R_PDE^2 + |R_sig|^2 + kappa w^2 + lambda_psi * gamma^-."""
    return (
        (problem.Lambda / problem.alpha) * residuals.R_PDE.pow(2)
        + residuals.R_sig.pow(2)
        + problem.kappa * residuals.w.pow(2)
        + residuals.R_com
    )


def phi_interface_from_residuals(
    problem: BaseProblem, residuals: Residuals, gamma_t: torch.Tensor, gamma_x: torch.Tensor
) -> torch.Tensor:
    """Phi_int = |A^{1/2} grad_x w|^2 + (Lambda/alpha) (d_t w)^2, via
    grad_x w = 1_{gamma>0} grad_x gamma, d_t w = 1_{gamma>0} d_t gamma."""
    active = (residuals.gamma > 0).to(gamma_t.dtype)
    w_t = active * gamma_t
    w_x = active * gamma_x
    return problem.A * w_x.pow(2) + (problem.Lambda / problem.alpha) * w_t.pow(2)


def psi_majorant_from_gap(problem: BaseProblem, gamma_t: torch.Tensor, gamma_x: torch.Tensor) -> torch.Tensor:
    """Psi = |A^{1/2} grad_x gamma|^2 + (Lambda/alpha) (d_t gamma)^2 (majorant)."""
    return problem.A * gamma_x.pow(2) + (problem.Lambda / problem.alpha) * gamma_t.pow(2)


def _eval_residuals_and_derivatives(problem: BaseProblem, triple: NeuralTriple, t: torch.Tensor, x: torch.Tensor):
    """Exactly ONE eval_triple_derivatives call, shared by all three public
    functions below -- phi_interface previously paid for two (one inside a
    compute_residuals_certification call, one of its own for u_t/u_x), a pure
    waste since both come from the same network evaluation. R_0 is unused by
    every density formula here, so `x0=t, u0_pred=u` are inert placeholders
    (computed and immediately discarded) rather than a second network call.
    Freezes `triple` on entry (mirrors classify_cells / compute_raw_estimator's
    own redundant-freeze pattern), so each function stays independently
    callable on an unfrozen triple, per this module's own docstring."""
    triple.eval()
    for p in triple.parameters():
        p.requires_grad_(False)
    u, sigma, lam, u_t, u_x, sigma_x = eval_triple_derivatives(triple, t, x)
    residuals = assemble_residuals(problem, t, x, t, u, sigma, lam, u_t, u_x, sigma_x, u0_pred=u)
    return residuals, u_t, u_x


def phi_smooth(problem: BaseProblem, triple: NeuralTriple, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    residuals, _, _ = _eval_residuals_and_derivatives(problem, triple, t, x)
    return phi_smooth_from_residuals(problem, residuals)


def phi_interface(problem: BaseProblem, triple: NeuralTriple, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    residuals, u_t, u_x = _eval_residuals_and_derivatives(problem, triple, t, x)
    gamma_t, gamma_x = eval_gap_derivatives(problem, t, x, u_t, u_x)
    return phi_interface_from_residuals(problem, residuals, gamma_t, gamma_x)


def psi_majorant(problem: BaseProblem, triple: NeuralTriple, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    _, u_t, u_x = _eval_residuals_and_derivatives(problem, triple, t, x)
    gamma_t, gamma_x = eval_gap_derivatives(problem, t, x, u_t, u_x)
    return psi_majorant_from_gap(problem, gamma_t, gamma_x)


def psi_majorant_and_R_PDE(
    problem: BaseProblem, triple: NeuralTriple, t: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """(psi, R_PDE) from ONE eval_triple_derivatives call -- for
    level2_remainder.py's shared midpoint pass (OPTION_B_..._PLAN.md addendum
    A5), which needs both Psi(z_K) (E_h_unr) and R_PDE(z_K) (Option B's
    m_R_h) at the same cell midpoints. `_eval_residuals_and_derivatives`
    already computes `residuals` (hence residuals.R_PDE) as a side effect of
    getting u_t/u_x for psi_majorant -- the plain `psi_majorant` above
    discards it; this sibling keeps it, so a caller needing both never pays
    for two network evaluations at the same points.
    """
    residuals, u_t, u_x = _eval_residuals_and_derivatives(problem, triple, t, x)
    gamma_t, gamma_x = eval_gap_derivatives(problem, t, x, u_t, u_x)
    psi = psi_majorant_from_gap(problem, gamma_t, gamma_x)
    return psi, residuals.R_PDE


def compute_boundary_densities(
    problem: BaseProblem, triple: NeuralTriple, x0: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (Phi_0, Phi_T), each evaluated at x0 (no derivatives needed)."""
    zeros_t = torch.zeros_like(x0)
    T_t = torch.full_like(x0, problem.T)

    with torch.no_grad():
        u0_pred = triple.state_net(zeros_t, x0)
        Phi_0 = (problem.u0(x0) - u0_pred).pow(2) + torch.relu(problem.g(zeros_t, x0) - u0_pred).pow(2)

        uT_pred = triple.state_net(T_t, x0)
        Phi_T = torch.relu(problem.g(T_t, x0) - uT_pred).pow(2)

    return Phi_0, Phi_T
