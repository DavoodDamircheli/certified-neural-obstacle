"""Residual definitions shared by training and estimation, specification section 2.

These formulas are written against the problem's general (A, b, c), never
hardcoded to a specific problem's constants, so that training, the raw
estimator, and the Level-2 certificate are all testing/bounding the same
mathematical objects regardless of which problem is plugged in.

Two public entry points, single shared formula core (`assemble_residuals`), so
the R_PDE/R_sig/R_com/... algebra is written exactly once:

  compute_residuals               -- the TRAINING path. `state_net`, `flux_net`,
                                      `multiplier_net` are plain callables
                                      (t, x) -> tensor, not necessarily the
                                      trained nn.Module triple -- this is what
                                      lets tests drive this exact routine with
                                      a closed-form exact solution instead of a
                                      network. Derivatives come from
                                      compute_derivatives (create_graph=True).

  compute_residuals_certification -- the CERTIFICATION path. Takes the full
                                      NeuralTriple `model` (not three loose
                                      callables) and gets its derivatives from
                                      eval_triple_derivatives (create_graph=
                                      False, docs/history/MEMORY_OOM_DEBUGGING_AND_SAFE_RUN_POLICY.md
                                      section 1). Freezes `model`'s parameters
                                      on entry (mirrors the existing redundant-
                                      freeze pattern already used by
                                      cell_classifier.classify_cells and
                                      raw_estimator.compute_raw_estimator) so
                                      this is safely callable standalone, not
                                      only from an already-frozen certify()
                                      run.
"""

import math
from typing import Callable, NamedTuple

import torch

from certified_obstacle_nn.autograd.derivatives import compute_derivatives, eval_triple_derivatives
from certified_obstacle_nn.problems.base_problem import BaseProblem

NetFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class Residuals(NamedTuple):
    R_PDE: torch.Tensor
    R_sig: torch.Tensor
    R_0: torch.Tensor
    R_obs: torch.Tensor
    R_com: torch.Tensor
    gamma: torch.Tensor
    gamma_minus: torch.Tensor
    w: torch.Tensor


def assemble_residuals(
    problem: BaseProblem,
    t: torch.Tensor,
    x: torch.Tensor,
    x0: torch.Tensor,
    u: torch.Tensor,
    sigma: torch.Tensor,
    lam: torch.Tensor,
    u_t: torch.Tensor,
    u_x: torch.Tensor,
    sigma_x: torch.Tensor,
    u0_pred: torch.Tensor,
) -> Residuals:
    """The formula core, specification section 2, applied to whatever (u, sigma,
    lam, u_t, u_x, sigma_x, u0_pred) an already-run derivative pass produced
    -- this function itself takes no derivative and touches no network."""
    g = problem.g(t, x)
    gamma = g - u
    w = torch.relu(gamma)
    gamma_minus = torch.relu(-gamma)

    R_PDE = problem.f(t, x) - (u_t + sigma_x) - problem.b * u_x - problem.c * u + lam
    R_sig = sigma / math.sqrt(problem.A) + math.sqrt(problem.A) * u_x
    R_obs = w
    R_com = lam * gamma_minus
    R_0 = problem.u0(x0) - u0_pred

    return Residuals(
        R_PDE=R_PDE,
        R_sig=R_sig,
        R_0=R_0,
        R_obs=R_obs,
        R_com=R_com,
        gamma=gamma,
        gamma_minus=gamma_minus,
        w=w,
    )


def compute_residuals(
    problem: BaseProblem,
    state_net: NetFn,
    flux_net: NetFn,
    multiplier_net: NetFn,
    t: torch.Tensor,
    x: torch.Tensor,
    x0: torch.Tensor,
) -> Residuals:
    u, sigma, lam, u_t, u_x, sigma_x = compute_derivatives(state_net, flux_net, multiplier_net, t, x)
    u0_pred = state_net(torch.zeros_like(x0), x0)
    return assemble_residuals(problem, t, x, x0, u, sigma, lam, u_t, u_x, sigma_x, u0_pred)


def compute_residuals_certification(
    problem: BaseProblem,
    model,
    t: torch.Tensor,
    x: torch.Tensor,
    x0: torch.Tensor,
) -> Residuals:
    """Certification-only sibling of compute_residuals: same Residuals, same
    formulas (via assemble_residuals), derivatives from eval_triple_derivatives
    (create_graph=False) instead of compute_derivatives (create_graph=True).
    Every field of the returned Residuals is therefore already detached --
    no autograd graph escapes this function.
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    u, sigma, lam, u_t, u_x, sigma_x = eval_triple_derivatives(model, t, x)
    with torch.no_grad():
        u0_pred = model.state_net(torch.zeros_like(x0), x0)
    return assemble_residuals(problem, t, x, x0, u, sigma, lam, u_t, u_x, sigma_x, u0_pred.detach())
