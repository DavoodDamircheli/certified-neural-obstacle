"""NEW_RUNS.md Experiment 2, sections 2.3-2.4 -- kink-aware collocation for
the American put.

Two independent mechanisms, matching NEW_RUNS.md's own menu exactly:

  2.3 (static): sample_interior_with_kink_band draws a fixed fraction of
  interior points from a narrow band around x_star=log(K) (the payoff kink),
  the remainder uniform over the whole domain -- "a fixed fraction of points
  drawn from a narrow band around the kink," explicitly NOT all points at the
  strike itself.

  2.4 (dynamic): run_contact_set_adaptive re-uses training/adaptive.py's own
  validated run_adaptive_loop (Dorfler marking on rho_penetration_K -- the
  obstacle-violation density R_obs=w=(g-u)^+, a function of the network's OWN
  current state only, never the reference solution) via its new
  initial_points argument, warm-started from the 2.3 sample. mark_on=
  "rho_penetration_K" IS the "near the learned contact set" signal NEW_RUNS.md
  asks for: it is large exactly where the network's current prediction is
  close to or violating the obstacle, i.e. near ITS OWN current estimate of
  the exercise region -- and never touches problem.free_boundary_exact /
  the frozen reference (American put has no closed-form free boundary at all;
  the frozen PSOR reference is loaded only by errors/true_error.py's own
  E_true computation, never by anything in this file), satisfying "the
  sampling procedure must remain independent of the reference boundary" by
  construction, not by a checked-but-separate promise.
"""

from __future__ import annotations

import torch

from certified_obstacle_nn.certification.partition import Partition
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.base_problem import BaseProblem
from certified_obstacle_nn.training.adaptive import AdaptiveIterationRecord, run_adaptive_loop

BAND_FRACTION = 0.3  # engineering choice, documented here (not a specification formula): 30% of the interior budget goes to the kink band.
BAND_HALFWIDTH = 0.4  # log-price units; domain width R-L=8, so 2*0.4=0.8 is 10% of the domain -- "narrow", not "all points at the strike."


def sample_interior_with_kink_band(
    problem: BaseProblem, n_interior: int, n_initial: int, seed: int,
    band_fraction: float = BAND_FRACTION, band_halfwidth: float = BAND_HALFWIDTH,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_star = getattr(problem, "x_star", None)
    assert x_star is not None, "sample_interior_with_kink_band requires a problem with a known kink location (problem.x_star)"

    generator = torch.Generator().manual_seed(seed)
    a, b = problem.omega_bounds
    T = problem.T

    n_band = int(round(band_fraction * n_interior))
    n_uniform = n_interior - n_band

    t_uniform = torch.rand(n_uniform, 1, generator=generator, dtype=torch.float64) * T
    x_uniform = torch.rand(n_uniform, 1, generator=generator, dtype=torch.float64) * (b - a) + a

    lo, hi = max(a, x_star - band_halfwidth), min(b, x_star + band_halfwidth)
    t_band = torch.rand(n_band, 1, generator=generator, dtype=torch.float64) * T
    x_band = torch.rand(n_band, 1, generator=generator, dtype=torch.float64) * (hi - lo) + lo

    t = torch.cat([t_uniform, t_band], dim=0)
    x = torch.cat([x_uniform, x_band], dim=0)
    x0 = torch.rand(n_initial, 1, generator=generator, dtype=torch.float64) * (b - a) + a

    return t.requires_grad_(True), x.requires_grad_(True), x0


def run_contact_set_adaptive(
    problem: BaseProblem, triple: NeuralTriple, partition: Partition, *,
    n_interior: int, n_initial: int, l_max: int, seed: int,
    n_adam_epochs: int, adam_lr: float, n_lbfgs_epochs: int, lbfgs_lr: float,
    theta: float = 0.5, n_qp: int = 4, alpha_marked_fraction: float = 0.7,
    band_fraction: float = BAND_FRACTION, band_halfwidth: float = BAND_HALFWIDTH,
) -> tuple[NeuralTriple, list[AdaptiveIterationRecord]]:
    initial = sample_interior_with_kink_band(problem, n_interior, n_initial, seed, band_fraction, band_halfwidth)
    return run_adaptive_loop(
        problem, triple, partition,
        method="dorfler", theta=theta, mark_on="rho_penetration_K",
        n_interior=n_interior, n_initial=n_initial, l_max=l_max, seed=seed,
        n_adam_epochs=n_adam_epochs, adam_lr=adam_lr, n_lbfgs_epochs=n_lbfgs_epochs, lbfgs_lr=lbfgs_lr,
        n_qp=n_qp, alpha_marked_fraction=alpha_marked_fraction,
        initial_points=initial,
    )
