"""Adaptive collocation refinement (2026-09-11), specification section 7's own
candidate target ("a candidate target for the adaptive refinement phase
(training/adaptive.py)").

Loop: Train -> Estimate rho_K (per-cell) -> Mark -> Resample -> warm-start
retrain. The partition used for per-cell rho_K estimates is FIXED across
every iteration (the SAME certification/partition.py grid throughout --
"adaptive" here refers to the COLLOCATION points' distribution over Q, never
to the certification partition's own resolution, which stays constant), and
its quadrature points are checked disjoint from the CURRENT collocation set
at every iteration via compute_raw_estimator's own `training_points`
argument (specification section 12 invariant 1). The fixed partition's nodes are
deterministic grid midpoints and the collocation points are continuous
random draws, so disjointness holds by construction with probability 1 --
the check still runs explicitly every iteration, not assumed.

IMPORTANT gotcha this module works around: compute_raw_estimator calls
_freeze(triple) internally (raw_estimator.py's own helper: triple.eval() +
requires_grad_(False) on every parameter), since certification code must
never need gradients w.r.t. parameters. But the adaptive loop's whole point
is to CONTINUE training the SAME triple after estimation -- so every
iteration here explicitly restores requires_grad_(True) on all parameters
immediately after the estimation call, before the next warm-start retrain.
Skipping this would silently produce zero gradients (Adam would run every
step with grad=None/0) from iteration 1 onward -- tests/test_adaptive.py's
test_warmstart_actually_updates_parameters exists specifically to catch a
regression here.

Marking strategies (both act on a caller-chosen per-cell column -- the
combined rho_total_K by default, or rho_penetration_K / rho_complementarity_K
for the component-marking ablation in experiments/stefan_adaptive.py):

  Dorfler(theta): the classical greedy marking -- sort cells by rho_K^2
  descending, mark the smallest prefix whose SUMMED rho_K^2 is >= theta *
  total rho_K^2. theta=0.5 (user-specified) marks "just over half the
  estimator's energy."

  Probabilistic(q): P(K) proportional to rho_K^q (q=1 or 2, user-specified)
  -- a SOFT alternative to Dorfler's hard threshold: every cell gets some
  resampling probability weighted by its own estimator value, no marked/
  unmarked binary split.

Resample: draws N_TOTAL (identical every iteration -- see
experiments/stefan_adaptive.py's own N_total_adaptive==N_total_uniform
assertion) new (t,x) points as a MIXTURE: an ALPHA_MARKED_FRACTION share
drawn by (1) choosing a cell according to the marking weights (Dorfler:
uniform over the marked set; Probabilistic: the P(K) distribution itself),
then (2) drawing uniformly within that cell's own (dt,dx) rectangle; the
remaining (1-ALPHA_MARKED_FRACTION) share drawn uniformly over the WHOLE
domain, preserving global coverage (a marked/high-weight set need not touch
every boundary or far-field region, and should not starve them entirely).
ALPHA_MARKED_FRACTION=0.7 is this module's own default -- an engineering
choice, not a specification formula, documented here since it materially shapes
resampling behavior.

Per specification section 12 invariant 2, the FULL certified bound (E_cert_U,
via certify_option_b) is never computed inside this loop -- only the raw
per-cell estimator (compute_raw_estimator), which the algorithm needs at
every iteration to decide where to resample and cannot reasonably be pushed
out to a separate process each time. experiments/stefan_adaptive.py invokes
the full certification through certify_cli.py/true_error_cli.py as separate
subprocesses on each method's FINAL trained weights, exactly as
experiments/stefan_core.py and stefan_stress.py already do.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from certified_obstacle_nn.certification.partition import Partition
from certified_obstacle_nn.certification.raw_estimator import DEFAULT_MAX_RSS_GB, DEFAULT_POINTS_PER_CHUNK, compute_raw_estimator
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.base_problem import BaseProblem
from certified_obstacle_nn.training.trainer import LBFGSOutcome, train

ALPHA_MARKED_FRACTION = 0.7
VALID_MARKING_METHODS = ("dorfler", "probabilistic")


def dorfler_mark(rho_K: np.ndarray, theta: float = 0.5) -> np.ndarray:
    """Boolean mask, True for cells in the Dorfler-marked set: the smallest
    greedy prefix (by descending rho_K^2) whose summed rho_K^2 is
    >= theta * sum(rho_K^2)."""
    rho_sq = np.asarray(rho_K, dtype=np.float64) ** 2
    order = np.argsort(-rho_sq)
    cumsum = np.cumsum(rho_sq[order])
    total = rho_sq.sum()
    target = theta * total
    n_marked = int(np.searchsorted(cumsum, target) + 1) if total > 0 else 1
    n_marked = min(max(n_marked, 1), len(rho_K))
    mask = np.zeros(len(rho_K), dtype=bool)
    mask[order[:n_marked]] = True
    return mask


def probabilistic_weights(rho_K: np.ndarray, q: float) -> np.ndarray:
    """P(K) ~ rho_K^q, normalized to sum to 1. Falls back to uniform if
    every rho_K is exactly zero (degenerate, should not occur in practice)."""
    w = np.abs(np.asarray(rho_K, dtype=np.float64)) ** q
    total = w.sum()
    if total <= 0:
        return np.full(len(rho_K), 1.0 / len(rho_K))
    return w / total


def cell_weights_from_marking(rho_K: np.ndarray, method: str, theta: float = 0.5, q: float = 1.0) -> np.ndarray:
    """Unifies both marking strategies into one per-cell weight vector
    (summing to 1) so a single resample_points call handles either."""
    if method == "dorfler":
        mask = dorfler_mark(rho_K, theta=theta)
        n_marked = int(mask.sum())
        return np.where(mask, 1.0 / n_marked, 0.0)
    elif method == "probabilistic":
        return probabilistic_weights(rho_K, q=q)
    raise ValueError(f"unknown marking method {method!r}, expected one of {VALID_MARKING_METHODS}")


def resample_points(
    partition: Partition,
    weights: np.ndarray,
    n_total: int,
    generator: torch.Generator,
    alpha_marked_fraction: float = ALPHA_MARKED_FRACTION,
) -> tuple[torch.Tensor, torch.Tensor]:
    """n_total points total (exact -- callers rely on this for the
    N_total_adaptive==N_total_uniform invariant), split as
    alpha_marked_fraction from the weighted-cell mixture component and the
    rest uniform over the whole domain."""
    n_weighted = int(round(alpha_marked_fraction * n_total))
    n_uniform = n_total - n_weighted

    weights_t = torch.tensor(weights, dtype=torch.float64)
    parts_t, parts_x = [], []

    if n_weighted > 0:
        cell_idx = torch.multinomial(weights_t, n_weighted, replacement=True, generator=generator)
        t_centers = partition.t_center[cell_idx]
        x_centers = partition.x_center[cell_idx]
        u_t = (torch.rand(n_weighted, generator=generator, dtype=torch.float64) - 0.5) * partition.dt
        u_x = (torch.rand(n_weighted, generator=generator, dtype=torch.float64) - 0.5) * partition.dx
        parts_t.append(t_centers + u_t)
        parts_x.append(x_centers + u_x)

    if n_uniform > 0:
        a = float(partition.x_edges[0])
        b = float(partition.x_edges[-1])
        T = float(partition.t_edges[-1])
        parts_t.append(torch.rand(n_uniform, generator=generator, dtype=torch.float64) * T)
        parts_x.append(torch.rand(n_uniform, generator=generator, dtype=torch.float64) * (b - a) + a)

    t = torch.cat(parts_t).reshape(-1, 1)
    x = torch.cat(parts_x).reshape(-1, 1)
    return t, x


@dataclass(frozen=True)
class AdaptiveIterationRecord:
    iteration: int
    n_marked: int  # 0 for probabilistic marking (no hard marked/unmarked split)
    rho_tilde_estimate: float  # informal per-iteration estimate from the SAME fixed partition -- not a final certified value
    t_points: torch.Tensor  # collocation points TRAINED ON this iteration (for Figure 7 snapshots)
    x_points: torch.Tensor
    training_time: float
    lbfgs_outcome: LBFGSOutcome


def _unfreeze(triple: NeuralTriple) -> None:
    """Undoes compute_raw_estimator's internal _freeze -- see module
    docstring's "IMPORTANT gotcha" section. triple.train() is a no-op for
    behavior here (tanh MLPs have no dropout/batchnorm) but is called for
    symmetry with .eval(); requires_grad_(True) is the part that actually
    matters for the next warm-start retrain to update parameters at all."""
    triple.train()
    for p in triple.parameters():
        p.requires_grad_(True)


def run_adaptive_loop(
    problem: BaseProblem,
    triple: NeuralTriple,
    partition: Partition,
    *,
    method: str,
    n_interior: int,
    n_initial: int,
    l_max: int,
    seed: int,
    n_adam_epochs: int,
    adam_lr: float,
    n_lbfgs_epochs: int,
    lbfgs_lr: float,
    theta: float = 0.5,
    q: float = 1.0,
    mark_on: str = "rho_total_K",
    n_qp: int = 4,
    points_per_chunk: int = DEFAULT_POINTS_PER_CHUNK,
    max_rss_gb: float = DEFAULT_MAX_RSS_GB,
    alpha_marked_fraction: float = ALPHA_MARKED_FRACTION,
    initial_points: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[NeuralTriple, list[AdaptiveIterationRecord]]:
    """Runs l_max+1 rounds of Train -> Estimate -> Mark -> Resample (the
    last round trains but does not mark/resample -- nothing downstream would
    use it). `triple` is trained IN PLACE (mutated) and also returned for
    convenience. Iteration 0's collocation set is a plain uniform sample by
    default (there is no rho_K yet); every later iteration's set comes from
    resample_points using the PRIOR iteration's estimator.

    initial_points (NEW_RUNS.md Experiment 2, 2026-09-13): optional
    (t, x, x0) override for iteration 0, e.g. a strike-band-stratified sample
    (experiments/new_runs/american_put_sampler.py) for problems with a known
    kink (American put's payoff at x=log(K)) that a plain uniform draw would
    under-resolve before the first round's own estimator has a chance to see
    it. Default None preserves the EXACT prior uniform-draw behavior (every
    existing Stefan call site omits this argument, so nothing about an
    already-published run changes)."""
    if method not in VALID_MARKING_METHODS:
        raise ValueError(f"unknown marking method {method!r}, expected one of {VALID_MARKING_METHODS}")

    generator = torch.Generator().manual_seed(seed)
    a, b = problem.omega_bounds
    T = problem.T

    if initial_points is not None:
        t, x, x0 = initial_points
        assert t.shape[0] == n_interior and x.shape[0] == n_interior and x0.shape[0] == n_initial, (
            "initial_points must match (n_interior, n_interior, n_initial)"
        )
        t, x, x0 = t.detach().clone(), x.detach().clone(), x0.detach().clone()
    else:
        t = torch.rand(n_interior, 1, generator=generator, dtype=torch.float64) * T
        x = torch.rand(n_interior, 1, generator=generator, dtype=torch.float64) * (b - a) + a
        x0 = torch.rand(n_initial, 1, generator=generator, dtype=torch.float64) * (b - a) + a

    history: list[AdaptiveIterationRecord] = []

    for it in range(l_max + 1):
        t_req = t.clone().requires_grad_(True)
        x_req = x.clone().requires_grad_(True)

        wall_start = time.perf_counter()
        _, outcome = train(
            problem, triple, t_req, x_req, x0,
            n_epochs=n_adam_epochs, lr=adam_lr,
            n_lbfgs_epochs=n_lbfgs_epochs, lbfgs_lr=lbfgs_lr,
        )
        training_time = time.perf_counter() - wall_start

        per_cell_df, summary = compute_raw_estimator(
            problem, triple, partition, n_qp=n_qp,
            training_points=(t.detach(), x.detach()),
            points_per_chunk=points_per_chunk, max_rss_gb=max_rss_gb,
        )
        _unfreeze(triple)  # undo compute_raw_estimator's internal _freeze -- see module docstring

        n_marked = 0
        t_this_iter, x_this_iter = t_req.detach().clone(), x_req.detach().clone()

        if it < l_max:
            rho_K = per_cell_df[mark_on].to_numpy()
            weights = cell_weights_from_marking(rho_K, method=method, theta=theta, q=q)
            if method == "dorfler":
                n_marked = int((weights > 0).sum())
            t, x = resample_points(partition, weights, n_interior, generator, alpha_marked_fraction)
            assert t.shape[0] == n_interior and x.shape[0] == n_interior, "resample_points must return exactly n_interior points"

        history.append(AdaptiveIterationRecord(
            iteration=it, n_marked=n_marked, rho_tilde_estimate=summary.rho_tilde,
            t_points=t_this_iter, x_points=x_this_iter,
            training_time=training_time, lbfgs_outcome=outcome,
        ))

    return triple, history
