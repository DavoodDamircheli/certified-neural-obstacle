"""Adam + L-BFGS trainer, specification section 3.

Adam runs first; L-BFGS then refines the SAME triple, warm-started from
wherever Adam left off (no reset -- it is simply the same parameters,
continuing to be optimized), using a fixed closure and strong_wolfe line
search. Every loss component is logged every epoch in both phases -- never
only the total -- to a single pandas DataFrame with a `phase` column
distinguishing the two, then optionally written to CSV.

L-BFGS is aborted -- not silently absorbed -- on the usual failure modes: a
non-finite (NaN/Inf) loss, or a line search that fails to make progress
despite a nonzero gradient (which strong_wolfe can do once the L-BFGS
curvature history stops being a good model of the objective). The step that
triggered the failure is never appended to the history; the caller gets back
whatever clean history was collected up to that point plus an LBFGSOutcome
explaining why it stopped.

This module has no knowledge of certification/ at all: it only ever touches
the training collocation points handed to it and the neural triple's own
parameters.
"""

import math
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import torch

from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.base_problem import BaseProblem
from certified_obstacle_nn.residuals.residuals import compute_residuals
from certified_obstacle_nn.training.losses import LossComponents, compute_loss


class LBFGSOutcome(NamedTuple):
    reason: str
    n_steps_completed: int


def _loss_record(epoch: int, phase: str, loss: LossComponents) -> dict:
    return {
        "epoch": epoch,
        "phase": phase,
        "loss_total": loss.total.item(),
        "loss_pde": loss.pde.item(),
        "loss_sigma": loss.sigma.item(),
        "loss_initial": loss.initial.item(),
        "loss_penetration": loss.penetration.item(),
        "loss_complementarity": loss.complementarity.item(),
    }


def _total_grad_norm(triple: NeuralTriple) -> float:
    total = 0.0
    for p in triple.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return math.sqrt(total)


def train(
    problem: BaseProblem,
    triple: NeuralTriple,
    t: torch.Tensor,
    x: torch.Tensor,
    x0: torch.Tensor,
    n_epochs: int,
    lr: float = 1e-3,
    n_lbfgs_epochs: int = 0,
    lbfgs_lr: float = 1.0,
    w_r: float = 1.0,
    w_s: float = 1.0,
    w_0: float = 1.0,
    w_p: float = 1.0,
    w_c: float = 1.0,
    csv_path: str | None = None,
    monitor=None,
) -> tuple[pd.DataFrame, LBFGSOutcome]:
    """monitor (v20, optional, read-only): callable(step, phase, loss) invoked after Adam steps 0, 500, 1000, ...,
    after the last Adam step and after L-BFGS. It must not touch parameters, gradients or any RNG; with
    monitor=None the run is unchanged bit for bit."""
    weights = dict(w_r=w_r, w_s=w_s, w_0=w_0, w_p=w_p, w_c=w_c)
    records: list[dict] = []

    # --- phase 1: Adam ---
    adam_optimizer = torch.optim.Adam(triple.parameters(), lr=lr)
    for epoch in range(n_epochs):
        adam_optimizer.zero_grad()
        residuals = compute_residuals(problem, triple.state_net, triple.flux_net, triple.multiplier_net, t, x, x0)
        loss = compute_loss(residuals, **weights)
        loss.total.backward()
        adam_optimizer.step()
        records.append(_loss_record(epoch, "adam", loss))
        if monitor is not None and (epoch % 500 == 0 or epoch == n_epochs - 1):
            monitor(epoch, "adam", loss)

    # --- phase 2: L-BFGS, warm-started from the Adam state above ---
    lbfgs_optimizer = torch.optim.LBFGS(
        triple.parameters(),
        lr=lbfgs_lr,
        max_iter=1,
        line_search_fn="strong_wolfe",
    )

    holder: dict = {}

    def closure():
        lbfgs_optimizer.zero_grad()
        residuals = compute_residuals(problem, triple.state_net, triple.flux_net, triple.multiplier_net, t, x, x0)
        loss = compute_loss(residuals, **weights)
        loss.total.backward()
        holder["loss"] = loss
        return loss.total

    steps_completed = 0
    prev_total: float | None = None
    reason = "skipped"

    for step_idx in range(n_lbfgs_epochs):
        try:
            lbfgs_optimizer.step(closure)
            # strong_wolfe's internal line-search trials evaluate the closure
            # at temporary points it then undoes, so the last call captured
            # in `holder` during step() is not reliably the accepted point.
            # Re-evaluate once, explicitly, at the now-finalized parameters.
            closure()
        except RuntimeError as exc:
            reason = f"runtime_error: {exc}"
            break

        loss = holder.get("loss")
        if loss is None or not torch.isfinite(loss.total):
            reason = "non_finite_loss"
            break

        total = loss.total.item()
        grad_norm = _total_grad_norm(triple)

        if prev_total is not None:
            if total > prev_total + 1e-9:
                reason = "loss_increased"
                break
            if abs(total - prev_total) < 1e-12 and grad_norm > 1e-6:
                reason = "line_search_stalled"
                break

        prev_total = total
        steps_completed += 1
        records.append(_loss_record(n_epochs + step_idx, "lbfgs", loss))
    else:
        reason = "completed" if n_lbfgs_epochs > 0 else "skipped"

    lbfgs_outcome = LBFGSOutcome(reason=reason, n_steps_completed=steps_completed)
    if monitor is not None:
        monitor(n_epochs + steps_completed, "lbfgs_end", holder.get("loss"))

    history = pd.DataFrame.from_records(records)

    if csv_path is not None:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        history.to_csv(path, index=False)

    return history, lbfgs_outcome
