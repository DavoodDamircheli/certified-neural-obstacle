"""NEW_RUNS.md Experiment 3 -- certificate-aware / Lipschitz-controlled
training loop.

training/trainer.py's own train() is NOT modified. This is a standalone,
parallel Adam+L-BFGS loop (same structure, same stall/failure-detection
logic, same hyperparameter names) that adds one extra scalar term,
gamma * R_Lip, to what is actually backpropagated -- an additive,
opt-in regularizer belonging to this experiment, not a change to the
validated core trainer every other experiment depends on.

R_Lip := sum over the three sub-networks (state, flux, multiplier) of the
PRODUCT of that sub-network's own per-layer operator (spectral) norms
||W_1||_2 * ||W_2||_2 * ... -- exactly Pi_N (specification section 6), the
quantity that already appears in Lip(R_PDE)=L_R and hence in
T1_pde_residual, the term specification's own section 8 addendum identifies as
the dominant driver of E_h_lip's blowup at larger architectures. Penalizing
it directly targets the diagnosed cause.

_differentiable_spectral_product is a DELIBERATE, DOCUMENTED duplicate of
certification/network_bounds.py's spectral_norms: that function is
frozen/no_grad by design (specification section 12 invariant 2 -- no gradient may
flow inside certification/ code), so it cannot be reused for a training-time
penalty; this training-time twin lives here instead of that invariant being
loosened.

gamma=0 safety: at gamma=0, the penalty's OWN gradient contribution must be
exactly zero, not merely small -- but torch.linalg.matrix_norm's backward
(through an SVD) can produce Inf at repeated/near-repeated singular values
even when the incoming upstream gradient is zero (0*Inf = NaN in IEEE754).
To guarantee gamma=0 is training/trainer.py-BYTE-IDENTICAL (never a source of
a stray NaN), the penalty is computed under torch.no_grad() (logged, not
backpropagated) whenever gamma==0.0, and the backward pass only ever sees
`loss.total` in that case, exactly as training/trainer.py's own train() does.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import torch

from certified_obstacle_nn.models.mlp import MLP
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.base_problem import BaseProblem
from certified_obstacle_nn.residuals.residuals import compute_residuals
from certified_obstacle_nn.training.losses import compute_loss
from certified_obstacle_nn.training.trainer import LBFGSOutcome


def _total_grad_norm(triple: NeuralTriple) -> float:
    total = 0.0
    for p in triple.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return math.sqrt(total)


def _differentiable_spectral_product(mlp: MLP) -> torch.Tensor:
    prod = None
    for w in mlp.weights:
        norm = torch.linalg.matrix_norm(w, ord=2)
        prod = norm if prod is None else prod * norm
    return prod


def lipschitz_penalty(triple: NeuralTriple) -> torch.Tensor:
    return (
        _differentiable_spectral_product(triple.state_net.net)
        + _differentiable_spectral_product(triple.flux_net.net)
        + _differentiable_spectral_product(triple.multiplier_net.net)
    )


def _loss_record(epoch: int, phase: str, loss, penalty_value: float, gamma: float) -> dict:
    return {
        "epoch": epoch, "phase": phase,
        "loss_total_unregularized": loss.total.item(),
        "loss_pde": loss.pde.item(), "loss_sigma": loss.sigma.item(),
        "loss_initial": loss.initial.item(), "loss_penetration": loss.penetration.item(),
        "loss_complementarity": loss.complementarity.item(),
        "lipschitz_penalty": penalty_value, "gamma": gamma,
        "loss_total_regularized": loss.total.item() + gamma * penalty_value,
    }


def train_with_lipschitz_penalty(
    problem: BaseProblem, triple: NeuralTriple, t: torch.Tensor, x: torch.Tensor, x0: torch.Tensor,
    n_epochs: int, gamma: float,
    lr: float = 1e-3, n_lbfgs_epochs: int = 0, lbfgs_lr: float = 1.0,
    w_r: float = 1.0, w_s: float = 1.0, w_0: float = 1.0, w_p: float = 1.0, w_c: float = 1.0,
    csv_path: str | None = None,
) -> tuple[pd.DataFrame, LBFGSOutcome]:
    weights = dict(w_r=w_r, w_s=w_s, w_0=w_0, w_p=w_p, w_c=w_c)
    records: list[dict] = []

    def _loss_and_backward_target():
        residuals = compute_residuals(problem, triple.state_net, triple.flux_net, triple.multiplier_net, t, x, x0)
        loss = compute_loss(residuals, **weights)
        if gamma != 0.0:
            penalty = lipschitz_penalty(triple)
            total_for_backward = loss.total + gamma * penalty
        else:
            with torch.no_grad():
                penalty = lipschitz_penalty(triple)
            total_for_backward = loss.total
        return loss, penalty, total_for_backward

    adam_optimizer = torch.optim.Adam(triple.parameters(), lr=lr)
    for epoch in range(n_epochs):
        adam_optimizer.zero_grad()
        loss, penalty, total_for_backward = _loss_and_backward_target()
        total_for_backward.backward()
        adam_optimizer.step()
        records.append(_loss_record(epoch, "adam", loss, penalty.item(), gamma))

    lbfgs_optimizer = torch.optim.LBFGS(triple.parameters(), lr=lbfgs_lr, max_iter=1, line_search_fn="strong_wolfe")
    holder: dict = {}

    def closure():
        lbfgs_optimizer.zero_grad()
        loss, penalty, total_for_backward = _loss_and_backward_target()
        total_for_backward.backward()
        holder["loss"] = loss
        holder["penalty"] = penalty
        holder["total_regularized"] = total_for_backward
        return total_for_backward

    steps_completed = 0
    prev_total: float | None = None
    reason = "skipped"

    for step_idx in range(n_lbfgs_epochs):
        try:
            lbfgs_optimizer.step(closure)
            closure()
        except RuntimeError as exc:
            reason = f"runtime_error: {exc}"
            break

        loss = holder.get("loss")
        total_regularized = holder.get("total_regularized")
        if loss is None or not torch.isfinite(total_regularized):
            reason = "non_finite_loss"
            break

        total = total_regularized.item()
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
        records.append(_loss_record(n_epochs + step_idx, "lbfgs", loss, holder["penalty"].item(), gamma))
    else:
        reason = "completed" if n_lbfgs_epochs > 0 else "skipped"

    lbfgs_outcome = LBFGSOutcome(reason=reason, n_steps_completed=steps_completed)
    history = pd.DataFrame.from_records(records)
    if csv_path is not None:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        history.to_csv(path, index=False)
    return history, lbfgs_outcome
