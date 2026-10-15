"""Qualitative solution comparison figure, specification section 9 (Stefan).

Three panels, each a pair (or triple) of heatmaps over Q = (0,T) x Omega:
  (a) u_theta, exact u, and the error u - u_theta
  (b) sigma_phi and exact sigma
  (c) lambda_psi and the exact indicator 1_{t<x}

Panel (c) is expected to show visible smoothing of lambda_psi across the free
boundary t=x: a tanh-based squared head is a continuous function and cannot
reproduce the exact solution's discontinuous indicator. That mismatch is a
real result about the multiplier head's structural limitation, not a bug --
never adjust color scales, exact-solution smoothing, or training to hide it.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.base_problem import BaseProblem


def _heatmap(fig, ax, data, extent, title, vmin=None, vmax=None, cmap="viridis"):
    im = ax.imshow(data, extent=extent, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)


def plot_solution(
    problem: BaseProblem,
    triple: NeuralTriple,
    n_t: int = 200,
    n_x: int = 200,
    save_path: str | None = None,
):
    triple.eval()
    a, b = problem.omega_bounds
    with torch.no_grad():
        t_lin = torch.linspace(0.0, problem.T, n_t, dtype=torch.float64)
        x_lin = torch.linspace(a, b, n_x, dtype=torch.float64)
        t_grid, x_grid = torch.meshgrid(t_lin, x_lin, indexing="ij")
        t_flat = t_grid.reshape(-1, 1)
        x_flat = x_grid.reshape(-1, 1)

        u_pred = triple.state_net(t_flat, x_flat).reshape(n_t, n_x).numpy()
        sigma_pred = triple.flux_net(t_flat, x_flat).reshape(n_t, n_x).numpy()
        lam_pred = triple.multiplier_net(t_flat, x_flat).reshape(n_t, n_x).numpy()

        u_exact = problem.u_exact(t_flat, x_flat).reshape(n_t, n_x).numpy()
        sigma_exact = problem.sigma_exact(t_flat, x_flat).reshape(n_t, n_x).numpy()
        lam_exact = problem.lambda_exact(t_flat, x_flat).reshape(n_t, n_x).numpy()

    error_u = u_pred - u_exact

    extent = [a, b, 0.0, problem.T]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    u_vmin, u_vmax = min(u_pred.min(), u_exact.min()), max(u_pred.max(), u_exact.max())
    _heatmap(fig, axes[0, 0], u_pred, extent, "u_theta", vmin=u_vmin, vmax=u_vmax)
    _heatmap(fig, axes[0, 1], u_exact, extent, "u exact", vmin=u_vmin, vmax=u_vmax)
    err_abs = max(abs(error_u.min()), abs(error_u.max()))
    _heatmap(fig, axes[0, 2], error_u, extent, "u_theta - u exact", vmin=-err_abs, vmax=err_abs, cmap="coolwarm")

    sig_vmin, sig_vmax = min(sigma_pred.min(), sigma_exact.min()), max(sigma_pred.max(), sigma_exact.max())
    _heatmap(fig, axes[1, 0], sigma_pred, extent, "sigma_phi", vmin=sig_vmin, vmax=sig_vmax)
    _heatmap(fig, axes[1, 1], sigma_exact, extent, "sigma exact", vmin=sig_vmin, vmax=sig_vmax)
    axes[1, 2].axis("off")

    lam_vmin, lam_vmax = 0.0, max(lam_pred.max(), lam_exact.max())
    _heatmap(fig, axes[2, 0], lam_pred, extent, "lambda_psi", vmin=lam_vmin, vmax=lam_vmax)
    _heatmap(fig, axes[2, 1], lam_exact, extent, "1_{t<x} exact", vmin=lam_vmin, vmax=lam_vmax)
    axes[2, 2].axis("off")

    fig.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    return fig
