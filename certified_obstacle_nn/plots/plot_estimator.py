"""Estimator heat map, specification sections 4 and 7.

Heat map of rho_tilde_K (the per-cell total estimator, `rho_total_K` in
certification/raw_estimator.py's per-cell DataFrame) over Q, with the exact
free boundary overlaid, plus separate panels for its three specification section 4
components: rho_r_K (PDE + flux + initial residual), rho_penetration_K
(obstacle violation), rho_complementarity_K.

The scientific question this exists to answer: the estimator formula (section
4) has no dependence on the free boundary's location whatsoever -- it is a
function of the trained triple's residuals alone. Does it nonetheless grow
near the true free boundary, "discovering" it without being told? That is
measured as the ratio of the mean per-cell indicator inside a band
|t-x| < band_width to the mean outside it, computed by `band_ratio` and
annotated directly on the figure.
"""

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from certified_obstacle_nn.certification.partition import Partition
from certified_obstacle_nn.problems.base_problem import BaseProblem

FreeBoundaryFn = Callable[[torch.Tensor], torch.Tensor]

_PANELS = [
    ("rho_total_K", "rho_tilde_K (total)"),
    ("rho_r_K", "rho_r_K (PDE + flux + initial)"),
    ("rho_penetration_K", "rho_penetration_K (obstacle)"),
    ("rho_complementarity_K", "rho_complementarity_K"),
]


def band_ratio(per_cell_df: pd.DataFrame, band_width: float = 0.05, value_col: str = "rho_total_K") -> dict:
    diff = (per_cell_df["t_center"] - per_cell_df["x_center"]).abs()
    inside = diff < band_width
    mean_inside = per_cell_df.loc[inside, value_col].mean()
    mean_outside = per_cell_df.loc[~inside, value_col].mean()
    return {
        "band_width": band_width,
        "value_col": value_col,
        "n_inside": int(inside.sum()),
        "n_outside": int((~inside).sum()),
        "mean_inside": float(mean_inside),
        "mean_outside": float(mean_outside),
        "ratio": float(mean_inside / mean_outside),
    }


def _grid_from_df(per_cell_df: pd.DataFrame, col: str, n_t: int, n_x: int) -> np.ndarray:
    grid = np.zeros((n_t, n_x))
    grid[per_cell_df["i"].to_numpy(), per_cell_df["j"].to_numpy()] = per_cell_df[col].to_numpy()
    return grid


def plot_estimator(
    per_cell_df: pd.DataFrame,
    partition: Partition,
    problem: BaseProblem,
    free_boundary_fn: FreeBoundaryFn | None = None,
    band_width: float = 0.05,
    save_path: str | None = None,
) -> tuple[plt.Figure, dict]:
    a, b = problem.omega_bounds
    T = problem.T
    extent = [a, b, 0.0, T]
    n_t, n_x = partition.n_t, partition.n_x

    t_line = None
    x_line = None
    if free_boundary_fn is not None:
        t_line = np.linspace(0.0, T, 200)
        x_line = free_boundary_fn(torch.tensor(t_line, dtype=torch.float64)).numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, (col, title) in zip(axes.flat, _PANELS):
        grid = _grid_from_df(per_cell_df, col, n_t, n_x)
        im = ax.imshow(grid, extent=extent, origin="lower", aspect="auto", cmap="inferno")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        if x_line is not None:
            ax.plot(x_line, t_line, color="cyan", linestyle="--", linewidth=1.5, label="exact free boundary")
            ax.legend(loc="upper right", fontsize=8)

    ratio_info = band_ratio(per_cell_df, band_width=band_width, value_col="rho_total_K")
    fig.suptitle(
        f"mean(rho_tilde_K): inside |t-x|<{band_width} = {ratio_info['mean_inside']:.4e}  "
        f"({ratio_info['n_inside']} cells)   outside = {ratio_info['mean_outside']:.4e}  "
        f"({ratio_info['n_outside']} cells)   ratio = {ratio_info['ratio']:.3f}"
    )
    fig.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    return fig, ratio_info
