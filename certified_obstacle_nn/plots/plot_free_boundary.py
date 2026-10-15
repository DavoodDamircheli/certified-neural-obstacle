"""American put exercise-boundary comparison plot, specification section 10
(2026-09-12): "Extract and plot the exercise boundary against the frozen
reference." (Stefan's own free-boundary overlay lives inline in
plots/plot_estimator.py -- this file is for the American put comparison
specifically, against reference/american_put_reference.py's frozen PSOR
solution, never against the network's own certification pipeline.)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_exercise_boundary_comparison(
    tau_reference: np.ndarray,
    boundary_reference: np.ndarray,
    tau_neural: np.ndarray,
    boundary_neural: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))

    valid_ref = np.isfinite(boundary_reference)
    valid_neural = np.isfinite(boundary_neural)

    ax.plot(boundary_reference[valid_ref], tau_reference[valid_ref], color="black", linewidth=2, label="frozen reference (CN+PSOR)")
    ax.plot(boundary_neural[valid_neural], tau_neural[valid_neural], color="tab:red", linestyle="--", linewidth=1.5, label="neural triple")

    ax.set_xlabel("x = log(S) (exercise boundary)")
    ax.set_ylabel("tau (time-to-maturity)")
    ax.set_title("American put exercise boundary: neural vs. frozen reference")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig
