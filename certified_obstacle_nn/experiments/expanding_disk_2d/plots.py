"""Figures A-H (roadmap section 21) at t in {0.25, 0.5, 0.75} on a 256^2 grid per slice.
Style follows plots/plot_solution.py (imshow heatmaps, viridis for fields, magma for errors).
Every figure is saved as PNG and PDF."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from certified_obstacle_nn.experiments.expanding_disk_2d import exact_solution as ex  # noqa: E402
from certified_obstacle_nn.experiments.expanding_disk_2d import interface_metrics  # noqa: E402

TIMES = (0.25, 0.5, 0.75)
N_GRID = 256


def _save(fig, path_stem):
    fig.tight_layout()
    fig.savefig(f"{path_stem}.png", dpi=150)
    fig.savefig(f"{path_stem}.pdf")
    plt.close(fig)
    return [f"{path_stem}.png", f"{path_stem}.pdf"]


def _circle(ax, t_val, **kw):
    th = np.linspace(0, 2 * np.pi, 400)
    Rt = ex.R0 + ex.R_SPEED * t_val
    ax.plot(ex.C0 + Rt * np.cos(th), ex.C0 + Rt * np.sin(th), **kw)


@torch.no_grad()
def _slices(triple, kappa):
    lin = torch.linspace(0, 1, N_GRID, dtype=torch.float64)
    X, Y = torch.meshgrid(lin, lin, indexing="xy")
    x, y = X.reshape(-1, 1), Y.reshape(-1, 1)
    out = {}
    for tv in TIMES:
        t = torch.full_like(x, tv)
        sh = (N_GRID, N_GRID)
        out[tv] = {"u": ex.u_exact(t, x, y, kappa).reshape(sh).numpy(),
                   "u_th": triple.u(t, x, y).reshape(sh).numpy(),
                   "lam": ex.lambda_exact(t, x, y, kappa).reshape(sh).numpy(),
                   "lam_th": triple.lam(t, x, y).reshape(sh).numpy(),
                   "gap_th": (triple.u(t, x, y) - ex.g(t, x, y)).reshape(sh).numpy()}
    return out


def _row_figure(data, keys_titles, cmaps, path_stem, shared=False):
    fig, axes = plt.subplots(1, len(TIMES), figsize=(4.6 * len(TIMES), 4.0))
    vals = [data[tv][keys_titles[0]] for tv in TIMES]
    vmin, vmax = (min(v.min() for v in vals), max(v.max() for v in vals)) if shared else (None, None)
    for ax, tv, v in zip(axes, TIMES, vals):
        im = ax.imshow(v, extent=[0, 1, 0, 1], origin="lower", aspect="equal", cmap=cmaps, vmin=vmin, vmax=vmax)
        _circle(ax, tv, color="w", lw=0.8, ls="--")
        ax.set_title(f"{keys_titles[1]}, t={tv}")
        ax.set_xlabel("x"), ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, path_stem)


def make_run_figures(triple, cfg, fig_dir) -> list[str]:
    triple.eval()
    d = _slices(triple, cfg.kappa)
    for tv in TIMES:
        d[tv]["err_u"] = np.abs(d[tv]["u"] - d[tv]["u_th"])
        d[tv]["err_lam"] = np.abs(d[tv]["lam"] - d[tv]["lam_th"])
    paths = []
    specs = [("A_u_exact", "u", "exact u", "viridis", True), ("B_u_neural", "u_th", r"$u_\theta$", "viridis", True),
             ("C_u_error", "err_u", r"$|u-u_\theta|$", "magma", False),
             ("E_lambda_exact", "lam", r"exact $\lambda$", "viridis", True),
             ("F_lambda_neural", "lam_th", r"$\lambda_\psi$", "viridis", True),
             ("G_lambda_error", "err_lam", r"$|\lambda-\lambda_\psi|$", "magma", False)]
    for stem, key, title, cmap, shared in specs:
        paths += _row_figure(d, (key, title), cmap, str(fig_dir / stem), shared)
    # Figure D: exact vs learned interface (both criteria: contour + ray-marched radii)
    fig, axes = plt.subplots(1, len(TIMES), figsize=(4.6 * len(TIMES), 4.2))
    lin = np.linspace(0, 1, N_GRID)
    th = np.arange(interface_metrics.N_ANGLES) * 2 * np.pi / interface_metrics.N_ANGLES
    for ax, tv in zip(axes, TIMES):
        ax.imshow(d[tv]["lam_th"], extent=[0, 1, 0, 1], origin="lower", cmap="viridis", alpha=0.6)
        ax.contour(lin, lin, d[tv]["lam_th"], levels=[cfg.kappa], colors="tab:orange", linewidths=1.0)
        ax.contour(lin, lin, d[tv]["gap_th"], levels=[cfg.eps_gamma], colors="tab:cyan", linewidths=1.0)
        for crit, col in (("lambda", "tab:orange"), ("gap", "tab:cyan")):
            r, _ = interface_metrics.interface_radii(triple, tv, crit, cfg.kappa, cfg.eps_gamma)
            ax.plot(ex.C0 + r * np.cos(th), ex.C0 + r * np.sin(th), ".", ms=1.0, color=col)
        _circle(ax, tv, color="red", lw=1.2, ls="--")
        ax.set_xlim(0.1, 0.9), ax.set_ylim(0.1, 0.9)
        ax.set_title(f"interface, t={tv}")
        ax.set_xlabel("x"), ax.set_ylabel("y")
    axes[0].plot([], [], color="red", ls="--", label="exact r=R(t)")
    axes[0].plot([], [], color="tab:orange", label=r"$\lambda_\psi=\kappa$")
    axes[0].plot([], [], color="tab:cyan", label=r"$u_\theta-g=\varepsilon_\Gamma$")
    axes[0].legend(loc="lower left", fontsize=7)
    paths += _save(fig, str(fig_dir / "D_interface"))
    return paths


def loss_figure(run_dir) -> list[str]:
    import pandas as pd
    h = pd.read_csv(run_dir / "loss_history.csv")
    mon = pd.read_csv(run_dir / "monitor_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for c in ("loss_total", "loss_pde", "loss_sigma", "loss_initial", "loss_penetration", "loss_complementarity"):
        axes[0].semilogy(h["step"], h[c].clip(lower=1e-30), label=c.replace("loss_", ""), lw=0.8)
    axes[0].axvline((h["phase"] == "adam").sum(), color="k", ls=":", lw=0.8)
    axes[0].legend(fontsize=7), axes[0].set_xlabel("step"), axes[0].set_title("loss components")
    axes[1].semilogy(mon["step"], mon["E_true"], "o-", ms=3, label=r"$E_{true}$ (monitor set)")
    axes[1].semilogy(mon["step"], mon["rho_tilde"], "s-", ms=3, label=r"$\tilde\rho$ (monitor set)")
    axes[1].legend(fontsize=8), axes[1].set_xlabel("step"), axes[1].set_title("error / estimator during training")
    return _save(fig, str(run_dir / "figures" / "training_history"))


def figure_H(rows, out_stem, label_key="label") -> list[str]:
    """E_true vs rho_tilde across configurations. rows: dicts with label, E_true, rho_tilde
    (optionally *_std)."""
    fig, ax = plt.subplots(figsize=(max(6.5, 1.1 * len(rows)), 4.8))
    xs = np.arange(len(rows))
    for key, mk, lab in (("E_true", "o", r"$E_{true}$"), ("rho_tilde", "s", r"$\tilde\rho$")):
        ys = [r[key] for r in rows]
        err = [r.get(f"{key}_std", 0.0) for r in rows]
        ax.errorbar(xs, ys, yerr=err, marker=mk, capsize=3, label=lab, ls="-")
    ax.set_yscale("log")
    ax.set_xticks(xs, [r[label_key] for r in rows], rotation=30, ha="right", fontsize=7)
    ax.legend()
    ax.set_title("2D expanding disk: true error vs raw estimator")
    return _save(fig, out_stem)
