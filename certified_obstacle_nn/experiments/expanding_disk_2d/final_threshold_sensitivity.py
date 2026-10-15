"""Final-phase Task 1: interface-threshold sensitivity (no retraining).

Learned interface = level set lambda_psi = c, c in {2.5, 5.0, 7.5} (exact jump 0 -> 2 kappa = 10),
extracted with the existing ray-marching + bisection (interface_metrics.interface_radii, 360 rays,
t in {0.25, 0.5, 0.75, 1}). Radial error |r_theta - R(t)| per ray. Per run, per threshold, per t:
mean and max over rays (plus the signed mean
offset r_theta - R, + = contour outside the circle; never-crossing rays sit at r = 0.5 by the
existing fallback and are counted in no_crossing); the time average is the mean over the four slices (for "max" this is the
mean of the per-slice maxima, the same convention as the headline iface_lambda_max), and the worst
slice is reported as max_over_t. Aggregated as mean +- std over seeds per architecture.

The u_theta - g = eps_Gamma contour is drawn in the figure as a secondary visualization only; it is
NOT a metric (u - g ~ kappa psi^2 has zero normal slope at the interface: ill-conditioned).

Outputs (results/expanding_disk_2d/final/):
  interface_threshold_sensitivity.csv          one row per (arch, seed, threshold, t | 'avg')
  interface_threshold_sensitivity_summary.csv  mean/std over seeds per (arch, threshold, t | 'avg')
  interface_threshold_sensitivity.tex          LaTeX table (time-averaged)
  fig_interface_thresholds.{png,pdf}           6x64 seed 0, exact circle + three contours
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from certified_obstacle_nn.experiments.expanding_disk_2d import exact_solution as ex  # noqa: E402
from certified_obstacle_nn.experiments.expanding_disk_2d import interface_metrics as im  # noqa: E402
from certified_obstacle_nn.experiments.expanding_disk_2d.final_common import ARCH_RUNS, final_dir, load_run  # noqa: E402

THRESHOLDS = (2.5, 5.0, 7.5)
FIG_TIMES = (0.25, 0.5, 0.75)
N_GRID = 256


def per_run_rows(arch, seed, run):
    triple, cfg = load_run(run)
    rows = []
    for c in THRESHOLDS:
        means, maxs, signs = [], [], []
        for tv in im.TIMES:
            # criterion "lambda" uses margin lambda_psi - <threshold>; the threshold is passed as `kappa`
            radii, n_miss = im.interface_radii(triple, tv, "lambda", c, cfg.eps_gamma)
            signed = radii - float(ex.R0 + ex.R_SPEED * tv)
            err = np.abs(signed)
            means.append(err.mean())
            maxs.append(err.max())
            signs.append(signed.mean())
            rows.append({"arch": arch, "seed": seed, "run": run, "threshold": c, "t": f"{tv:g}",
                         "mean_err": err.mean(), "max_err": err.max(), "signed_mean": signed.mean(),
                         "no_crossing": n_miss})
        rows.append({"arch": arch, "seed": seed, "run": run, "threshold": c, "t": "avg",
                     "mean_err": float(np.mean(means)), "max_err": float(np.mean(maxs)),
                     "max_err_over_t": float(np.max(maxs)), "signed_mean": float(np.mean(signs)),
                     "no_crossing": np.nan})
    return rows


def to_tex(s: pd.DataFrame) -> str:
    a = s[s["t"] == "avg"]
    lines = [r"\begin{tabular}{lrrllll}", r"\toprule",
             r"Network & Seeds & $c$ & mean $|r_\theta-R|$ & max $|r_\theta-R|$ (avg.\ over $t$) & "
             r"max $|r_\theta-R|$ (worst $t$) & signed mean $r_\theta-R$ \\", r"\midrule"]

    def pm(m, sd):
        return f"{m:.4f}" if not np.isfinite(sd) else f"{m:.4f} $\\pm$ {sd:.4f}"

    for _, r in a.iterrows():
        lines.append(" & ".join([r["arch"], str(int(r["n_seeds"])), f"{r['threshold']:g}",
                                 pm(r["mean_err_mean"], r["mean_err_std"]), pm(r["max_err_mean"], r["max_err_std"]),
                                 pm(r["max_err_over_t_mean"], r["max_err_over_t_std"]),
                                 pm(r["signed_mean_mean"], r["signed_mean_std"])]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


@torch.no_grad()
def figure(out_stem):
    triple, cfg = load_run(ARCH_RUNS["6x64"][0])
    lin = torch.linspace(0, 1, N_GRID, dtype=torch.float64)
    X, Y = torch.meshgrid(lin, lin, indexing="xy")
    x, y = X.reshape(-1, 1), Y.reshape(-1, 1)
    styles = {2.5: ("tab:blue", ":"), 5.0: ("tab:orange", "-"), 7.5: ("tab:green", "-.")}
    th = np.linspace(0, 2 * np.pi, 400)
    fig, axes = plt.subplots(1, len(FIG_TIMES), figsize=(4.2 * len(FIG_TIMES), 4.2))
    for ax, tv in zip(axes, FIG_TIMES):
        t = torch.full_like(x, tv)
        lam = triple.lam(t, x, y).reshape(N_GRID, N_GRID).numpy()
        gap = (triple.u(t, x, y) - ex.g(t, x, y)).reshape(N_GRID, N_GRID).numpy()
        Rt = ex.R0 + ex.R_SPEED * tv
        ax.plot(ex.C0 + Rt * np.cos(th), ex.C0 + Rt * np.sin(th), color="red", lw=1.6, ls="--")
        for c, (col, ls) in styles.items():
            ax.contour(lin.numpy(), lin.numpy(), lam, levels=[c], colors=col, linestyles=ls, linewidths=1.1)
        ax.contour(lin.numpy(), lin.numpy(), gap, levels=[cfg.eps_gamma], colors="0.55", linestyles="--",
                   linewidths=0.7)
        ax.set_xlim(0.1, 0.9), ax.set_ylim(0.1, 0.9), ax.set_aspect("equal")
        ax.set_title(f"t={tv}")
        ax.set_xlabel("x"), ax.set_ylabel("y")
    axes[0].plot([], [], color="red", ls="--", lw=1.6, label="exact r=R(t)")
    for c, (col, ls) in styles.items():
        axes[0].plot([], [], color=col, ls=ls, label=rf"$\lambda_\psi={c:g}$")
    axes[0].plot([], [], color="0.55", ls="--", lw=0.7, label=r"$u_\theta-g=10^{-3}$ (secondary)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    paths = [f"{out_stem}.png", f"{out_stem}.pdf"]
    fig.savefig(paths[0], dpi=150)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include_long", action="store_true",
                    help="also evaluate the finished long-schedule 6x64 runs (group '6x64-long')")
    a = ap.parse_args()
    out = final_dir()
    groups = dict(ARCH_RUNS)
    if a.include_long:
        groups["6x64-long"] = {int(p.name.split("_s")[1].split("_")[0]): str(p)
                               for p in sorted(out.glob("long_6x64_*")) if (p / "metrics.json").exists()}
    rows = []
    for arch, runs in groups.items():
        for seed, run in runs.items():
            rows += per_run_rows(arch, seed, run)
            print(f"done {arch} seed {seed}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "interface_threshold_sensitivity.csv", index=False)
    cols = ["mean_err", "max_err", "max_err_over_t", "signed_mean", "no_crossing"]
    g = df.groupby(["arch", "threshold", "t"], sort=False)
    s = g[cols].mean().add_suffix("_mean").join(g[cols].std(ddof=1).add_suffix("_std"))
    s["n_seeds"] = g.size()
    s = s.reset_index()
    s.to_csv(out / "interface_threshold_sensitivity_summary.csv", index=False)
    (out / "interface_threshold_sensitivity.tex").write_text(to_tex(s))
    figure(str(out / "fig_interface_thresholds"))
    pd.set_option("display.width", 200)
    print(s[s["t"] == "avg"][["arch", "threshold", "n_seeds", "mean_err_mean", "mean_err_std", "max_err_mean",
                              "max_err_std", "max_err_over_t_mean", "signed_mean_mean"]].to_string(index=False))
    print(s[(s["arch"] == "6x64")][["threshold", "t", "mean_err_mean", "mean_err_std", "max_err_mean",
                                    "no_crossing_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
