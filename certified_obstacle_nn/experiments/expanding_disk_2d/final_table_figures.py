"""Final-phase Task 5: main table and paper figures (no retraining; frozen checkpoints only).

Table  final/table_2d.{csv,tex}: rows 4x64 (seed 0), 6x64 (seeds 0-4, the selected architecture),
6x128 (seed 0), plus the long-schedule 6x64 row only with --include_long (Task 4 accepted). All
numbers are read from the stored metrics.json of each run (1e6-point Sobol evaluation).

Figures (final/figures/, PNG + PDF), representative run = 6x64 seed 0 (stage3_6x64_N20000_s0_k5);
with --include_long the same set for the long-schedule seed 0 goes to final/figures_long/;
t in {0.25, 0.5, 0.75}, 256^2 grid per slice, same style as plots.py (viridis fields, magma errors,
exact circle r = R(t) dashed):
  fig1_u_exact, fig2_u_neural (shared colour scale), fig3_u_error,
  fig4_free_boundary (lambda_psi = kappa = 5 contour vs exact circle),
  fig5_lambda_exact, fig6_lambda_neural (shared colour scale), fig7_lambda_error,
  fig8_error_vs_estimator (E_true vs rho_tilde, mean +- std over seeds),
  fig_combined (rows: u exact / neural / error, lambda exact / neural / error; columns: times).
Band diagnostic for Fig. 7: fraction of ||lambda - lambda_psi||^2_{L2(Q)} from |r - R(t)| < 0.02 and
< 0.05 (evaluate.evaluate_all band split, original evaluation scrambles), five 6x64 seeds ->
final/lambda_band_fractions.csv.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib  # noqa: E402
import matplotlib.ticker  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from certified_obstacle_nn.experiments.expanding_disk_2d import evaluate  # noqa: E402
from certified_obstacle_nn.experiments.expanding_disk_2d import exact_solution as ex  # noqa: E402
from certified_obstacle_nn.experiments.expanding_disk_2d.config import RESULTS_ROOT  # noqa: E402
from certified_obstacle_nn.experiments.expanding_disk_2d.final_common import (  # noqa: E402
    ARCH_RUNS, REFERENCE_6X64, final_dir, load_run)

TIMES = (0.25, 0.5, 0.75)
N_GRID = 256
BANDS = (0.02, 0.05, 0.10)  # 0.02 and 0.05 required; 0.10 ~ half-width of the smeared lambda ramp (Task 1)
COLS = ["E_true", "rho_tilde", "I_eff_raw", "err_u_L2", "err_sigma_L2", "err_lambda_L2", "iface_lambda_mean",
        "iface_lambda_max", "rho_r", "rho_p_tilde", "rho_c", "n_params"]


# ----------------------------------------------------------------------------------------- table
def long_runs() -> dict:
    """Long-schedule runs (final_long_train.py) that have finished: {seed: absolute run dir}."""
    return {int(p.name.split("_s")[1].split("_")[0]): str(p) for p in sorted(final_dir().glob("long_6x64_*"))
            if (p / "metrics.json").exists()}


def table_rows(include_long: bool) -> pd.DataFrame:
    groups = [(f"{a}" + (" (selected)" if a == "6x64" else ""), a, runs, "5000 Adam + L-BFGS")
              for a, runs in ARCH_RUNS.items()]
    if include_long:
        groups.append(("6x64 long schedule", "6x64", long_runs(), "20000 Adam (lr decay) + L-BFGS"))
    out = []
    for label, arch, runs, sched in groups:
        ms = []
        for s, run in runs.items():
            d = Path(run) if Path(run).is_absolute() else RESULTS_ROOT / run
            c = json.loads((d / "config.json").read_text())
            assert c["arch"] == arch and c["n_int"] == 20_000 and c["kappa"] == 5.0 and c["w_p"] == c["w_c"] == 10.0
            ms.append(json.loads((d / "metrics.json").read_text()))
        df = pd.DataFrame([{k: m[k] for k in COLS} for m in ms])
        row = {"network": label, "arch": arch, "schedule": sched, "N_int": 20_000, "n_seeds": len(ms),
               "seeds": ",".join(map(str, sorted(runs))), "n_params": int(df["n_params"].iloc[0])}
        for k in COLS[:-1]:
            row[f"{k}_mean"] = df[k].mean()
            row[f"{k}_std"] = df[k].std(ddof=1) if len(ms) > 1 else np.nan
        out.append(row)
    return pd.DataFrame(out)


def _pm(m, s, nd):
    return f"{m:.{nd}f}" if not np.isfinite(s) else f"{m:.{nd}f} $\\pm$ {s:.{nd}f}"


def table_tex(t: pd.DataFrame) -> str:
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{lrrllllllc}", r"\toprule",
             r"Network & $N_{\rm int}$ & Seeds & $E_{\rm true}$ & $\widetilde\rho$ & $I_{\rm eff}^{\rm raw}$ & "
             r"$\|u-u_\theta\|$ & $\|\sigma-\sigma_\phi\|$ & $\|\lambda-\lambda_\psi\|$ & Interface mean / max \\",
             r"\midrule"]
    for _, r in t.iterrows():
        iface = (f"{r['iface_lambda_mean_mean']:.4f} / {r['iface_lambda_max_mean']:.3f}")
        lines.append(" & ".join([
            r["network"].replace("x", r"$\times$", 1), f"{r['N_int'] // 1000}k", str(r["n_seeds"]),
            _pm(r["E_true_mean"], r["E_true_std"], 3), _pm(r["rho_tilde_mean"], r["rho_tilde_std"], 3),
            _pm(r["I_eff_raw_mean"], r["I_eff_raw_std"], 2), _pm(r["err_u_L2_mean"], r["err_u_L2_std"], 4),
            _pm(r["err_sigma_L2_mean"], r["err_sigma_L2_std"], 3),
            _pm(r["err_lambda_L2_mean"], r["err_lambda_L2_std"], 2), iface]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{2D expanding-disk benchmark ($\kappa=5$, $w_p=w_c=10$): true error, raw estimator "
              r"and effectivity; mean $\pm$ std over seeds where more than one seed exists. Norms are "
              r"$L^2(Q)$; interface error $|r_\theta-R(t)|$ of the level set $\lambda_\psi=\kappa$ along 360 rays, "
              r"averaged over $t\in\{0.25,0.5,0.75,1\}$ (max = mean over $t$ of the per-slice maximum). "
              r"$6\times64$ is the selected architecture among those tested.}",
              r"\label{tab:disk2d}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------- figures
def _save(fig, stem):
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    return [f"{stem}.png", f"{stem}.pdf"]


def _circle(ax, tv, **kw):
    th = np.linspace(0, 2 * np.pi, 400)
    Rt = ex.R0 + ex.R_SPEED * tv
    ax.plot(ex.C0 + Rt * np.cos(th), ex.C0 + Rt * np.sin(th), **kw)


@torch.no_grad()
def slices(triple, kappa):
    lin = torch.linspace(0, 1, N_GRID, dtype=torch.float64)
    X, Y = torch.meshgrid(lin, lin, indexing="xy")
    x, y = X.reshape(-1, 1), Y.reshape(-1, 1)
    sh = (N_GRID, N_GRID)
    d = {}
    for tv in TIMES:
        t = torch.full_like(x, tv)
        u, uth = ex.u_exact(t, x, y, kappa).reshape(sh).numpy(), triple.u(t, x, y).reshape(sh).numpy()
        lam, lth = ex.lambda_exact(t, x, y, kappa).reshape(sh).numpy(), triple.lam(t, x, y).reshape(sh).numpy()
        d[tv] = {"u": u, "u_th": uth, "err_u": np.abs(u - uth), "lam": lam, "lam_th": lth, "err_lam": np.abs(lam - lth)}
    return d


SPECS = {  # key: (title, cmap, shared-scale group)
    "u": ("exact $u$", "viridis", "u"), "u_th": (r"neural $u_\theta$", "viridis", "u"),
    "err_u": (r"$|u-u_\theta|$", "magma", "err_u"),
    "lam": (r"exact $\lambda$", "viridis", "lam"), "lam_th": (r"neural $\lambda_\psi$", "viridis", "lam"),
    "err_lam": (r"$|\lambda-\lambda_\psi|$", "magma", "err_lam")}


def _limits(d, keys):
    lim = {}
    for k in keys:
        grp = SPECS[k][2]
        v = [d[tv][kk] for tv in TIMES for kk in keys if SPECS[kk][2] == grp]
        lim[grp] = (min(a.min() for a in v), max(a.max() for a in v))
    return lim


def _panel(ax, d, tv, key, lim, circle_color="w"):
    title, cmap, grp = SPECS[key]
    im = ax.imshow(d[tv][key], extent=[0, 1, 0, 1], origin="lower", cmap=cmap, vmin=lim[grp][0], vmax=lim[grp][1])
    _circle(ax, tv, color=circle_color, lw=0.8, ls="--")
    ax.set_xticks([0, 0.5, 1]), ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=7)
    return im


def row_figure(d, key, lim, stem):
    fig, axes = plt.subplots(1, len(TIMES), figsize=(3.0 * len(TIMES) + 0.6, 3.0), constrained_layout=True)
    for ax, tv in zip(axes, TIMES):
        im = _panel(ax, d, tv, key, lim)
        ax.set_title(f"{SPECS[key][0]}, $t={tv}$", fontsize=9)
    fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.02).ax.tick_params(labelsize=7)
    return _save(fig, stem)


def combined_figure(d, lim, stem):
    keys = ["u", "u_th", "err_u", "lam", "lam_th", "err_lam"]
    fig, axes = plt.subplots(len(keys), len(TIMES), figsize=(2.3 * len(TIMES) + 0.9, 2.2 * len(keys)),
                             constrained_layout=True)
    for i, k in enumerate(keys):
        for j, tv in enumerate(TIMES):
            im = _panel(axes[i, j], d, tv, k, lim)
            if i == 0:
                axes[i, j].set_title(f"$t={tv}$", fontsize=9)
            if j == 0:
                axes[i, j].set_ylabel(SPECS[k][0], fontsize=9)
        fig.colorbar(im, ax=list(axes[i]), fraction=0.03, pad=0.02).ax.tick_params(labelsize=7)
    return _save(fig, stem)


@torch.no_grad()
def free_boundary_figure(triple, d, kappa, stem):
    lin = np.linspace(0, 1, N_GRID)
    fig, axes = plt.subplots(1, len(TIMES), figsize=(3.0 * len(TIMES), 3.2), constrained_layout=True)
    for ax, tv in zip(axes, TIMES):
        ax.contour(lin, lin, d[tv]["lam_th"], levels=[kappa], colors="tab:orange", linewidths=1.3)
        _circle(ax, tv, color="k", lw=1.2, ls="--")
        ax.set_xlim(0.1, 0.9), ax.set_ylim(0.1, 0.9), ax.set_aspect("equal")
        ax.set_title(f"$t={tv}$", fontsize=9)
        ax.tick_params(labelsize=7)
    axes[0].plot([], [], color="k", ls="--", label="exact $r=R(t)$")
    axes[0].plot([], [], color="tab:orange", label=r"learned $\lambda_\psi=\kappa$")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    return _save(fig, stem)


LABEL_OFFSETS = {"4x64": (-62, 8), "6x128": (6, -12), "6x64 long schedule": (8, -12)}


def error_vs_estimator_figure(t: pd.DataFrame, stem):
    fig, ax = plt.subplots(figsize=(4.4, 4.0), constrained_layout=True)
    markers = ["o", "s", "^", "D"]
    for (_, r), mk in zip(t.iterrows(), markers):
        ax.errorbar(r["E_true_mean"], r["rho_tilde_mean"], xerr=np.nan_to_num(r["E_true_std"]),
                    yerr=np.nan_to_num(r["rho_tilde_std"]), marker=mk, ms=7, capsize=3, ls="none", color="0.15",
                    mfc="white" if r["n_seeds"] == 1 else "0.15")
        ax.annotate(f"{r['network']} ({r['n_seeds']} seed{'s' if r['n_seeds'] > 1 else ''})",
                    (r["E_true_mean"], r["rho_tilde_mean"]), textcoords="offset points",
                    xytext=LABEL_OFFSETS.get(r["network"], (7, -10)), fontsize=7)
    lo, hi = 0.06, 0.6
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.8, ls="--", label=r"$\widetilde\rho=E_{\rm true}$")
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlim(lo, hi), ax.set_ylim(lo, hi), ax.set_aspect("equal")
    ticks = [0.07, 0.1, 0.2, 0.3, 0.5]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        axis.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{v:g}" for v in ticks]))
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"$E_{\rm true}$"), ax.set_ylabel(r"$\widetilde\rho$")
    ax.legend(fontsize=8, loc="upper left")
    return _save(fig, stem)


def band_fractions(groups: dict) -> pd.DataFrame:
    rows = []
    for group, runs in groups.items():
        for s, run in runs.items():
            triple, cfg = load_run(run)
            row = {"group": group, "seed": s, "run": Path(run).name}
            for b in BANDS:
                evaluate.BAND = b  # existing band split, half-width set per call
                m = evaluate.evaluate_all(triple, cfg)
                tot = m["err_lambda_L2"] ** 2
                row.update({f"band{b:g}_volume_frac": m["band_volume_fraction"],
                            f"band{b:g}_lam_err2_frac": m["err_lambda_band"] ** 2 / tot,
                            f"band{b:g}_E_vol2_frac": m["E_vol_band"] ** 2 / (m["E_vol_band"] ** 2 + m["E_vol_away"] ** 2),
                            "err_lambda_L2": m["err_lambda_L2"]})
            rows.append(row)
    evaluate.BAND = 0.02
    return pd.DataFrame(rows)


def figure_set(run, figdir, t) -> list[str]:
    figdir.mkdir(exist_ok=True)
    triple, cfg = load_run(run)
    d = slices(triple, cfg.kappa)
    lim = _limits(d, list(SPECS))
    paths = []
    for stem, key in (("fig1_u_exact", "u"), ("fig2_u_neural", "u_th"), ("fig3_u_error", "err_u"),
                      ("fig5_lambda_exact", "lam"), ("fig6_lambda_neural", "lam_th"), ("fig7_lambda_error", "err_lam")):
        paths += row_figure(d, key, lim, str(figdir / stem))
    paths += free_boundary_figure(triple, d, cfg.kappa, str(figdir / "fig4_free_boundary"))
    paths += error_vs_estimator_figure(t, str(figdir / "fig8_error_vs_estimator"))
    paths += combined_figure(d, lim, str(figdir / "fig_combined"))
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include_long", action="store_true", help="add the long-schedule row (Task 4 accepted)")
    a = ap.parse_args()
    out = final_dir()

    t = table_rows(a.include_long)
    t.to_csv(out / "table_2d.csv", index=False)
    (out / "table_2d.tex").write_text(table_tex(t))
    pd.set_option("display.width", 250)
    print(t[["network", "n_seeds", "E_true_mean", "E_true_std", "rho_tilde_mean", "I_eff_raw_mean",
             "err_lambda_L2_mean", "iface_lambda_mean_mean", "iface_lambda_max_mean"]].to_string(index=False))

    groups = {"6x64 baseline": REFERENCE_6X64}
    if a.include_long:
        groups["6x64 long schedule"] = long_runs()
    bf = band_fractions(groups)
    bf.to_csv(out / "lambda_band_fractions.csv", index=False)
    print(bf.to_string(index=False))
    print(bf.drop(columns=["run", "seed"]).groupby("group").agg(["mean", "std"]).T.to_string())

    # primary figure set: baseline seed 0 (the run behind the retained five-seed headline)
    paths = figure_set(REFERENCE_6X64[0], out / "figures", t)
    if a.include_long:
        paths += figure_set(long_runs()[0], out / "figures_long", t)
    print("\n".join(paths))


if __name__ == "__main__":
    main()
