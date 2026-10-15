"""TODO section 24 figures (+ prompt section 8 bracket figure). Reads ONLY saved artifacts:
docs/history/LEVEL2_COMPARISON_TABLE.csv, runs/adaptive/*/{history.csv,result.json,cells.npz}. Writes PNG + PDF to
results/level2_local/figures/. The exact circle in plot 5 is drawn for visualization only.

python3 -m certified_obstacle_nn.experiments.level2.plots
"""

import csv
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from certified_obstacle_nn.experiments.level2.common import OUT_ROOT, REPO_ROOT, RUNS  # noqa: E402

FIG = OUT_ROOT / "figures"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"legacy": "#2a78d6", "legacy+local-radius": "#eb6834", "upper-sum-T3": "#1baf7a",
          "upper-sum-T3q(TM2)": "#eda100", "adaptive": "#e87ba4", "upper-sum-T2(affine-jac)": "#4a3aa7"}
LABEL = {"legacy": "legacy global (Option B)", "legacy+local-radius": "legacy + local r_K",
         "upper-sum-T3": "upper sum, affine (T3)", "upper-sum-T3q(TM2)": "upper sum, Taylor-2 (T3q)",
         "upper-sum-T2(affine-jac)": "upper sum, centered (T2a)", "adaptive": "adaptive upper sum"}
CASES = [("1d", "1D Stefan 4x64 s0"), ("2d_base", "2D disk baseline s0"), ("2d_long", "2D disk long s0")]


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, lw=0.6, which="major")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=160)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def table():
    return list(csv.DictReader(open(REPO_ROOT / "docs/history/LEVEL2_COMPARISON_TABLE.csv")))


def adaptive_runs(case):
    out = []
    for d in sorted((RUNS / "adaptive").glob(f"{case}_T*")):
        if not (d / "history.csv").exists():
            continue
        res = json.loads((d / "result.json").read_text())
        hist = list(csv.DictReader(open(d / "history.csv")))
        out.append((d.name, res, hist))
    return out


def _largest_adaptive(case):
    runs = adaptive_runs(case)
    return max(runs, key=lambda r: r[1]["n_cells"]) if runs else None


def _uniform_series(rows, case, variant, col):
    pts = sorted((int(r["cells"]), float(r[col])) for r in rows if r["case"] == case and r["method_variant"] == variant
                 and r[col] not in ("", None))
    return [p[0] for p in pts], [p[1] for p in pts]


def plot_metric(rows, col, name, ylabel, use_floor=False, adaptive_col=None):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (case, title) in zip(axes, CASES):
        any_ = False
        for v in ("legacy", "legacy+local-radius", "upper-sum-T3", "upper-sum-T2(affine-jac)", "upper-sum-T3q(TM2)"):
            x, y = _uniform_series(rows, case, v, col)
            y = [abs(t) for t in y]
            if x:
                any_ = True
                ax.plot(x, y, "-o", color=SERIES[v], lw=2, ms=4, label=LABEL[v])
        ad = _largest_adaptive(case)
        if ad and adaptive_col:
            _, res, hist = ad
            Et = res["case_meta"]["E_true"]
            x = [int(h["cells"]) for h in hist]
            y = [adaptive_col(h, Et) for h in hist]
            ax.plot(x, y, "-", color=SERIES["adaptive"], lw=2, label=LABEL["adaptive"] + f" ({res['method']})")
            any_ = True
        if use_floor:
            fl = [float(r["floor"]) for r in rows if r["case"] == case]
            if fl:
                ax.axhline(fl[0], color=INK2, ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        if use_floor and fl:
            ax.annotate(f"floor sqrt(C_eq C_rel) I_eff^raw = {fl[0]:.1f}", xy=(0.01, fl[0]),
                        xycoords=("axes fraction", "data"), xytext=(0, -10 if case == "1d" else 3), textcoords="offset points",
                        color=INK2, fontsize=7)
            ax.set_ylim(bottom=fl[0] * 0.6)
        _style(ax, title, "cells", ylabel)
    handles = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(list(handles.values()), list(handles.keys()), loc="lower center", ncol=min(len(handles), 6),
               fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=160)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    return
    _save(fig, name)


def plot_runtime(rows):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (case, title) in zip(axes, CASES):
        groups = {}
        for r in rows:
            if r["case"] != case or r["runtime"] in ("", None):
                continue
            v = r["method_variant"]
            key = "adaptive" if v.startswith("adaptive") else v
            if key not in SERIES:
                continue
            groups.setdefault(key, []).append((float(r["runtime"]), float(r["certificate_effectivity"])))
        for k, pts in groups.items():
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o", color=SERIES[k], ms=6, label=LABEL[k])
        ax.set_xscale("log")
        ax.set_yscale("log")
        _style(ax, title, "runtime [s]", "E_cert,U / E_true")
    axes[0].legend(fontsize=7, frameon=False)
    _save(fig, "plot4_runtime_vs_effectivity")


def plot_cells(case="2d_base"):
    ad = _largest_adaptive(case)
    if not ad:
        return
    name, res, _ = ad
    z = np.load(RUNS / "adaptive" / name / "cells.npz")
    lo, hi, depth = z["lo"], z["hi"], z["depth"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    cmap = plt.get_cmap("Blues")
    dmin, dmax = int(np.percentile(depth, 1)), int(depth.max())
    for ax, tq in zip(axes, (0.25, 0.5, 0.75)):
        sel = (lo[:, 0] <= tq) & (hi[:, 0] > tq)
        patches = [Rectangle((lo[i, 1], lo[i, 2]), hi[i, 1] - lo[i, 1], hi[i, 2] - lo[i, 2]) for i in np.where(sel)[0]]
        pc = PatchCollection(patches, facecolor=[cmap(0.1 + 0.9 * min(max(d - dmin, 0) / max(dmax - dmin, 1), 1.0))
                                                 for d in depth[sel]], edgecolor="none")
        ax.add_collection(pc)
        th = np.linspace(0, 2 * math.pi, 400)
        Rt = 0.15 + 0.2 * tq
        ax.plot(0.5 + Rt * np.cos(th), 0.5 + Rt * np.sin(th), color="#eb6834", lw=1.5,
                label="exact free boundary (visualization only)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        _style(ax, f"t = {tq}: {int(sel.sum())} cells cut", "x", "y")
        ax.grid(False)
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    sm = plt.cm.ScalarMappable(cmap=plt.cm.colors.LinearSegmentedColormap.from_list(
        "b", [cmap(0.1), cmap(1.0)]), norm=plt.Normalize(dmin, dmax))
    fig.colorbar(sm, ax=axes, shrink=0.8, label="refinement depth")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"plot5_adaptive_cells_{case}.png", dpi=160)
    fig.savefig(FIG / f"plot5_adaptive_cells_{case}.pdf")
    plt.close(fig)


def plot_bracket(case="2d_base"):
    ad = _largest_adaptive(case)
    if not ad:
        return
    _, res, hist = ad
    x = [int(h["cells"]) for h in hist]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(x, [float(h["U"]) for h in hist], "-", color="#2a78d6", lw=2, label="U (certified upper sum)")
    ax.plot(x, [float(h["L"]) for h in hist], "-", color="#eb6834", lw=2, label="L (lower sum, diagnostic)")
    ax.plot(x, [float(h["rho_h_sq"]) for h in hist], "-", color="#1baf7a", lw=2, label="midpoint rho_h^2")
    ax.axhline(res["case_meta"]["rho_ref"] ** 2, color=INK2, ls="--", lw=1)
    ax.text(x[0], res["case_meta"]["rho_ref"] ** 2 * 1.2, " reference rho_tilde^2 (Sobol 1e6)", color=INK2, fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, f"{case}: bracket [L, U] around rho_tilde^2 ({res['method']})", "cells", "value")
    ax.legend(fontsize=7, frameon=False)
    _save(fig, f"plot6_bracket_{case}")


def main():
    rows = table()
    plot_metric(rows, "certificate_effectivity", "plot1_effectivity_vs_cells", "E_cert,U / E_true", use_floor=True,
                adaptive_col=lambda h, Et: float(h["E_cert_U"]) / Et)
    plot_metric(rows, "E_h_lip", "plot2_Ehlip_vs_cells", "E_h_lip (smooth part of U - rho_h^2)")
    plot_metric(rows, "unresolved_fraction", "plot3_unresolved_vs_cells", "unresolved volume fraction",
                adaptive_col=lambda h, Et: max(float(h["unresolved_fraction"]), 1e-6))
    plot_runtime(rows)
    for c in ("2d_base", "2d_long"):
        plot_cells(c)
        plot_bracket(c)
    print("figures in", FIG)


if __name__ == "__main__":
    main()
