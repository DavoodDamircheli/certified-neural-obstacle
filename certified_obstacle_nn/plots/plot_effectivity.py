"""The central M18 figure, specification section 8 (2026-09-11).

Two panels, both against parameter count (log x-axis), aggregated per
architecture (mean +/- std over seeds -- specification's own "single-seed
results are not evidence"):

  left:  E_true, rho_tilde_h, sqrt(rho_tilde_h^2 + E_h) (= E_quad),
         E_cert, E_cert_U -- the whole accuracy-vs-certified-bound picture
         on one log-log axis. This is what makes T3's finding visible: E_true
         goes DOWN with architecture size while E_cert_U goes UP.
  right: I_eff_cert (with the theoretical floor 1/sqrt(432), specification
         section 9, drawn as a horizontal line) and I_eff_raw (NO floor
         line -- specification section 8 is explicit that I_eff_raw has none;
         plotted alongside only for visual contrast with I_eff_cert, never
         with a floor of its own).

Expects a Table-1-shaped DataFrame (experiments/stefan_core.py's own
table1.csv: one row per (Architecture, Seed), with param_count, E_true,
rho_tilde, E_quad, E_cert, E_cert_U, I_eff_raw, I_eff_cert columns) --
this module does no certification/training of its own, purely plotting.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

I_EFF_CERT_FLOOR = 1.0 / (432.0 ** 0.5)  # specification section 9: 1/sqrt(C_eq*C_rel) = 1/sqrt(432)

_LEFT_PANEL_SERIES = [
    ("E_true", "E_true", "o"),
    ("rho_tilde", "rho_tilde_h", "s"),
    ("E_quad", "sqrt(rho_tilde_h^2 + E_h)", "^"),
    ("E_cert", "E_cert", "D"),
    ("E_cert_U", "E_cert_U", "*"),
]


def _aggregate_by_architecture(table1_df: pd.DataFrame, arch_order: list[str]) -> pd.DataFrame:
    g = table1_df.groupby("Architecture", observed=True)
    agg = g.agg(
        param_count=("param_count", "mean"),
        **{f"{c}_mean": (c, "mean") for c in [
            "E_true", "rho_tilde", "E_quad", "E_cert", "E_cert_U", "I_eff_raw", "I_eff_cert",
        ]},
        **{f"{c}_std": (c, "std") for c in [
            "E_true", "rho_tilde", "E_quad", "E_cert", "E_cert_U", "I_eff_raw", "I_eff_cert",
        ]},
    )
    return agg.reindex(arch_order)


def plot_effectivity(
    table1_df: pd.DataFrame,
    arch_order: list[str] | None = None,
    save_path: str | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    if arch_order is None:
        arch_order = sorted(table1_df["Architecture"].unique(), key=lambda s: table1_df.loc[table1_df["Architecture"] == s, "param_count"].iloc[0])

    agg = _aggregate_by_architecture(table1_df, arch_order)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

    # --- left panel: the five error/bound quantities vs param_count ---
    for col, label, marker in _LEFT_PANEL_SERIES:
        ax_left.errorbar(
            agg["param_count"], agg[f"{col}_mean"], yerr=agg[f"{col}_std"],
            marker=marker, label=label, capsize=3, linewidth=1.5,
        )
    ax_left.set_xscale("log")
    ax_left.set_yscale("log")
    ax_left.set_xlabel("parameter count")
    ax_left.set_ylabel("error / bound (log scale)")
    ax_left.set_title("Accuracy vs. certified bound by architecture size")
    ax_left.legend(loc="best", fontsize=9)
    ax_left.grid(True, which="both", alpha=0.3)

    # --- right panel: I_eff_cert (with floor) and I_eff_raw (no floor) ---
    ax_right.errorbar(
        agg["param_count"], agg["I_eff_cert_mean"], yerr=agg["I_eff_cert_std"],
        marker="D", color="tab:red", label="I_eff_cert", capsize=3, linewidth=1.5,
    )
    ax_right.axhline(
        I_EFF_CERT_FLOOR, color="tab:red", linestyle="--", linewidth=1.2,
        label=f"I_eff_cert floor = 1/sqrt(432) = {I_EFF_CERT_FLOOR:.4f}",
    )
    ax_right.errorbar(
        agg["param_count"], agg["I_eff_raw_mean"], yerr=agg["I_eff_raw_std"],
        marker="o", color="tab:blue", label="I_eff_raw (no theoretical floor)", capsize=3, linewidth=1.5,
    )
    ax_right.set_xscale("log")
    ax_right.set_yscale("log")
    ax_right.set_xlabel("parameter count")
    ax_right.set_ylabel("effectivity (log scale)")
    ax_right.set_title("Effectivity: I_eff_cert (floored) vs. I_eff_raw (unfloored)")
    ax_right.legend(loc="best", fontsize=9)
    ax_right.grid(True, which="both", alpha=0.3)

    for arch, row in agg.iterrows():
        ax_left.annotate(arch, (row["param_count"], row["E_cert_U_mean"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax_right.annotate(arch, (row["param_count"], row["I_eff_cert_mean"]), textcoords="offset points", xytext=(4, 4), fontsize=8)

    fig.suptitle("M18: certified bound and effectivity vs. architecture size")
    fig.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    return fig, agg


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    table1_path = repo_root / "results" / "stefan_m18" / "table1.csv"
    df = pd.read_csv(table1_path)
    out_path = repo_root / "results" / "stefan_m18_effectivity.png"
    _, agg = plot_effectivity(df, arch_order=["4x32", "4x64", "6x64", "6x128"], save_path=str(out_path))
    print(agg)
    print(f"figure -> {out_path}")
