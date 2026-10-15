"""v14 Stage 7: build every v14 table from results/v14_*/<run>/summary.csv only
(no hand-typed numbers, no recomputation).

    python3 -m certified_obstacle_nn.scripts.make_tables [--include-dirty] [--out DIR]

Writes <out>/tables.md and one CSV per table (default out:
results/v14_tables/<timestamp>/). Runs whose summary rows record
git_dirty = True (code not committed when the run was made) are excluded
unless --include-dirty is given; the exclusion count is printed in tables.md.
Every table carries the status column of the rows it is built from.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from certified_obstacle_nn.experiments.v14.common import read_summary

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ("v14_stefan_validated", "v14_disk_validated", "v14_ellipse_validated")


def load(include_dirty: bool):
    frames = []
    for exp in EXPERIMENTS:
        for f in sorted(glob.glob(str(REPO_ROOT / "results" / exp / "*" / "summary.csv"))):
            df = read_summary(f)
            df["run_dir"] = str(Path(f).parent.relative_to(REPO_ROOT))
            df["experiment"] = exp
            frames.append(df)
    if not frames:
        return pd.DataFrame(), 0
    df = pd.concat(frames, ignore_index=True)
    if "problem" not in df:
        df["problem"] = np.nan
    df["problem"] = df["problem"].fillna("stefan")
    if "schedule" not in df:
        df["schedule"] = np.nan
    df["schedule"] = df["schedule"].fillna("4x64")
    df["size"] = df["n_cells"]
    if "budget" in df:
        df.loc[df["mode"] == "adaptive", "size"] = df.loc[df["mode"] == "adaptive", "budget"].fillna(
            df.loc[df["mode"] == "adaptive", "n_cells"])
    n_dirty = int(df["git_dirty"].astype(bool).sum())
    if not include_dirty:
        df = df[~df["git_dirty"].astype(bool)]
    # repeated evaluations of the same (checkpoint, partition, backend): keep the latest run directory
    df = df.sort_values("run_dir").drop_duplicates(
        ["experiment", "problem", "schedule", "seed", "mode", "size", "backend"], keep="last")
    return df, n_dirty


def fmt(x, nd=4):
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    ax = abs(x)
    if ax != 0 and (ax < 1e-3 or ax >= 1e4):
        return f"{x:.3e}"
    return f"{x:.{nd}g}"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def status_of(s: pd.Series) -> str:
    u = sorted(set(s.astype(str)))
    return u[0] if len(u) == 1 else "MIXED:" + "/".join(u)


BOUND_COLS = ["U_r", "U_p", "U_c", "U_r_hat", "E_U_uniform_216_comparison_convention", "E_U_uniform", "E_U_weighted",
              "E_U_delta_1_6", "E_U_delta_1_12", "E_U_delta_star", "E_U_best", "E_U_projected", "E_U_original_via_projection", "E_true_v", "E_true_vhat",
              "ratio_E_U_weighted", "ratio_E_U_projected_vs_vhat", "ratio_E_U_original", "grad_bound_rel",
              "flux_bound_rel", "grad_bound_best_rel"]


def preferred_backend(df):
    """arb (reference) rows where present, else numpy (D7)."""
    keys = ["problem", "schedule", "seed", "mode", "size"]
    has_arb = df[df["backend"] == "arb"].set_index(keys).index
    pick = (df["backend"] == "arb") | ~df.set_index(keys).index.isin(has_arb)
    return df[pick.values]


def headline(df):
    d = preferred_backend(df[(df["problem"] == "stefan") & (df["seed"] == 0)])
    d = d.sort_values(["mode", "n_cells"])
    return d[["mode", "n_cells", "backend", "status"] + BOUND_COLS].assign(ratios_status=RATIO_STATUS)


# The row 'status' labels the U's and every bound computed from them. E_true_*, norm_*, ratio_* and *_rel
# columns use float64 quadrature (E_true, ||grad u||, ||sigma||): they are diagnostics, labelled per row.
RATIO_STATUS = "diagnostic_float64"
STAGE23_COLS = [
    "problem", "schedule", "mode", "size", "seed", "n_cells", "backend", "status", "C_eq", "C_rel",
    "U_r", "U_p", "U_c", "U_r_hat",
    "E_U_uniform_216_comparison_convention", "E_U_uniform", "E_U_weighted",
    "E_U_delta_1_6", "E_U_delta_1_12", "delta_star_rational", "E_U_delta_star", "E_U_best",
    "E_U_projected", "E_U_original_via_projection",
    "grad_bound", "flux_bound", "grad_bound_best", "flux_bound_best",
    "ratios_status", "norm_grad_u", "norm_sigma", "grad_bound_rel", "flux_bound_rel",
    "E_true_v", "E_true_vhat", "ratio_E_U_weighted", "ratio_E_U_projected_vs_vhat", "ratio_E_U_original",
]


def stage23_bounds(df):
    """Every Stage 2/3 bound variant per run and seed (projected vs E_true(v_hat), original vs E_true(v))."""
    d = preferred_backend(df).assign(ratios_status=RATIO_STATUS)
    return d.sort_values(["problem", "schedule", "mode", "size", "seed"])[STAGE23_COLS]


def seed_stats(df):
    d = preferred_backend(df)
    out = []
    for key, g in d.groupby(["problem", "schedule", "mode", "size"]):
        row = {"problem": key[0], "schedule": key[1], "mode": key[2], "size": key[3], "n_seeds": g["seed"].nunique(),
               "seeds": ",".join(str(int(s)) for s in sorted(g["seed"].unique())), "backend": status_of(g["backend"]),
               "status": status_of(g["status"]), "ratios_status": RATIO_STATUS}
        for c in ("E_true_v", "E_U_weighted", "ratio_E_U_weighted", "E_U_projected", "ratio_E_U_projected_vs_vhat",
                  "E_U_original_via_projection", "grad_bound_rel", "flux_bound_rel"):
            v = g[c].astype(float)
            row[c + "_mean"] = v.mean()
            row[c + "_sd"] = v.std(ddof=1) if len(v) > 1 else np.nan
        out.append(row)
    return pd.DataFrame(out)


def per_seed(df):
    d = preferred_backend(df)
    return d.sort_values(["problem", "schedule", "mode", "size", "seed"])[
        ["problem", "schedule", "mode", "size", "seed", "n_cells", "backend", "status", "U_r", "U_p", "U_c",
         "E_true_v", "E_U_weighted", "ratio_E_U_weighted", "E_U_projected", "E_true_vhat",
         "ratio_E_U_projected_vs_vhat", "E_U_original_via_projection", "grad_bound_rel", "flux_bound_rel"]
    ].assign(ratios_status=RATIO_STATUS)


def largest_per_seed(df):
    d = preferred_backend(df)
    d = d.sort_values("n_cells")
    return d.groupby(["problem", "schedule", "seed"], as_index=False).tail(1).sort_values(["problem", "schedule", "seed"])


def bound_decomposition(df):
    """E_U_weighted / E_true = sqrt(C_eq B / U) [constant] x sqrt(U / rho^2) [enclosure slack] x rho / E_true
    [raw effectivity], rho^2 = diag_r2 + diag_p2 + diag_c2 (converged float64 integrals); projected:
    sqrt(6 C_eq) x sqrt((U_r_hat + U_c) / (I_r_hat + I_c)) x sqrt(I_r_hat + I_c) / E_true(v_hat). Largest cell
    budget per (problem, schedule, seed). A gradient/flux bound above ||grad u|| / ||sigma|| is NOT informative."""
    d = largest_per_seed(df).copy()
    f = lambda c: d[c].astype(float)
    U = f("U_r") + f("U_p") + f("U_c")
    rho2 = f("diag_r2") + f("diag_p2") + f("diag_c2")
    I_proj = f("diag_r2_hat") + f("diag_c2")
    d["constant_factor"] = np.sqrt(f("C_eq") * f("B_weighted") / U)
    d["enclosure_slack_factor"] = np.sqrt(U / rho2)
    d["raw_effectivity"] = np.sqrt(rho2) / f("E_true_v")
    d["proj_constant_factor"] = np.sqrt(6 * f("C_eq"))
    d["proj_slack_factor"] = np.sqrt((f("U_r_hat") + f("U_c")) / I_proj)
    d["proj_raw_effectivity"] = np.sqrt(I_proj) / f("E_true_vhat")
    d["dominant_factor"] = np.where(d["constant_factor"] >= d["enclosure_slack_factor"], "constant", "enclosure_slack")
    d["gradient_bound"] = np.where(f("grad_bound_rel") < 1, "informative", "NOT informative")
    d["flux_bound"] = np.where(f("flux_bound_rel") < 1, "informative", "NOT informative")
    d["ratios_status"] = RATIO_STATUS
    return d[["problem", "schedule", "seed", "mode", "size", "n_cells", "backend", "status", "ratio_E_U_weighted",
              "constant_factor", "enclosure_slack_factor", "raw_effectivity", "dominant_factor",
              "slack_U_r", "slack_U_p", "slack_U_c", "ratio_E_U_projected_vs_vhat", "proj_constant_factor",
              "proj_slack_factor", "proj_raw_effectivity", "grad_bound_rel", "gradient_bound", "flux_bound_rel",
              "flux_bound", "ratios_status"]]


def cost_summary(df):
    """Leaf-evaluation cost per cell by problem and backend (wall at the recorded worker count, and CPU)."""
    rows = []
    for key, g in df.groupby(["problem", "backend"]):
        rows.append({"problem": key[0], "backend": key[1], "n_runs": len(g),
                     "workers": ",".join(str(int(w)) for w in sorted(g["workers"].unique())),
                     "ms_per_cell_wall_median": 1e3 * g["s_per_cell_wall"].astype(float).median(),
                     "ms_per_cell_cpu_median": 1e3 * g["s_per_cell_cpu"].astype(float).median(),
                     "max_cells": int(g["n_cells"].max()),
                     "max_t_total_post_training_h": g["t_total_post_training_s"].astype(float).max() / 3600})
    return pd.DataFrame(rows)


def meanvalue_comparison(_df):
    """Mean-value vs TM1 enclosure on the same frozen Stefan partition (D16), clean runs only."""
    frames = []
    for f in sorted(glob.glob(str(REPO_ROOT / "results" / "v14_stefan_meanvalue" / "*" / "summary.csv"))):
        d = read_summary(f)
        d["run_dir"] = str(Path(f).parent.relative_to(REPO_ROOT))
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d = d[~d["git_dirty"].astype(bool)]
    return d.sort_values(["seed", "budget", "backend", "enclosure"])[
        ["seed", "budget", "n_cells", "backend", "enclosure", "status", "containment_gate", "slack_U_r", "slack_U_p",
         "slack_U_c", "slack_U_r_hat", "ratio_E_U_weighted", "ratio_E_U_projected_vs_vhat",
         "median_R_width_ratio_mv_over_tm1", "frac_cells_mv_narrower", "s_per_cell_wall", "workers",
         "partition_hash_matches_committed", "run_dir"]]


def accuracy(df):
    d = preferred_backend(df).drop_duplicates(["problem", "schedule", "seed"])
    return d.sort_values(["problem", "schedule", "seed"])[
        ["problem", "schedule", "seed", "E_true_v", "E_true_grad", "E_true_flux", "E_true_graph", "E_true_init",
         "E_true_term", "E_true_mult_L2", "E_true_vhat", "norm_grad_u", "norm_sigma", "E_true_rel_change"]]


def slack(df):
    d = preferred_backend(df)
    cols = ["problem", "schedule", "mode", "size", "seed", "n_cells", "max_cell_diam", "slack_U_r", "slack_U_p",
            "slack_U_c", "slack_U_r_hat"]
    t = d.sort_values(cols[:5])[cols].copy()
    t.insert(5, "label", "diagnostic ratio (validated U / float64 quadrature)")
    return t


def cost(df):
    cols = ["problem", "schedule", "mode", "size", "seed", "backend", "prec_bits", "n_cells", "workers",
            "t_partition_s", "t_leaf_eval_s", "t_trace_s", "t_reduction_s", "s_per_cell_wall", "s_per_cell_cpu",
            "t_total_post_training_s", "peak_rss_gb", "training_time_s", "network"]
    return df.sort_values(cols[:6])[[c for c in cols if c in df]]


def backend_agreement(df):
    rows = []
    for key, g in df.groupby(["problem", "schedule", "seed", "mode", "size"]):
        if set(g["backend"]) >= {"arb", "numpy"}:
            a = g[g["backend"] == "arb"].iloc[0]
            n = g[g["backend"] == "numpy"].iloc[0]
            rows.append({"problem": key[0], "schedule": key[1], "seed": key[2], "mode": key[3], "size": key[4],
                         "same_partition": a["partition_hash"] == n["partition_hash"],
                         **{f"rel_diff_{c}": abs(a[c] - n[c]) / abs(a[c]) for c in ("U_r", "U_p", "U_c", "U_r_hat")}})
    return pd.DataFrame(rows)


def training_runs():
    """Training records of the problems trained in v14 (results/v14_ellipse_train/<run>/summary.csv)."""
    frames = []
    for f in sorted(glob.glob(str(REPO_ROOT / "results" / "v14_ellipse_train" / "*" / "summary.csv"))):
        d = read_summary(f)
        d.insert(0, "run_dir", str(Path(f).parent.relative_to(REPO_ROOT)))
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    cols = ["problem", "variant", "seed", "train_time_s", "lbfgs_outcome", "lbfgs_steps_completed", "final_loss",
            "E_true", "E_true_vhat", "E_grad", "E_flux", "E_graph", "E_mult_L2", "norm_grad_u", "r2", "p2", "c2",
            "status_E_true", "git_dirty", "run_dir"]
    return d.sort_values(["problem", "variant", "seed"])[[c for c in cols if c in d]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-dirty", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df, n_dirty = load(a.include_dirty)
    out = Path(a.out) if a.out else REPO_ROOT / "results/v14_tables" / time.strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "headline_stefan_seed0": ("Stefan seed 0 (4x64): every bound variant against E_true "
                                  "(projected bound against E_true(v_hat))", headline),
        "stage23_bounds": ("Stage 2/3 bounds, every variant, every seed (projected vs E_true(v_hat); "
                           "original-output vs E_true(v); ratio columns diagnostic_float64)", stage23_bounds),
        "seed_statistics": ("Mean and sample standard deviation over seeds", seed_stats),
        "per_seed": ("Individual seeds", per_seed),
        "accuracy": ("True-error components (diagnostic_float64, exact solution)", accuracy),
        "enclosure_slack": ("Enclosure slack: validated U / converged float64 integral (diagnostic ratio)", slack),
        "bound_decomposition": ("Bound decomposition at the largest budget per seed: constant x enclosure slack x raw "
                                "effectivity (ratios diagnostic_float64)", bound_decomposition),
        "cost_summary": ("Cost per cell by problem and backend (medians over runs)", cost_summary),
        "meanvalue_comparison": ("Mean-value vs TM1 enclosure, same frozen Stefan partition (D16)",
                                 meanvalue_comparison),
        "cost": ("Cost and resources", cost),
        "backend_agreement": ("Reference (arb) vs fast (numpy) backend on the same frozen partition", backend_agreement),
    }
    parts = [f"# v14 tables\n\nBuilt by `scripts/make_tables.py` from {df['run_dir'].nunique() if len(df) else 0} "
             f"run directories ({len(df)} summary rows). Runs with uncommitted code (git_dirty) "
             f"{'INCLUDED' if a.include_dirty else 'excluded'}: {n_dirty} rows.\n"
             "Ratios 'E_U / E_true' are bound / true error; 'grad_bound_rel' = sqrt(B/alpha) / ||grad u|| "
             "(> 1 means the gradient guarantee is weaker than the trivial scale).\n"
             "Row 'status' applies to the U's and to every bound derived from them; E_true_*, norm_*, ratio_* and "
             "*_rel columns use float64 quadrature and are diagnostic_float64 ('ratios_status').\n"]
    manifest_runs = sorted(df["run_dir"].unique()) if len(df) else []
    tables["training"] = ("v14 training runs (ellipse; float64 diagnostics at the end of training)", None)
    for name, (title, fn) in tables.items():
        t = training_runs() if fn is None else (fn(df) if len(df) else pd.DataFrame())
        t.to_csv(out / f"{name}.csv", index=False)
        parts.append(f"## {title}\n\n`{name}.csv`\n\n" + (md(t) if len(t) else "(no rows)") + "\n")
    (out / "tables.md").write_text("\n".join(parts))
    (out / "sources.json").write_text(json.dumps({"runs": manifest_runs, "include_dirty": a.include_dirty}, indent=1))
    print(out)


if __name__ == "__main__":
    main()
