"""Aggregate run metrics into results/expanding_disk_2d/{summary.csv, summary.tex} (mean +- std
over seeds per configuration) and Figure H (E_true vs rho_tilde across configurations).

  python summarize.py --runs <glob> [<glob> ...] --out_prefix summary
Configurations are grouped by (arch, n_int, kappa, tag, band_fraction, weights).
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from certified_obstacle_nn.experiments.expanding_disk_2d import plots  # noqa: E402
from certified_obstacle_nn.experiments.expanding_disk_2d.config import RESULTS_ROOT  # noqa: E402

COLS = ["E_true", "rho_tilde", "I_eff_raw", "err_u_L2", "err_sigma_L2", "err_lambda_L2",
        "iface_lambda_mean", "iface_lambda_max", "iface_gap_mean", "iface_gap_max", "E_FB", "train_time_s",
        "rho_r", "rho_p_tilde", "rho_c", "E_vol_band", "E_vol_away", "err_lambda_band", "err_lambda_away"]


def load(patterns):
    rows = []
    for pat in patterns:
        for d in sorted(glob.glob(pat)):
            d = Path(d)
            if not (d / "metrics.json").exists():
                continue
            c = json.loads((d / "config.json").read_text())
            m = json.loads((d / "metrics.json").read_text())
            w = "/".join(f"{c[k]:g}" for k in ("w_r", "w_s", "w_0", "w_p", "w_c"))
            rows.append({"run": d.name, "arch": c["arch"], "n_int": c["n_int"], "seed": c["seed"],
                         "kappa": c["kappa"], "tag": c["tag"], "band_fraction": c["band_fraction"],
                         "weights": w, "n_params": m["n_params"], **{k: m.get(k, np.nan) for k in COLS}})
    return pd.DataFrame(rows)


def aggregate(df):
    keys = ["arch", "n_int", "kappa", "tag", "band_fraction", "weights"]
    g = df.groupby(keys, sort=False)
    out = g[COLS].mean().add_suffix("_mean").join(g[COLS].std(ddof=1).add_suffix("_std"))
    out["n_seeds"] = g.size()
    out["seeds"] = g["seed"].apply(lambda s: ",".join(map(str, sorted(s))))
    out["n_params"] = g["n_params"].first()
    return out.reset_index()


def _pm(m, s, fmt="{:.3e}"):
    return fmt.format(m) if not np.isfinite(s) else (fmt + r" $\pm$ " + fmt).format(m, s)


def to_tex(agg) -> str:
    lines = [r"\begin{tabular}{lrrlllllllr}", r"\toprule",
             r"Network & Params & $N_{\rm int}$ & $E_{\rm true}$ & $\tilde\rho$ & $I_{\rm eff}^{\rm raw}$ & "
             r"$\|u-u_\theta\|$ & $\|\sigma-\sigma_\phi\|$ & $\|\lambda-\lambda_\psi\|$ & "
             r"interface err (mean/max) & train [s] \\", r"\midrule"]
    for _, r in agg.iterrows():
        tag = r["tag"] if isinstance(r["tag"], str) else ""
        name = r["arch"] + (f" ({tag})" if tag else "") + (f" $\\kappa$={r['kappa']:g}, w={r['weights']}")
        lines.append(" & ".join([
            name, f"{int(r['n_params'])}", f"{int(r['n_int'])}",
            _pm(r["E_true_mean"], r["E_true_std"]), _pm(r["rho_tilde_mean"], r["rho_tilde_std"]),
            _pm(r["I_eff_raw_mean"], r["I_eff_raw_std"], "{:.2f}"),
            _pm(r["err_u_L2_mean"], r["err_u_L2_std"]), _pm(r["err_sigma_L2_mean"], r["err_sigma_L2_std"]),
            _pm(r["err_lambda_L2_mean"], r["err_lambda_L2_std"]),
            f"{r['iface_lambda_mean_mean']:.2e} / {r['iface_lambda_max_mean']:.2e}",
            f"{r['train_time_s_mean']:.0f}"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out_prefix", default="summary")
    a = ap.parse_args()
    df = load(a.runs)
    agg = aggregate(df)
    df.to_csv(RESULTS_ROOT / f"{a.out_prefix}_runs.csv", index=False)
    agg.to_csv(RESULTS_ROOT / f"{a.out_prefix}.csv", index=False)
    (RESULTS_ROOT / f"{a.out_prefix}.tex").write_text(to_tex(agg))
    rows = [{"label": f"{r['arch']} N={int(r['n_int']) // 1000}k k={r['kappa']:g} w={r['weights']}"
             + (f" {r['tag']}" if isinstance(r["tag"], str) and r["tag"] else ""),
             "E_true": r["E_true_mean"], "E_true_std": np.nan_to_num(r["E_true_std"]),
             "rho_tilde": r["rho_tilde_mean"], "rho_tilde_std": np.nan_to_num(r["rho_tilde_std"])}
            for _, r in agg.iterrows()]
    plots.figure_H(rows, str(RESULTS_ROOT / f"H_{a.out_prefix}_error_vs_estimator"))
    pd.set_option("display.width", 250)
    print(agg[["arch", "n_int", "tag", "n_seeds", "E_true_mean", "E_true_std", "rho_tilde_mean", "I_eff_raw_mean",
               "err_lambda_L2_mean", "iface_lambda_mean_mean", "iface_lambda_max_mean", "train_time_s_mean"]])


if __name__ == "__main__":
    main()
