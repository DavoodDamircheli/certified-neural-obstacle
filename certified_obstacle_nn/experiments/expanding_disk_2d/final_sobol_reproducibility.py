"""Final-phase Task 2: Sobol integration reproducibility (no retraining).

For the five 6x64 reference checkpoints, re-evaluate E_true, rho_tilde (rho_r, rho_p_tilde, rho_c)
and I_eff_raw with 3 independent fresh Sobol scrambles at the original sizes (n_eval = 1e6 in Q,
2e5 per trace set at t = 0 and t = T), each set with its own fresh scramble seed:
  replicate k = 1..3:  Q: 1_000_000 + 1000 k + s,  t=0: 2_000_000 + 1000 k + s,  t=T: 3_000_000 + 1000 k + s
(disjoint from every training / monitor / original-eval seed). Replicate k = 0 re-runs the ORIGINAL
scrambles as a determinism check against the stored metrics.json; it is reported but not used in
the variability statistics.

  integration std (per seed) = std over the 3 fresh scrambles (ddof = 1); averaged over seeds
  seed std                    = std over the 5 seeds per scramble (ddof = 1); averaged over scrambles
  ratio                       = integration std / seed std   (expected << 1)

Outputs: results/expanding_disk_2d/final/sobol_reproducibility.csv (one row per seed x replicate)
         results/expanding_disk_2d/final/sobol_reproducibility_summary.csv (per quantity)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd  # noqa: E402

from certified_obstacle_nn.experiments.expanding_disk_2d import evaluate  # noqa: E402
from certified_obstacle_nn.experiments.expanding_disk_2d.config import RESULTS_ROOT  # noqa: E402
from certified_obstacle_nn.experiments.expanding_disk_2d.final_common import REFERENCE_6X64, final_dir, load_run  # noqa: E402

QUANTITIES = ["E_true", "rho_tilde", "rho_r", "rho_p_tilde", "rho_c", "I_eff_raw"]
N_REPLICATES = 3


def scramble_seeds(k: int, s: int, cfg) -> tuple[int, int, int]:
    if k == 0:
        return cfg.sobol_seeds["eval"], cfg.sobol_seeds["eval_t0"], cfg.sobol_seeds["eval_T"]
    return 1_000_000 + 1000 * k + s, 2_000_000 + 1000 * k + s, 3_000_000 + 1000 * k + s


def main():
    out = final_dir()
    rows = []
    for s, run in REFERENCE_6X64.items():
        triple, cfg = load_run(run)
        stored = json.loads((RESULTS_ROOT / run / "metrics.json").read_text())
        for k in range(N_REPLICATES + 1):
            sd = scramble_seeds(k, s, cfg)
            m = evaluate.evaluate_all(triple, cfg, n_eval=cfg.n_eval, n_trace=cfg.n_eval_trace, seeds=sd)
            row = {"seed": s, "run": run, "replicate": k, "original_scramble": k == 0,
                   "sobol_seed_Q": sd[0], "sobol_seed_t0": sd[1], "sobol_seed_T": sd[2],
                   "n_eval": cfg.n_eval, "n_eval_trace": cfg.n_eval_trace, **{q: m[q] for q in QUANTITIES}}
            if k == 0:
                row["max_rel_diff_vs_stored"] = max(abs(m[q] - stored[q]) / abs(stored[q]) for q in QUANTITIES)
            rows.append(row)
            print(f"seed {s} rep {k}: " + " ".join(f"{q}={m[q]:.5f}" for q in QUANTITIES), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "sobol_reproducibility.csv", index=False)

    fresh = df[~df["original_scramble"]]
    summ = []
    for q in QUANTITIES:
        integ = fresh.groupby("seed")[q].std(ddof=1)        # per seed, across scrambles
        seedv = fresh.groupby("replicate")[q].std(ddof=1)   # per scramble, across seeds
        summ.append({"quantity": q, "mean_over_all": fresh[q].mean(),
                     **{f"integration_std_seed{s}": integ[s] for s in integ.index},
                     "integration_std_mean": integ.mean(), "integration_std_max": integ.max(),
                     "seed_std_mean": seedv.mean(), "ratio_integration_over_seed": integ.mean() / seedv.mean(),
                     "integration_rel_std_mean": (integ / fresh.groupby("seed")[q].mean()).mean()})
    sdf = pd.DataFrame(summ)
    sdf.to_csv(out / "sobol_reproducibility_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(sdf[["quantity", "mean_over_all", "integration_std_mean", "integration_std_max", "seed_std_mean",
               "ratio_integration_over_seed", "integration_rel_std_mean"]].to_string(index=False))
    print("max rel diff of original-scramble re-run vs stored metrics:", df["max_rel_diff_vs_stored"].max())
    bad = sdf[sdf["ratio_integration_over_seed"] > 0.1]
    if len(bad):
        print("WARNING: integration/seed std ratio > 0.1 for:", list(bad["quantity"]))


if __name__ == "__main__":
    main()
