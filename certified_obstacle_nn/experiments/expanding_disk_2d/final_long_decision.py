"""Final-phase Task 4: accept/reject the long Adam schedule against the SEED-0 baseline.

Rule (the 2D finalization protocol, Task 4): accept if E_true improves by > 10 % OR
||lambda - lambda_psi|| improves by > 15 % (relative to the seed-0 baseline run, not the five-seed
mean). Reads only stored metrics.json / train_info.json.

Output: results/expanding_disk_2d/final/long_training_comparison.csv; prints ACCEPT / REJECT.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd  # noqa: E402

from certified_obstacle_nn.experiments.expanding_disk_2d.config import RESULTS_ROOT  # noqa: E402
from certified_obstacle_nn.experiments.expanding_disk_2d.final_common import REFERENCE_6X64, final_dir  # noqa: E402

KEYS = ["E_true", "rho_tilde", "I_eff_raw", "err_u_L2", "err_sigma_L2", "err_lambda_L2", "iface_lambda_mean",
        "iface_lambda_max", "E_FB", "rho_r", "rho_p_tilde", "rho_c", "train_time_s"]
TOL_E, TOL_LAM = 0.10, 0.15


def main():
    base_dir = RESULTS_ROOT / REFERENCE_6X64[0]
    long_dir = final_dir() / "long_6x64_N20000_s0_k5"
    rows = []
    for name, d in (("baseline_seed0", base_dir), ("long_seed0", long_dir)):
        m = json.loads((d / "metrics.json").read_text())
        info = json.loads((d / "train_info.json").read_text())
        rows.append({"run": name, "dir": str(d), **{k: m[k] for k in KEYS},
                     "adam_time_s": info["adam_time_s"], "torch_threads": info["torch_threads"],
                     "lbfgs_outcome": info["lbfgs_outcome"], "lbfgs_steps": info["lbfgs_steps_completed"]})
    df = pd.DataFrame(rows).set_index("run")
    b, lg = df.loc["baseline_seed0"], df.loc["long_seed0"]
    rel = {k: (b[k] - lg[k]) / b[k] for k in KEYS if k not in ("I_eff_raw", "train_time_s")}  # + = improvement
    df.loc["rel_improvement"] = {k: rel.get(k) for k in df.columns}
    accept = rel["E_true"] > TOL_E or rel["err_lambda_L2"] > TOL_LAM
    df["decision"] = ""
    df.loc["rel_improvement", "decision"] = "ACCEPT" if accept else "REJECT"
    df.to_csv(final_dir() / "long_training_comparison.csv")
    pd.set_option("display.width", 250)
    print(df.T.to_string())
    print(f"E_true improvement {100 * rel['E_true']:+.1f}% (threshold {100 * TOL_E:.0f}%), "
          f"lambda-error improvement {100 * rel['err_lambda_L2']:+.1f}% (threshold {100 * TOL_LAM:.0f}%) -> "
          f"{'ACCEPT' if accept else 'REJECT'}")


if __name__ == "__main__":
    main()
