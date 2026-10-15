"""Final-phase Task 6: Stage-0 continuity record for f (scaling test, decision D0.1).

Same straddling pairs as verify_exact_problem.py check 7 (20 000 random (t, theta), generator seed 1,
float64): points at radii R(t) -+ delta on the same ray. For delta in {1e-3, ..., 1e-8} record the raw
jump |f(R+delta) - f(R-delta)| (max and mean over pairs), the successive ratio jump(10 delta)/jump(delta)
(continuity with a kink => ~10, i.e. O(delta); a genuine jump => ~1), and the slope jump/(2 delta)
against the analytic kink bound kappa (R' + 4/R0). exact_solution.py is not modified.

Output: results/expanding_disk_2d/final/f_continuity_scaling.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from certified_obstacle_nn.experiments.expanding_disk_2d import exact_solution as ex  # noqa: E402
from certified_obstacle_nn.experiments.expanding_disk_2d.final_common import final_dir  # noqa: E402

DTYPE = torch.float64
N_PAIRS = 20_000
SEED = 0  # verify_exact_problem.SEED; pairs use generator seed SEED + 1, as there
DELTAS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)


def main(kappa: float = ex.KAPPA_DEFAULT):
    gen = torch.Generator().manual_seed(SEED + 1)
    t = torch.rand(N_PAIRS, 1, generator=gen, dtype=DTYPE)
    theta = torch.rand(N_PAIRS, 1, generator=gen, dtype=DTYPE) * (2.0 * np.pi)
    Rt, c, s = ex.R(t), torch.cos(theta), torch.sin(theta)
    bound = kappa * (ex.R_SPEED + 4.0 / ex.R0)
    rows, prev = [], None
    for d in DELTAS:
        f_in = ex.f_exact(t, ex.C0 + (Rt - d) * c, ex.C0 + (Rt - d) * s, kappa)
        f_out = ex.f_exact(t, ex.C0 + (Rt + d) * c, ex.C0 + (Rt + d) * s, kappa)
        jump = (f_out - f_in).abs()
        jmax, jmean = jump.max().item(), jump.mean().item()
        rows.append({"delta": d, "total_distance": 2 * d, "jump_max": jmax, "jump_mean": jmean,
                     "ratio_prev_over_this_max": (prev[0] / jmax) if prev else np.nan,
                     "ratio_prev_over_this_mean": (prev[1] / jmean) if prev else np.nan,
                     "slope_max": jmax / (2 * d), "kink_slope_bound": bound, "n_pairs": N_PAIRS, "kappa": kappa})
        prev = (jmax, jmean)
    df = pd.DataFrame(rows)
    df.to_csv(final_dir() / "f_continuity_scaling.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    ok = df["ratio_prev_over_this_max"].dropna().between(9.0, 11.0).all() and (df["slope_max"] <= 1.01 * bound).all()
    print("scaling test:", "PASS (jump = O(delta))" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
