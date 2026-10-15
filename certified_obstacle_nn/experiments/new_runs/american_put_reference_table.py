"""NEW_RUNS.md Experiment 2, section 2.1 -- reference-convergence table.

Does NOT modify reference/american_put_reference.py in any way (NEW_RUNS.md:
"Do not modify the reference solver merely to make the neural error look
smaller"). Calls the SAME unmodified solve_psor/PSORConfig at the SAME six
refinement levels experiments/american_put_reference_convergence.py already
established (REFINEMENT_LEVELS, imported read-only from that module) and adds
one thing that script did not report: price at THREE representative points
(below/near/above strike), not just the single at-the-money point. The
near-strike point is exactly problem.x_star (= log(K)), so its column is a
direct cross-check against that script's own already-saved
value_at_reference_point_per_level (results/american_put_reference_convergence.json)
-- both must agree to machine precision, verified below.

The finest level (n=6400) is the ALREADY-FROZEN reference
(results/american_put_reference/, config_hash 5f43facd47d2251c) -- used for
every neural comparison in this experiment, per NEW_RUNS.md's "the finest
accepted reference should be used for all neural comparisons."
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from certified_obstacle_nn.experiments.american_put_reference_convergence import REFINEMENT_LEVELS
from certified_obstacle_nn.experiments.new_runs.common import NEW_RUNS_DIR
from certified_obstacle_nn.problems.american_put import AmericanPutProblem
from certified_obstacle_nn.reference.american_put_reference import FROZEN_DIR, PSORConfig, load_reference, solve_psor

EXPERIMENT = "american_put_improved"
OUT_DIR = NEW_RUNS_DIR / EXPERIMENT

# S below/at/above strike (K=100), converted to the log-price coordinate the
# solver actually works in -- all three comfortably inside omega_bounds=(-1,7).
S_POINTS = {"price_below_strike_S70": 70.0, "price_near_strike_S100": 100.0, "price_above_strike_S130": 130.0}


def _price_at(x_grid, u_row, problem: AmericanPutProblem, S: float) -> float:
    """V(tau, S) = u(tau, x) + u_lift(x) at x = log(S), linear interpolation
    on the already-solved grid (same interpolation convention
    solve_psor itself uses for value_at_reference_point)."""
    import numpy as np
    x = math.log(S)
    u_val = float(np.interp(x, x_grid, u_row))
    u_lift = float(problem.C * (problem.R - x) / (problem.R - problem.L))
    return u_val + u_lift


def main() -> pd.DataFrame:
    problem = AmericanPutProblem()
    rows = []
    prev_boundary = None
    prev_tau = None

    for n in REFINEMENT_LEVELS:
        config = PSORConfig(n_x=n, n_tau=n)
        sol = solve_psor(problem, config)
        print(f"n_x=n_tau={n}: value_at_reference_point={sol.value_at_reference_point:.8f}  "
              f"wall_clock={sol.wall_clock_seconds:.2f}s", flush=True)

        row = {"Nx": n, "Nt": n, "runtime_seconds": sol.wall_clock_seconds}
        for label, S in S_POINTS.items():
            row[label] = _price_at(sol.x_grid, sol.u[-1, :], problem, S)

        if prev_boundary is not None:
            import numpy as np
            prev_on_cur = np.interp(sol.tau_grid, prev_tau, prev_boundary, left=np.nan, right=np.nan)
            valid = ~(np.isnan(prev_on_cur) | np.isnan(sol.exercise_boundary))
            row["boundary_error_vs_previous_level"] = float(np.abs(sol.exercise_boundary[valid] - prev_on_cur[valid]).max())
        else:
            row["boundary_error_vs_previous_level"] = None

        rows.append(row)
        prev_boundary, prev_tau = sol.exercise_boundary, sol.tau_grid

        # near-strike point must reproduce value_at_reference_point exactly
        # (both are u(tau=T, x=log K) + u_lift(log K) vs. the RAW LIFTED
        # value_at_reference_point -- so compare after adding u_lift back).
        u_lift_at_star = problem.C * (problem.R - problem.x_star) / (problem.R - problem.L)
        expected = sol.value_at_reference_point + u_lift_at_star
        assert abs(row["price_near_strike_S100"] - expected) < 1e-9, (
            f"near-strike price ({row['price_near_strike_S100']}) does not match "
            f"solve_psor's own value_at_reference_point+u_lift ({expected}) at n={n} -- cross-check failed"
        )

    df = pd.DataFrame.from_records(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "reference_convergence.csv"
    df.to_csv(out_path, index=False)
    print(f"\nreference_convergence.csv -> {out_path}")
    print(df.to_string(index=False))

    frozen_config = json.loads((FROZEN_DIR / "config.json").read_text())
    print(f"\nFrozen reference in use for all neural comparisons: {FROZEN_DIR} "
          f"(n_x=n_tau={frozen_config['n_x']}, config_hash={frozen_config['config_hash']})")

    return df


if __name__ == "__main__":
    main()
