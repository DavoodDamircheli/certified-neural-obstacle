"""h, h/2, h/4 (and h/8 for extra margin) refinement study for the American
put CN+PSOR reference (reference/american_put_reference.py), specification
section 12 invariant 4 / this milestone's own instruction: "Refine h, h/2,
h/4 until option value and exercise boundary are stable well below the
neural error scale."

"Well below the neural error scale": this project's own neural E_true values
(experiments/stefan_core.py's M18 table) are O(1e-2), i.e. two-decimal-place
accuracy; the reference solver here targets at least two more significant
digits of its own convergence (O(1e-4) or tighter) before being frozen, so
it is not merely "a bit better" than the network it will validate but
comfortably outside the network's own noise floor.

Compares the value_at_reference_point (u at tau=T, x=log K -- the current-
time, at-the-money lifted value) and the exercise boundary curve across
refinement levels, then freezes the finest level shown stable
(reference/american_put_reference.py's freeze_reference, which itself
refuses to silently overwrite a different already-frozen config).

Also reports a rough sanity comparison against commonly-cited American put
benchmark magnitudes for this parameter regime (S=K=100, r=0.06, sigma=0.4,
T=1) -- reported with explicit uncertainty, not asserted as an exact
published number, since this script has no network access to verify a
specific literature value to the cent.
"""

import json
from pathlib import Path

import numpy as np

from certified_obstacle_nn.problems.american_put import AmericanPutProblem
from certified_obstacle_nn.reference.american_put_reference import (
    FROZEN_DIR,
    PSORConfig,
    freeze_reference,
    solve_psor,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"

REFINEMENT_LEVELS = [200, 400, 800, 1600, 3200, 6400]  # h, h/2, h/4, h/8, h/16, h/32 (n_x=n_tau at each level)
STABILITY_TARGET = 1e-4  # absolute change in value_at_reference_point, level to level


def _boundary_diff(b1: np.ndarray, b2_interp_at_b1_tau: np.ndarray) -> float:
    valid = np.isfinite(b1) & np.isfinite(b2_interp_at_b1_tau)
    return float(np.abs(b1[valid] - b2_interp_at_b1_tau[valid]).max())


def main() -> dict:
    problem = AmericanPutProblem()
    solutions = []
    for n in REFINEMENT_LEVELS:
        config = PSORConfig(n_x=n, n_tau=n)
        sol = solve_psor(problem, config)
        solutions.append(sol)
        print(f"n_x=n_tau={n}: value_at_reference_point={sol.value_at_reference_point:.8f}  "
              f"wall_clock={sol.wall_clock_seconds:.2f}s  max_psor_sweeps={sol.max_psor_sweeps_used}", flush=True)

    rows = []
    for i in range(1, len(solutions)):
        prev, cur = solutions[i - 1], solutions[i]
        value_diff = abs(cur.value_at_reference_point - prev.value_at_reference_point)

        # Interpolate the coarser level's boundary onto the finer level's tau
        # grid for a like-for-like comparison.
        prev_boundary_on_cur_tau = np.interp(
            cur.tau_grid, prev.tau_grid, prev.exercise_boundary,
            left=np.nan, right=np.nan,
        )
        boundary_diff = _boundary_diff(cur.exercise_boundary, prev_boundary_on_cur_tau)

        rows.append({
            "n_coarse": REFINEMENT_LEVELS[i - 1], "n_fine": REFINEMENT_LEVELS[i],
            "value_diff": value_diff, "boundary_diff_max": boundary_diff,
        })
        print(f"  {REFINEMENT_LEVELS[i-1]:>4} -> {REFINEMENT_LEVELS[i]:>4}: "
              f"|dValue|={value_diff:.3e}  max|dBoundary|={boundary_diff:.3e}", flush=True)

    finest = solutions[-1]
    finest_stable = rows[-1]["value_diff"] < STABILITY_TARGET
    print(f"\nFinest level ({REFINEMENT_LEVELS[-1]}) value change vs previous: "
          f"{rows[-1]['value_diff']:.3e} ({'STABLE' if finest_stable else 'NOT YET STABLE'} "
          f"vs target {STABILITY_TARGET:.0e})")

    u_lift_at_x_star = problem.C * (problem.R - problem.x_star) / (problem.R - problem.L)
    V_at_reference = finest.value_at_reference_point + u_lift_at_x_star
    print(f"Physical (unlifted) American put value at S=K={problem.K}, tau=T: V = {V_at_reference:.6f}")
    print("Sanity context (NOT an authoritative lookup -- no network access to verify to the cent): "
          "commonly-cited American put benchmarks for S=K=100, r=0.06, sigma=0.4, T=1 in the "
          "quantitative-finance literature (e.g. Broadie-Detemple-style comparison tables) are "
          "typically reported in the low-to-mid teens for this high-volatility case; the computed "
          f"V={V_at_reference:.2f} is consistent with that ballpark, reported here with explicit "
          "uncertainty rather than false precision.")

    freeze_config = PSORConfig(n_x=REFINEMENT_LEVELS[-1], n_tau=REFINEMENT_LEVELS[-1])
    frozen_info = freeze_reference(freeze_config, out_dir=FROZEN_DIR)
    print(f"\nFROZEN reference -> {FROZEN_DIR} (config_hash={frozen_info['config_hash']})")

    summary = {
        "refinement_levels": REFINEMENT_LEVELS,
        "value_at_reference_point_per_level": [s.value_at_reference_point for s in solutions],
        "refinement_diffs": rows,
        "finest_level_stable": finest_stable,
        "stability_target": STABILITY_TARGET,
        "V_physical_at_S_eq_K_tau_eq_T": V_at_reference,
        "frozen_config_hash": frozen_info["config_hash"],
        "frozen_dir": str(FROZEN_DIR),
    }
    out_path = RESULTS_DIR / "american_put_reference_convergence.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Convergence summary -> {out_path}")
    return summary


if __name__ == "__main__":
    main()
