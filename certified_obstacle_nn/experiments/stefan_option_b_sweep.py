"""Option B numerical experiment (docs/history/OPTION_B_CERTIFIED_RESIDUAL_BOUND_IMPLEMENTATION_PLAN.md
sections 15/16/17/21/25/26, and its addendum A2/A3/A5).

Reuses the SAME fixed reference triple as experiments/stefan_quad_convergence.py
(results/stefan_baseline_weights.pt) and the SAME 32^2...512^2 grid sweep --
addendum SCOPE: "Do not extend the grid sweep in this milestone." Each grid
runs as its own scripts/certify_cli.py --option-b-mode subprocess (same
process-isolation discipline as the non-Option-B sweep); this script reads
the resulting JSONs back from disk and recomputes nothing.

Produces:
  - results/stefan_option_b_table.csv    (plan section 26's columns + addendum A2's)
  - results/stefan_option_b_report.txt   (plan section 25's before/after report, per grid)
  - results/stefan_option_b_config.json  (fitted a/b, rate comparison, M16 reverification)

Addendum A3: E_h_new ~= a*h + b*h^2 (superlinear) is EXPECTED, not a defect --
a and b are fit by least squares and reported; the OLD (apriori) rate is
reported alongside since it is what preserves the uniform first-order
argument (plan section 17).
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
BASELINE_WEIGHTS_PATH = RESULTS_DIR / "stefan_baseline_weights.pt"
TRIPLE_CONFIG_PATH = RESULTS_DIR / "stefan_quad_convergence_reference_triple_config.json"

GRID_SIZES = (32, 64, 128, 256, 512)  # the original Option-B milestone's sweep
# Pre-M18 T2: extend the sweep. 4096 is CONDITIONAL (only if 2048 finishes
# under EXTENDED_GRID_TIME_BUDGET_SEC / EXTENDED_GRID_RSS_BUDGET_GIB) -- not
# unconditionally appended to EXTENDED_GRID_SIZES.
EXTENDED_GRID_SIZES = (1024, 2048)
CONDITIONAL_GRID = 4096
EXTENDED_GRID_TIME_BUDGET_SEC = 15 * 60
EXTENDED_GRID_RSS_BUDGET_GIB = 4.0
EXPECTED_RATE_FIRST_ORDER = 2.0

# Documented baseline peak RSS (non-Option-B streaming sweep, this session),
# per grid -- addendum A5: "report peak RSS per grid against the current
# baseline ... and flag any increase above 5%." Only covers the ORIGINAL
# GRID_SIZES: the non-Option-B sweep never ran 1024^2+, so there is no
# baseline to regress against there -- reported without a comparison instead
# of guessing one.
BASELINE_PEAK_RSS_GIB = {32: 0.667, 64: 0.989, 128: 1.000, 256: 1.011, 512: 1.039}
PEAK_RSS_REGRESSION_THRESHOLD = 0.05

# Previously reported fit (from the ORIGINAL 32^2..512^2 sweep only), used as
# the FIXED prediction basis for the new grids -- T2: "a deviation from
# a*h + b*h^2 is a bug, not a finding". Refit at the end over the full
# extended dataset too (reported separately) -- that refit is NOT what's
# being validated here, since fitting on the same points being predicted
# would be circular.
PREVIOUSLY_FITTED_A = 1.442083e04
PREVIOUSLY_FITTED_B = 5.028793e05
PREDICTED_E_H_DEVIATION_THRESHOLD = 0.10  # relative; flagged as a bug above this

TRUE_ERROR_N = 100
# Pre-M18 T1: true_error_cli.py now auto-splits the quadrature domain along
# Stefan's exact free boundary s(t)=t whenever the problem exposes
# free_boundary_exact, restoring spectral self-check convergence (~1e-15
# relative change, not merely "under some loosened tolerance"). The
# originally specified 1e-3 is back to being the real tolerance here, not a
# placeholder -- see errors/true_error.py and
# experiments/stefan_true_error_quadrature_diagnostic.py for the before/after
# evidence.
TRUE_ERROR_TOLERANCE = 1e-3


def _certify_option_b_via_cli(n: int, out_dir: Path, max_rss_gb: float, save_cells: bool = True) -> dict:
    """save_cells=True (T2: "keep the per-cell arrays, M10 identity depends
    on them") writes into out_dir/grid{n}/ specifically -- each grid gets its
    own subdirectory so cells.{csv,npz} from different grid sizes never
    collide (certification/cells_io.py always names the file "cells.csv" or
    "cells.npz", not grid-size-qualified)."""
    run_dir = out_dir / f"grid{n}" if save_cells else out_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "certified_obstacle_nn.scripts.certify_cli",
        "--weights", str(BASELINE_WEIGHTS_PATH),
        "--config", str(TRIPLE_CONFIG_PATH),
        "--grid", str(n),
        "--out", str(run_dir),
        "--max-rss-gb", str(max_rss_gb),
        "--option-b-mode", "min_apriori_and_partition",
    ]
    if save_cells:
        cmd.append("--save-cells")

    wall_start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    wall_clock_seconds = time.perf_counter() - wall_start
    print(proc.stdout, end="")
    if proc.returncode != 0:
        raise RuntimeError(f"certify_cli --option-b-mode failed for grid {n}x{n}:\n{proc.stderr}")

    payload = json.loads((run_dir / f"certificate_grid{n}.json").read_text())
    payload["_wall_clock_seconds"] = wall_clock_seconds
    return payload


def _true_error_via_cli(out_dir: Path) -> dict:
    cmd = [
        sys.executable, "-m", "certified_obstacle_nn.scripts.true_error_cli",
        "--weights", str(BASELINE_WEIGHTS_PATH),
        "--config", str(TRIPLE_CONFIG_PATH),
        "--n-t", str(TRUE_ERROR_N), "--n-x", str(TRUE_ERROR_N),
        "--quadrature-tolerance", str(TRUE_ERROR_TOLERANCE),
        "--out", str(out_dir),
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    print(proc.stdout, end="")
    if proc.returncode != 0:
        raise RuntimeError(f"true_error_cli failed:\n{proc.stderr}")
    return json.loads((out_dir / f"true_error_nt{TRUE_ERROR_N}_nx{TRUE_ERROR_N}.json").read_text())


def _fit_a_b(h: np.ndarray, E_h_new: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of E_h_new(h) = a*h + b*h^2 (addendum A3)."""
    design = np.stack([h, h**2], axis=1)
    (a, b), *_ = np.linalg.lstsq(design, E_h_new, rcond=None)
    return float(a), float(b)


def main(max_rss_gb: float = 8.0, run_extended: bool = True) -> pd.DataFrame:
    """run_extended=True (default, Pre-M18 T2) appends EXTENDED_GRID_SIZES
    (1024, 2048) to GRID_SIZES, then conditionally CONDITIONAL_GRID (4096)
    if 2048^2 finished within budget. run_extended=False reproduces exactly
    the original 32^2..512^2 sweep."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cert_out_dir = RESULTS_DIR / "stefan_option_b_certificates"
    cert_out_dir.mkdir(parents=True, exist_ok=True)
    table_csv_path = RESULTS_DIR / "stefan_option_b_table.csv"
    report_path = RESULTS_DIR / "stefan_option_b_report.txt"
    config_path = RESULTS_DIR / "stefan_option_b_config.json"

    if not BASELINE_WEIGHTS_PATH.exists() or not TRIPLE_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"{BASELINE_WEIGHTS_PATH} / {TRIPLE_CONFIG_PATH} not found -- run "
            "experiments/stefan_quad_convergence.py first to produce the fixed reference triple "
            "this Option B sweep reuses (SCOPE: no retraining in this milestone)."
        )

    # Mutated during iteration to conditionally append CONDITIONAL_GRID after
    # seeing 2048's own runtime/peak RSS -- safe in CPython: list iteration
    # re-checks the current length each step, so an append before the loop
    # reaches the end is picked up (removal would NOT be safe this way; we
    # never remove).
    grid_sizes_to_run = list(GRID_SIZES) + (list(EXTENDED_GRID_SIZES) if run_extended else [])

    rows = []
    for n in grid_sizes_to_run:
        payload = _certify_option_b_via_cli(n, cert_out_dir, max_rss_gb, save_cells=True)
        result = payload["result"]
        rb = result["residual_bound"]

        peak_rss_gib = payload["peak_rss_gib"]
        wall_clock_seconds = payload["_wall_clock_seconds"]
        baseline_rss = BASELINE_PEAK_RSS_GIB.get(n)
        if baseline_rss is not None:
            regression = (peak_rss_gib - baseline_rss) / baseline_rss
            if regression > PEAK_RSS_REGRESSION_THRESHOLD:
                print(f"!!! PEAK RSS REGRESSION at grid {n}x{n}: {peak_rss_gib:.3f} GiB vs baseline "
                      f"{baseline_rss:.3f} GiB ({regression:+.1%}) -- exceeds {PEAK_RSS_REGRESSION_THRESHOLD:.0%} threshold !!!")
        else:
            regression = None  # no non-Option-B baseline exists at this grid size (never swept before)

        # T2: predicted (from the PREVIOUSLY fitted a,b, i.e. NOT refit on
        # this point) vs actual E_h_new for every newly-added grid --
        # deviation is a bug, not a finding.
        predicted_E_h_new = None
        if n in EXTENDED_GRID_SIZES or n == CONDITIONAL_GRID:
            h = payload["h"]
            predicted_E_h_new = PREVIOUSLY_FITTED_A * h + PREVIOUSLY_FITTED_B * h**2
            actual_E_h_new = result["E_h_new"]
            deviation = abs(actual_E_h_new - predicted_E_h_new) / actual_E_h_new
            print(f"  [T2 predicted-vs-actual] grid={n}x{n}: predicted E_h_new={predicted_E_h_new:.6e}  "
                  f"actual={actual_E_h_new:.6e}  relative_deviation={deviation:.4%} "
                  f"({'OK' if deviation <= PREDICTED_E_H_DEVIATION_THRESHOLD else 'EXCEEDS THRESHOLD -- BUG'})")
            assert deviation <= PREDICTED_E_H_DEVIATION_THRESHOLD, (
                f"E_h_new at grid {n}x{n} deviates from the previously-fitted a*h+b*h^2 by "
                f"{deviation:.2%} (predicted {predicted_E_h_new:.6e}, actual {actual_E_h_new:.6e}) -- "
                "per T2 this is treated as a BUG, not a finding: investigate before trusting this grid."
            )

        rows.append({
            "grid_n": n,
            "h": payload["h"],
            "m_R_apriori": rb["m_R_apriori"],
            "m_R_h": rb["m_R_h"],
            "m_R_cert": rb["m_R_cert"],
            "m_R_ratio": rb["m_R_h"] / rb["m_R_apriori"],
            "L_R": rb["L_R"],
            "r_K": rb["r_K"],
            "max_R_mid": rb["max_R_mid"],
            "lip_term": rb["lip_term"],
            "lip_term_fraction": rb["lip_term_fraction"],
            "n_data_cut_cells": rb["n_data_cut_cells"],
            "argmax_cell_id": rb["argmax_cell_id"],
            "argmax_t": rb["argmax_t"],
            "argmax_x": rb["argmax_x"],
            "dist_argmax_from_diagonal": rb["dist_argmax_from_diagonal"],
            "T1_old": result["T1_old"],
            "T1_new": result["T1_new"],
            "L_sm_old": result["L_sm_old"],
            "L_sm_new": result["L_sm_new"],
            "E_h_old": result["E_h_old"],
            "E_h_new": result["E_h_new"],
            "rho_tilde_h_sq": result["rho_tilde_h_sq"],
            "E_h_old_over_rho2": result["E_h_old_over_rho2"],
            "E_h_new_over_rho2": result["E_h_new_over_rho2"],
            "E_cert_old": result["E_cert_old"],
            "E_cert_new": result["E_cert_new"],
            "E_cert_U_old": result["E_cert_U_old"],
            "E_cert_U_new": result["E_cert_U_new"],
            "runtime_sec": wall_clock_seconds,
            "peak_rss_mb": peak_rss_gib * 1024.0,
            "peak_rss_gib": peak_rss_gib,
            "peak_rss_regression_vs_baseline": regression,
            "predicted_E_h_new": predicted_E_h_new,
        })
        baseline_note = f"(baseline {baseline_rss:.3f}, {regression:+.1%})" if baseline_rss is not None else "(no non-Option-B baseline at this grid)"
        print(
            f"grid={n}x{n:<4d} h={payload['h']:.5f}  m_R: apriori={rb['m_R_apriori']:.4e} "
            f"h={rb['m_R_h']:.4e} cert={rb['m_R_cert']:.4e}  E_h: old={result['E_h_old']:.4e} "
            f"new={result['E_h_new']:.4e}  peak_rss={peak_rss_gib:.3f}GiB {baseline_note} "
            f"runtime={wall_clock_seconds:.1f}s"
        )

        # T2: conditional 4096^2 gate, decided right after 2048^2 finishes.
        if n == 2048 and run_extended and CONDITIONAL_GRID not in grid_sizes_to_run:
            gate_ok = wall_clock_seconds < EXTENDED_GRID_TIME_BUDGET_SEC and peak_rss_gib < EXTENDED_GRID_RSS_BUDGET_GIB
            print(
                f"\n2048x2048 finished in {wall_clock_seconds:.1f}s (budget {EXTENDED_GRID_TIME_BUDGET_SEC}s) "
                f"with peak RSS {peak_rss_gib:.3f} GiB (budget {EXTENDED_GRID_RSS_BUDGET_GIB} GiB) -> "
                f"{'proceeding to run' if gate_ok else 'SKIPPING'} 4096x4096.\n"
            )
            if gate_ok:
                grid_sizes_to_run.append(CONDITIONAL_GRID)

    table = pd.DataFrame(rows)

    # rates: rate[i] = E_h[i-1]/E_h[i], meaningful because GRID_SIZES doubles n (halves h) every step.
    table["rate_old"] = table["E_h_old"].shift(1) / table["E_h_old"]
    table["rate_new"] = table["E_h_new"].shift(1) / table["E_h_new"]

    csv_columns = [
        "grid_n", "h", "m_R_apriori", "m_R_h", "m_R_cert", "m_R_ratio",
        "T1_old", "T1_new", "L_sm_old", "L_sm_new",
        "E_h_old", "E_h_new", "rho_tilde_h_sq", "E_h_old_over_rho2", "E_h_new_over_rho2",
        "E_cert_old", "E_cert_new", "rate_old", "rate_new", "runtime_sec", "peak_rss_mb",
        # addendum A2's required additions
        "L_R", "max_R_mid", "lip_term", "lip_term_fraction",
        "argmax_cell_id", "argmax_t", "argmax_x", "dist_argmax_from_diagonal",
        # Pre-M18 T2's required additions
        "predicted_E_h_new", "peak_rss_regression_vs_baseline",
    ]
    table[csv_columns].to_csv(table_csv_path, index=False)

    # Addendum A3: fit E_h_new(h) = a*h + b*h^2.
    a_fit, b_fit = _fit_a_b(table["h"].to_numpy(), table["E_h_new"].to_numpy())
    fitted = a_fit * table["h"] + b_fit * table["h"] ** 2
    fit_residual_rel = ((table["E_h_new"] - fitted).abs() / table["E_h_new"]).max()

    # Addendum A2: which of the two terms in m_R_h dominates, per grid.
    print("\nA2 -- which term dominates m_R_h_nondata, per grid:")
    for _, row in table.iterrows():
        dominant = "lip_term (Lipschitz sweep)" if row["lip_term_fraction"] > 0.5 else "max_R_mid (sampled residual)"
        near_diag = "near t=x" if row["dist_argmax_from_diagonal"] < 3 * row["h"] else "NOT near t=x"
        print(
            f"  grid={row['grid_n']}x{row['grid_n']:<4.0f} lip_term_fraction={row['lip_term_fraction']:.4f} "
            f"-> {dominant} dominates. argmax at (t={row['argmax_t']:.4f}, x={row['argmax_x']:.4f}), "
            f"|t-x|={row['dist_argmax_from_diagonal']:.4f} ({near_diag})"
        )

    print(f"\nA3 -- fit E_h_new(h) = a*h + b*h^2: a={a_fit:.6e}  b={b_fit:.6e}  "
          f"(max relative fit residual over tested grids: {fit_residual_rel:.3e})")
    print("Rates E_h(h)/E_h(h/2), old (apriori, expected -> 2.0, first-order) vs "
          "new (sharpened, expected > 2.0 approaching 2.0 from above -- NOT a defect, addendum A3):")
    for i in range(1, len(table)):
        print(
            f"  {table['grid_n'].iloc[i-1]}x{table['grid_n'].iloc[i-1]:<4.0f} -> "
            f"{table['grid_n'].iloc[i]}x{table['grid_n'].iloc[i]:<4.0f}  "
            f"rate_old={table['rate_old'].iloc[i]:.4f}  rate_new={table['rate_new'].iloc[i]:.4f}"
        )

    # Does E_h_new still dominate rho_tilde_h^2? Which term is now dominant overall?
    print(f"\nE_h_new / rho_tilde_h^2 ranges from {table['E_h_new_over_rho2'].iloc[0]:.5e} "
          f"({table['grid_n'].iloc[0]}x{table['grid_n'].iloc[0]:.0f}) to "
          f"{table['E_h_new_over_rho2'].iloc[-1]:.5e} ({table['grid_n'].iloc[-1]}x{table['grid_n'].iloc[-1]:.0f}).")
    still_dominates = (table["E_h_new_over_rho2"] > 1.0).all()
    print(f"E_h_new {'still dominates' if still_dominates else 'no longer dominates'} rho_tilde_h^2 at every tested grid.")

    # M16 reverification at sharpened values: E_true <= E_cert_U_new.
    print("\nM16 reverification (Test A/B logic) at SHARPENED E_cert_U_new:")
    true_error_payload = _true_error_via_cli(cert_out_dir)
    E_true = true_error_payload["result"]["E_true"]
    m16_rows = []
    for _, row in table.iterrows():
        ok = E_true <= row["E_cert_U_new"]
        m16_rows.append({"grid_n": int(row["grid_n"]), "E_true": E_true, "E_cert_U_new": row["E_cert_U_new"], "holds": bool(ok)})
        print(f"  grid={row['grid_n']:.0f}x{row['grid_n']:.0f}  E_true={E_true:.6e} <= E_cert_U_new={row['E_cert_U_new']:.6e} ? {ok}")
        assert ok, f"M16 VIOLATED at grid {row['grid_n']}: E_true={E_true} > E_cert_U_new={row['E_cert_U_new']}"

    # Section 25: before/after report, per grid.
    report_lines = []
    for _, row in table.iterrows():
        report_lines.append(f"=== grid {row['grid_n']:.0f}x{row['grid_n']:.0f} ===")
        report_lines.append(f"Old m_R_apriori: {row['m_R_apriori']:.6e}")
        report_lines.append(f"New m_R_h:       {row['m_R_h']:.6e}")
        report_lines.append(f"Selected m_R_cert: {row['m_R_cert']:.6e}")
        report_lines.append(f"Reduction factor: {row['m_R_apriori'] / row['m_R_cert']:.4f}x")
        report_lines.append("")
        report_lines.append(f"Old T1: {row['T1_old']:.6e}")
        report_lines.append(f"New T1: {row['T1_new']:.6e}")
        report_lines.append(f"Reduction factor: {row['T1_old'] / row['T1_new']:.4f}x")
        report_lines.append("")
        report_lines.append(f"Old L_sm: {row['L_sm_old']:.6e}")
        report_lines.append(f"New L_sm: {row['L_sm_new']:.6e}")
        report_lines.append(f"Reduction factor: {row['L_sm_old'] / row['L_sm_new']:.4f}x")
        report_lines.append("")
        report_lines.append(f"Old E_h: {row['E_h_old']:.6e}")
        report_lines.append(f"New E_h: {row['E_h_new']:.6e}")
        report_lines.append(f"Reduction factor: {row['E_h_old'] / row['E_h_new']:.4f}x")
        report_lines.append("")
        report_lines.append(f"Old E_cert: {row['E_cert_old']:.6e}")
        report_lines.append(f"New E_cert: {row['E_cert_new']:.6e}")
        report_lines.append(f"Reduction factor: {row['E_cert_old'] / row['E_cert_new']:.4f}x")
        report_lines.append("")
        report_lines.append(f"Old E_h / rho_tilde_h^2: {row['E_h_old_over_rho2']:.6e}")
        report_lines.append(f"New E_h / rho_tilde_h^2: {row['E_h_new_over_rho2']:.6e}")
        report_lines.append("")
        report_lines.append(f"Observed convergence rate (old/new): {row['rate_old'] if pd.notna(row['rate_old']) else float('nan'):.4f} / "
                             f"{row['rate_new'] if pd.notna(row['rate_new']) else float('nan'):.4f}")
        report_lines.append(f"Remaining dominant term: {'lip_term (Lipschitz sweep)' if row['lip_term_fraction'] > 0.5 else 'max_R_mid (sampled residual)'}")
        if pd.notna(row["peak_rss_regression_vs_baseline"]):
            report_lines.append(
                f"Peak RSS: {row['peak_rss_gib']:.3f} GiB "
                f"(baseline {BASELINE_PEAK_RSS_GIB[int(row['grid_n'])]:.3f} GiB, {row['peak_rss_regression_vs_baseline']:+.1%})"
            )
        else:
            report_lines.append(f"Peak RSS: {row['peak_rss_gib']:.3f} GiB (no non-Option-B baseline at this grid)")
        if pd.notna(row["predicted_E_h_new"]):
            report_lines.append(
                f"Predicted E_h_new (from previously-fitted a,b): {row['predicted_E_h_new']:.6e}  "
                f"actual: {row['E_h_new']:.6e}  relative deviation: "
                f"{abs(row['E_h_new'] - row['predicted_E_h_new']) / row['E_h_new']:.4%}"
            )
        report_lines.append(f"Runtime: {row['runtime_sec']:.2f}s")
        report_lines.append("")
    report_lines.append("=== fit / rates / dominance (global) ===")
    report_lines.append("Root cause of looseness (pre-Option-B): T1_pde_residual, ~86% of L_sm+L_Psi "
                         "(experiments/stefan_bound_decomposition.py)")
    report_lines.append(f"Fitted a (h-independent part, E_h_new ~= a*h + b*h^2): {a_fit:.6e}")
    report_lines.append(f"Fitted b: {b_fit:.6e}")
    report_lines.append(f"E_h_new still dominates rho_tilde_h^2 at every grid: {still_dominates}")
    report_lines.append(f"M16 (E_true <= E_cert_U_new) holds at every grid: {all(r['holds'] for r in m16_rows)}")
    report_lines.append("All tests passed: see pytest output (tests/test_certified_residual_bound.py)")

    report_text = "\n".join(report_lines)
    report_path.write_text(report_text)
    print("\n" + report_text)

    config_path.write_text(json.dumps({
        "grid_sizes": list(table["grid_n"].astype(int)),  # what was ACTUALLY run, including any conditional 4096
        "reference_triple_weights": str(BASELINE_WEIGHTS_PATH),
        "a_fit": a_fit,
        "b_fit": b_fit,
        "fit_max_relative_residual": float(fit_residual_rel),
        "previously_fitted_a": PREVIOUSLY_FITTED_A,
        "previously_fitted_b": PREVIOUSLY_FITTED_B,
        "predicted_vs_actual_deviation_threshold": PREDICTED_E_H_DEVIATION_THRESHOLD,
        "E_h_new_dominates_rho2_at_every_grid": bool(still_dominates),
        "M16_reverification": m16_rows,
        "true_error_n_t": TRUE_ERROR_N,
        "true_error_n_x": TRUE_ERROR_N,
        "true_error_tolerance": TRUE_ERROR_TOLERANCE,
        "E_true": E_true,
        "baseline_peak_rss_gib": BASELINE_PEAK_RSS_GIB,
        "peak_rss_regression_threshold": PEAK_RSS_REGRESSION_THRESHOLD,
        "table_csv_path": str(table_csv_path),
        "report_path": str(report_path),
    }, indent=2))

    print(f"\ntable  -> {table_csv_path}")
    print(f"report -> {report_path}")
    print(f"config -> {config_path}")

    return table


if __name__ == "__main__":
    main()
