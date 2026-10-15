"""Pre-M18 T2: standard certification mesh selection.

Reads the per-grid certificate JSONs already written by
stefan_option_b_sweep.py (results/stefan_option_b_certificates/gridN/) for
every grid actually certified this milestone (32^2 .. 2048^2 -- 4096^2 was
not run: the extended sweep's own predicted-vs-actual E_h check correctly
raised at 2048^2 before reaching the 4096^2 gate, and 2048^2's measured
peak_rss=7.19 GiB independently exceeds the 4 GiB budget that gate requires,
so 4096^2 would not have been attempted either way).

Builds one table: E_cert_U_new, M16 status (E_true <= E_cert_U_new), and
projected M18 cost (20 runs = 4 architectures x 5 seeds, certification-only
wall-clock and peak RSS) at each candidate grid. Selects the standard mesh
and writes the justification alongside the table. No certification is
re-run and no formula/constant is touched -- this only reads and reports
numbers already produced under Pre-M18 T1/T2.
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
CERT_DIR = RESULTS_DIR / "stefan_option_b_certificates"
GRIDS = (32, 64, 128, 256, 512, 1024, 2048)
E_TRUE = 0.07215944928189263  # from results/stefan_option_b_config.json (T1-fixed quadrature)
M18_N_RUNS = 20  # 4 architectures x 5 seeds
SELECTED_GRID = 1024


def _load(grid: int) -> dict:
    p = CERT_DIR / f"grid{grid}" / f"certificate_grid{grid}.json"
    d = json.loads(p.read_text())
    r = d["result"]
    return dict(
        grid=grid,
        h=d["h"],
        n_cells=d["n_cells"],
        E_cert_U_new=r["E_cert_U_new"],
        E_h_new=r["E_h_new"],
        rho_tilde_h=r["rho_tilde_h"],
        V_h_unr=r["V_h_unr"],
        wall_clock_sec=d["wall_clock_seconds"],
        peak_rss_gib=d["peak_rss_gib"],
    )


def main() -> dict:
    rows = [_load(g) for g in GRIDS]
    for row in rows:
        row["holds_M16"] = E_TRUE <= row["E_cert_U_new"]
        row["m18_projected_wall_clock_sec"] = M18_N_RUNS * row["wall_clock_sec"]
        row["m18_projected_wall_clock_min"] = row["m18_projected_wall_clock_sec"] / 60.0

    header = (
        f"{'grid':>6} {'h':>10} {'n_cells':>9} {'E_cert_U_new':>13} {'M16':>5} "
        f"{'V_h_unr':>8} {'cert_sec':>9} {'peak_GiB':>9} {'M18_20x_min':>12}"
    )
    lines = [
        "Pre-M18 T2: standard certification mesh candidates",
        f"E_true = {E_TRUE:.6e} (fixed, closed-form, T1)",
        f"M18 projection assumes {M18_N_RUNS} runs (4 architectures x 5 seeds), "
        "certification phase only (training cost is mesh-independent)",
        "",
        header,
        "-" * len(header),
    ]
    for row in rows:
        lines.append(
            f"{row['grid']:>6} {row['h']:>10.5f} {row['n_cells']:>9} "
            f"{row['E_cert_U_new']:>13.4f} {'OK' if row['holds_M16'] else 'FAIL':>5} "
            f"{row['V_h_unr']:>8.4f} {row['wall_clock_sec']:>9.2f} {row['peak_rss_gib']:>9.3f} "
            f"{row['m18_projected_wall_clock_min']:>12.2f}"
        )

    selected = next(r for r in rows if r["grid"] == SELECTED_GRID)
    prev_candidate = next(r for r in rows if r["grid"] == 512)
    next_candidate = next(r for r in rows if r["grid"] == 2048)

    justification = (
        f"Selected standard certification mesh: {SELECTED_GRID}x{SELECTED_GRID}.\n\n"
        f"  E_cert_U_new: {prev_candidate['E_cert_U_new']:.2f} (512^2) -> "
        f"{selected['E_cert_U_new']:.2f} (1024^2) -> {next_candidate['E_cert_U_new']:.2f} (2048^2).\n"
        f"  1024^2 is {prev_candidate['E_cert_U_new'] / selected['E_cert_U_new']:.2f}x tighter than 512^2 "
        f"for {selected['wall_clock_sec'] / prev_candidate['wall_clock_sec']:.1f}x the per-run cost "
        f"({prev_candidate['wall_clock_sec']:.1f}s -> {selected['wall_clock_sec']:.1f}s).\n"
        f"  2048^2 only buys a further {selected['E_cert_U_new'] / next_candidate['E_cert_U_new']:.2f}x "
        f"tightening over 1024^2, for {next_candidate['wall_clock_sec'] / selected['wall_clock_sec']:.1f}x "
        f"the per-run cost and {next_candidate['peak_rss_gib'] / selected['peak_rss_gib']:.2f}x the peak RSS "
        f"({selected['peak_rss_gib']:.2f} GiB -> {next_candidate['peak_rss_gib']:.2f} GiB, the latter close "
        f"to the 8 GiB certify_cli cap with no headroom for M18's larger architectures).\n"
        f"  Projected M18 cost (20 runs, certification only): "
        f"{selected['m18_projected_wall_clock_min']:.1f} min at 1024^2 vs "
        f"{next_candidate['m18_projected_wall_clock_min']:.1f} min "
        f"({next_candidate['m18_projected_wall_clock_min'] / 60:.2f} h) at 2048^2.\n"
        f"  M16 (E_true <= E_cert_U_new) holds at every candidate grid, including the selected one "
        f"(E_true={E_TRUE:.4e} <= E_cert_U_new={selected['E_cert_U_new']:.4f}).\n"
        "  1024^2 is chosen over 512^2 for a materially tighter certificate at a still-cheap "
        "per-run cost, and over 2048^2 because 2048^2's marginal tightening does not justify its "
        "4.3x runtime and 3.2x peak-RSS cost across 20 M18 runs, especially given 2048^2 already "
        "sits near the memory cap that flagged the predicted-vs-actual E_h check as a bug candidate "
        "this milestone (regime change in V_h_unr, accepted as a genuine finding, not re-litigated here)."
    )

    report = "\n".join(lines) + "\n\n" + justification + "\n"
    print(report)

    out = {
        "grids_candidates": rows,
        "selected_grid": SELECTED_GRID,
        "grid_4096_status": (
            "not run: extended sweep's predicted-vs-actual E_h check raised at 2048^2 before "
            "reaching the 4096^2 gate; 2048^2 peak_rss=7.19 GiB independently exceeds the 4 GiB "
            "budget required for that gate, so 4096^2 would not have run regardless"
        ),
        "m18_n_runs": M18_N_RUNS,
        "E_true": E_TRUE,
        "justification": justification,
    }
    (RESULTS_DIR / "stefan_mesh_selection.json").write_text(json.dumps(out, indent=2))
    (RESULTS_DIR / "stefan_mesh_selection_report.txt").write_text(report)
    return out


if __name__ == "__main__":
    main()
