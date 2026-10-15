"""Builds the final new_runs_master.csv (NEW_RUNS.md "Master CSV
Requirement") by concatenating each experiment's own all_runs.csv --
never recomputes anything, purely a file-level aggregation, matching this
project's own "plots/aggregators read only from already-saved files"
convention (specification's run-artifact convention section).

Run this LAST, after adaptive_multiseed.py, american_put_improved.py, and
certificate_aware.py have all completed (threshold_sensitivity.py's output
is intentionally NOT folded in here -- it is a diagnostic re-evaluation
sweep over existing checkpoints, not a set of new training runs, and stays
in its own free_boundary_threshold_sensitivity.csv per NEW_RUNS.md's own
packaging structure).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from certified_obstacle_nn.experiments.new_runs.common import MASTER_CSV_PATH, NEW_RUNS_DIR, aggregate_master_csv

SOURCES = [
    NEW_RUNS_DIR / "adaptive_multiseed" / "adaptive_multiseed_all_runs.csv",
    NEW_RUNS_DIR / "american_put_improved" / "all_runs.csv",
    NEW_RUNS_DIR / "certificate_aware" / "all_runs.csv",
]


def main() -> pd.DataFrame:
    rows: list[dict] = []
    for path in SOURCES:
        if not path.exists():
            print(f"WARNING: {path} does not exist yet -- skipping (run that experiment's script first)")
            continue
        df = pd.read_csv(path)
        rows.extend(df.to_dict(orient="records"))
        print(f"{path}: {len(df)} rows")

    master = aggregate_master_csv(rows, out_path=MASTER_CSV_PATH)
    print(f"\nnew_runs_master.csv -> {MASTER_CSV_PATH} ({len(master)} rows, {len(master.columns)} columns)")

    n_failed = int((master["status"] == "failed").sum()) if len(master) else 0
    n_violated = int(master["certificate_inequality_violated"].fillna(False).astype(bool).sum()) if len(master) else 0
    print(f"status=failed rows: {n_failed}")
    print(f"certificate_inequality_violated rows: {n_violated}")
    if n_violated:
        print("*** WARNING: certificate inequality violations present -- flagged for review, not hidden ***")
        print(master[master["certificate_inequality_violated"].fillna(False).astype(bool)][["experiment", "subexperiment", "strategy", "seed", "gamma", "E_true", "E_cert_U"]])

    return master


if __name__ == "__main__":
    main()
