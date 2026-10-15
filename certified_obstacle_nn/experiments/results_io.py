"""Uniform experiment run-artifact layout, 2026-09-12 (user-specified).

Every experiment run writes to its own timestamped directory, never
overwritten (specification section 12 invariant 7: "every reported quantity must
be reproducible from the saved config + weights" -- a fixed, reused path per
experiment cannot satisfy that once a second run happens):

    results/<experiment>/<timestamp>/
        config.json   -- seed, architecture, head, optimizer schedule,
                          N_train, grid, eps_* -- everything needed to
                          reproduce this exact run, nothing computed from it.
        losses.csv    -- per-epoch, all five loss components. This is just
                          training/trainer.py's own `train(..., csv_path=...)`
                          output pointed at this directory, not a second
                          logging path.
        cells.csv     -- per-cell indicators, when certification ran. This is
                          just certification/cells_io.py's save_per_cell_arrays
                          pointed at this directory (writes cells.npz instead
                          above CELLS_CSV_CELL_THRESHOLD cells -- unchanged
                          existing behavior, not a new format).
        summary.csv   -- ONE row: every quantity in specification's Tables 1-6
                          that this run produces (rho_r/rho_p_tilde/rho_c,
                          E_h_lip/E_h_unr/E_h_dat/E_h_0T, E_true, E_cert,
                          E_cert_U, ... -- specification section 12 invariant 5:
                          never merge these into one number, so summary.csv
                          gets one COLUMN per quantity, not one column per
                          formula).
        weights.pt    -- torch.save(triple.state_dict()).

Plots must read ONLY from these five files -- never recompute a quantity
(e.g. never re-run compute_true_error or certify() from a plotting script).
This module only provides the directory/file plumbing; it has no opinion on
what a particular experiment's config or summary dict contains.
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"


def make_run_dir(experiment: str, results_dir: Path = RESULTS_DIR) -> Path:
    """Creates and returns results/<experiment>/<timestamp>/ -- a FRESH
    directory every call (exist_ok=False): two runs launched in the same
    process would otherwise race on the same second-resolution timestamp,
    which must fail loudly rather than silently share a directory."""
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    # v19 Task C (DECISIONS D46): an opt-in prefix routes a whole pipeline's run directories to new experiment names
    # (e.g. CERT_EXPERIMENT_PREFIX=v19_C_ -> results/v19_C_v17_split/<ts>/); unset = unchanged behaviour
    experiment = os.environ.get("CERT_EXPERIMENT_PREFIX", "") + experiment
    run_dir = results_dir / experiment / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_config(run_dir: Path, config: dict) -> Path:
    path = run_dir / "config.json"
    path.write_text(json.dumps(config, indent=2))
    return path


def losses_csv_path(run_dir: Path) -> Path:
    """Not a writer itself -- training/trainer.py's own `train(csv_path=...)`
    writes losses.csv directly (it already logs all five components per
    epoch, specification section 3's "log every component every epoch"); this is
    just the one place the filename `losses.csv` is spelled, so a caller
    doesn't have to repeat it."""
    return run_dir / "losses.csv"


def write_summary(run_dir: Path, summary: dict) -> Path:
    """ONE-row summary.csv holding every quantity a plot might need --
    specification section 12 invariant 5's separately-reported components (the
    three rho_* parts, the four E_h parts) go in as separate columns, never
    pre-merged. Values are written as given (no rounding, no recomputation)."""
    path = run_dir / "summary.csv"
    pd.DataFrame([summary]).to_csv(path, index=False)
    return path


def write_weights(run_dir: Path, triple) -> Path:
    path = run_dir / "weights.pt"
    torch.save(triple.state_dict(), path)
    return path


def read_summary(run_dir: Path) -> pd.Series:
    """Reads back the one-row summary.csv as a Series -- the read-side of
    write_summary, for plots (module docstring: "plots read from these files
    only")."""
    df = pd.read_csv(run_dir / "summary.csv")
    assert len(df) == 1, f"expected exactly one row in {run_dir / 'summary.csv'}, found {len(df)}"
    return df.iloc[0]
