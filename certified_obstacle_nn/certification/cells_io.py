"""Per-cell array persistence, Pre-M18 milestone T2.

`compute_raw_estimator`'s per-cell DataFrame (rho_pde_K, rho_sigma_K, ...,
rho_total_K, phi_sm_K, phi_int_K -- specification section 4/M10) is kept, never
dropped, at every grid size: the M10 localization identity
(sum_K rho_total_K^2 == rho_tilde^2) is only checkable from these arrays.
Above 1e6 cells (1024^2 and up), a CSV of the underlying float64 values
becomes a needlessly large text file for no accuracy benefit -- this module
writes `.npz` instead above that threshold (avoids adding a new parquet
dependency; `np.savez_compressed` is already available via numpy, an
existing dependency). Below the threshold, `.csv` stays human-readable, as
it already was.

Predicted (raw in-memory bytes: n_cells * n_columns * 8, float64/int64
uniformly) vs measured (actual bytes on disk) is logged for every write --
csv's ASCII float encoding is typically LARGER than the raw binary size;
npz's compression is typically smaller. Both are real, reportable facts
about the two formats, not just a sanity check.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

CELLS_CSV_CELL_THRESHOLD = 1_000_000
BYTES_PER_VALUE = 8  # float64 and int64 are both 8 bytes


def predicted_bytes(n_cells: int, n_columns: int) -> int:
    return n_cells * n_columns * BYTES_PER_VALUE


def save_per_cell_arrays(per_cell_df: pd.DataFrame, out_dir: Path, n_cells: int) -> dict:
    """Writes per_cell_df to `out_dir`, csv below CELLS_CSV_CELL_THRESHOLD
    cells, npz above. Returns and logs a provenance dict (format, path,
    predicted/measured bytes) -- also written as a small sidecar JSON next
    to the cells file itself, so a caller that only has a directory path
    (e.g. a later analysis script reading results back from disk, per this
    session's established "recompute nothing, read from files" discipline)
    can recover it without re-deriving anything.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    predicted = predicted_bytes(n_cells, len(per_cell_df.columns))

    if n_cells <= CELLS_CSV_CELL_THRESHOLD:
        path = out_dir / "cells.csv"
        per_cell_df.to_csv(path, index=False)
        fmt = "csv"
    else:
        path = out_dir / "cells.npz"
        np.savez_compressed(path, **{col: per_cell_df[col].to_numpy() for col in per_cell_df.columns})
        fmt = "npz"

    measured = path.stat().st_size
    provenance = {
        "format": fmt,
        "path": str(path),
        "n_cells": n_cells,
        "n_columns": len(per_cell_df.columns),
        "predicted_bytes": predicted,
        "measured_bytes": measured,
        "measured_over_predicted": measured / predicted if predicted > 0 else float("nan"),
    }
    (out_dir / f"{path.stem}_provenance.json").write_text(json.dumps(provenance, indent=2))

    print(
        f"[cells_io] wrote {path} format={fmt} n_cells={n_cells} "
        f"predicted_bytes={predicted:,} measured_bytes={measured:,} "
        f"(measured/predicted={provenance['measured_over_predicted']:.3f})",
        flush=True,
    )
    return provenance


def load_per_cell_arrays(out_dir: Path) -> pd.DataFrame:
    """Reads back whichever of cells.csv / cells.npz save_per_cell_arrays
    wrote, by checking the provenance sidecar first (never guesses the
    format from a bare file-existence check, which could pick up a stale
    file from a previous run at a different grid size in the same
    directory)."""
    csv_provenance = out_dir / "cells_provenance.json"
    if csv_provenance.exists():
        provenance = json.loads(csv_provenance.read_text())
        path = Path(provenance["path"])
        if provenance["format"] == "csv":
            return pd.read_csv(path)
        npz = np.load(path)
        return pd.DataFrame({col: npz[col] for col in npz.files})
    raise FileNotFoundError(f"no cells_provenance.json in {out_dir} -- was save_per_cell_arrays ever called here?")
