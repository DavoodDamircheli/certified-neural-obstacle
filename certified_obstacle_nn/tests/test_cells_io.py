"""Tests for certification/cells_io.py (Pre-M18 T2)."""

import numpy as np
import pandas as pd
import pytest

from certified_obstacle_nn.certification.cells_io import (
    CELLS_CSV_CELL_THRESHOLD,
    load_per_cell_arrays,
    save_per_cell_arrays,
)


def _fake_df(n):
    return pd.DataFrame({
        "cell_id": np.arange(n),
        "rho_total_K": np.random.default_rng(0).random(n),
    })


def test_small_grid_writes_csv(tmp_path):
    df = _fake_df(100)
    provenance = save_per_cell_arrays(df, tmp_path, n_cells=100)
    assert provenance["format"] == "csv"
    assert (tmp_path / "cells.csv").exists()
    assert (tmp_path / "cells_provenance.json").exists()


def test_large_grid_writes_npz(tmp_path):
    df = _fake_df(10)  # content size doesn't matter, n_cells argument decides format
    provenance = save_per_cell_arrays(df, tmp_path, n_cells=CELLS_CSV_CELL_THRESHOLD + 1)
    assert provenance["format"] == "npz"
    assert (tmp_path / "cells.npz").exists()
    assert not (tmp_path / "cells.csv").exists()


def test_threshold_boundary_is_csv_inclusive(tmp_path):
    df = _fake_df(5)
    provenance = save_per_cell_arrays(df, tmp_path, n_cells=CELLS_CSV_CELL_THRESHOLD)
    assert provenance["format"] == "csv"


def test_roundtrip_csv(tmp_path):
    df = _fake_df(50)
    save_per_cell_arrays(df, tmp_path, n_cells=50)
    loaded = load_per_cell_arrays(tmp_path)
    pd.testing.assert_frame_equal(df, loaded)


def test_roundtrip_npz(tmp_path):
    df = _fake_df(50)
    save_per_cell_arrays(df, tmp_path, n_cells=CELLS_CSV_CELL_THRESHOLD + 1)
    loaded = load_per_cell_arrays(tmp_path)
    pd.testing.assert_frame_equal(df, loaded)


def test_predicted_vs_measured_bytes_are_reported(tmp_path):
    df = _fake_df(1000)
    provenance = save_per_cell_arrays(df, tmp_path, n_cells=1000)
    assert provenance["predicted_bytes"] == 1000 * len(df.columns) * 8
    assert provenance["measured_bytes"] > 0
    assert provenance["measured_over_predicted"] > 0


def test_load_without_provenance_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_per_cell_arrays(tmp_path)
