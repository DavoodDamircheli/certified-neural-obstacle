"""v17: a chunked evaluation reproduces the unchunked one exactly (per-cell evaluation is independent)."""

import json
from pathlib import Path

import pandas as pd

from certified_obstacle_nn.experiments.v14.common import REPO_ROOT
from certified_obstacle_nn.experiments.v17 import chunked_certify as CC
from certified_obstacle_nn.experiments.v17 import split_certify as SC
from certified_obstacle_nn.experiments.v17.stage5_pilot import LONG0_1E5
from certified_obstacle_nn.tests.stored_data import needs


@needs('results/v17_partitions')
def test_chunked_equals_unchunked_on_pilot_partition(tmp_path, monkeypatch):
    parts = sorted((REPO_ROOT / "results/v17_partitions").glob("*/partition.json"))
    part_dir = str(parts[0].parent.relative_to(REPO_ROOT))
    # write job and output dirs under tmp_path, not results/
    from certified_obstacle_nn.experiments import results_io

    monkeypatch.setattr(results_io, "RESULTS_DIR", tmp_path)
    orig = results_io.make_run_dir
    monkeypatch.setattr(results_io, "make_run_dir", lambda exp, results_dir=tmp_path: orig(exp, tmp_path))
    monkeypatch.setattr(CC, "REPO_ROOT", REPO_ROOT)
    job = CC.init(LONG0_1E5, "TM1", 3, part_dir)
    for i in range(3):
        CC.chunk(job, i, 8)
    d = CC.finalize(job, 8)
    chunked = pd.read_csv(d / "summary.csv", float_precision="round_trip", dtype={"partition_hash": str}).iloc[0]
    meta = json.loads((job / "job.json").read_text())
    _, r = CC._load(job, 8)
    config, rows, cells, r, _ = SC.run(r.run_dir, ["numpy"], 8, 2, r=r, enclosure="TM1", log=lambda *_: None)
    for k in ("U_R", "U_S", "U_I", "U_C", "U_r", "U_p", "U_c", "U_r_hat", "flux_bound_direct_split",
              "grad_bound_direct_split", "E_dir_split"):
        assert float(chunked[k]) == rows[0][k], k
    assert chunked["partition_hash"] == meta["partition_hash"]
