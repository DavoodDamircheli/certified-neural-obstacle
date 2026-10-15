"""v17 split driver on the small stored Stefan 32^2 reference run (the reproduce_one reference)."""

import math

from certified_obstacle_nn.experiments.v14.reproduce_one import find_reference
from certified_obstacle_nn.experiments.v17 import split_certify as SC


def test_split_certify_stefan_32_reproduces_and_reports():
    ref = find_reference()
    assert ref is not None
    from pathlib import Path

    config, rows, cells, r, _ = SC.run(Path(ref).parent, ["numpy"], 4, 2, log=lambda *_: None)
    (row,) = rows
    assert row["status"] == "validated_upper_bound"
    assert row["reproduces_stored_aggregates"] is True  # same engine + partition -> identical aggregates
    assert row["split_consistency_rel"] <= 1e-14
    assert row["ratio_status"] == "validated_upper_bound"  # analytic Stefan denominator
    assert row["flux_bound_direct_split"] <= row["flux_bound_direct"]
    assert row["grad_bound_direct_split"] <= row["grad_bound_direct_split"] * 1 <= row["grad_bound_direct_check_theta_half"]
    # per-class sums add up to the totals (each class reduced separately in arb, upward)
    for nm in ("U_R", "U_S", "U_C"):
        assert row[f"{nm}_res"] + row[f"{nm}_unres"] >= row[nm] * (1 - 1e-14)
    # every validated per-term sum dominates its float64 integral on the same partition (slack >= 1)
    for nm in ("U_R", "U_S", "U_I", "U_C"):
        assert row[f"slack_{nm}"] >= 1.0 or math.isnan(row[f"slack_{nm}"])
    # sampled sup <= validated cell bound -> enclosure factor >= 1; sampled sup >= mean -> majorant factor >= 1
    for nm in ("U_R", "U_S"):
        assert row[f"enclosure_factor_{nm}"] >= 1.0 and row[f"majorant_factor_{nm}"] >= 1.0
    for k in ("checkpoint_sha256", "partition_hash", "trace0_hash", "traceT_hash", "config_hash", "enclosure"):
        assert row[k]
