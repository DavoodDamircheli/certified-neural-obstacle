"""v17: reproduce one direct certificate end to end from the frozen checkpoint (training excluded, enclosure and
reduction included) and compare it with the stored v17 record.

    python3 -m certified_obstacle_nn.experiments.v17.reproduce_one [--ref results/v17_split/<ts>] [--workers 20]

Default reference: the Stefan seed-0 512^2 split certificate (Stage 2). The stored run's source partition and
traces are reconstructed and hash-checked (load_stored_run), the checkpoint hash is checked, the per-term sums and
every direct bound are recomputed with the stored backend(s) and compared EXACTLY (bitwise floats; validated
evaluation is deterministic). Prints REPRODUCED or the differing quantities; writes nothing into results/.
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from certified_obstacle_nn.experiments.v14.common import REPO_ROOT
from certified_obstacle_nn.experiments.v17 import split_certify as SC

DEFAULT_REF = "results/v17_split/20261009T191750Z"
KEYS = ("U_R", "U_S", "U_I", "U_C", "U_r", "U_p", "U_c", "U_r_hat", "D", "L", "flux_bound_direct_split",
        "grad_bound_direct_split", "grad_bound_direct_check_theta_half", "E_dir_split", "flux_bound_direct",
        "grad_bound_direct", "E_U_direct", "flux_bound_direct_split_rel", "grad_bound_direct_split_rel")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=DEFAULT_REF)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--backends", nargs="+", default=None)
    a = ap.parse_args()
    ref_dir = REPO_ROOT / a.ref
    cfg = json.loads((ref_dir / "config.json").read_text())
    ref = pd.read_csv(ref_dir / "summary.csv", dtype=str).set_index("backend")
    backends = a.backends or list(ref.index)
    _, rows, _, r, _ = SC.run(cfg["source_run"], backends, a.workers, int(cfg["diagnostic_quadrature"].split()[0].split("=")[1]),
                              enclosure=cfg["enclosure"], log=lambda m: print(m, flush=True))
    ok = r.checkpoint_sha256 == cfg["checkpoint_sha256"]
    if not ok:
        print("checkpoint hash differs")
    for row in rows:
        s = ref.loc[row["backend"]]
        for k in ("partition_hash", "trace0_hash", "traceT_hash", "status", "enclosure"):
            if str(row[k]) != s[k]:
                ok = False
                print(f"DIFF {row['backend']} {k}: stored {s[k]} new {row[k]}")
        for k in KEYS:
            if float(s[k]) != row[k]:
                ok = False
                print(f"DIFF {row['backend']} {k}: stored {float(s[k])!r} new {row[k]!r}")
    print("REPRODUCED" if ok else "NOT REPRODUCED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
