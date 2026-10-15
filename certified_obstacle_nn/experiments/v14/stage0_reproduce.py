"""v14 Stage 0 gate: one command reproduces the 40x40 CertificateResult of the
Stefan baseline checkpoint from saved weights.

    python3 -m certified_obstacle_nn.experiments.v14.stage0_reproduce

Compares the raw-estimator components (rho_r, rho_p_tilde, rho_c, rho_tilde_h;
unaffected by A2) against the pre-A2 reference run recorded at the start of
the v14 session, within 1e-10 relative error, and records the change in the
global-Lipschitz remainder caused by the A2 spectral-norm fix. Also re-runs
certify() a second time and requires bitwise-identical output (determinism).
"""

import json
import sys
from dataclasses import asdict

from certified_obstacle_nn.certification.certificate import certify
from certified_obstacle_nn.certification.partition import build_partition
from certified_obstacle_nn.experiments.v14.common import REPO_ROOT, STEFAN_BASELINE, load_frozen_stefan, sha256_file

REF = REPO_ROOT / "results/v14/stage0/reference_pre_A2_baseline_grid40.json"
RTOL = 1e-10


def main():
    p, tr = load_frozen_stefan(STEFAN_BASELINE)
    part = build_partition(p, 40, 40)
    r1 = asdict(certify(tr, p, part))
    r2 = asdict(certify(tr, p, part))
    ref = json.loads(REF.read_text())["result"]
    rows = {}
    ok = True
    for k in ("rho_r", "rho_p_tilde", "rho_c", "rho_tilde_h"):
        e = abs(r1[k] - ref[k]) / abs(ref[k])
        rows[k] = {"now": r1[k], "reference": ref[k], "rel_err": e}
        ok &= e <= RTOL
    deterministic = json.dumps(r1, default=str) == json.dumps(r2, default=str)
    ok &= deterministic
    a2 = {k: {"pre_A2": ref["remainder"][k], "post_A2": r1["remainder"][k],
              "ratio": r1["remainder"][k] / ref["remainder"][k] if ref["remainder"][k] else None}
          for k in ("E_h_lip", "E_h_unr", "E_h_0T", "E_h")}
    a2["E_cert_U"] = {"pre_A2": ref["E_cert_U"], "post_A2": r1["E_cert_U"], "ratio": r1["E_cert_U"] / ref["E_cert_U"]}
    out = {
        "checkpoint": str(STEFAN_BASELINE.relative_to(REPO_ROOT)),
        "checkpoint_sha256": sha256_file(STEFAN_BASELINE),
        "grid": 40,
        "estimator_reproduction": rows,
        "deterministic_rerun_identical": deterministic,
        "A2_effect_on_global_lipschitz_baseline": a2,
        "result_post_A2": r1,
        "gate_passed": bool(ok),
    }
    dst = REPO_ROOT / "results/v14/stage0/stage0_gate.json"
    dst.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: out[k] for k in ("estimator_reproduction", "deterministic_rerun_identical", "A2_effect_on_global_lipschitz_baseline", "gate_passed")}, indent=1, default=str))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
