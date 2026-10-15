"""v14 Stages 1-3 (and 4/5 when a validated source is requested): every bound
variant from ONE function call on ONE frozen checkpoint.

    python3 -m certified_obstacle_nn.experiments.v14.evaluate_variants --seed 0

U sources (each row of the table carries its own status):
  * ``float64_quadrature``  -- converged composite Gauss integrals of every
    component (status ``diagnostic_float64``: these are approximations of the
    integrals, NOT upper bounds; the derived "bounds" are diagnostics).
  * ``global_lipschitz``    -- specification section 7 remainder split per v14 A4
    (status ``unvalidated_enclosure``).
  * ``validated_<backend>`` -- per-cell outward enclosures from
    certification/enclose (status ``validated_upper_bound`` when every leaf
    and trace was evaluated and no NaN occurred).

E_true is evaluated for each output separately: E_true(v) for every
unprojected bound, E_true(v_hat) for the projected bound (Duffy split rule,
status ``diagnostic_float64``).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict

from certified_obstacle_nn.certification import reliability as rel
from certified_obstacle_nn.certification.certificate import certify
from certified_obstacle_nn.certification.components import converged_components
from certified_obstacle_nn.certification.partition import build_partition
from certified_obstacle_nn.errors.true_error_v14 import converged_true_error
from certified_obstacle_nn.experiments.v14.common import (
    REPO_ROOT, STEFAN_SEEDS, git_commit, load_frozen_stefan, sha256_file, sha256_json,
)


def true_errors(problem, triple, n_raw=64, n_proj=256) -> dict:
    raw = converged_true_error(problem, triple, n_raw, "duffy", projected=False)
    proj = converged_true_error(problem, triple, n_proj, "duffy", projected=True)
    keep = ("grad", "flux", "init", "term", "graph", "weak", "E_true", "mult", "norm_grad_u", "norm_sigma",
            "rel_change_E_true", "n", "rule", "status")
    return {"raw": {k: raw[k] for k in keep}, "projected": {k: proj[k] for k in keep}}


def bound_rows(problem, U: dict, status: str, source: str, te: dict) -> dict:
    b = rel.all_bounds(problem, U["U_r"], U["U_p"], U["U_c"], status=status, U_r_hat=U.get("U_r_hat"),
                       norm_grad_u=te["raw"]["norm_grad_u"], norm_sigma=te["raw"]["norm_sigma"])
    b["source"] = source
    E, Eh = te["raw"]["E_true"], te["projected"]["E_true"]
    for k in ("E_U_uniform_216_comparison_convention", "E_U_uniform", "E_U_weighted", "E_U_delta_1_12",
              "E_U_delta_1_6", "E_U_delta_star", "E_U_best"):
        b[f"ratio_{k}"] = b[k] / E
    if isinstance(b.get("E_U_original_via_projection"), float):
        b["ratio_E_U_original_via_projection"] = b["E_U_original_via_projection"] / E
    if "E_U_projected" in b:
        b["ratio_E_U_projected_vs_E_true_vhat"] = b["E_U_projected"] / Eh
    b["E_true_v"] = E
    b["E_true_vhat"] = Eh
    b["grad_err_v"] = math.sqrt(te["raw"]["grad"])
    b["flux_err_v"] = math.sqrt(te["raw"]["flux"])
    return b


def evaluate(seed: int = 0, n_quad: int = 128, global_grid: int = 40, validated: dict | None = None) -> dict:
    """One frozen checkpoint, every variant. `validated` (optional) maps a
    source name to a dict with U_r, U_p, U_c, U_r_hat and status."""
    t0 = time.time()
    path = STEFAN_SEEDS[seed]
    problem, triple = load_frozen_stefan(path)
    te = true_errors(problem, triple)
    comp = converged_components(problem, triple, n_quad, 4)
    U_q = {"U_r": comp["r2"], "U_p": comp["p2"], "U_c": comp["c2"], "U_r_hat": comp["r2_hat"]}
    cert = certify(triple, problem, build_partition(problem, global_grid, global_grid))
    U_g = {k: cert.components[k] for k in ("U_r", "U_p", "U_c")}
    rows = [
        bound_rows(problem, U_q, "diagnostic_float64", "float64_quadrature", te),
        bound_rows(problem, U_g, "unvalidated_enclosure", f"global_lipschitz_{global_grid}x{global_grid}", te),
    ]
    for name, U in (validated or {}).items():
        rows.append(bound_rows(problem, U, U["status"], name, te))
    return {
        "seed": seed,
        "checkpoint": str(path.relative_to(REPO_ROOT)),
        "checkpoint_sha256": sha256_file(path),
        "git_commit": git_commit(),
        "problem_constants": rel.problem_constants_record(problem),
        "true_error": te,
        "components_float64": {k: v for k, v in comp.items() if k != "weights"},
        "component_weights": comp["weights"],
        "global_lipschitz_certificate": {k: v for k, v in asdict(cert).items() if k != "bounds"},
        "rows": rows,
        "wall_clock_s": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = evaluate(args.seed)
    out = REPO_ROOT / (args.out or f"results/v14/stage1_3/seed{args.seed}_variants.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    res["config_hash"] = sha256_json({k: res[k] for k in ("seed", "checkpoint", "problem_constants")})
    out.write_text(json.dumps(res, indent=1, default=str))
    import pandas as pd

    pd.DataFrame(res["rows"]).to_csv(out.with_suffix(".csv"), index=False)
    for r in res["rows"]:
        print(r["source"], r["status"], {k: (round(r[k], 4) if isinstance(r.get(k), float) else r.get(k)) for k in
              ("E_U_uniform_216_comparison_convention", "E_U_uniform", "E_U_weighted", "E_U_delta_star",
               "E_U_projected", "E_U_original_via_projection", "grad_bound_rel", "flux_bound_rel")})
    print("E_true(v)", res["true_error"]["raw"]["E_true"], "E_true(v_hat)", res["true_error"]["projected"]["E_true"])


if __name__ == "__main__":
    main()
