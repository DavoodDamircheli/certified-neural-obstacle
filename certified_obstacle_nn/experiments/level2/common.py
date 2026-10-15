"""Shared helpers for the Level-2 upgrade drivers: the three frozen cases, their recorded
E_true / reference rho_tilde, output directories, JSON writing.

Every checkpoint is loaded read-only, float64, eval mode, requires_grad=False.
"""

import json
import math
import subprocess
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO_ROOT / "results" / "level2_local"
RUNS = OUT_ROOT / "runs"

C_REL = 216.0

CASES = {
    "1d": {
        "kind": "stefan",
        "dir": REPO_ROOT / "results" / "stefan_m18" / "4x64" / "seed0",
        "true_error_json": "true_error/true_error_nt60_nx60.json",
        "legacy_json": "certification/certificate_grid1024.json",
    },
    "2d_base": {"kind": "disk", "dir": REPO_ROOT / "results" / "expanding_disk_2d" / "stage3_6x64_N20000_s0_k5"},
    "2d_long": {"kind": "disk", "dir": REPO_ROOT / "results" / "expanding_disk_2d" / "final" / "long_6x64_N20000_s0_k5"},
}
for _s in (1, 2):
    CASES[f"2d_base_s{_s}"] = {"kind": "disk",
                               "dir": REPO_ROOT / "results" / "expanding_disk_2d" / f"stage5_6x64_N20000_s{_s}_k5"}
    CASES[f"2d_long_s{_s}"] = {"kind": "disk",
                               "dir": REPO_ROOT / "results" / "expanding_disk_2d" / "final" / f"long_6x64_N20000_s{_s}_k5"}


def git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def load_case(name: str):
    """Returns (model, meta). model is the frozen legacy object (NeuralTriple / Triple2D)."""
    c = CASES[name]
    torch.set_default_dtype(torch.float64)
    if c["kind"] == "stefan":
        from certified_obstacle_nn.problems.stefan import StefanProblem
        from certified_obstacle_nn.scripts.certify_cli import load_frozen_triple
        problem = StefanProblem()
        cfg = json.loads((c["dir"] / "config.json").read_text())
        model = load_frozen_triple(problem, cfg, c["dir"] / "weights.pt").double()
        te = json.loads((c["dir"] / c["true_error_json"]).read_text())["result"]
        leg = json.loads((c["dir"] / c["legacy_json"]).read_text())["result"]
        meta = {"case": name, "checkpoint": str(c["dir"]), "E_true": te["E_true"], "problem": problem,
                "rho_ref": leg["rho_tilde_h"], "rho_ref_source": "legacy grid1024 Gauss n_qp=4",
                "C_eq": 2.0, "config": cfg}
    else:
        from certified_obstacle_nn.experiments.expanding_disk_2d.final_common import load_run
        model, cfg = load_run(c["dir"])
        m = json.loads((c["dir"] / "metrics.json").read_text())
        meta = {"case": name, "checkpoint": str(c["dir"]), "E_true": m["E_true"], "cfg": cfg,
                "rho_ref": m["rho_tilde"], "rho_ref_source": "metrics.json Sobol 1e6",
                "C_eq": 1.0, "config": cfg.to_dict()}
    for p in model.parameters():
        assert not p.requires_grad
    meta["floor"] = math.sqrt(meta["C_eq"] * C_REL) * meta["rho_ref"] / meta["E_true"]
    return model, meta


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_default))
    return path


def _default(o):
    if hasattr(o, "_asdict"):
        return o._asdict()
    if hasattr(o, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, torch.Tensor):
        return o.tolist()
    return str(o)


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S")


def legacy_constants(model, meta) -> dict:
    """The legacy GLOBAL constants (computed by the unchanged legacy functions)."""
    if "cfg" not in meta:
        from certified_obstacle_nn.certification.network_bounds import compute_triple_bounds
        b = compute_triple_bounds(meta["problem"], model)
        return {"L_sm": b.L_sm, "L_Psi": b.L_Psi, "L_gamma": b.L_gamma, "L_Phi0": b.L_Phi0, "L_PhiT": b.L_PhiT,
                "L_R": b.lip_R_PDE, "m_R_apriori": b.m_R}
    from certified_obstacle_nn.experiments.expanding_disk_2d.level2_diagnostic import data_bounds, triple_bounds
    cfg = meta["cfg"]
    bd = triple_bounds(model, cfg, data_bounds(cfg.kappa))
    return {"L_sm": bd["L_sm_old"], "L_Psi": bd["L_Psi"], "L_gamma": bd["L_gamma"], "L_Phi0": bd["L_Phi0"],
            "L_PhiT": bd["L_PhiT"], "L_R": bd["L_R"], "m_R_apriori": bd["m_R_apriori"]}


TABLE_COLS = ["case", "checkpoint", "method", "method_variant", "grid", "cells", "cells_evaluated",
              "unresolved_fraction", "E_h_lip", "E_h_unr", "E_h_0T", "E_h_total", "U", "L", "rho_h_sq", "E_cert_U",
              "E_true", "certificate_effectivity", "floor", "slack", "runtime", "max_depth", "notes"]
ALL_ROWS = OUT_ROOT / "tables" / "all_rows.csv"


def record(rows):
    """Append rows (dedup on case/method_variant/grid) and rebuild docs/history/LEVEL2_COMPARISON_TABLE.csv."""
    import csv
    import fcntl
    ALL_ROWS.parent.mkdir(parents=True, exist_ok=True)
    lock = open(ALL_ROWS.parent / ".lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    old = list(csv.DictReader(open(ALL_ROWS))) if ALL_ROWS.exists() else []
    key = lambda r: (r["case"], r["method_variant"], str(r["grid"]))  # noqa: E731
    new_keys = {key(r) for r in rows}
    allr = [r for r in old if key(r) not in new_keys] + [{c: r.get(c, "") for c in TABLE_COLS} for r in rows]
    for path in (ALL_ROWS, REPO_ROOT / "docs/history/LEVEL2_COMPARISON_TABLE.csv"):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=TABLE_COLS)
            w.writeheader()
            w.writerows(allr)
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()


def upper_sum_row(meta, res, method, variant, grid, notes=""):
    """Table row of an upper-sum result (uniform or adaptive)."""
    smooth = sum(res["U_terms"][k] - res["mid_terms"][k] for k in ("R", "sig", "kw", "c"))
    E_cu = res["E_cert_U"]
    return {"case": meta["case"], "checkpoint": meta["checkpoint"].replace(str(REPO_ROOT) + "/", ""),
            "method": method, "method_variant": variant, "grid": grid, "cells": res["n_cells"],
            "cells_evaluated": res.get("cells_evaluated", res["n_cells"]),
            "unresolved_fraction": res["V_unr_frac"], "E_h_lip": smooth,
            "E_h_unr": res["U_terms"]["int"] - res["mid_terms"]["int"],
            "E_h_0T": res["U_0"] + res["U_T"] - res["rho_mid_0"] - res["rho_mid_T"],
            "E_h_total": res["U"] - res["rho_h_sq"], "U": res["U"], "L": res["L"], "rho_h_sq": res["rho_h_sq"],
            "E_cert_U": E_cu, "E_true": meta["E_true"], "certificate_effectivity": E_cu / meta["E_true"],
            "floor": meta["floor"], "slack": E_cu / meta["E_true"] / meta["floor"], "runtime": res["wall_s"],
            "max_depth": res.get("max_depth", 0), "notes": notes}


def legacy_row(meta, r, variant, grid, method="global/uniform", notes=""):
    """Table row of a legacy-formula result (dict with rho_tilde_h, E_h_*_new, E_cert_U_new)."""
    rho2 = r["rho_tilde_h"] ** 2
    return {"case": meta["case"], "checkpoint": meta["checkpoint"].replace(str(REPO_ROOT) + "/", ""),
            "method": method, "method_variant": variant, "grid": grid, "cells": r["n_cells"],
            "cells_evaluated": r["n_cells"], "unresolved_fraction": r["V_unr_frac"], "E_h_lip": r["E_h_lip_new"],
            "E_h_unr": r["E_h_unr"], "E_h_0T": r["E_h_0T"], "E_h_total": r["E_h_new"], "U": rho2 + r["E_h_new"],
            "L": "", "rho_h_sq": rho2, "E_cert_U": r["E_cert_U_new"], "E_true": meta["E_true"],
            "certificate_effectivity": r["E_cert_U_new"] / meta["E_true"], "floor": meta["floor"],
            "slack": r["E_cert_U_new"] / meta["E_true"] / meta["floor"], "runtime": r["wall_s"], "max_depth": 0,
            "notes": notes}
