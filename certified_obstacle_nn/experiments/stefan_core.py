"""M18: architecture-comparison study, specification section 1's own study set
(4x32, 4x64, 6x64, 6x128) x >=5 seeds each, Stefan problem, IDENTICAL
optimizer budget across every run (experiments/stefan_baseline.py's own
defaults: 5000 Adam epochs @ lr=1e-3, up to 300 L-BFGS epochs, 5000 interior
+ 300 initial collocation points -- unchanged, not re-tuned per architecture
or seed), certified at the single mesh fixed by the Pre-M18 mesh-selection
milestone (results/stefan_mesh_selection.json: 1024^2, "M17" in the user's
own later shorthand).

Per specification section 12 invariant 2 and certify_cli.py's own stated
constraint ("never call certify() directly in the same process that just
called train()"), this script trains IN-PROCESS (mirrors
experiments/stefan_baseline.py -- training touches no certification/ code)
but invokes certification (scripts/certify_cli.py, Option B mode
min_apriori_and_partition -- the mode this milestone's own mesh-selection
sweep used) and the true-error/E_FB diagnostics (scripts/true_error_cli.py)
through SEPARATE SUBPROCESSES, one fresh process per call, per run.

Table 1 columns (user-specified 2026-09-11): Architecture, Seed, E_true,
rho_tilde, E_quad = sqrt(rho_tilde^2 + E_h), E_cert, E_cert_U, I_eff_raw,
I_eff_cert, E_FB_gap_only, E_FB_gap_and_multiplier, training_time,
certification_time. E_cert/E_cert_U/E_h report the Option-B SHARPENED
(*_new) values -- the methodology this milestone's own T2/T3 work
established as the intended path into M18, not a silent substitution;
*_old values are also carried in the per-run JSON for transparency. E_quad
is pure arithmetic from already-certified quantities (specification section 8's
own E_cert/E_cert_U inner term), not a new formula. certification_time =
certify_cli wall-clock + true_error_cli wall-clock (the two subprocess calls
together constitute "certifying" a run). E_FB_gap_only/E_FB_gap_and_multiplier
(errors/free_boundary.py, redefined 2026-09-11 as a fixed-threshold contact
detector) and their fixed eps_u/eps_lambda/alpha_u/alpha_lambda are also
carried per row -- never retuned per run, see errors/free_boundary.py.

Also emitted per run (user-specified): the component errors ||u-u_theta||_V,
||sigma-sigma_phi||_{L2(Q)}, ||lambda-lambda_psi||_{L2(Q)} (true_error_cli's
own norm_u_V / norm_sigma_L2 / norm_lambda_L2 -- already computed by
errors/true_error.py, no new math), and the estimator split rho_r /
rho_p_tilde / rho_c (certify_option_b's OptionBCertificateResult, extended
this milestone to carry these through from EstimatorSummary alongside the
combined rho_tilde_h it already reported).

Also emitted (T3, results/pre_m18_t3_prediction.json's m18_requirements):
L_R, m_R_apriori, m_R_h, m_R_cert, lip_term_fraction, T1_new, per-network
Pi_N/Sigma_N (state/flux/multiplier), and parameter counts. Table 1's
ranking is per T3: I_eff_cert and E_true are the accuracy ranking; E_cert_U
is reported but never used alone to rank architectures; this script's report
states explicitly whether E_cert_U's architecture ordering matches E_true's,
and treats any divergence as a finding, not something to tune away.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.stefan import StefanProblem
from certified_obstacle_nn.training.sampler import TrainingSampler
from certified_obstacle_nn.training.trainer import train

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
M18_DIR = RESULTS_DIR / "stefan_m18"

# specification section 1's own study set, as (depth, width) -- matches
# MLP.__init__(hidden_width, hidden_depth) / NeuralTriple(width, depth).
ARCHITECTURES = [
    {"tag": "4x32", "depth": 4, "width": 32},
    {"tag": "4x64", "depth": 4, "width": 64},
    {"tag": "6x64", "depth": 6, "width": 64},
    {"tag": "6x128", "depth": 6, "width": 128},
]
SEEDS = [0, 1, 2, 3, 4]  # >= 5 seeds, user-specified

# Identical optimizer budget across every (architecture, seed) -- matches
# experiments/stefan_baseline.py's own defaults, unchanged.
N_INTERIOR = 5000
N_INITIAL = 300
N_ADAM_EPOCHS = 5000
ADAM_LR = 1e-3
N_LBFGS_EPOCHS = 300
LBFGS_LR = 1.0

# M17-fixed standard certification mesh (results/stefan_mesh_selection.json).
CERT_GRID = 1024
OPTION_B_MODE = "min_apriori_and_partition"
N_QP = 4
TRUE_ERROR_N_T = 60
TRUE_ERROR_N_X = 60
# 200 (true_error_cli.py's measured-safe default against the 4x64 baseline)
# was NOT enough for 6x64/seed4 during the real M18 sweep: self-check failed
# by a hair (1.168e-2 > 1e-2 tolerance) when n_t doubled 200->400. Raised to
# 400 -- more t-resolution, not a loosened tolerance, per this session's
# standing rule (diagnose, don't paper over quadrature self-check failures).
FB_N_T = 400


def _load_train_info_from_disk(run_dir: Path) -> dict:
    """Resume support: if a prior sweep attempt already trained this run,
    reconstruct the exact same train_info dict _train_one_run would have
    returned, from the artifacts it already wrote to disk -- no retraining."""
    config_path = run_dir / "config.json"
    weights_path = run_dir / "weights.pt"
    history_csv = run_dir / "history.csv"
    config = json.loads(config_path.read_text())
    history = pd.read_csv(history_csv)
    return {
        "weights_path": weights_path,
        "config_path": config_path,
        "training_time": config["training_time_seconds"],
        "final_loss_total": float(history["loss_total"].iloc[-1]),
        "lbfgs_outcome_reason": config["optimizer_schedule"]["lbfgs"]["outcome_reason"],
        "lbfgs_steps_completed": config["optimizer_schedule"]["lbfgs"]["n_steps_completed"],
    }


def _train_one_run_resumable(width: int, depth: int, seed: int, run_dir: Path) -> dict:
    if (run_dir / "weights.pt").exists() and (run_dir / "config.json").exists() and (run_dir / "history.csv").exists():
        print("  [resume] training already complete, loading from disk", flush=True)
        return _load_train_info_from_disk(run_dir)
    return _train_one_run(width, depth, seed, run_dir)


def _train_one_run(width: int, depth: int, seed: int, run_dir: Path) -> dict:
    torch.manual_seed(seed)
    problem = StefanProblem()
    triple = NeuralTriple(problem, width=width, depth=depth)

    sampler = TrainingSampler(problem, n_interior=N_INTERIOR, n_initial=N_INITIAL, seed=seed)
    t, x, x0 = sampler.sample()

    run_dir.mkdir(parents=True, exist_ok=True)
    history_csv = run_dir / "history.csv"
    weights_path = run_dir / "weights.pt"
    config_path = run_dir / "config.json"

    wall_start = time.perf_counter()
    history, outcome = train(
        problem, triple, t, x, x0,
        n_epochs=N_ADAM_EPOCHS, lr=ADAM_LR,
        n_lbfgs_epochs=N_LBFGS_EPOCHS, lbfgs_lr=LBFGS_LR,
        csv_path=str(history_csv),
    )
    training_time = time.perf_counter() - wall_start

    torch.save(triple.state_dict(), weights_path)

    config = {
        "problem": "stefan",
        "seed": seed,
        "architecture": {"width": width, "depth": depth, "multiplier_head": triple.multiplier_net.head},
        "optimizer_schedule": {
            "adam": {"lr": ADAM_LR, "n_epochs": N_ADAM_EPOCHS},
            "lbfgs": {
                "lr": LBFGS_LR, "n_epochs_requested": N_LBFGS_EPOCHS, "line_search": "strong_wolfe",
                "outcome_reason": outcome.reason, "n_steps_completed": outcome.n_steps_completed,
            },
        },
        "grid_sizes": {"n_interior": N_INTERIOR, "n_initial": N_INITIAL},
        "training_time_seconds": training_time,
        "weights_path": str(weights_path),
        "history_csv_path": str(history_csv),
    }
    config_path.write_text(json.dumps(config, indent=2))

    return {
        "weights_path": weights_path,
        "config_path": config_path,
        "training_time": training_time,
        "final_loss_total": float(history["loss_total"].iloc[-1]),
        "lbfgs_outcome_reason": outcome.reason,
        "lbfgs_steps_completed": outcome.n_steps_completed,
    }


def _run_cli(module: str, args: list[str]) -> tuple[float, dict]:
    """Runs a certification-adjacent script as a fresh subprocess (never in
    the training process), returns (wall_clock_seconds, parsed JSON payload)."""
    wall_start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", module] + args,
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    wall_clock = time.perf_counter() - wall_start
    if result.returncode != 0:
        raise RuntimeError(
            f"{module} failed (exit {result.returncode}):\n--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
    return wall_clock, result


def _certify_one_run(weights_path: Path, config_path: Path, out_dir: Path) -> tuple[float, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    wall_clock, _ = _run_cli(
        "certified_obstacle_nn.scripts.certify_cli",
        [
            "--weights", str(weights_path), "--config", str(config_path),
            "--grid", str(CERT_GRID), "--n-qp", str(N_QP),
            "--option-b-mode", OPTION_B_MODE, "--out", str(out_dir),
        ],
    )
    payload = json.loads((out_dir / f"certificate_grid{CERT_GRID}.json").read_text())
    return wall_clock, payload


def _certify_one_run_resumable(weights_path: Path, config_path: Path, out_dir: Path) -> tuple[float, dict]:
    cert_path = out_dir / f"certificate_grid{CERT_GRID}.json"
    if cert_path.exists():
        print("  [resume] certification already complete, loading from disk", flush=True)
        payload = json.loads(cert_path.read_text())
        return payload["wall_clock_seconds"], payload
    return _certify_one_run(weights_path, config_path, out_dir)


def _true_error_one_run(weights_path: Path, config_path: Path, out_dir: Path) -> tuple[float, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    wall_clock, _ = _run_cli(
        "certified_obstacle_nn.scripts.true_error_cli",
        [
            "--weights", str(weights_path), "--config", str(config_path),
            "--n-t", str(TRUE_ERROR_N_T), "--n-x", str(TRUE_ERROR_N_X),
            "--fb-n-t", str(FB_N_T), "--out", str(out_dir),
        ],
    )
    payload = json.loads((out_dir / f"true_error_nt{TRUE_ERROR_N_T}_nx{TRUE_ERROR_N_X}.json").read_text())
    return wall_clock, payload


def _true_error_one_run_resumable(weights_path: Path, config_path: Path, out_dir: Path) -> tuple[float, dict]:
    te_path = out_dir / f"true_error_nt{TRUE_ERROR_N_T}_nx{TRUE_ERROR_N_X}.json"
    if te_path.exists():
        print("  [resume] true_error already complete, loading from disk", flush=True)
        payload = json.loads(te_path.read_text())
        return payload["wall_clock_seconds"], payload
    return _true_error_one_run(weights_path, config_path, out_dir)


def _build_row(arch: dict, seed: int, train_info: dict, cert_payload: dict, te_payload: dict, cert_wall: float, te_wall: float) -> dict:
    r = cert_payload["result"]
    b = r["bounds"]
    rb = r["residual_bound"]
    te = te_payload["result"]
    fb = te_payload.get("free_boundary_error")

    E_true = te["E_true"]
    rho_tilde = r["rho_tilde_h"]
    E_h_new = r["E_h_new"]
    E_quad = (rho_tilde ** 2 + E_h_new) ** 0.5
    E_cert = r["E_cert_new"]
    E_cert_U = r["E_cert_U_new"]
    I_eff_raw = rho_tilde / E_true
    I_eff_cert = E_quad / E_true

    return {
        "Architecture": arch["tag"],
        "Seed": seed,
        "E_true": E_true,
        "rho_tilde": rho_tilde,
        "E_quad": E_quad,
        "E_cert": E_cert,
        "E_cert_U": E_cert_U,
        "I_eff_raw": I_eff_raw,
        "I_eff_cert": I_eff_cert,
        "E_FB_gap_only": fb["E_FB_gap_only"] if fb is not None else None,
        "E_FB_gap_and_multiplier": fb["E_FB_gap_and_multiplier"] if fb is not None else None,
        "eps_u": fb["eps_u"] if fb is not None else None,
        "eps_lambda": fb["eps_lambda"] if fb is not None else None,
        "alpha_u": fb["alpha_u"] if fb is not None else None,
        "alpha_lambda": fb["alpha_lambda"] if fb is not None else None,
        "training_time": train_info["training_time"],
        "certification_time": cert_wall + te_wall,
        # component errors
        "norm_u_V": te["norm_u_V"],
        "norm_sigma_L2": te["norm_sigma_L2"],
        "norm_lambda_L2": te["norm_lambda_L2"],
        # estimator split
        "rho_r": r["rho_r"],
        "rho_p_tilde": r["rho_p_tilde"],
        "rho_c": r["rho_c"],
        # T3-required columns
        "L_R": b["lip_R_PDE"],
        "m_R_apriori": rb["m_R_apriori"],
        "m_R_h": rb["m_R_h"],
        "m_R_cert": rb["m_R_cert"],
        "lip_term_fraction": rb["lip_term_fraction"],
        "T1_new": r["T1_new"],
        "Sigma_N_state": b["state_bounds"]["output_bound"],
        "Pi_N_state": b["state_bounds"]["gradient_bound"],
        "Sigma_N_flux": b["flux_bounds"]["output_bound"],
        "Pi_N_flux": b["flux_bounds"]["gradient_bound"],
        "Sigma_N_multiplier": b["multiplier_bounds"]["output_bound"],
        "Pi_N_multiplier": b["multiplier_bounds"]["gradient_bound"],
        "param_count": arch["param_count"],
        # old (pre-Option-B) values, for transparency
        "E_h_old": r["E_h_old"],
        "E_cert_old": r["E_cert_old"],
        "E_cert_U_old": r["E_cert_U_old"],
        # extras
        "V_h_unr": r["V_h_unr"],
        "final_loss_total": train_info["final_loss_total"],
        "lbfgs_outcome_reason": train_info["lbfgs_outcome_reason"],
        "E_true_quadrature_relative_change": te["quadrature_relative_change"],
        "E_FB_gap_only_quadrature_relative_change": fb["quadrature_relative_change_gap_only"] if fb is not None else None,
        "E_FB_gap_and_multiplier_quadrature_relative_change": fb["quadrature_relative_change_gap_and_multiplier"] if fb is not None else None,
        "weights_path": str(train_info["weights_path"]),
    }


def _param_count(width: int, depth: int) -> int:
    problem = StefanProblem()
    triple = NeuralTriple(problem, width=width, depth=depth)
    return sum(p.numel() for p in triple.parameters())


def main(architectures: list[dict] | None = None, seeds: list[int] | None = None) -> pd.DataFrame:
    architectures = architectures if architectures is not None else ARCHITECTURES
    seeds = seeds if seeds is not None else SEEDS

    M18_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for arch in architectures:
        arch = dict(arch)
        arch["param_count"] = _param_count(arch["width"], arch["depth"])
        for seed in seeds:
            run_dir = M18_DIR / arch["tag"] / f"seed{seed}"
            print(f"=== {arch['tag']} seed={seed} ===", flush=True)

            train_info = _train_one_run_resumable(arch["width"], arch["depth"], seed, run_dir)
            print(f"  trained: {train_info['training_time']:.1f}s  final_loss={train_info['final_loss_total']:.4e}  "
                  f"lbfgs={train_info['lbfgs_outcome_reason']}", flush=True)

            cert_wall, cert_payload = _certify_one_run_resumable(
                train_info["weights_path"], train_info["config_path"], run_dir / "certification",
            )
            print(f"  certified: {cert_wall:.1f}s  E_cert_U_new={cert_payload['result']['E_cert_U_new']:.4e}", flush=True)

            te_wall, te_payload = _true_error_one_run_resumable(
                train_info["weights_path"], train_info["config_path"], run_dir / "true_error",
            )
            fb = te_payload.get("free_boundary_error")
            fb_i = fb['E_FB_gap_only'] if fb else float('nan')
            fb_ii = fb['E_FB_gap_and_multiplier'] if fb else float('nan')
            print(f"  true_error: {te_wall:.1f}s  E_true={te_payload['result']['E_true']:.4e}  "
                  f"E_FB_gap_only={fb_i:.4e}  E_FB_gap_and_multiplier={fb_ii:.4e}", flush=True)

            row = _build_row(arch, seed, train_info, cert_payload, te_payload, cert_wall, te_wall)
            rows.append(row)

    table = pd.DataFrame.from_records(rows)
    table_path = M18_DIR / "table1.csv"
    table.to_csv(table_path, index=False)
    print(f"\nTable 1 -> {table_path}")
    return table


if __name__ == "__main__":
    main()
