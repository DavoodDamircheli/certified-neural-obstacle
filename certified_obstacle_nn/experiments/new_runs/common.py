"""Shared scaffolding for the four NEW_RUNS.md experiments (Phase 1, 2026-09-13).

Reuses existing validated machinery rather than re-implementing it:
  - training: training/trainer.py's own `train()`, called in-process, exactly
    as experiments/stefan_core.py already does.
  - certification / E_true / E_FB: scripts/certify_cli.py and
    scripts/true_error_cli.py, invoked as fresh subprocesses -- never
    certify()/compute_true_error() called directly in the training process
    (specification section 12 invariant 2; docs/history/MEMORY_OOM_DEBUGGING_AND_SAFE_RUN_POLICY.md
    section 4 point 4).
  - result layout: experiments/results_io.py's config.json/summary.csv/
    weights.pt convention, extended (not replaced) with a status.json and a
    richer summary row.

Field-name mapping decided in the Phase 0 report (unchanged here):
  - L_R := bounds.lip_R_PDE (Lip(R_PDE)) -- this exact alias is already
    documented in certification/certificate_option_b.py's own OptionBCertificateResult
    docstring ("L_R = bounds.lip_R_PDE"), not invented for this task.
  - E_FB is two numbers, not one (errors/free_boundary.py's E_FB_gap_only and
    E_FB_gap_and_multiplier). Both are always recorded; the single `E_FB`
    alias column required by NEW_RUNS.md's Master CSV uses
    E_FB_gap_and_multiplier (the stricter of the two, since it additionally
    requires the multiplier to have engaged) -- documented in
    NEW_RUNS_METADATA.md, not silently picked.
  - PDE_residual_RMS / constitutive_residual_RMS / complementarity_residual_RMS
    did not exist as named fields anywhere in the main pipeline (only in the
    one-off experiments/stefan_stress_s2_residual_diagnostic.py, R_PDE/R_sig
    only). compute_residual_rms() below generalizes that script's exact
    sqrt(mean(residual**2)) pattern to all three residuals used in NEW_RUNS.md,
    adding R_com (complementarity) which the diagnostic script did not need.
  - spectral_norm_layer_k (NEW_RUNS.md section 3.6) already exists as
    certification/network_bounds.py's spectral_norms(mlp) but was never
    persisted anywhere; extract_spectral_norms() below just calls it per
    sub-network and flattens the result into loggable fields.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from certified_obstacle_nn.autograd.derivatives import eval_triple_derivatives
from certified_obstacle_nn.certification.network_bounds import spectral_norms
from certified_obstacle_nn.residuals.residuals import assemble_residuals
from certified_obstacle_nn.scripts.certify_cli import build_problem, load_frozen_triple
from certified_obstacle_nn.training.losses import compute_loss

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results"
NEW_RUNS_DIR = RESULTS_DIR / "new_runs"
MASTER_CSV_PATH = NEW_RUNS_DIR / "new_runs_master.csv"

# Exact column list from NEW_RUNS.md's "Master CSV Requirement" section, in
# the order given there. Extra columns (the E_FB breakdown, rho_* split, the
# four E_h parts, checkpoint paths, ...) are appended after these, never
# instead of them -- NEW_RUNS.md: "Fields that do not apply to a particular
# experiment may be blank/NaN", not "columns may be renamed or dropped".
MASTER_CSV_COLUMNS = [
    "experiment", "subexperiment", "strategy", "seed",
    "architecture", "width", "depth", "n_parameters",
    "regularization_type", "regularization_strength",
    "training_steps", "collocation_budget", "certification_grid", "training_time_seconds",
    "E_true", "E_ref", "rho_tilde_h", "E_cert", "E_cert_U", "E_FB",
    "raw_effectivity", "rigorous_effectivity",
    "PDE_residual_RMS", "constitutive_residual_RMS", "complementarity_residual_RMS",
    "L_R", "quadrature_remainder",
    "final_loss", "best_loss",
    "status", "notes",
]

# Extra (non-spec, but required by NEW_RUNS.md's per-experiment sections)
# columns appended after MASTER_CSV_COLUMNS -- kept in one fixed order so
# every experiment's rows line up in the same shape.
EXTRA_CSV_COLUMNS = [
    "E_FB_gap_only", "E_FB_gap_and_multiplier",
    "eps_u", "eps_lambda", "alpha_u", "alpha_lambda",
    "rho_r", "rho_p_tilde", "rho_c",
    "E_h_lip", "E_h_unr", "E_h_dat", "E_h_0T",
    "V_h_unr", "V_h_dat",
    "n_adam_epochs", "n_lbfgs_epochs", "n_interior", "n_initial",
    "adam_lr", "lbfgs_lr",
    "weights_path", "best_weights_path", "config_path",
    "certificate_inequality_violated",
    "gamma",
    "exercise_boundary_error", "max_price_error", "L2_price_error",
]

ALL_CSV_COLUMNS = MASTER_CSV_COLUMNS + EXTRA_CSV_COLUMNS


# --------------------------------------------------------------------------
# Directory / status plumbing
# --------------------------------------------------------------------------

def new_run_dir(experiment: str, *parts: str) -> Path:
    """results/new_runs/<experiment>/<parts...>/ -- created idempotently
    (exist_ok=True), unlike results_io.make_run_dir's fresh-timestamp
    directories: NEW_RUNS.md's example layout uses fixed, meaningful names
    (strategy_uniform/seed_0/...) so a multi-hour sweep can be resumed after
    an interruption (NEW_RUNS.md's own allowance: rerun only for genuine
    hardware/interruption failures, which requires being able to tell what
    already finished)."""
    run_dir = NEW_RUNS_DIR / experiment / Path(*parts)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_status(run_dir: Path, status: str, notes: str = "") -> Path:
    assert status in ("ok", "failed"), f"status must be 'ok' or 'failed', got {status!r}"
    path = run_dir / "status.json"
    path.write_text(json.dumps({"status": status, "notes": notes}, indent=2))
    return path


def is_run_complete(run_dir: Path) -> bool:
    """True only if a PRIOR run finished with status=='ok'. A status=='failed'
    run dir must NOT be treated as complete -- NEW_RUNS.md distinguishes
    "rerun because of a genuine hardware/interruption failure" (allowed,
    expected to happen automatically on the next invocation of a resumable
    sweep) from "rerun because the numerical result looks bad" (forbidden).
    A crashed run is the former, not the latter."""
    status_path = run_dir / "status.json"
    summary_path = run_dir / "summary.csv"
    if not (status_path.exists() and summary_path.exists()):
        return False
    return json.loads(status_path.read_text())["status"] == "ok"


# --------------------------------------------------------------------------
# Residual RMS and per-layer spectral norms (not persisted anywhere upstream)
# --------------------------------------------------------------------------

def compute_residual_rms(triple, problem, t: torch.Tensor, x: torch.Tensor, x0: torch.Tensor) -> dict[str, float]:
    """sqrt(mean(residual**2)) over the given collocation points, for the
    three residuals NEW_RUNS.md names explicitly. Same pattern as
    experiments/stefan_stress_s2_residual_diagnostic.py's _residual_rms
    (R_PDE, R_sig), extended with R_com (residuals/residuals.py section 2:
    R_com = lambda_psi * gamma^-) for "complementarity_residual_RMS"."""
    u, sigma, lam, u_t, u_x, sigma_x = eval_triple_derivatives(triple, t, x)
    with torch.no_grad():
        u0_pred = triple.state_net(torch.zeros_like(x0), x0)
    res = assemble_residuals(problem, t, x, x0, u, sigma, lam, u_t, u_x, sigma_x, u0_pred)
    return {
        "PDE_residual_RMS": res.R_PDE.pow(2).mean().sqrt().item(),
        "constitutive_residual_RMS": res.R_sig.pow(2).mean().sqrt().item(),
        "complementarity_residual_RMS": res.R_com.pow(2).mean().sqrt().item(),
    }


def extract_spectral_norms(triple) -> dict[str, float]:
    """Per-layer ||W_k||_2 (certification/network_bounds.py's spectral_norms,
    already computed internally by compute_triple_bounds but never exposed)
    for each of the three sub-networks, plus the running product
    (Pi_N == product_of_spectral_norms, network_bounds.py's own Pi_j
    definition). `triple` may be a PhysicalUnitsTriple (American put) --
    unwrapped via .base_triple, the same attribute certify_cli.py's own
    stress-corruption path already relies on existing."""
    base = getattr(triple, "base_triple", triple)
    out: dict[str, float] = {}
    for name, net in (("state", base.state_net.net), ("flux", base.flux_net.net), ("multiplier", base.multiplier_net.net)):
        norms = spectral_norms(net)
        prod = 1.0
        for i, v in enumerate(norms, start=1):
            out[f"spectral_norm_{name}_layer_{i}"] = float(v)
            prod *= v
        out[f"product_of_spectral_norms_{name}"] = float(prod)
    return out


def compute_diagnostic_loss(
    triple, problem, *, n_interior: int = 5000, n_initial: int = 300, seed: int = 999_999,
    loss_weights: dict | None = None,
) -> float:
    """training/adaptive.py's run_adaptive_loop discards every round's loss
    history (it only returns the outcome, not the LossComponents) -- there is
    no literal "final training loss" available for an adaptive-method run the
    way a single trainer.train(csv_path=...) call gives one for uniform.
    Rather than inventing two different final_loss definitions per arm (exact
    in-sample last-batch loss for uniform, nothing for adaptive), this
    computes the SAME thing for every method: total loss (training/losses.py's
    compute_loss, default weights) on one FIXED, FRESH, out-of-sample draw
    (seed independent of any run's own training seed) -- comparable across
    every strategy and seed by construction, at the cost of not being
    bit-identical to any individual method's own in-sample training loss."""
    loss_weights = loss_weights or {}
    a, b = problem.omega_bounds
    generator = torch.Generator().manual_seed(seed)
    t = torch.rand(n_interior, 1, generator=generator, dtype=torch.float64) * problem.T
    x = torch.rand(n_interior, 1, generator=generator, dtype=torch.float64) * (b - a) + a
    x0 = torch.rand(n_initial, 1, generator=generator, dtype=torch.float64) * (b - a) + a

    u, sigma, lam, u_t, u_x, sigma_x = eval_triple_derivatives(triple, t, x)
    with torch.no_grad():
        u0_pred = triple.state_net(torch.zeros_like(x0), x0)
    res = assemble_residuals(problem, t, x, x0, u, sigma, lam, u_t, u_x, sigma_x, u0_pred)
    loss = compute_loss(res, **loss_weights)
    return float(loss.total.item())


def load_frozen_triple_from_run(config_path: Path, weights_path: Path):
    """Reloads a just-trained (or historical) run's frozen triple from disk,
    for residual-RMS / spectral-norm diagnostics -- reuses
    scripts/certify_cli.py's own load_frozen_triple/build_problem rather than
    re-implementing "reconstruct triple from config + load state_dict", so
    this can never silently drift from what certify_cli.py actually
    certifies."""
    config = json.loads(config_path.read_text())
    problem = build_problem(config["problem"])
    triple = load_frozen_triple(problem, config, weights_path)
    return problem, triple


# --------------------------------------------------------------------------
# Certification / true-error subprocess wrappers (generalizes
# experiments/stefan_core.py's _run_cli/_certify_one_run/_true_error_one_run
# to be problem-agnostic, so every new experiment shares one implementation)
# --------------------------------------------------------------------------

def _run_cli(module: str, args: list[str]) -> float:
    wall_start = time.perf_counter()
    result = subprocess.run([sys.executable, "-m", module] + args, cwd=REPO_ROOT, capture_output=True, text=True)
    wall_clock = time.perf_counter() - wall_start
    if result.returncode != 0:
        raise RuntimeError(f"{module} failed (exit {result.returncode}):\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}")
    return wall_clock


def certify_run(
    weights_path: Path, config_path: Path, out_dir: Path, *,
    grid: int, n_qp: int = 4, option_b_mode: str = "min_apriori_and_partition",
) -> tuple[float, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cert_path = out_dir / f"certificate_grid{grid}.json"
    if cert_path.exists():
        payload = json.loads(cert_path.read_text())
        return payload["wall_clock_seconds"], payload
    wall_clock = _run_cli(
        "certified_obstacle_nn.scripts.certify_cli",
        ["--weights", str(weights_path), "--config", str(config_path),
         "--grid", str(grid), "--n-qp", str(n_qp), "--option-b-mode", option_b_mode, "--out", str(out_dir)],
    )
    payload = json.loads(cert_path.read_text())
    return wall_clock, payload


def true_error_run(
    weights_path: Path, config_path: Path, out_dir: Path, *,
    n_t: int = 60, n_x: int = 60, fb_n_t: int = 400,
) -> tuple[float, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    te_path = out_dir / f"true_error_nt{n_t}_nx{n_x}.json"
    if te_path.exists():
        payload = json.loads(te_path.read_text())
        return payload["wall_clock_seconds"], payload
    wall_clock = _run_cli(
        "certified_obstacle_nn.scripts.true_error_cli",
        ["--weights", str(weights_path), "--config", str(config_path),
         "--n-t", str(n_t), "--n-x", str(n_x), "--fb-n-t", str(fb_n_t), "--out", str(out_dir)],
    )
    payload = json.loads(te_path.read_text())
    return wall_clock, payload


# --------------------------------------------------------------------------
# Sanity checks (NEW_RUNS.md "Sanity Checks Before Accepting Any Result")
# --------------------------------------------------------------------------

def check_certificate_inequality(E_true: float, E_cert_U: float, *, label: str = "", rtol: float = 1e-9) -> bool:
    """The guaranteed inequality bounds E_true by E_cert_U specifically (specification
    section 8) -- substituting the plain (non-U) energy-norm certificate here
    is explicitly forbidden by both specification and NEW_RUNS.md; this function
    must never be called with anything but E_cert_U as its second argument.
    A violation does not crash the sweep (it may
    be a genuine, reportable finding, and NEW_RUNS.md says failed/bad runs
    must be recorded, never deleted) but is always surfaced loudly via
    warnings.warn AND returned as a boolean the caller must store."""
    violated = E_true > E_cert_U * (1.0 + rtol)
    if violated:
        warnings.warn(
            f"CERTIFICATE INEQUALITY VIOLATED{' (' + label + ')' if label else ''}: "
            f"E_true={E_true!r} > E_cert_U={E_cert_U!r}. Flagged for review, run NOT discarded.",
            RuntimeWarning, stacklevel=2,
        )
    return violated


def check_no_nan_inf(row: dict[str, Any], *, label: str = "") -> None:
    """NEW_RUNS.md: 'Assert that final reported quantities contain no NaN/Inf
    ... unless a run is explicitly marked failed.' Hard-fails (AssertionError)
    for status=='ok' rows -- unlike the certificate-inequality check, a NaN in
    a non-failed run's reported quantity is a pipeline bug, not a scientific
    finding, so it must not be silently aggregated."""
    if row.get("status") != "ok":
        return
    bad = []
    for key, value in row.items():
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)) and (math.isnan(value) or math.isinf(value)):
            bad.append((key, value))
    assert not bad, f"NaN/Inf found in non-failed row{' (' + label + ')' if label else ''}: {bad}"


def check_matched_configs(configs: list[dict[str, Any]], keys: list[str], *, label: str = "") -> None:
    """For matched-budget comparisons (adaptive-vs-uniform strategies;
    certificate-aware gamma sweep): fail fast if any of `keys` differs across
    the given list of per-run config dicts. `configs` entries may be nested
    (e.g. config["optimizer_schedule"]["adam"]["lr"]) -- keys are given as
    dotted paths."""
    def _get(d: dict, dotted_key: str):
        cur: Any = d
        for part in dotted_key.split("."):
            cur = cur[part]
        return cur

    if not configs:
        return
    reference = configs[0]
    for key in keys:
        ref_val = _get(reference, key)
        for i, cfg in enumerate(configs[1:], start=1):
            val = _get(cfg, key)
            assert val == ref_val, (
                f"matched-comparison check failed{' (' + label + ')' if label else ''}: "
                f"key {key!r} differs (run 0 = {ref_val!r}, run {i} = {val!r}) -- "
                f"matched-budget comparisons must not vary anything except the compared factor"
            )


# --------------------------------------------------------------------------
# Master-row builder + CSV aggregator
# --------------------------------------------------------------------------

def build_master_row(
    *,
    experiment: str,
    subexperiment: str | None = None,
    strategy: str | None = None,
    seed: int,
    architecture_tag: str,
    width: int,
    depth: int,
    n_parameters: int,
    regularization_type: str | None = None,
    regularization_strength: float | None = None,
    gamma: float | None = None,
    n_adam_epochs: int,
    n_lbfgs_epochs: int,
    n_interior: int,
    n_initial: int,
    adam_lr: float,
    lbfgs_lr: float,
    certification_grid: int | None,
    training_time_seconds: float,
    final_loss: float,
    best_loss: float | None = None,
    E_ref: float | None = None,
    cert_payload: dict | None = None,
    te_payload: dict | None = None,
    residual_rms: dict[str, float] | None = None,
    weights_path: str | None = None,
    best_weights_path: str | None = None,
    config_path: str | None = None,
    status: str = "ok",
    notes: str = "",
) -> dict[str, Any]:
    """Flattens (train_info, cert_payload, te_payload, residual_rms) into one
    row matching ALL_CSV_COLUMNS. Generalizes
    experiments/stefan_core.py's _build_row to be problem/experiment-agnostic
    (extra keyword args for strategy/regularization instead of hardcoded
    architecture-study columns). best_loss defaults to final_loss: the
    existing codebase has no intermediate/early-stopping checkpoint
    selection anywhere (training/trainer.py's train() only ever returns the
    state after the full Adam+L-BFGS schedule completes), so "best checkpoint
    per the same criterion as existing code" is, truthfully, the same
    checkpoint as "final" -- not a new selection rule invented for this task."""
    row: dict[str, Any] = {col: None for col in ALL_CSV_COLUMNS}
    row.update({
        "experiment": experiment,
        "subexperiment": subexperiment,
        "strategy": strategy,
        "seed": seed,
        "architecture": architecture_tag,
        "width": width,
        "depth": depth,
        "n_parameters": n_parameters,
        "regularization_type": regularization_type,
        "regularization_strength": regularization_strength,
        "gamma": gamma,
        "training_steps": n_adam_epochs + n_lbfgs_epochs,
        "collocation_budget": n_interior + n_initial,
        "certification_grid": certification_grid,
        "training_time_seconds": training_time_seconds,
        "final_loss": final_loss,
        "best_loss": best_loss if best_loss is not None else final_loss,
        "n_adam_epochs": n_adam_epochs,
        "n_lbfgs_epochs": n_lbfgs_epochs,
        "n_interior": n_interior,
        "n_initial": n_initial,
        "adam_lr": adam_lr,
        "lbfgs_lr": lbfgs_lr,
        "E_ref": E_ref,
        "weights_path": weights_path,
        "best_weights_path": best_weights_path if best_weights_path is not None else weights_path,
        "config_path": config_path,
        "status": status,
        "notes": notes,
    })

    if residual_rms:
        row.update(residual_rms)

    if cert_payload is not None:
        r = cert_payload["result"]
        b = r["bounds"]
        row.update({
            "rho_tilde_h": r["rho_tilde_h"],
            "E_cert": r["E_cert_new"],
            "E_cert_U": r["E_cert_U_new"],
            "quadrature_remainder": r["E_h_new"],
            "L_R": b["lip_R_PDE"],
            "rho_r": r["rho_r"],
            "rho_p_tilde": r["rho_p_tilde"],
            "rho_c": r["rho_c"],
            "E_h_lip": r["E_h_lip_new"],
            "E_h_unr": r["E_h_unr"],
            "E_h_dat": r["E_h_dat"],
            "E_h_0T": r["E_h_0T"],
            "V_h_unr": r["V_h_unr"],
            "V_h_dat": r["V_h_dat"],
        })

    if te_payload is not None:
        te = te_payload["result"]
        E_true = te["E_true"]
        row["E_true"] = E_true
        fb = te_payload.get("free_boundary_error")
        if fb is not None:
            row.update({
                "E_FB_gap_only": fb["E_FB_gap_only"],
                "E_FB_gap_and_multiplier": fb["E_FB_gap_and_multiplier"],
                "E_FB": fb["E_FB_gap_and_multiplier"],
                "eps_u": fb["eps_u"],
                "eps_lambda": fb["eps_lambda"],
                "alpha_u": fb["alpha_u"],
                "alpha_lambda": fb["alpha_lambda"],
            })
        if cert_payload is not None and E_true not in (None, 0) and not (isinstance(E_true, float) and math.isnan(E_true)):
            rho_tilde_h = row["rho_tilde_h"]
            E_h_new = cert_payload["result"]["E_h_new"]
            row["raw_effectivity"] = rho_tilde_h / E_true
            row["rigorous_effectivity"] = math.sqrt(rho_tilde_h ** 2 + E_h_new) / E_true

    if row["E_true"] is not None and row["E_cert_U"] is not None:
        row["certificate_inequality_violated"] = check_certificate_inequality(
            row["E_true"], row["E_cert_U"], label=f"{experiment}/{subexperiment}/{strategy}/seed{seed}",
        )
    else:
        row["certificate_inequality_violated"] = None

    return row


def write_row_csv(run_dir: Path, row: dict[str, Any]) -> Path:
    """Per-run summary.csv -- results_io.write_summary's exact convention
    (one row, no rounding), just pointed at a dict built by build_master_row
    instead of an experiment-specific summary dict."""
    path = run_dir / "summary.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    return path


def aggregate_master_csv(rows: list[dict[str, Any]], out_path: Path = MASTER_CSV_PATH) -> pd.DataFrame:
    """Writes new_runs_master.csv with ALL_CSV_COLUMNS in a fixed order,
    filling any row's missing keys with NaN -- valid (correct header, zero
    data rows) even when `rows` is empty, which is what Phase 1 ships before
    any real experiment has run."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        df = pd.DataFrame(columns=ALL_CSV_COLUMNS)
    else:
        df = pd.DataFrame.from_records(rows)
        for col in ALL_CSV_COLUMNS:
            if col not in df.columns:
                df[col] = None
        extra_cols = [c for c in df.columns if c not in ALL_CSV_COLUMNS]
        df = df[ALL_CSV_COLUMNS + extra_cols]
    df.to_csv(out_path, index=False)
    return df


# --------------------------------------------------------------------------
# Figure helper
# --------------------------------------------------------------------------

def save_figure(fig, path_stem: Path) -> tuple[Path, Path]:
    """Every NEW_RUNS.md figure must exist as both PDF and PNG."""
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = path_stem.with_suffix(".pdf")
    png_path = path_stem.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=150)
    return pdf_path, png_path
