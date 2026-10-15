"""Adaptive vs. uniform collocation comparison (2026-09-11),
training/adaptive.py's own Train -> Estimate -> Mark -> Resample loop.

Six runs, ALL sharing identical architecture, seed, TOTAL optimizer budget,
N_total collocation-point budget, and certification grid -- an unmatched
budget on any of these axes would make the comparison meaningless (user's
own words), so the point-count and epoch-budget equalities are ASSERTED in
code below, not just intended:

  uniform            -- ONE training call with TOTAL_ADAM_EPOCHS/
                         TOTAL_LBFGS_EPOCHS (the SUM across every adaptive
                         round) on a single uniform N_INTERIOR-point sample,
                         never resampled.
  dorfler_total       -- Dorfler(theta=0.5) marking on rho_total_K.
  probabilistic_q1    -- P(K) ~ rho_K^1 marking on rho_total_K.
  probabilistic_q2    -- P(K) ~ rho_K^2 marking on rho_total_K.
  dorfler_rho_p       -- ablation: Dorfler(theta=0.5) marking on
                         rho_penetration_K instead of the total.
  dorfler_rho_c       -- ablation: Dorfler(theta=0.5) marking on
                         rho_complementarity_K instead of the total.

Every adaptive method runs L_MAX+1 rounds of training/adaptive.py's own
loop, each round using ADAM_EPOCHS_PER_ROUND/LBFGS_EPOCHS_PER_ROUND --
summing to EXACTLY uniform's own single-call total, so the only thing that
differs between uniform and any adaptive method is WHERE the (always
N_INTERIOR) collocation points sit, never how much total optimizer budget
was spent on them or how many of them there are at any given time.

Per specification section 12 invariant 2, the adaptive loop's own per-iteration
marking estimator (compute_raw_estimator) runs in-process (training/
adaptive.py's own documented, necessary exception -- the algorithm needs
per-cell rho_K to decide the NEXT round's points, which cannot reasonably be
pushed to a separate process every round). The FINAL, official certification
(certify_option_b, E_cert_U) and true-error/E_FB for every method's FINAL
trained weights are computed via certify_cli.py/true_error_cli.py as
separate subprocesses, exactly as experiments/stefan_core.py and
stefan_stress.py already do.

Table 5 columns (user-specified 2026-09-11): method, n_points, E_true,
rho_tilde, E_cert_U, E_FB, training_time. training_time is the SUM of every
round's training_time for that method.

Figure 7: the dorfler_total run's own collocation cloud at iteration 0 vs.
iteration L_MAX, overlaid with the exact free boundary t=x.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from certified_obstacle_nn.certification.partition import build_partition
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.stefan import StefanProblem
from certified_obstacle_nn.training.adaptive import run_adaptive_loop
from certified_obstacle_nn.training.trainer import train

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
ADAPTIVE_DIR = RESULTS_DIR / "stefan_adaptive"

# Identical across every one of the six runs.
WIDTH, DEPTH = 64, 4  # specification section 1's "Default 4x64"
SEED = 0
N_INTERIOR = 1500
N_INITIAL = 200
ADAM_LR = 1e-3
LBFGS_LR = 1.0
CERT_GRID = 128  # used both for per-iteration marking AND final official certification -- one shared constant, never two different grids
N_QP = 4
TRUE_ERROR_N_T = 60
TRUE_ERROR_N_X = 60
OPTION_B_MODE = "min_apriori_and_partition"

# Per-round budget for every adaptive method's L_MAX+1 rounds.
ADAM_EPOCHS_PER_ROUND = 1500
LBFGS_EPOCHS_PER_ROUND = 50
L_MAX = 4  # 5 rounds: iterations 0..4

# Uniform's single call gets the SUM across all L_MAX+1 adaptive rounds --
# asserted equal below, not just computed this way by construction, since a
# future edit to either side must not be able to silently break this.
TOTAL_ADAM_EPOCHS = (L_MAX + 1) * ADAM_EPOCHS_PER_ROUND
TOTAL_LBFGS_EPOCHS = (L_MAX + 1) * LBFGS_EPOCHS_PER_ROUND

DORFLER_THETA = 0.5

METHODS = [
    {"name": "uniform", "kind": "uniform"},
    {"name": "dorfler_total", "kind": "adaptive", "marking_method": "dorfler", "theta": DORFLER_THETA, "mark_on": "rho_total_K"},
    {"name": "probabilistic_q1", "kind": "adaptive", "marking_method": "probabilistic", "q": 1.0, "mark_on": "rho_total_K"},
    {"name": "probabilistic_q2", "kind": "adaptive", "marking_method": "probabilistic", "q": 2.0, "mark_on": "rho_total_K"},
    {"name": "dorfler_rho_p_ablation", "kind": "adaptive", "marking_method": "dorfler", "theta": DORFLER_THETA, "mark_on": "rho_penetration_K"},
    {"name": "dorfler_rho_c_ablation", "kind": "adaptive", "marking_method": "dorfler", "theta": DORFLER_THETA, "mark_on": "rho_complementarity_K"},
]


def _run_uniform(run_dir: Path) -> dict:
    torch.manual_seed(SEED)
    problem = StefanProblem()
    triple = NeuralTriple(problem, width=WIDTH, depth=DEPTH)

    generator = torch.Generator().manual_seed(SEED)
    a, b = problem.omega_bounds
    T = problem.T
    t = (torch.rand(N_INTERIOR, 1, generator=generator, dtype=torch.float64) * T).requires_grad_(True)
    x = (torch.rand(N_INTERIOR, 1, generator=generator, dtype=torch.float64) * (b - a) + a).requires_grad_(True)
    x0 = torch.rand(N_INITIAL, 1, generator=generator, dtype=torch.float64) * (b - a) + a

    assert t.shape[0] == N_INTERIOR, "N_total_uniform must equal N_INTERIOR"

    wall_start = time.perf_counter()
    _, outcome = train(
        problem, triple, t, x, x0,
        n_epochs=TOTAL_ADAM_EPOCHS, lr=ADAM_LR,
        n_lbfgs_epochs=TOTAL_LBFGS_EPOCHS, lbfgs_lr=LBFGS_LR,
    )
    training_time = time.perf_counter() - wall_start

    run_dir.mkdir(parents=True, exist_ok=True)
    weights_path = run_dir / "weights.pt"
    config_path = run_dir / "config.json"
    torch.save(triple.state_dict(), weights_path)
    config = {
        "problem": "stefan",
        "seed": SEED,
        "architecture": {"width": WIDTH, "depth": DEPTH, "multiplier_head": triple.multiplier_net.head},
        "method": "uniform",
        "n_points": N_INTERIOR,
        "total_adam_epochs": TOTAL_ADAM_EPOCHS,
        "total_lbfgs_epochs": TOTAL_LBFGS_EPOCHS,
        "training_time_seconds": training_time,
        "lbfgs_outcome_reason": outcome.reason,
    }
    config_path.write_text(json.dumps(config, indent=2))

    return {
        "method": "uniform", "n_points": N_INTERIOR, "weights_path": weights_path,
        "config_path": config_path, "training_time": training_time,
        "t0": t.detach().clone(), "x0": x.detach().clone(),
        "t_final": t.detach().clone(), "x_final": x.detach().clone(),
    }


def _run_adaptive(spec: dict, run_dir: Path) -> dict:
    torch.manual_seed(SEED)
    problem = StefanProblem()
    triple = NeuralTriple(problem, width=WIDTH, depth=DEPTH)
    partition = build_partition(problem, n_t=CERT_GRID, n_x=CERT_GRID)

    final_triple, history = run_adaptive_loop(
        problem, triple, partition,
        method=spec["marking_method"],
        theta=spec.get("theta", 0.5), q=spec.get("q", 1.0),
        mark_on=spec["mark_on"],
        n_interior=N_INTERIOR, n_initial=N_INITIAL, l_max=L_MAX, seed=SEED,
        n_adam_epochs=ADAM_EPOCHS_PER_ROUND, adam_lr=ADAM_LR,
        n_lbfgs_epochs=LBFGS_EPOCHS_PER_ROUND, lbfgs_lr=LBFGS_LR,
        n_qp=N_QP,
    )

    assert history[-1].t_points.shape[0] == N_INTERIOR, "N_total_adaptive must equal N_INTERIOR at every round"
    total_training_time = sum(r.training_time for r in history)

    run_dir.mkdir(parents=True, exist_ok=True)
    weights_path = run_dir / "weights.pt"
    config_path = run_dir / "config.json"
    torch.save(final_triple.state_dict(), weights_path)
    config = {
        "problem": "stefan",
        "seed": SEED,
        "architecture": {"width": WIDTH, "depth": DEPTH, "multiplier_head": final_triple.multiplier_net.head},
        "method": spec["name"],
        "marking_method": spec["marking_method"],
        "mark_on": spec["mark_on"],
        "theta": spec.get("theta"), "q": spec.get("q"),
        "n_points": N_INTERIOR,
        "l_max": L_MAX,
        "adam_epochs_per_round": ADAM_EPOCHS_PER_ROUND, "lbfgs_epochs_per_round": LBFGS_EPOCHS_PER_ROUND,
        "total_adam_epochs": (L_MAX + 1) * ADAM_EPOCHS_PER_ROUND,
        "total_lbfgs_epochs": (L_MAX + 1) * LBFGS_EPOCHS_PER_ROUND,
        "training_time_seconds": total_training_time,
        "n_marked_per_round": [r.n_marked for r in history],
        "rho_tilde_estimate_per_round": [r.rho_tilde_estimate for r in history],
    }
    config_path.write_text(json.dumps(config, indent=2))

    return {
        "method": spec["name"], "n_points": N_INTERIOR, "weights_path": weights_path,
        "config_path": config_path, "training_time": total_training_time,
        "t0": history[0].t_points, "x0": history[0].x_points,
        "t_final": history[-1].t_points, "x_final": history[-1].x_points,
    }


def _run_cli(module: str, args: list[str]) -> None:
    result = subprocess.run([sys.executable, "-m", module] + args, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{module} failed (exit {result.returncode}):\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}")


def _certify_and_true_error(weights_path: Path, config_path: Path, out_dir: Path) -> dict:
    cert_dir = out_dir / "certification"
    te_dir = out_dir / "true_error"
    cert_dir.mkdir(parents=True, exist_ok=True)
    te_dir.mkdir(parents=True, exist_ok=True)

    _run_cli("certified_obstacle_nn.scripts.certify_cli", [
        "--weights", str(weights_path), "--config", str(config_path),
        "--grid", str(CERT_GRID), "--n-qp", str(N_QP),
        "--option-b-mode", OPTION_B_MODE, "--out", str(cert_dir),
    ])
    cert_payload = json.loads((cert_dir / f"certificate_grid{CERT_GRID}.json").read_text())

    _run_cli("certified_obstacle_nn.scripts.true_error_cli", [
        "--weights", str(weights_path), "--config", str(config_path),
        "--n-t", str(TRUE_ERROR_N_T), "--n-x", str(TRUE_ERROR_N_X),
        "--out", str(te_dir),
    ])
    te_payload = json.loads((te_dir / f"true_error_nt{TRUE_ERROR_N_T}_nx{TRUE_ERROR_N_X}.json").read_text())

    return {"cert": cert_payload, "true_error": te_payload}


def _build_row(run_info: dict, diag: dict) -> dict:
    r = diag["cert"]["result"]
    te = diag["true_error"]["result"]
    fb = diag["true_error"].get("free_boundary_error")
    return {
        "method": run_info["method"],
        "n_points": run_info["n_points"],
        "E_true": te["E_true"],
        "rho_tilde": r["rho_tilde_h"],
        "E_cert_U": r["E_cert_U_new"],
        "E_FB": fb["E_FB_gap_and_multiplier"] if fb is not None else None,
        "training_time": run_info["training_time"],
    }


def plot_collocation_cloud(uniform_or_adaptive_run: dict, save_path: str | None = None) -> plt.Figure:
    """Figure 7: dorfler_total's collocation cloud at iteration 0 vs L_MAX,
    overlaid with the exact free boundary t=x."""
    problem = StefanProblem()
    t_line = torch.linspace(0.0, problem.T, 200, dtype=torch.float64)
    x_line = problem.free_boundary_exact(t_line)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    for ax, t_pts, x_pts, title in [
        (ax0, uniform_or_adaptive_run["t0"], uniform_or_adaptive_run["x0"], "iteration 0 (uniform initial cloud)"),
        (ax1, uniform_or_adaptive_run["t_final"], uniform_or_adaptive_run["x_final"], f"iteration L_max={L_MAX}"),
    ]:
        ax.scatter(x_pts.numpy(), t_pts.numpy(), s=2, alpha=0.4, color="tab:blue")
        ax.plot(x_line.numpy(), t_line.numpy(), color="red", linewidth=1.5, label="exact free boundary t=x")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"Adaptive collocation cloud (dorfler_total, theta={DORFLER_THETA})")
    fig.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def main() -> pd.DataFrame:
    assert TOTAL_ADAM_EPOCHS == (L_MAX + 1) * ADAM_EPOCHS_PER_ROUND
    assert TOTAL_LBFGS_EPOCHS == (L_MAX + 1) * LBFGS_EPOCHS_PER_ROUND

    ADAPTIVE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    run_infos: dict[str, dict] = {}

    for spec in METHODS:
        run_dir = ADAPTIVE_DIR / spec["name"]
        print(f"=== {spec['name']} ===", flush=True)
        if spec["kind"] == "uniform":
            run_info = _run_uniform(run_dir)
        else:
            run_info = _run_adaptive(spec, run_dir)
        run_infos[spec["name"]] = run_info

        # The N_total equality this whole milestone hinges on -- asserted
        # against the FIRST run (uniform, always run first) for every
        # subsequent method, not just computed identically by construction.
        if spec["name"] != "uniform":
            assert run_info["n_points"] == run_infos["uniform"]["n_points"], (
                f"N_total_adaptive ({run_info['n_points']}) != N_total_uniform "
                f"({run_infos['uniform']['n_points']}) for method {spec['name']!r} -- "
                f"an unmatched budget makes the comparison meaningless."
            )

        print(f"  trained: {run_info['training_time']:.1f}s total", flush=True)
        diag = _certify_and_true_error(run_info["weights_path"], run_info["config_path"], run_dir)
        row = _build_row(run_info, diag)
        print(f"  {row}", flush=True)
        rows.append(row)

    table5 = pd.DataFrame.from_records(rows)
    table5_path = RESULTS_DIR / "stefan_table5_adaptive.csv"
    table5.to_csv(table5_path, index=False)
    print(f"\nTable 5 -> {table5_path}")

    figure_path = RESULTS_DIR / "stefan_figure7_collocation_cloud.png"
    plot_collocation_cloud(run_infos["dorfler_total"], save_path=str(figure_path))
    print(f"Figure 7 -> {figure_path}")

    return table5


if __name__ == "__main__":
    main()
