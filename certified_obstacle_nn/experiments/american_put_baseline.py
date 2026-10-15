"""American put baseline training + certification (2026-09-12), specification
section 10 (Phase 4). First real Phase-4 run: demonstrates the normalized-
training / physical-units-at-certification pipeline end to end, and the two
ways certification differs from Stefan (large C_eq since alpha=0.08; a
nonempty Gamma_data at x=log(K) giving E_h_dat != 0, reported separately).

Per this milestone's own instruction, verify_boundary_and_initial_conditions
runs BEFORE training starts -- a numerical, not just algebraic, pre-flight
check (problems/american_put.py).

Training uses NormalizedAmericanPutProblem (g/f/u0 divided by K) so the loss
is computed against ubar/gbar/fbar; the triple's own state_net/flux_net/
multiplier_net are built with x_rescale=problem.xi (specification section 10's
xi=2(x-L)/(R-L)-1, chain-rule-correct per tests/test_x_rescale_chain_rule.py).
The trained (normalized-output) triple is saved as-is; certify_cli.py/
true_error_cli.py's shared load_frozen_triple (scripts/certify_cli.py, 2026-
09-12) detects problem.xi/problem.K generically and wraps the loaded triple
in PhysicalUnitsTriple before handing it to certification -- so the config
saved here must record "problem": "american_put" and nothing else needs to
know about normalization at certification time.

Per specification section 12 invariant 2, training happens in-process; the
certificate is computed via certify_cli.py as a separate subprocess, exactly
as experiments/stefan_core.py and stefan_stress.py already do.

Run-artifact layout (2026-09-12, user-specified; experiments/results_io.py):
this is the FIRST experiment wired to the uniform
results/<experiment>/<timestamp>/{config.json, losses.csv, cells.csv,
summary.csv, weights.pt} layout -- a fresh timestamped directory every run,
never overwritten, so a later run can never silently invalidate an earlier
one's saved numbers. certify_cli.py/true_error_cli.py's own detailed JSON
payloads (certificate_grid*.json, true_error_nt*_nx*.json) are ALSO written
into this same directory for provenance, but summary.csv is the one file any
plot may read from -- specification section 12 invariant 5's separately-reported
components (rho_r/rho_p_tilde/rho_c, the four E_h parts) are written as
separate summary.csv columns, never pre-merged.
"""

import json
import subprocess
import sys
import time

import torch

from certified_obstacle_nn.experiments.results_io import REPO_ROOT, losses_csv_path, make_run_dir, write_config, write_summary
from certified_obstacle_nn.models.neural_triple import NeuralTriple
from certified_obstacle_nn.problems.american_put import AmericanPutProblem, NormalizedAmericanPutProblem, verify_boundary_and_initial_conditions
from certified_obstacle_nn.training.sampler import TrainingSampler
from certified_obstacle_nn.training.trainer import train

WIDTH, DEPTH = 64, 4  # specification section 1's "Default 4x64"
SEED = 0
N_INTERIOR = 5000
N_INITIAL = 300
N_ADAM_EPOCHS = 5000
ADAM_LR = 1e-3
N_LBFGS_EPOCHS = 300
LBFGS_LR = 1.0

CERT_GRID = 256  # first exploratory Phase-4 run -- moderate grid, not M17's 1024^2
OPTION_B_MODE = "min_apriori_and_partition"
N_QP = 4

# E_true quadrature order (2026-09-12): the frozen reference is only a
# piecewise interpolant of a finite (6400x6400) grid, not a smooth closed
# form, so it needs more points than Stefan's smooth exact solution to clear
# the same self-check tolerance -- measured 2026-09-12 against this exact
# baseline: n_t=n_x=320 cleared 1e-3 with a ~2x margin (rel_change=4.8e-4),
# after fixing two distinct root causes (errors/true_error.py's
# vertical_interface_x split at x=log(K), and switching
# reference/american_put_reference.py's interpolant from bilinear to a C^1
# bicubic Catmull-Rom spline) -- a resolution choice, not a loosened
# tolerance.
TRUE_ERROR_N = 320
TRUE_ERROR_TOLERANCE = 1e-3


def _run_cli(module: str, args: list[str]) -> None:
    result = subprocess.run([sys.executable, "-m", module] + args, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{module} failed (exit {result.returncode}):\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}")


def main() -> dict:
    problem = AmericanPutProblem()
    verify_boundary_and_initial_conditions(problem)
    print("verify_boundary_and_initial_conditions: PASSED (g(t,L)=g(t,R)=0, u0=g(0,.) hold numerically)")

    normalized_problem = NormalizedAmericanPutProblem()

    torch.manual_seed(SEED)
    triple = NeuralTriple(problem, width=WIDTH, depth=DEPTH, x_rescale=problem.xi)

    sampler = TrainingSampler(normalized_problem, n_interior=N_INTERIOR, n_initial=N_INITIAL, seed=SEED)
    t, x, x0 = sampler.sample()

    run_dir = make_run_dir("american_put_baseline")
    weights_path = run_dir / "weights.pt"
    config_path = run_dir / "config.json"

    wall_start = time.perf_counter()
    history, outcome = train(
        normalized_problem, triple, t, x, x0,
        n_epochs=N_ADAM_EPOCHS, lr=ADAM_LR,
        n_lbfgs_epochs=N_LBFGS_EPOCHS, lbfgs_lr=LBFGS_LR,
        csv_path=str(losses_csv_path(run_dir)),
    )
    training_time = time.perf_counter() - wall_start
    final_loss = float(history["loss_total"].iloc[-1])
    print(f"trained: {training_time:.1f}s  final_loss={final_loss:.4e}  lbfgs={outcome.reason}")

    torch.save(triple.state_dict(), weights_path)
    config = {
        "problem": "american_put",
        "seed": SEED,
        "architecture": {"width": WIDTH, "depth": DEPTH, "multiplier_head": triple.multiplier_net.head},
        "optimizer_schedule": {
            "adam": {"lr": ADAM_LR, "n_epochs": N_ADAM_EPOCHS},
            "lbfgs": {"lr": LBFGS_LR, "n_epochs_requested": N_LBFGS_EPOCHS, "outcome_reason": outcome.reason, "n_steps_completed": outcome.n_steps_completed},
        },
        "N_train": {"n_interior": N_INTERIOR, "n_initial": N_INITIAL},
        "grid": CERT_GRID,
        "eps_u": None,  # E_FB (errors/free_boundary.py) is not computed for American put --
        "eps_lambda": None,  # its free boundary is not known in closed form (getattr(problem, "free_boundary_exact", None) is None).
        "training_time_seconds": training_time,
        "final_loss_total": final_loss,
        "trained_on": "NormalizedAmericanPutProblem (ubar/gbar/fbar)",
    }
    write_config(run_dir, config)

    _run_cli("certified_obstacle_nn.scripts.certify_cli", [
        "--weights", str(weights_path), "--config", str(config_path),
        "--grid", str(CERT_GRID), "--n-qp", str(N_QP),
        "--option-b-mode", OPTION_B_MODE, "--save-cells", "--out", str(run_dir),
    ])
    cert_payload = json.loads((run_dir / f"certificate_grid{CERT_GRID}.json").read_text())
    r = cert_payload["result"]

    print(f"\nC_eq = {r['C_eq']:.4f}  (Stefan's own C_eq = 2 -- American put's alpha=0.08 makes 1/alpha dominate)")
    print(f"V_h_dat = {r['V_h_dat']:.6e}  (nonzero: Gamma_data={{log(K)}} is hit by the partition)")
    print(f"M_Psi = {r['bounds']['M_Psi']:.6e}")
    print(f"E_h_dat = {r['E_h_dat']:.6e}  (= M_Psi * V_h_dat = {r['bounds']['M_Psi'] * r['V_h_dat']:.6e}, reported SEPARATELY from E_h_unr/E_h_lip/E_h_0T)")
    print(f"E_h_new (total) = {r['E_h_new']:.6e}")
    print(f"E_cert_U_new = {r['E_cert_U_new']:.6e}")

    # E_true against the FROZEN reference/american_put_reference.py solution
    # (specification section 12 invariant 4) -- never against the network itself.
    # true_error_cli.py detects American put generically via
    # getattr(problem, "x_star", None) and loads the frozen reference from
    # its default --reference-dir; a config-hash mismatch there raises loudly
    # rather than silently comparing against a different resolution.
    _run_cli("certified_obstacle_nn.scripts.true_error_cli", [
        "--weights", str(weights_path), "--config", str(config_path),
        "--n-t", str(TRUE_ERROR_N), "--n-x", str(TRUE_ERROR_N),
        "--quadrature-tolerance", str(TRUE_ERROR_TOLERANCE), "--out", str(run_dir),
    ])
    te_payload = json.loads((run_dir / f"true_error_nt{TRUE_ERROR_N}_nx{TRUE_ERROR_N}.json").read_text())
    te = te_payload["result"]
    E_true = te["E_true"]

    print(f"\nE_true = {E_true:.6e}  (quadrature_relative_change={te['quadrature_relative_change']:.3e}, reference_config_hash={te_payload['reference_config_hash']})")
    print(f"E_true <= E_cert_U: {E_true} <= {r['E_cert_U_new']} -> {E_true <= r['E_cert_U_new']}")
    assert E_true <= r["E_cert_U_new"], "M16 (E_true <= E_cert_U) violated -- do not report this run"

    # Every quantity specification's Tables 1-6 would need from this run, as
    # separate columns (invariant 5: never merge rho_r/rho_p_tilde/rho_c or
    # the four E_h parts into one number) -- this is the ONLY file a plot may
    # read from (module docstring / experiments/results_io.py).
    summary = {
        "seed": SEED, "architecture_tag": f"{WIDTH}x{DEPTH}",
        "training_time_seconds": training_time, "final_loss_total": final_loss,
        "rho_r": r["rho_r"], "rho_p_tilde": r["rho_p_tilde"], "rho_c": r["rho_c"],
        "rho_tilde_h": r["rho_tilde_h"],
        "V_h_unr": r["V_h_unr"], "V_h_dat": r["V_h_dat"],
        "E_h_lip": r["E_h_lip_new"], "E_h_unr": r["E_h_unr"], "E_h_dat": r["E_h_dat"], "E_h_0T": r["E_h_0T"],
        "E_h": r["E_h_new"],
        "C_rel": r["C_rel"], "C_eq": r["C_eq"],
        "E_cert": r["E_cert_new"], "E_cert_U": r["E_cert_U_new"],
        "cert_grid": CERT_GRID,
        "E_true": E_true, "E_true_n": TRUE_ERROR_N,
        "E_true_quadrature_relative_change": te["quadrature_relative_change"],
        "reference_config_hash": te_payload["reference_config_hash"],
    }
    write_summary(run_dir, summary)
    print(f"\nrun_dir -> {run_dir}")
    return summary


if __name__ == "__main__":
    main()
