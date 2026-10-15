"""Stage 0: verify the manufactured 2D expanding-disk solution before any neural training.

Runs the checks required by the 2D benchmark specification section 5, using
>= 1e6 scrambled-Sobol points in Q = (0,1)^3 (see NOTES_existing_code.md finding 1: Sobol is
new to this codebase, but the prompt is explicit and authoritative on this point for Stage 0).

Fails loudly: any check outside its stated tolerance causes this script to raise SystemExit(1)
after printing the full summary table (never a silent partial failure), per ground rule 5
("verification scripts must raise / exit non-zero when a check fails").

float64 throughout; torch/numpy/random all seeded (ground rule 4).
"""

import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from certified_obstacle_nn.experiments import results_io  # noqa: E402
from certified_obstacle_nn.experiments.expanding_disk_2d import exact_solution as ex  # noqa: E402

SEED = 0
DTYPE = torch.float64

N_MAIN = 1_000_000
N_AUTOGRAD = 20_000
N_BOUNDARY_PER_SIDE = 100_000
N_CONTINUITY_PAIRS = 20_000

PSI_EXCLUSION = 1e-6
CONTINUITY_HALF_DISTANCE = 5e-8  # total straddle distance = 1e-7

TOL_OBSTACLE = -1e-14
TOL_LAMBDA_NONNEG = 0.0
TOL_COMPLEMENTARITY = 1e-14
TOL_PDE_EXPLICIT = 1e-10
TOL_PDE_AUTOGRAD = 1e-8
TOL_DERIV_VS_AUTOGRAD = 1e-10
TOL_CONTINUITY_F = 1e-5
TOL_BOUNDARY = 1e-14
TOL_INITIAL = 1e-14


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _sobol_points(n: int, dim: int, seed: int) -> torch.Tensor:
    engine = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=seed)
    return engine.draw(n).to(DTYPE)


class CheckResult:
    def __init__(self, name: str, value: float, tolerance: float, passed: bool, note: str = ""):
        self.name = name
        self.value = value
        self.tolerance = tolerance
        self.passed = passed
        self.note = note

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "note": self.note,
        }


def _autograd_u_derivatives(t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, kappa: float):
    """u, u_t, u_x, u_y, Delta u -- purely via torch.autograd of ex.u_exact. Independent of
    every explicit-formula function in exact_solution.py (ground rule 4: autograd is used
    only here, as the verification script's cross-check, never in exact_solution.py itself)."""
    t = t.clone().requires_grad_(True)
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    u = ex.u_exact(t, x, y, kappa)
    ones = torch.ones_like(u)
    u_t, u_x, u_y = torch.autograd.grad(u, (t, x, y), grad_outputs=ones, create_graph=True)
    (u_xx,) = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True)
    (u_yy,) = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), retain_graph=True)
    lap_u = u_xx + u_yy
    return u.detach(), u_t.detach(), u_x.detach(), u_y.detach(), lap_u.detach()


def run_checks(kappa: float = ex.KAPPA_DEFAULT) -> tuple[list[CheckResult], dict]:
    _seed_everything(SEED)
    checks: list[CheckResult] = []
    extras: dict = {}

    # ---- main 1e6-point Sobol cloud over Q = (0,1)^3 ----
    main = _sobol_points(N_MAIN, 3, SEED)
    t, x, y = main[:, 0:1], main[:, 1:2], main[:, 2:3]

    with torch.no_grad():
        u = ex.u_exact(t, x, y, kappa)
        g = ex.g(t, x, y)
        lam = ex.lambda_exact(t, x, y, kappa)
        psi_val = ex.psi(t, x, y)

        # 1. min(u - g) >= -1e-14
        min_u_minus_g = (u - g).min().item()
        checks.append(CheckResult("min(u - g)", min_u_minus_g, TOL_OBSTACLE, min_u_minus_g >= TOL_OBSTACLE))

        # 2. min(lambda) >= 0
        min_lambda = lam.min().item()
        checks.append(CheckResult("min(lambda)", min_lambda, TOL_LAMBDA_NONNEG, min_lambda >= TOL_LAMBDA_NONNEG))

        # 3. max|lambda * (u - g)| <= 1e-14
        max_comp = (lam * (u - g)).abs().max().item()
        checks.append(CheckResult("max|lambda*(u-g)|", max_comp, TOL_COMPLEMENTARITY, max_comp <= TOL_COMPLEMENTARITY))

        # 4. PDE residual, explicit formulas, away from the interface
        away = (psi_val.abs() > PSI_EXCLUSION).squeeze(-1)
        t_a, x_a, y_a = t[away], x[away], y[away]
        u_t_e = ex.u_t_exact(t_a, x_a, y_a, kappa)
        lap_u_e = ex.lap_u_exact(t_a, x_a, y_a, kappa)
        f_e = ex.f_exact(t_a, x_a, y_a, kappa)
        lam_a = ex.lambda_exact(t_a, x_a, y_a, kappa)
        residual_explicit = (u_t_e - lap_u_e - f_e - lam_a).abs().max().item()
        checks.append(
            CheckResult(
                "max|u_t - Delta u - f - lambda| (explicit, |psi|>1e-6)",
                residual_explicit, TOL_PDE_EXPLICIT, residual_explicit <= TOL_PDE_EXPLICIT,
                note=f"n_points={int(away.sum().item())}",
            )
        )

        # 9. Mask compatibility: u_exact / B finite and bounded where B > 1e-8
        mask = ex.B(x, y)
        interior = (mask > 1e-8).squeeze(-1)
        ratio = u[interior] / mask[interior]
        finite = torch.isfinite(ratio).all().item()
        max_ratio = ratio.abs().max().item()
        checks.append(
            CheckResult(
                "u_exact/B finite on interior (B>1e-8); max|u/B| reported",
                max_ratio, float("nan"), bool(finite),
                note="informational bound-reported check; pass/fail is finiteness only",
            )
        )
        extras["max_abs_u_over_B"] = max_ratio

    # ---- autograd cross-check (second-derivative Laplacian), smaller subset ----
    away_idx = away.nonzero(as_tuple=True)[0]
    perm = torch.randperm(away_idx.shape[0], generator=torch.Generator().manual_seed(SEED))[:N_AUTOGRAD]
    sub_idx = away_idx[perm]
    t_s, x_s, y_s = t[sub_idx], x[sub_idx], y[sub_idx]

    u_ag, u_t_ag, u_x_ag, u_y_ag, lap_u_ag = _autograd_u_derivatives(t_s, x_s, y_s, kappa)
    f_s = ex.f_exact(t_s, x_s, y_s, kappa)
    lam_s = ex.lambda_exact(t_s, x_s, y_s, kappa)

    # 5. PDE residual with autograd derivatives of u_exact
    residual_autograd = (u_t_ag - lap_u_ag - f_s - lam_s).abs().max().item()
    checks.append(
        CheckResult(
            "max|u_t - Delta u - f - lambda| (autograd u, |psi|>1e-6)",
            residual_autograd, TOL_PDE_AUTOGRAD, residual_autograd <= TOL_PDE_AUTOGRAD,
            note=f"n_points={N_AUTOGRAD}",
        )
    )

    # 6. explicit grad_u_exact / u_t_exact vs autograd
    gu_x_e, gu_y_e = ex.grad_u_exact(t_s, x_s, y_s, kappa)
    u_t_e_sub = ex.u_t_exact(t_s, x_s, y_s, kappa)
    max_diff_grad = torch.maximum((gu_x_e - u_x_ag).abs().max(), (gu_y_e - u_y_ag).abs().max()).item()
    max_diff_ut = (u_t_e_sub - u_t_ag).abs().max().item()
    checks.append(
        CheckResult(
            "max|grad_u_exact(explicit) - grad_u(autograd)|",
            max_diff_grad, TOL_DERIV_VS_AUTOGRAD, max_diff_grad <= TOL_DERIV_VS_AUTOGRAD,
        )
    )
    checks.append(
        CheckResult(
            "max|u_t_exact(explicit) - u_t(autograd)|",
            max_diff_ut, TOL_DERIV_VS_AUTOGRAD, max_diff_ut <= TOL_DERIV_VS_AUTOGRAD,
        )
    )

    # 7. continuity of f across psi=0.
    #
    # The prompt's literal check (|f_in - f_out| <= 1e-5 at straddle distance 1e-7) is
    # inconsistent with the prompt's own f: f is continuous but has a radial KINK at r=R(t),
    # with one-sided inner slope kappa*(2(R'+2/R) + 4/R) (= 268.7 at R=0.15, kappa=5), so a
    # straddling pair at total distance d differs by ~kappa*(R' + 4/R)*d = 1.34e-5 at d=1e-7.
    # That is a Lipschitz increment, not a jump. The literal check is kept and reported
    # (informational, never gates), and continuity is gated by what actually distinguishes
    # "continuous" from "discontinuous": the difference must scale LINEARLY in d (a jump would
    # give an O(1) difference, ~2*kappa, independent of d), must be <= 1e-5 at d=1e-9, and the
    # slope |df|/d must not exceed the analytic kink bound (decision logged in PROGRESS.md).
    gen = torch.Generator().manual_seed(SEED + 1)
    t_c = torch.rand(N_CONTINUITY_PAIRS, 1, generator=gen, dtype=DTYPE)
    theta = torch.rand(N_CONTINUITY_PAIRS, 1, generator=gen, dtype=DTYPE) * (2.0 * np.pi)
    Rt = ex.R(t_c)
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)

    def _max_f_jump(dist: float) -> float:
        r_in, r_out = Rt - dist / 2.0, Rt + dist / 2.0
        f_in = ex.f_exact(t_c, ex.C0 + r_in * cos_t, ex.C0 + r_in * sin_t, kappa)
        f_out = ex.f_exact(t_c, ex.C0 + r_out * cos_t, ex.C0 + r_out * sin_t, kappa)
        return (f_in - f_out).abs().max().item()

    jump_1e7 = _max_f_jump(2.0 * CONTINUITY_HALF_DISTANCE)
    jump_1e9 = _max_f_jump(1e-9)
    checks.append(
        CheckResult(
            "[informational] prompt-literal |f jump| at distance 1e-7",
            jump_1e7, TOL_CONTINUITY_F, True,
            note=f"literal tol 1e-5 would {'pass' if jump_1e7 <= TOL_CONTINUITY_F else 'FAIL'}: "
                 "tol is below Lip(f)*1e-7 for this f (kink, not jump); not gating",
        )
    )
    checks.append(
        CheckResult(
            "continuity of f: |f jump| at distance 1e-9",
            jump_1e9, TOL_CONTINUITY_F, jump_1e9 <= TOL_CONTINUITY_F,
            note=f"n_pairs={N_CONTINUITY_PAIRS}",
        )
    )
    ratio = jump_1e7 / max(jump_1e9, 1e-300)
    checks.append(
        CheckResult(
            "continuity of f: jump(1e-7)/jump(1e-9) (linear scaling => ~100)",
            ratio, 100.0, 90.0 <= ratio <= 110.0,
            note="a genuine discontinuity would give ratio ~1",
        )
    )
    kink_slope_bound = kappa * (ex.R_SPEED + 4.0 / ex.R0)
    slope = jump_1e7 / (2.0 * CONTINUITY_HALF_DISTANCE)
    checks.append(
        CheckResult(
            "continuity of f: |f jump|/d <= kappa*(R' + 4/R0) (analytic kink slope)",
            slope, kink_slope_bound, slope <= 1.01 * kink_slope_bound,
            note="1% slack for the smooth part g_t - Delta g",
        )
    )

    # 8. boundary condition u=0 on the four sides of partial Omega, for t in [0,1]
    gen_b = torch.Generator().manual_seed(SEED + 2)
    max_boundary = 0.0
    for side in ("x=0", "x=1", "y=0", "y=1"):
        t_b = torch.rand(N_BOUNDARY_PER_SIDE, 1, generator=gen_b, dtype=DTYPE)
        s_b = torch.rand(N_BOUNDARY_PER_SIDE, 1, generator=gen_b, dtype=DTYPE)
        if side == "x=0":
            x_b, y_b = torch.zeros_like(s_b), s_b
        elif side == "x=1":
            x_b, y_b = torch.ones_like(s_b), s_b
        elif side == "y=0":
            x_b, y_b = s_b, torch.zeros_like(s_b)
        else:
            x_b, y_b = s_b, torch.ones_like(s_b)
        u_b = ex.u_exact(t_b, x_b, y_b, kappa)
        max_boundary = max(max_boundary, u_b.abs().max().item())
    checks.append(
        CheckResult(
            "max|u| on partial Omega x (0,1) (4 sides, 1e5 pts each)",
            max_boundary, TOL_BOUNDARY, max_boundary <= TOL_BOUNDARY,
        )
    )

    # 10. initial condition u_exact(0,.) == u0_exact
    gen_i = torch.Generator().manual_seed(SEED + 3)
    x_i = torch.rand(100_000, 1, generator=gen_i, dtype=DTYPE)
    y_i = torch.rand(100_000, 1, generator=gen_i, dtype=DTYPE)
    t_i0 = torch.zeros_like(x_i)
    diff_ic = (ex.u_exact(t_i0, x_i, y_i, kappa) - ex.u0_exact(x_i, y_i, kappa)).abs().max().item()
    checks.append(
        CheckResult(
            "max|u_exact(0,.) - u0_exact|",
            diff_ic, TOL_INITIAL, diff_ic <= TOL_INITIAL,
        )
    )

    # Geometry facts (section 2.4), deterministic -- not sampled, see prompt's own wording
    # ("Geometry facts to rely on (and verify)").
    max_R = ex.R0 + ex.R_SPEED * 1.0
    dist_c0_to_boundary = ex.C0  # distance from (0.5,0.5) to the nearest side of (0,1)^2
    disk_stays_interior = max_R < dist_c0_to_boundary
    checks.append(
        CheckResult(
            "disk stays interior: max R(t) < dist(c0, partial Omega)",
            max_R, dist_c0_to_boundary, bool(disk_stays_interior),
            note=f"max_R={max_R}, dist={dist_c0_to_boundary}",
        )
    )

    return checks, extras


def _print_table(checks: list[CheckResult]) -> bool:
    all_passed = True
    name_w = max(len(c.name) for c in checks) + 2
    print(f"{'check':{name_w}} {'value':>16} {'tolerance':>12} {'status':>8}  note")
    print("-" * (name_w + 44))
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        if not c.passed:
            all_passed = False
        tol_str = f"{c.tolerance:.3e}" if c.tolerance == c.tolerance else "n/a"
        print(f"{c.name:{name_w}} {c.value:16.6e} {tol_str:>12} {status:>8}  {c.note}")
    return all_passed


def _plot_exact_solution(save_dir: Path, kappa: float, n_grid: int = 256) -> list[Path]:
    """Figures: u, g, u-g, lambda, f at t in {0, 0.25, 0.5, 0.75, 1}, with the circle
    r = R(t) overlaid, matching plots/plot_solution.py's heatmap style (viridis for raw
    fields, coolwarm for the signed difference)."""
    save_dir.mkdir(parents=True, exist_ok=True)
    lin = torch.linspace(0.0, 1.0, n_grid, dtype=DTYPE)
    x_grid, y_grid = torch.meshgrid(lin, lin, indexing="xy")
    x_flat, y_flat = x_grid.reshape(-1, 1), y_grid.reshape(-1, 1)

    times = [0.0, 0.25, 0.5, 0.75, 1.0]
    saved: list[Path] = []

    with torch.no_grad():
        for t_val in times:
            t_flat = torch.full_like(x_flat, t_val)
            u_g = ex.u_exact(t_flat, x_flat, y_flat, kappa).reshape(n_grid, n_grid).numpy()
            g_g = ex.g(t_flat, x_flat, y_flat).reshape(n_grid, n_grid).numpy()
            lam_g = ex.lambda_exact(t_flat, x_flat, y_flat, kappa).reshape(n_grid, n_grid).numpy()
            f_g = ex.f_exact(t_flat, x_flat, y_flat, kappa).reshape(n_grid, n_grid).numpy()
            gap_g = u_g - g_g

            R_t = ex.R0 + ex.R_SPEED * t_val
            theta_circ = np.linspace(0.0, 2.0 * np.pi, 400)
            circle_x = ex.C0 + R_t * np.cos(theta_circ)
            circle_y = ex.C0 + R_t * np.sin(theta_circ)

            fig, axes = plt.subplots(1, 5, figsize=(24, 4.5))
            panels = [
                ("u", u_g, "viridis"),
                ("g", g_g, "viridis"),
                ("u - g", gap_g, "viridis"),
                ("lambda", lam_g, "viridis"),
                ("f", f_g, "coolwarm"),
            ]
            for ax, (title, data, cmap) in zip(axes, panels):
                im = ax.imshow(data, extent=[0, 1, 0, 1], origin="lower", aspect="equal", cmap=cmap)
                ax.plot(circle_x, circle_y, color="red", linewidth=1.5)
                ax.set_title(f"{title} (t={t_val})")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                fig.colorbar(im, ax=ax, fraction=0.046)

            fig.tight_layout()
            out_path = save_dir / f"exact_solution_t{t_val:.2f}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            saved.append(out_path)

    return saved


def main(kappa: float = ex.KAPPA_DEFAULT) -> None:
    checks, extras = run_checks(kappa)
    all_passed = _print_table(checks)
    print(f"\nmax|u_exact/B| on interior (B>1e-8): {extras['max_abs_u_over_B']:.6e}")

    run_dir = results_io.make_run_dir("expanding_disk_2d_stage0_verification")
    figures_dir = run_dir / "figures"
    figure_paths = _plot_exact_solution(figures_dir, kappa)

    report = {
        "seed": SEED,
        "kappa": kappa,
        "n_main": N_MAIN,
        "n_autograd": N_AUTOGRAD,
        "n_boundary_per_side": N_BOUNDARY_PER_SIDE,
        "n_continuity_pairs": N_CONTINUITY_PAIRS,
        "psi_exclusion": PSI_EXCLUSION,
        "all_passed": all_passed,
        "checks": [c.to_dict() for c in checks],
        "extras": extras,
        "figures": [str(p) for p in figure_paths],
    }
    report_path = run_dir / "verification_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(f"\nreport  -> {report_path}")
    print(f"figures -> {figures_dir}")

    if not all_passed:
        print("\nVERIFICATION FAILED", file=sys.stderr)
        sys.exit(1)

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
