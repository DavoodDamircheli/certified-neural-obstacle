"""NEW_RUNS.md Experiment 2 -- required figures (2.8), Table B, and the
success/failure verdict (2.9/2.10). Reads ONLY already-saved files (this
project's own run-artifact convention) -- never recomputes a quantity.

Representative run for the surface/boundary/slice/decomposition figures:
the FINAL configuration's MEDIAN-E_true seed (not the best-looking one) --
same "not chosen for visual appeal" discipline as Experiment 1's collocation
clouds.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from certified_obstacle_nn.experiments.new_runs.american_put_improved import TAU_SLICES
from certified_obstacle_nn.experiments.new_runs.common import NEW_RUNS_DIR, save_figure

EXPERIMENT = "american_put_improved"
EXP_DIR = NEW_RUNS_DIR / EXPERIMENT
FIG_DIR = EXP_DIR / "figures"

BASELINE_ROW = {
    # results/american_put_baseline_summary.json (exact values read from that file, not
    # rounded) / experiments/american_put_baseline.py, 4x64 seed 0, plain uniform sampling
    # -- the run this experiment set out to improve on.
    "architecture": "4x64 (baseline)", "E_true": 3.6909647884785324, "E_cert": None, "E_cert_U": 59530719.204360664,
    "exercise_boundary_error": None, "max_price_error": None, "L2_price_error": None,
    "runtime": 382.1062846020013,
}


def _representative_seed() -> int:
    df = pd.read_csv(EXP_DIR / "final_configuration_all_seeds.csv")
    df = df.sort_values("E_true").reset_index(drop=True)
    return int(df.iloc[len(df) // 2]["seed"])


def _representative_run_dir(seed: int) -> Path:
    best_arch = pd.read_csv(EXP_DIR / "final_configuration_all_seeds.csv")["architecture"].iloc[0]
    return EXP_DIR / "final" / best_arch / f"seed_{seed}"


def _fig_surfaces(run_dir: Path) -> None:
    data = np.load(run_dir / "reference_comparison.npz")
    tau, x, V_neural, V_ref = data["tau_grid"], data["x_grid"], data["V_neural"], data["V_ref"]

    for name, Z, title in [("fig_american_put_reference", V_ref, "Reference option-value surface (frozen PSOR)"),
                            ("fig_american_put_neural", V_neural, "Neural option-value surface")]:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        im = ax.pcolormesh(x, tau, Z, shading="auto", cmap="viridis")
        ax.set_xlabel("x = log(S)")
        ax.set_ylabel("tau")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="V (physical price)")
        fig.tight_layout()
        save_figure(fig, FIG_DIR / name)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.pcolormesh(x, tau, np.abs(V_neural - V_ref), shading="auto", cmap="magma")
    ax.set_xlabel("x = log(S)")
    ax.set_ylabel("tau")
    ax.set_title("Pointwise error |V_neural - V_ref|")
    fig.colorbar(im, ax=ax, label="|error|")
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "fig_american_put_error")
    plt.close(fig)


def _fig_exercise_boundary(run_dir: Path, seed: int) -> None:
    data = np.load(run_dir / "reference_comparison.npz")
    tau_b, neural_b, ref_b = data["tau_b"], data["neural_boundary"], data["ref_boundary"]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(tau_b, ref_b, label="reference (frozen PSOR)", color="tab:red", linewidth=2)
    ax.plot(tau_b, neural_b, label="neural (learned)", color="tab:blue", linewidth=1.5, linestyle="--")
    ax.set_xlabel("tau")
    ax.set_ylabel("exercise boundary x*(tau) = log(S*(tau))")
    ax.set_title(f"Exercise boundary: reference vs. neural (representative seed={seed}, median E_true)")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "fig_american_put_exercise_boundary")
    plt.close(fig)


def _fig_slices(run_dir: Path, seed: int) -> None:
    df = pd.read_csv(run_dir / "price_slices.csv")
    fig, axes = plt.subplots(1, len(TAU_SLICES), figsize=(4 * len(TAU_SLICES), 4.5), sharey=True)
    for ax, tau in zip(axes, TAU_SLICES):
        sub = df[df["tau"] == tau].sort_values("S")
        ax.plot(sub["S"], sub["V_ref"], "o-", label="reference", color="tab:red")
        ax.plot(sub["S"], sub["V_neural"], "s--", label="neural", color="tab:blue")
        ax.set_title(f"tau={tau}")
        ax.set_xlabel("S")
    axes[0].set_ylabel("V (physical price)")
    axes[0].legend()
    fig.suptitle(f"Price slices at representative seed={seed} (below/near/above strike, several tau)")
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "fig_american_put_slices")
    plt.close(fig)


def _fig_certificate_decomposition(run_dir: Path, seed: int) -> None:
    summary = pd.read_csv(run_dir / "summary.csv").iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    rho_parts = ["rho_r", "rho_p_tilde", "rho_c"]
    axes[0].bar(rho_parts, [summary[c] for c in rho_parts], color="tab:blue")
    axes[0].set_title(f"Raw estimator split (rho_tilde_h={summary['rho_tilde_h']:.3g})")
    axes[0].set_ylabel("value")

    eh_parts = ["E_h_lip", "E_h_unr", "E_h_dat", "E_h_0T"]
    axes[1].bar(eh_parts, [summary[c] for c in eh_parts], color="tab:orange")
    axes[1].set_title(f"E_h decomposition (E_h_dat != 0: Gamma_data={{log K}} hit -- American-put-specific)")
    axes[1].set_ylabel("value")
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle(f"Estimator/certificate decomposition (representative seed={seed})")
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "fig_american_put_certificate_decomposition")
    plt.close(fig)


def _table_b() -> pd.DataFrame:
    final = pd.read_csv(EXP_DIR / "final_configuration_all_seeds.csv")
    ref_conv = pd.read_csv(EXP_DIR / "reference_convergence.csv")
    finest = ref_conv.iloc[-1]

    rows = [
        {"row": "reference", "architecture": f"n={int(finest['Nx'])} PSOR (frozen)", "E_ref": None,
         "E_cert": None, "E_cert_U": None, "exercise_boundary_error": None,
         "max_price_error": None, "L2_price_error": None, "runtime_seconds": finest["runtime_seconds"]},
        {"row": "baseline neural", "architecture": BASELINE_ROW["architecture"], "E_ref": BASELINE_ROW["E_true"],
         "E_cert": BASELINE_ROW["E_cert"], "E_cert_U": BASELINE_ROW["E_cert_U"],
         "exercise_boundary_error": BASELINE_ROW["exercise_boundary_error"],
         "max_price_error": BASELINE_ROW["max_price_error"], "L2_price_error": BASELINE_ROW["L2_price_error"],
         "runtime_seconds": BASELINE_ROW["runtime"]},
        {"row": "best improved neural (mean +/- std, 5 seeds)",
         "architecture": final["architecture"].iloc[0],
         "E_ref": f"{final['E_true'].mean():.4g} +/- {final['E_true'].std():.4g}",
         "E_cert": f"{final['E_cert'].mean():.4g} +/- {final['E_cert'].std():.4g}",
         "E_cert_U": f"{final['E_cert_U'].mean():.4g} +/- {final['E_cert_U'].std():.4g}",
         "exercise_boundary_error": f"{final['exercise_boundary_error'].mean():.4g} +/- {final['exercise_boundary_error'].std():.4g}",
         "max_price_error": f"{final['max_price_error'].mean():.4g} +/- {final['max_price_error'].std():.4g}",
         "L2_price_error": f"{final['L2_price_error'].mean():.4g} +/- {final['L2_price_error'].std():.4g}",
         "runtime_seconds": f"{final['training_time_seconds'].mean():.1f} +/- {final['training_time_seconds'].std():.1f}"},
    ]
    df = pd.DataFrame.from_records(rows)
    df.to_csv(EXP_DIR / "table_B_improved_american_put.csv", index=False)
    return df


def _write_interpretation(table_b: pd.DataFrame) -> None:
    exp = pd.read_csv(EXP_DIR / "exploratory_architecture_study.csv")
    final = pd.read_csv(EXP_DIR / "final_configuration_all_seeds.csv")
    text = f"""# Experiment 2 -- Improved American Put: Verdict

## What was tried

- Section 2.2 (normalization): already satisfied by existing code (`problems/american_put.py`'s
  `xi`, `NormalizedAmericanPutProblem`) -- unchanged, reused.
- Section 2.3 (payoff kink): 30% of the interior collocation budget drawn from a
  narrow band (halfwidth 0.4 in log-price, ~10% of the domain) around x_star=log(K).
- Section 2.4 (learned contact set): 3 further rounds of Dorfler-marked adaptive
  resampling on `rho_penetration_K` (a function of the network's OWN obstacle-violation
  gap only -- never touches the reference), warm-started from the section-2.3 sample.
- Section 2.5: exploratory architecture study, 1 seed, {{4x64, 6x64, 6x128}}.
- Section 2.6: full per-epoch loss-component logging for the (majority) static phase.

## Exploratory architecture study (1 seed each)

{exp[["architecture", "E_true", "E_cert_U", "exercise_boundary_error"]].to_string(index=False)}

All three architectures scored WORSE on E_true than the flagged baseline
(4x64, plain uniform sampling, E_true=3.69). 6x64 was the best of the three
(E_true=4.93) and was carried forward to the 5-seed final sweep.

## Final configuration (6x64, 5 seeds)

{final[["seed", "E_true", "E_cert_U", "exercise_boundary_error", "max_price_error", "L2_price_error"]].to_string(index=False)}

Mean E_true = {final['E_true'].mean():.4g} +/- {final['E_true'].std():.4g} (baseline: 3.69).
**Every one of the 5 final seeds is worse than the baseline's single E_true=3.69** --
this is not seed noise, it is a consistent effect.

## Verdict against the section 2.9 success criterion

Section 2.9 requires ALL of: (1) option-value error clearly decreases vs. the
current run, (2) the learned boundary tracks the reference quantitatively,
(3) stability across seeds, (4) residual diagnostics consistent with the
observed error. **Criterion (1) fails outright** -- E_true got WORSE, not
better, in every architecture and every final seed. This experiment is
reported as a **stress-test / failure diagnostic (section 2.10)**, not a
successful application example. Per NEW_RUNS.md's explicit instruction, this
result is reported as-is and was not tuned further to chase a better number.

## Why this likely happened (diagnosis, not an excuse to keep tuning)

The two mechanisms most likely responsible, in order of suspicion:

1. **Aggressive resampling churn.** `training/adaptive.py`'s `resample_points`
   (alpha_marked_fraction=0.7, unchanged from its Stefan default) discards
   70% of the collocation set's spatial distribution each of the 3 adaptive
   rounds and re-concentrates it near the CURRENT (possibly still-inaccurate)
   estimate of the contact region. For Stefan (Experiment 1) this same
   mechanism only modestly hurt accuracy relative to uniform; American put's
   sharper payoff kink and data interface may make the network's early-round
   contact-region estimate less reliable, so each resampling round can pull
   coverage away from regions (e.g. near the domain boundaries x=L, x=R,
   where the Dirichlet conditions and u_lift's own domain-wide linear term
   matter) that the loss still needs.
2. **Short per-round budget.** Each of the 3 adaptive rounds gets only 250
   Adam + 20 L-BFGS epochs to re-converge to a drastically redistributed
   point set -- plausibly too little for a network this size on this problem.

Both are properties of the SPECIFIC (alpha_marked_fraction, rounds,
per-round-epochs) hyperparameters chosen here, not of the general idea in
2.3/2.4 -- a genuinely more careful schedule (e.g. a smaller marked fraction,
more rounds with smaller per-round changes) might do better, but that is
future work, not something to chase now under a "do not tune toward a
desired conclusion" instruction.

## Table B

{table_b.to_string(index=False)}
"""
    (EXP_DIR / "INTERPRETATION.md").write_text(text)
    print(f"INTERPRETATION.md -> {EXP_DIR / 'INTERPRETATION.md'}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    seed = _representative_seed()
    run_dir = _representative_run_dir(seed)
    print(f"Representative run (median E_true among final seeds): seed={seed}, {run_dir}")

    _fig_surfaces(run_dir)
    _fig_exercise_boundary(run_dir, seed)
    _fig_slices(run_dir, seed)
    _fig_certificate_decomposition(run_dir, seed)
    print(f"6 figures -> {FIG_DIR}")

    table_b = _table_b()
    print(f"table_B_improved_american_put.csv -> {EXP_DIR / 'table_B_improved_american_put.csv'}")
    print(table_b.to_string(index=False))

    _write_interpretation(table_b)


if __name__ == "__main__":
    main()
