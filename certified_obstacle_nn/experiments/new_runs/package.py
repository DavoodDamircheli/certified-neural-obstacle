"""NEW_RUNS.md "Packaging Instructions" -- assembles
certified_obstacle_nn_new_runs.zip with the exact structure specified there.
Copies files (never moves/deletes originals); checkpoints/ is a lean copy
(weights.pt + config.json + summary.csv per run only -- not the bulkier
certification/true_error subprocess JSON payloads or diagnostic npz files,
which are regenerable from weights+config and would otherwise dominate the
zip size for no reproducibility benefit, per NEW_RUNS.md's own "no
unnecessary large temporary files").
"""

from __future__ import annotations

import shutil
from pathlib import Path

from certified_obstacle_nn.experiments.new_runs.common import NEW_RUNS_DIR

REPO_ROOT = Path(__file__).resolve().parents[3]
STAGING_DIR = REPO_ROOT / "zips" / "certified_obstacle_nn_new_runs"
ZIP_PATH = REPO_ROOT / "zips" / "certified_obstacle_nn_new_runs.zip"

EXPERIMENT_TOP_LEVEL_FILES = {
    "adaptive_multiseed": ["adaptive_multiseed_all_runs.csv", "adaptive_multiseed_summary.csv",
                            "table_A_adaptive_sampling.csv", "INTERPRETATION.md"],
    "american_put_improved": ["reference_convergence.csv", "all_runs.csv", "summary.csv",
                               "exploratory_architecture_study.csv", "final_configuration_all_seeds.csv",
                               "table_B_improved_american_put.csv", "INTERPRETATION.md"],
    "certificate_aware": ["all_runs.csv", "summary.csv", "exploratory_gamma_sweep.csv",
                           "final_gamma_sweep_all_seeds.csv", "table_C_certificate_aware.csv", "INTERPRETATION.md"],
    "threshold_sensitivity": ["free_boundary_threshold_sensitivity.csv", "RANKING_STABILITY.md"],
}

CHECKPOINT_FILES = ["weights.pt", "config.json", "summary.csv", "status.json"]


def _copy_checkpoints(src_exp_dir: Path, dst_checkpoints_dir: Path) -> int:
    n = 0
    for weights_path in src_exp_dir.rglob("weights.pt"):
        run_dir = weights_path.parent
        rel = run_dir.relative_to(src_exp_dir)
        dst_run_dir = dst_checkpoints_dir / rel
        dst_run_dir.mkdir(parents=True, exist_ok=True)
        for fname in CHECKPOINT_FILES:
            src_file = run_dir / fname
            if src_file.exists():
                shutil.copy2(src_file, dst_run_dir / fname)
        n += 1
    return n


def main() -> Path:
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir(parents=True)

    shutil.copy2(NEW_RUNS_DIR / "NEW_RUNS_METADATA.md", STAGING_DIR / "NEW_RUNS_METADATA.md")
    shutil.copy2(NEW_RUNS_DIR / "new_runs_master.csv", STAGING_DIR / "new_runs_master.csv")

    for experiment, files in EXPERIMENT_TOP_LEVEL_FILES.items():
        src_dir = NEW_RUNS_DIR / experiment
        dst_dir = STAGING_DIR / experiment
        dst_dir.mkdir(parents=True, exist_ok=True)

        for fname in files:
            src = src_dir / fname
            if src.exists():
                shutil.copy2(src, dst_dir / fname)
            else:
                print(f"WARNING: {src} not found, skipping")

        figures_src = src_dir / "figures"
        if figures_src.exists():
            shutil.copytree(figures_src, dst_dir / "figures", dirs_exist_ok=True)

        if experiment != "threshold_sensitivity":
            checkpoints_dst = dst_dir / "checkpoints"
            checkpoints_dst.mkdir(parents=True, exist_ok=True)
            n = _copy_checkpoints(src_dir, checkpoints_dst)
            print(f"{experiment}: {n} checkpoints copied")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", root_dir=STAGING_DIR.parent, base_dir=STAGING_DIR.name)

    size_mb = ZIP_PATH.stat().st_size / (1024 ** 2)
    print(f"\n{ZIP_PATH} written ({size_mb:.1f} MB)")
    return ZIP_PATH


if __name__ == "__main__":
    main()
