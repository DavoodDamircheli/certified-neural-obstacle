"""Final-phase Task 3/4: reference configuration retrained from scratch with a longer Adam
schedule (20 000 steps, lr 1e-3 -> 3e-4 @ 8000 -> 1e-4 @ 14000), then the unchanged
baseline L-BFGS phase. Everything else is the reference configuration (6x64, N_int = 20k,
kappa = 5, w_p = w_c = 10).

  python certified_obstacle_nn/experiments/expanding_disk_2d/final_long_train.py --seed 0 --threads 12

Run dir: results/expanding_disk_2d/final/long_6x64_N20000_s<seed>_k5/ (same file set as
run_experiment.py; never overwritten).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from certified_obstacle_nn.experiments.expanding_disk_2d import config as config_mod  # noqa: E402
from certified_obstacle_nn.experiments.expanding_disk_2d.config import Config  # noqa: E402
from certified_obstacle_nn.experiments.expanding_disk_2d.run_experiment import run  # noqa: E402

LONG_SCHEDULE = [[0, 1e-3], [8000, 3e-4], [14000, 1e-4]]
N_ADAM_LONG = 20_000


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args(argv)
    # new outputs go under final/ (make_run_dir reads config_mod.RESULTS_ROOT at call time)
    config_mod.RESULTS_ROOT = config_mod.RESULTS_ROOT / "final"
    extra = {"adam_lr_schedule": LONG_SCHEDULE}
    if a.threads:
        extra["threads"] = a.threads
    cfg = Config(stage="long", arch="6x64", n_int=20_000, seed=a.seed, kappa=5.0, n_adam=N_ADAM_LONG,
                 n_lbfgs=300, w_p=10.0, w_c=10.0, extra=extra)
    run(cfg)


if __name__ == "__main__":
    main()
