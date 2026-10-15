"""v14 Stage 5 study: seeds x budgets x {uniform, adaptive}, diagnostics computed
once per seed and reused (they do not depend on the partition).

    nohup python3 -m certified_obstacle_nn.experiments.v14.stefan_study \
        --seeds 0 1 2 --uniform 32 64 128 256 512 --adaptive 1000 10000 100000 \
        --workers 20 > logs/v14_stefan_study.log 2>&1 &

Each (seed, partition) is one run directory under results/v14_stefan_validated/
(arb = reference backend, numpy = fast backend; both evaluate the same frozen
partition). A run whose directory already contains the same (seed, mode, size)
in results/v14_stefan_validated/index.jsonl is skipped (resumable).
"""

import argparse
import json
import time
from pathlib import Path

import torch

from certified_obstacle_nn.experiments.v14.common import REPO_ROOT, STEFAN_SEEDS, load_frozen_stefan
from certified_obstacle_nn.experiments.v14.stefan_validated import diagnostics, run, write_run

INDEX = REPO_ROOT / "results/v14_stefan_validated/index.jsonl"


def done_keys(index):
    if not index.exists():
        return set()
    return {(d["seed"], d["mode"], d["size"]) for d in map(json.loads, index.read_text().splitlines())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--uniform", type=int, nargs="*", default=[32, 64, 128, 256])
    ap.add_argument("--adaptive", type=int, nargs="*", default=[1000, 10000, 100000])
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--prec", type=int, default=64)
    ap.add_argument("--arb-max-cells", type=int, default=300000)
    ap.add_argument("--index", default=str(INDEX), help="resume index (a new file starts a fresh study)")
    a = ap.parse_args()
    index = REPO_ROOT / a.index if not a.index.startswith("/") else Path(a.index)
    torch.set_num_threads(4)
    index.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(index)

    def log(msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    for seed in a.seeds:
        p, tr = load_frozen_stefan(STEFAN_SEEDS[seed])
        diag = None
        jobs = [("uniform", n) for n in a.uniform] + [("adaptive", b) for b in a.adaptive]
        for mode, size in jobs:
            if (seed, mode, size) in done:
                log(f"skip seed {seed} {mode} {size} (done)")
                continue
            if diag is None:
                t0 = time.time()
                diag = diagnostics(p, tr)
                log(f"seed {seed}: diagnostics {time.time() - t0:.1f}s E_true={diag[1]['E_true']:.6f} "
                    f"E_true(vhat)={diag[2]['E_true']:.6f}")
            ncells = size * size if mode == "uniform" else size
            vb = "arb" if ncells <= a.arb_max_cells else None
            log(f"seed {seed} {mode} {size}: start")
            config, rows, part, vols, _ = run(seed, mode, size, size, "numpy", vb, a.workers, a.prec, 4096, 0.5,
                                              diag=diag, log=log)
            d = write_run(config, rows, part, vols, STEFAN_SEEDS[seed])
            with index.open("a") as fh:
                fh.write(json.dumps({"seed": seed, "mode": mode, "size": size, "run_dir": str(d.relative_to(REPO_ROOT))}) + "\n")
            for r in rows:
                log(f"  {r['backend']}: n={r['n_cells']} {r['status']} E_U_w={r['E_U_weighted']:.4f} "
                    f"ratio={r['ratio_E_U_weighted']:.2f} proj={r['E_U_projected']:.4f} orig={r['E_U_original_via_projection']:.4f} "
                    f"grad_rel={r['grad_bound_rel']:.3f} t_leaf={r['t_leaf_eval_s']:.1f}s")


if __name__ == "__main__":
    main()
