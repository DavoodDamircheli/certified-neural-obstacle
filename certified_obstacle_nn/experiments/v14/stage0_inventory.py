"""v14 Stage 0: inventory of checkpoints / results / scripts, with SHA-256 of
every checkpoint (frozen before any v14 certification)."""

import json

from certified_obstacle_nn.experiments.v14.common import REPO_ROOT, sha256_file


def main():
    ck = sorted(p for ext in ("*.pt",) for p in (REPO_ROOT / "results").rglob(ext))
    inv = {
        "checkpoints": {str(p.relative_to(REPO_ROOT)): {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in ck},
        "results_top_level": sorted(str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / "results").iterdir()),
        "experiment_scripts": sorted(str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / "certified_obstacle_nn/experiments").rglob("*.py")),
        "configs": sorted(str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / "results").rglob("config*.json")),
    }
    dst = REPO_ROOT / "results/v14/stage0/inventory.json"
    dst.write_text(json.dumps(inv, indent=1))
    print(len(inv["checkpoints"]), "checkpoints;", len(inv["configs"]), "configs;", len(inv["experiment_scripts"]), "scripts")


if __name__ == "__main__":
    main()
