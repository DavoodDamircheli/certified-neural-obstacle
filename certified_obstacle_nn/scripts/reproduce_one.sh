#!/usr/bin/env bash
# v14 Stage 7: re-run one full Stefan evaluation (seed 0, uniform 32x32, arb + numpy backends)
# from its frozen checkpoint and compare with the stored committed-code run.
# Usage (from the repository root): bash certified_obstacle_nn/scripts/reproduce_one.sh [--run-dir results/...]
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -m certified_obstacle_nn.experiments.v14.reproduce_one "$@"
