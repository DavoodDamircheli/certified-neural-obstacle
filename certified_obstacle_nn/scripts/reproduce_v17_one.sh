#!/usr/bin/env bash
# v17 Stage 7: reproduce one direct certificate end to end (training excluded, enclosure included) from its frozen
# checkpoint and compare bitwise with the stored record (default: Stefan seed 0, 512^2, numpy + arb, Stage 2).
# Usage (from the repository root): bash certified_obstacle_nn/scripts/reproduce_v17_one.sh [--ref results/v17_split/<ts>]
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -m certified_obstacle_nn.experiments.v17.reproduce_one "$@"
