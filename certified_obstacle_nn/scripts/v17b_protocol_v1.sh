#!/usr/bin/env bash
# v17b Task 1: run the frozen protocol v1 (docs/v17_frozen_protocol.json) on one stored run, in two jobs of
# <= 100 min each (prompt Section 1c). Same commands as the Stage 7 long-disk seeds; nothing is changed.
#
#   bash certified_obstacle_nn/scripts/v17b_protocol_v1.sh steps123 <stored run> [workers]   # steps 1-3
#   bash certified_obstacle_nn/scripts/v17b_protocol_v1.sh step4 <partition dir> <stored run> [workers]
#
# steps123: uniform 128^3 (frozen), TM1 evaluation in 2 chunks + finalize (intermediate result), seed the
#           selection state from the chunks, ONE Doerfler flux step capped at 3.5e6 cells (frozen + hashed).
# step4:    from-scratch TM1 evaluation of the frozen protocol partition in 3 chunks + finalize.
set -euo pipefail
cd "$(dirname "$0")/../.."
M="python3 -m certified_obstacle_nn.experiments.v17"
mode=$1
if [ "$mode" = steps123 ]; then
  src=$2; w=${3:-20}
  part=$($M.stage6 uniform --n 128 --source "$src" --workers "$w" | tail -n 1 | awk '{print $1}')
  echo "$(date +%T) uniform partition $part"
  job=$($M.chunked_certify init --source "$src" --enclosure TM1 --chunks 2 --partition "$part" | tail -n 1)
  echo "$(date +%T) uniform job $job"
  $M.chunked_certify chunk --job "$job" --index 0 1 --workers "$w"
  $M.chunked_certify finalize --job "$job" --workers "$w"
  seeded=$($M.stage6 seed-from-chunks --job "$job" | tail -n 1 | awk '{print $1}')
  echo "$(date +%T) seeded state $seeded"
  $M.stage6 adaptive --budget 3500000 --source "$src" --resume "$seeded" --workers "$w"
elif [ "$mode" = step4 ]; then
  part=$2; src=$3; w=${4:-20}
  job=$($M.chunked_certify init --source "$src" --enclosure TM1 --chunks 3 --partition "$part" | tail -n 1)
  echo "$(date +%T) protocol job $job"
  $M.chunked_certify chunk --job "$job" --index 0 1 2 --workers "$w"
  $M.chunked_certify finalize --job "$job" --workers "$w"
else
  echo "usage: $0 steps123|step4 ..." >&2; exit 2
fi
