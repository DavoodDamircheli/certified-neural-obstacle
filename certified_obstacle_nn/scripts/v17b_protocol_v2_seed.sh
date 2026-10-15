#!/usr/bin/env bash
# v17b Task 3: run the whole frozen protocol v2 (docs/v17_frozen_protocol_v2.json) on one seed as a sequence of jobs,
# each wrapped in `timeout 7200` and designed <= 100 min; the intermediate certificate of a budget (4 workers) runs
# concurrently with the next selection step (16 workers), the final from-scratch chunks use 20 workers.
#
#   bash certified_obstacle_nn/scripts/v17b_protocol_v2_seed.sh <seed> <stored run> <v1 partition> [log prefix] > logs/....log
set -euo pipefail
cd "$(dirname "$0")/../.."
P="timeout 7200 bash certified_obstacle_nn/scripts/v17b_protocol_v2.sh"
seed=$1; src=$2; prev=$3
L=${4:-logs/v17b_v2_s$seed}
pids=()
for b in 4500000 6000000 8000000; do
  $P select "$prev" "$src" "$b" 16 > "${L}_select$b.log" 2>&1
  prev=$(tail -n 1 "${L}_select$b.log" | awk '{print $1}')
  echo "$(date +%T) seed $seed budget $b partition $prev"
  for p in "${pids[@]}"; do wait "$p"; done; pids=()  # never more than one 4-worker job at a time
  if [ "$b" != 8000000 ]; then
    $P intermediate "$prev" "$src" 4 > "${L}_inter$b.log" 2>&1 &
    pids+=($!)
  fi
done
for p in "${pids[@]}"; do wait "$p"; done
job=$($P final-init "$prev" "$src" | tail -n 1 | awk '{print $NF}')
echo "$(date +%T) seed $seed final job $job"
$P final-chunks "$job" 0 1 -- 20 > "${L}_final_c01.log" 2>&1
$P final-chunks "$job" 2 3 -- 20 > "${L}_final_c23.log" 2>&1
$P final-finalize "$job" 20 > "${L}_final_fin.log" 2>&1
echo "$(date +%T) seed $seed done: $(tail -n 1 "${L}_final_fin.log")"
