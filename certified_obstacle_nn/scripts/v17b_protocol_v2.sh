#!/usr/bin/env bash
# v17b Task 3: frozen protocol v2 (docs/v17_frozen_protocol_v2.json), one job of <= 100 min per call.
#
#   bash certified_obstacle_nn/scripts/v17b_protocol_v2.sh select <previous partition> <stored run> <budget> [workers]
#   bash certified_obstacle_nn/scripts/v17b_protocol_v2.sh intermediate <partition> <stored run> [workers]
#   bash certified_obstacle_nn/scripts/v17b_protocol_v2.sh final-init <partition> <stored run>
#   bash certified_obstacle_nn/scripts/v17b_protocol_v2.sh final-chunks <job> <i> [<j> ...] -- [workers]
#   bash certified_obstacle_nn/scripts/v17b_protocol_v2.sh final-finalize <job> [workers]
#
# select:        Doerfler step(s) on the gradient marking, capped at <budget> cells (frozen + hashed partition).
# intermediate:  arb reduction of the selection-state TM1 densities (import-selection + finalize).
# final-*:       from-scratch TM1 evaluation in 4 chunks + finalize, split over several jobs.
set -euo pipefail
cd "$(dirname "$0")/../.."
M="python3 -m certified_obstacle_nn.experiments.v17"
mode=$1
case "$mode" in
  select)
    $M.stage6 adaptive --objective grad --budget "$4" --source "$3" --resume "$2" --workers "${5:-16}" ;;
  intermediate)
    job=$($M.chunked_certify import-selection --source "$3" --partition "$2" | tail -n 1)
    echo "$(date +%T) selection job $job"
    $M.chunked_certify finalize --job "$job" --workers "${4:-16}" ;;
  final-init)
    job=$($M.chunked_certify init --source "$3" --enclosure TM1 --chunks 4 --partition "$2" | tail -n 1)
    echo "$(date +%T) final job $job" ;;
  final-chunks)
    job=$2; shift 2; idx=()
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do idx+=("$1"); shift; done
    [ $# -gt 0 ] && shift
    $M.chunked_certify chunk --job "$job" --index "${idx[@]}" --workers "${1:-16}" ;;
  final-finalize)
    $M.chunked_certify finalize --job "$2" --workers "${3:-16}" ;;
  *)
    echo "usage: $0 select|intermediate|final-init|final-chunks|final-finalize ..." >&2; exit 2 ;;
esac
