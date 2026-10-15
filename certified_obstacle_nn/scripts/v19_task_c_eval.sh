#!/usr/bin/env bash
# v19 Task C2: evaluate one retrained ellipse network exactly as results/v19_protocols/ellipse_multiplier_protocol.json
# says: Stage-0 stored run (traces), frozen protocol v1 (scripts/v17b_protocol_v1.sh, unchanged), then the Task A
# exact-integral diagnostics on the frozen v1 partition. Every run directory goes to results/v19_C_<arm>_*/<ts>/ (D46).
#
#   bash certified_obstacle_nn/scripts/v19_task_c_eval.sh <arm> <training run dir> [workers]
set -euo pipefail
cd "$(dirname "$0")/../.."
arm=$1; train=$2; w=${3:-16}
export CERT_EXPERIMENT_PREFIX="v19_C_${arm}_" PYTHONPATH=.
ts=$(date -u +%Y%m%dT%H%M%SZ)
idx="results/v19_C_index/${ts}_${arm}.jsonl"
python3 -m certified_obstacle_nn.experiments.v14.ellipse_validated --seeds 3 --budgets 100000 --arb-seeds \
  --train-run "$train" --index "$idx" --workers "$w"
src=$(python3 -c "import json,sys;print(json.loads(open('$idx').read().splitlines()[-1])['run_dir'])")
echo "$(date +%T) stage0 $src"
part=$(bash certified_obstacle_nn/scripts/v17b_protocol_v1.sh steps123 "$src" "$w" | tee >(cat >&2) | tail -n 1 | awk '{print $1}')
echo "$(date +%T) protocol partition $part"
run=$(bash certified_obstacle_nn/scripts/v17b_protocol_v1.sh step4 "$part" "$src" "$w" | tee >(cat >&2) | tail -n 1 | awk '{print $NF}')
echo "$(date +%T) protocol run $run"
python3 -m certified_obstacle_nn.experiments.v19.substitution --run "$run"
echo "$(date +%T) DONE $arm stage0=$src partition=$part run=$run"
