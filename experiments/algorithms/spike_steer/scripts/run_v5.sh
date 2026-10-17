#!/usr/bin/env bash
# v5 search-fix runs: bash experiments/algorithms/spike_steer/scripts/run_v5.sh <tag> <seed> [workers] [extra pilot_v5 flags...]
#   v5b (balanced fitness):  run_v5.sh b 3 18 --fitness balanced
#   v5c (mirror-tied pair):  run_v5.sh c 3 14 --tied
# Works from any checkout/worktree; python from $PY if set.
set -u
TAG=${1:?tag}; SEED=${2:?seed}; WORKERS=${3:-$(( $(nproc) - 2 ))}; shift 3 || shift $#
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd); cd "$ROOT" || exit 1
PY=${PY:-/home/robot/miniconda3/envs/mujoco/bin/python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=runs/spike_steer/pilot_v5${TAG}_s${SEED}; ART=artifacts/spike_steer/v5${TAG}
mkdir -p "$OUT" "$ART"
echo "start $(date -Is) host=$(hostname) tag=$TAG seed=$SEED workers=$WORKERS flags=$*"
$PY -W ignore -m experiments.algorithms.spike_steer.pilot_v5 all --seed "$SEED" --workers "$WORKERS" --output "$OUT" "$@"
echo "exit=$? $(date -Is)"
cp "$OUT"/history.jsonl "$OUT"/validated.json "$OUT"/select.json "$OUT"/selected_raw.json "$OUT"/calibrate.json \
   "$OUT"/selected_calibrated.json "$OUT"/preregistration.json "$OUT"/test.json "$OUT"/neat.ini "$ART"/ 2>/dev/null
echo "end $(date -Is)"
