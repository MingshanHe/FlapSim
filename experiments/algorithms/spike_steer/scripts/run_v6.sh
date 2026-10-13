#!/usr/bin/env bash
# Plan B' arm run: bash experiments/algorithms/spike_steer/scripts/run_v6.sh <arm> <seed> [workers] [population] [generations] [extra pilot_v6 flags...]
#   arm: main | nodelay | fixed | ahrs | rate;  extra flags e.g. --tied --fitness balanced
# Works from any checkout/worktree: repo root = 4 levels above this script; python from $PY if set.
set -u
ARM=${1:?arm}; SEED=${2:?seed}; WORKERS=${3:-$(( $(nproc) - 2 ))}; POP=${4:-64}; GEN=${5:-40}
shift 5 2>/dev/null || shift $#
TAG=""; for f in "$@"; do case "$f" in --tied) TAG="${TAG}_tied";; balanced) TAG="${TAG}_bal";; esac; done
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd); cd "$ROOT" || exit 1
PY=${PY:-/home/robot/miniconda3/envs/mujoco/bin/python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=runs/spike_steer/pilot_v6_${ARM}${TAG}_s${SEED}; ART=artifacts/spike_steer/v6/${ARM}${TAG}_s${SEED}
mkdir -p "$OUT" "$ART"
echo "start $(date -Is) host=$(hostname) arm=$ARM seed=$SEED workers=$WORKERS pop=$POP gen=$GEN flags=$*"
$PY -W ignore -m experiments.algorithms.spike_steer.pilot_v6 all --arm "$ARM" --seed "$SEED" --workers "$WORKERS" \
    --population "$POP" --generations "$GEN" --output "$OUT" "$@"
echo "exit=$? $(date -Is)"
cp "$OUT"/history.jsonl "$OUT"/validated.json "$OUT"/select.json "$OUT"/selected_raw.json "$OUT"/calibrate.json \
   "$OUT"/selected_calibrated.json "$OUT"/preregistration.json "$OUT"/test.json "$OUT"/neat.ini "$ART"/ 2>/dev/null
$PY -W ignore -m experiments.algorithms.spike_steer.pilot_v6 analyze --spec "$ART"/selected_calibrated.json \
    --output "$ART"/mechanism.json --workers "$WORKERS" 2>&1 | tail -20
echo "end $(date -Is)"
