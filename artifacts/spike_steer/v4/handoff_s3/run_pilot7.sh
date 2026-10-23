#!/usr/bin/env bash
# Round-7 pilot (exploratory): one arm, one seed: setup, 30-generation search, 128-case evaluation of the saved centres,
# archive to artifacts/spike_steer/pilot7/<arm>/s<seed>/.   Usage: W=15 PY=python bash run_pilot7.sh <B|C> <seed>
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-python}; ARMX=${1:?arm}; SEED=${2:?seed}; W=${W:?W}
M=experiments.algorithms.spike_steer.pilot7
$PY -W ignore -m $M selftest --arm $ARMX || exit 1
[ -f runs/spike_steer/pilot7_$ARMX/train/imitate_spec_s$SEED.json ] || $PY -W ignore -m $M setup --arm $ARMX || exit 1
$PY -W ignore -m $M finetune --arm $ARMX --seed $SEED --generations 30 --hours 20 --workers $W || exit 1
$PY -W ignore -m $M evaluate --arm $ARMX --seed $SEED --workers $W || exit 1
D=artifacts/spike_steer/pilot7/$ARMX/s$SEED; mkdir -p $D
cp runs/spike_steer/pilot7_$ARMX/train/finetune_s$SEED/history.jsonl runs/spike_steer/pilot7_$ARMX/train/finetune_s$SEED/spec_val_g*.json runs/spike_steer/pilot7_$ARMX/eval_s$SEED.json $D/
echo "done arm $ARMX seed $SEED; commit $D and push"
