#!/usr/bin/env bash
# Phase estimators from raw IMU: train GRU, gradient SNN and NEAT SNN in parallel, then the offline test.
#   tmux new-session -d -s wingbeat_phase "bash experiments/algorithms/wingbeat_snn/run_phase.sh"
# Do not edit while a launched copy is running: bash reads scripts incrementally.
set -u
DATA=runs/wingbeat_snn/phase_data
OUT=runs/wingbeat_snn/phase_models
PYTHON="${PYTHON:-/home/robot/miniconda3/envs/mujoco/bin/python}"
cd "$(dirname "$0")/../../.."
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
M=experiments.algorithms.wingbeat_snn.phase_train
"$PYTHON" -W ignore -m $M gru --data $DATA --output $OUT --steps 2000 --threads 4 > $OUT/gru.log 2>&1 &
"$PYTHON" -W ignore -m $M snn_grad --data $DATA --output $OUT --steps 2000 --threads 4 > $OUT/snn_grad.log 2>&1 &
"$PYTHON" -W ignore -m $M snn_neat --data $DATA --output $OUT --generations 400 --population 64 --workers 10 \
  > $OUT/snn_neat.log 2>&1 &
wait
"$PYTHON" -W ignore -m $M offline_test --data $DATA --output $OUT > $OUT/offline_test.log 2>&1
echo "finished $(date -Is)" > "$OUT/DONE"
