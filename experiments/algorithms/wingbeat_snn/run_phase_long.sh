#!/usr/bin/env bash
# Longer gradient training (8000 steps) for the GRU and two SNN variants, offline test, then closed-loop flights
# with the retuned hand law using each phase source.
#   tmux new-session -d -s wingbeat_phase2 "bash experiments/algorithms/wingbeat_snn/run_phase_long.sh <scratch>"
# Do not edit while a launched copy is running: bash reads scripts incrementally.
set -u
SCRATCH="$1"
DATA=runs/wingbeat_snn/phase_data
OUT=runs/wingbeat_snn/phase_models_long
PYTHON="${PYTHON:-/home/robot/miniconda3/envs/mujoco/bin/python}"
cd "$(dirname "$0")/../../.."
mkdir -p "$OUT"
cp runs/wingbeat_snn/phase_models/snn_neat.json "$OUT/snn_neat.json"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
M=experiments.algorithms.wingbeat_snn.phase_train
"$PYTHON" -W ignore -m $M gru --data $DATA --output $OUT --steps 8000 --threads 5 --tag _long > $OUT/gru_long.log 2>&1 &
"$PYTHON" -W ignore -m $M snn_grad --data $DATA --output $OUT --steps 8000 --threads 5 --tag _h24 \
  > $OUT/snn_grad_h24.log 2>&1 &
"$PYTHON" -W ignore -m $M snn_grad --data $DATA --output $OUT --steps 8000 --threads 5 --tag _h48fast \
  --hidden 48 --tau-m 0.005 --tau-s 0.0025 > $OUT/snn_grad_h48fast.log 2>&1 &
wait
"$PYTHON" -W ignore -m $M offline_test --data $DATA --output $OUT > $OUT/offline_test.log 2>&1
"$PYTHON" -W ignore -m experiments.algorithms.wingbeat_snn.evaluate phase_closed_loop --scratch "$SCRATCH" \
  --workers 18 --retuned artifacts/wingbeat_snn_p1/retune.json --levels 0 1.0 1.25 1.5 --count 8 \
  --estimator "gru=$OUT/gru_long.pt" --estimator "snn_grad_h24=$OUT/snn_grad_h24.json" \
  --estimator "snn_grad_h48fast=$OUT/snn_grad_h48fast.json" --estimator "snn_neat=$OUT/snn_neat.json" \
  --output "$OUT/closed_loop.json" > "$OUT/closed_loop.log" 2>&1
echo "finished $(date -Is)" > "$OUT/DONE"
