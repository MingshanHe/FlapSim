#!/usr/bin/env bash
# After the pilots: retune B2+ on training cases, then compare everything on the held-out test split.
#   tmux new-session -d -s wingbeat_compare "bash experiments/algorithms/wingbeat_snn/run_compare.sh <scratch>"
# Waits for the v2 pilot to finish first. Do not edit while a launched copy is running.
set -u
SCRATCH="$1"
V1=runs/wingbeat_snn/pilot_20261012
V2=runs/wingbeat_snn/pilot_v2_20261012
OUT=artifacts/wingbeat_snn_p1
PYTHON="${PYTHON:-/home/robot/miniconda3/envs/mujoco/bin/python}"
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
while [ ! -f "$V2/DONE" ]; do sleep 30; done
mkdir -p "$OUT"
"$PYTHON" -W ignore -m experiments.algorithms.wingbeat_snn.evaluate retune --scratch "$SCRATCH" \
  --output "$OUT/retune.json" --workers 18 > "$OUT/retune.log" 2>&1
"$PYTHON" -W ignore -m experiments.algorithms.wingbeat_snn.evaluate compare --scratch "$SCRATCH" \
  --output "$OUT/compare_test.json" --workers 18 --retuned "$OUT/retune.json" \
  --levels 0 0.5 1.0 1.25 1.5 1.75 \
  --policy "v1_ann_neat=$V1/ann_neat_seed1/best_policy.json" \
  --policy "v2_snn_neat_delay=$V2/snn_neat_delay_seed1/best_policy.json" \
  --policy "v2_ann_neat=$V2/ann_neat_seed1/best_policy.json" \
  --policy "v2_snn_fixed=$V2/snn_fixed_seed1/best_policy.json" > "$OUT/compare_test.log" 2>&1
echo "finished $(date -Is)" > "$OUT/DONE"
