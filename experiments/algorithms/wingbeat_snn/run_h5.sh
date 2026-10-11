#!/usr/bin/env bash
# H5: unknown actuator phase lag. Robust hand tuning, lag-randomized residual training, held-out lag scan.
#   tmux new-session -d -s wingbeat_h5 "bash experiments/algorithms/wingbeat_snn/run_h5.sh <scratch>"
# Do not edit while a launched copy is running: bash reads scripts incrementally.
set -u
SCRATCH="$1"
OUT=artifacts/wingbeat_snn_h5
RUNS=runs/wingbeat_snn/h5_20261012
LAG=60
PYTHON="${PYTHON:-/home/robot/miniconda3/envs/mujoco/bin/python}"
cd "$(dirname "$0")/../../.."
mkdir -p "$OUT" "$RUNS"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# 1. The hand law tuned on the same kind of lag-randomized training cases (fair robust baseline).
"$PYTHON" -W ignore -m experiments.algorithms.wingbeat_snn.evaluate retune --scratch "$SCRATCH" --workers 18 \
  --lag-range $LAG --retune-peaks 1.0 1.25 --output "$OUT/retune_robust.json" > "$OUT/retune_robust.log" 2>&1
# 2. Residual networks on the P1-retuned law, trained with the same lag randomization.
for m in snn_neat_delay ann_neat snn_fixed; do
  "$PYTHON" -W ignore -m experiments.algorithms.wingbeat_snn.evolve --method "$m" --seed 1 --scratch "$SCRATCH" \
    --output "$RUNS/${m}_seed1" --population 20 --generations 15 --cases 4 --seconds 7 \
    --residual --residual-base artifacts/wingbeat_snn_p1/retune.json --readout-std 0.3 --spike-scale 5 \
    --levels 0.75 1.0 1.25 --validation-cases 8 --validation-levels 1.0 1.25 --lag-range $LAG \
    --validate-every 5 --workers 6 > "$RUNS/${m}_seed1.log" 2>&1 &
done
wait
# 3. Held-out lag scan.
"$PYTHON" -W ignore -m experiments.algorithms.wingbeat_snn.evaluate lag_scan --scratch "$SCRATCH" --workers 18 \
  --retuned artifacts/wingbeat_snn_p1/retune.json --retuned "$OUT/retune_robust.json" \
  --levels 1.0 1.25 --lags -60 -45 -30 0 30 45 60 \
  --policy "h5_snn_neat_delay=$RUNS/snn_neat_delay_seed1/best_policy.json" \
  --policy "h5_ann_neat=$RUNS/ann_neat_seed1/best_policy.json" \
  --policy "h5_snn_fixed=$RUNS/snn_fixed_seed1/best_policy.json" \
  --output "$OUT/lag_scan_trained.json" > "$OUT/lag_scan_trained.log" 2>&1
echo "finished $(date -Is)" > "$OUT/DONE"
