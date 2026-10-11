#!/usr/bin/env bash
# P1 pilot: one seed per arm, three arms in parallel (6 simulator workers each).
# Launch detached so it survives the calling shell:
#   tmux new-session -d -s wingbeat_pilot "bash experiments/algorithms/wingbeat_snn/run_pilot.sh <scratch> <out>"
set -u
SCRATCH="$1"
OUT="${2:-runs/wingbeat_snn/pilot_20261012}"
PYTHON="${PYTHON:-/home/robot/miniconda3/envs/mujoco/bin/python}"
cd "$(dirname "$0")/../../.."
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for m in snn_neat_delay ann_neat snn_fixed; do
  "$PYTHON" -W ignore -m experiments.algorithms.wingbeat_snn.evolve --method "$m" --seed 1 --scratch "$SCRATCH" \
    --output "$OUT/${m}_seed1" --population 20 --generations 15 --cases 4 --seconds 7 \
    --levels 0 0.75 1.0 1.25 --validation-cases 8 --validation-levels 1.0 1.25 --validate-every 5 --workers 6 \
    > "$OUT/${m}_seed1.log" 2>&1 &
done
wait
echo "pilot finished $(date -Is)" > "$OUT/DONE"
