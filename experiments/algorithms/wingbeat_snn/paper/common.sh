#!/usr/bin/env bash
# Shared settings for the paper campaign. Source this from the run_*.sh scripts.
#   PYTHON   interpreter with mujoco 3.8, numpy, torch, neat-python 2.0, gymnasium  (default: python)
#   WORKERS  simulator processes for evaluation stages                (default: nproc - 2)
#   QUICK=1  tiny smoke version of every stage, written to runs/wingbeat_paper/quick/
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO"
PYTHON="${PYTHON:-python}"
NPROC="$(nproc)"
WORKERS="${WORKERS:-$(( NPROC > 4 ? NPROC - 2 : 2 ))}"
ARM_WORKERS="${ARM_WORKERS:-$(( WORKERS / 3 > 0 ? WORKERS / 3 : 1 ))}"
RAW="${RAW:-$REPO/runs/wingbeat_paper}"
SCRATCH="${SCRATCH:-$RAW/scratch}"
OUT="${OUT:-$REPO/artifacts/wingbeat_paper}"
P1_RETUNE="$REPO/artifacts/wingbeat_snn_p1/retune.json"
H5_ROBUST="$REPO/artifacts/wingbeat_snn_h5/retune_robust.json"
PLL_TUNED_PILOT="$REPO/artifacts/wingbeat_snn_phase/long_run/pll_tuned.json"
EVAL=experiments.algorithms.wingbeat_snn.evaluate
TRAIN=experiments.algorithms.wingbeat_snn.phase_train

if [ "${QUICK:-0}" = 1 ]; then
  RAW="$REPO/runs/wingbeat_paper/quick"; SCRATCH="$RAW/scratch"; OUT="$RAW/out"
  E1_SEEDS=1; N_CURVE=1; CURVE_LEVELS="0 1.0"; N_LAG=1; LAGS="-30 0"; LAG_LEVELS="1.0"; N_ROB=1; ROB_LEVELS="1.0"
  N_ABL=1; ABL_LEVELS="1.0"; SEEDS="${SEEDS:-2}"; GRAD_STEPS=100; NEAT_GENS=2; NEAT_POP=8
  RES_GENS=1; RES_POP=4; RES_CASES=1; RES_SECONDS=1.6; N_SEEDCMP=1; SEEDCMP_LEVELS="1.0"; N_PHASECMP=1; PHASE_LEVELS="1.0"
  H5_LAGS="0 30"; N_H5=1; RECORD_SCALE=0.05
else
  RECORD_SCALE=1
  E1_SEEDS=16; N_CURVE=32; CURVE_LEVELS="0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0"
  N_LAG=16; LAGS="-90 -75 -60 -45 -30 -15 0 15 30 45 60 75 90"; LAG_LEVELS="1.0 1.25"
  N_ROB=16; ROB_LEVELS="1.0 1.25"; N_ABL=16; ABL_LEVELS="0 1.0 1.25 1.5"
  SEEDS="${SEEDS:-1 2 3 4 5}"; GRAD_STEPS=8000; NEAT_GENS=400; NEAT_POP=64
  RES_GENS=15; RES_POP=20; RES_CASES=4; RES_SECONDS=7; N_SEEDCMP=16; SEEDCMP_LEVELS="1.0 1.25 1.5 1.75"
  N_PHASECMP=16; PHASE_LEVELS="1.0 1.25 1.5"; H5_LAGS="-60 -30 0 30 60"; N_H5=8
fi
export PYTHON OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$RAW" "$SCRATCH" "$OUT"

# step NAME MARKER command...   runs the command unless MARKER already exists (non-empty); logs to $RAW/NAME.log
step() {
  local name="$1" marker="$2"; shift 2
  if [ -s "$marker" ]; then echo "[skip] $name ($marker exists)"; return 0; fi
  echo "[run ] $name  $(date -Is)"
  if "$@" > "$RAW/$name.log" 2>&1; then echo "[done] $name  $(date -Is)"
  else echo "[FAIL] $name  -> see $RAW/$name.log"; return 1; fi
}
echo "repo=$REPO python=$PYTHON nproc=$NPROC workers=$WORKERS arm_workers=$ARM_WORKERS quick=${QUICK:-0}"
echo "raw=$RAW out=$OUT"
