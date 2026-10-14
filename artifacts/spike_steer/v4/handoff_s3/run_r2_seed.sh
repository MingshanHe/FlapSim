#!/usr/bin/env bash
# Plan v4 round 2 (command-gated descending layer), any Linux machine: ES fine-tune of one seed on nod+dash, then the
# full 4-task blind test with 5 ablation variants. Inputs come from artifacts/spike_steer/plan_v4 (or runs/spike_steer/bench).
# Usage (repo root, mujoco env active):  W=18 GENS=60 bash artifacts/spike_steer/v4/handoff_s3/run_r2_seed.sh 2
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-python}
SEED=${1:?usage: run_r2_seed.sh SEED}
W=${W:?set W = cores minus 2}
GENS=${GENS:-60}
HOURS=${HOURS:-8}
OUT=runs/spike_steer/bench_r2
M=experiments.algorithms.spike_steer.descend_gated
log() { echo "[$(date -Iseconds)] $*"; }
[ -f "$OUT/baseline.json" ] || { log "prepare"; $PY -W ignore -m $M prepare --output "$OUT" --base-law law_tuned || exit 1; }
log "r2 finetune seed $SEED ($GENS generations, $W workers)"
$PY -W ignore -m $M finetune --output "$OUT" --seed "$SEED" --generations "$GENS" --hours "$HOURS" --workers "$W" || exit 1
log "r2 evaluate seed $SEED"
$PY -W ignore -m $M evaluate --output "$OUT" --seed "$SEED" --variants intact,clamp_fast,nodelay_all,noref,nopulse --workers "$W" || exit 1
ART=artifacts/spike_steer/plan_v4/r2; mkdir -p "$ART/finetune_s$SEED"
cp "$OUT"/preregistration_r2.json "$OUT"/evaluate_r2_s$SEED.json "$ART"/
cp "$OUT"/train/finetune_s$SEED/{history.jsonl,best.json,spec_best.json,spec_last.json} "$OUT"/train/finetune_s$SEED/spec_val_g*.json "$ART/finetune_s$SEED/"
log "done; commit artifacts/spike_steer/plan_v4/r2 and push"
