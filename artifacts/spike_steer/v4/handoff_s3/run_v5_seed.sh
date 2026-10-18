#!/usr/bin/env bash
# Plan v5 round 1 (safe ES: round-5 configuration + per-generation safety check with revert), any Linux machine:
# search of one seed, then archive every validated centre and the safety log. NO selection / blind test here (claude-ea).
# Usage (repo root, env active):  W=30 GENS=40 PY=python bash artifacts/spike_steer/v4/handoff_s3/run_v5_seed.sh 1
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-python}
SEED=${1:?usage: run_v5_seed.sh SEED}
W=${W:?set W}
GENS=${GENS:-40}
OUT=runs/spike_steer/bench_v5
ART=artifacts/spike_steer/plan_v5/r1
M=experiments.algorithms.spike_steer.descend_safe
log() { echo "[$(date -Iseconds)] $*"; }
$PY -W ignore -m $M selftest || exit 1
if [ ! -f "$OUT/train/imitate_spec_s$SEED.json" ]; then
  log "copy plan-v5 round-1 directory from $ART"
  mkdir -p "$OUT/train/upset_v2" "$OUT/upset_v2"
  cp "$ART/claude_ea/baseline.json" "$OUT/" && cp "$ART/train/baseline.json" "$ART"/train/imitate_spec_s*.json "$OUT/train/" || exit 1
  cp artifacts/spike_steer/plan_v4/r3/upset_v2/calibrate.json "$OUT/upset_v2/" && cp artifacts/spike_steer/plan_v4/r3/upset_v2/calibrate.json "$OUT/train/upset_v2/" || exit 1
fi
log "v5 safe-ES seed $SEED ($GENS generations, $W workers)"
$PY -W ignore -m $M finetune --output "$OUT" --seed "$SEED" --generations "$GENS" --workers "$W" || exit 1
mkdir -p "$ART/finetune_s$SEED"
cp "$OUT"/train/finetune_s$SEED/{history.jsonl,safety.jsonl,best.json,spec_best.json,spec_last.json} "$OUT"/train/finetune_s$SEED/spec_val_g*.json "$ART/finetune_s$SEED/"
log "done seed $SEED; commit $ART/finetune_s$SEED and push (claude-ea selects + blind-tests)"
