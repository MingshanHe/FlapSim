#!/usr/bin/env bash
# Plan v4 round 4 (gated descending layer + wingbeat refractory period), any Linux machine: ES fine-tune of one seed on
# nod+dash (penalty from 60 deg x0.3), then archive every validated centre. NO selection / blind test here: both run on
# claude-ea only (round-3 amendment 1). The round-4 directory (claude-ea baseline with the fresh blind set, start specs)
# is copied from the committed artifacts, never re-prepared.
# Usage (repo root, mujoco env active):  W=18 GENS=60 HOURS=14 PY=$PY bash artifacts/spike_steer/v4/handoff_s3/run_r4_seed.sh 4
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-python}
SEED=${1:?usage: run_r4_seed.sh SEED}
W=${W:?set W}
GENS=${GENS:-60}
HOURS=${HOURS:-14}
OUT=runs/spike_steer/bench_r4
ART=artifacts/spike_steer/plan_v4/r4
M=experiments.algorithms.spike_steer.descend_r4
log() { echo "[$(date -Iseconds)] $*"; }
$PY -W ignore -m $M selftest || exit 1
if [ ! -f "$OUT/train/imitate_spec_s$SEED.json" ]; then
  log "copy round-4 directory from $ART"
  mkdir -p "$OUT/train/upset_v2" "$OUT/upset_v2"
  cp "$ART/claude_ea/baseline.json" "$OUT/" && cp "$ART/train/baseline.json" "$ART"/train/imitate_spec_s*.json "$OUT/train/" || exit 1
  cp artifacts/spike_steer/plan_v4/r3/upset_v2/calibrate.json "$OUT/upset_v2/" && cp artifacts/spike_steer/plan_v4/r3/upset_v2/calibrate.json "$OUT/train/upset_v2/" || exit 1
fi
log "r4 finetune seed $SEED ($GENS generations, $W workers, refractory $($PY -c "import json;print(json.load(open('$OUT/train/baseline.json'))['prereg']['refractory_s'])") s)"
$PY -W ignore -m $M finetune --output "$OUT" --seed "$SEED" --generations "$GENS" --hours "$HOURS" --workers "$W" || exit 1
mkdir -p "$ART/finetune_s$SEED"
cp "$OUT"/train/finetune_s$SEED/{history.jsonl,best.json,spec_best.json,spec_last.json} "$OUT"/train/finetune_s$SEED/spec_val_g*.json "$ART/finetune_s$SEED/"
log "done seed $SEED; commit $ART/finetune_s$SEED and push (claude-ea selects + blind-tests)"
