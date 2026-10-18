#!/usr/bin/env bash
# Plan v4 round 6 (FINAL: round 5 + 16+16 ES cases per candidate, 30 generations), any Linux machine: ES fine-tune of one seed on
# nod+dash (nod:dash 2:1, 16+16 ES cases), then archive every validated centre. NO selection / blind test here: both run on
# claude-ea only (round-3 amendment 1). The round-6 directory (claude-ea baseline with the fresh blind set, start specs)
# is copied from the committed artifacts, never re-prepared.
# Usage (repo root, mujoco env active):  W=18 GENS=30 HOURS=40 PY=$PY bash artifacts/spike_steer/v4/handoff_s3/run_r6_seed.sh 4
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-python}
SEED=${1:?usage: run_r6_seed.sh SEED}
W=${W:?set W}
GENS=${GENS:-30}
HOURS=${HOURS:-40}
OUT=runs/spike_steer/bench_r6
ART=artifacts/spike_steer/plan_v4/r6
M=experiments.algorithms.spike_steer.descend_r6
log() { echo "[$(date -Iseconds)] $*"; }
$PY -W ignore -m $M selftest || exit 1
if [ ! -f "$OUT/train/imitate_spec_s$SEED.json" ]; then
  log "copy round-6 directory from $ART"
  mkdir -p "$OUT/train/upset_v2" "$OUT/upset_v2"
  cp "$ART/claude_ea/baseline.json" "$OUT/" && cp "$ART/train/baseline.json" "$ART"/train/imitate_spec_s*.json "$OUT/train/" || exit 1
  cp artifacts/spike_steer/plan_v4/r3/upset_v2/calibrate.json "$OUT/upset_v2/" && cp artifacts/spike_steer/plan_v4/r3/upset_v2/calibrate.json "$OUT/train/upset_v2/" || exit 1
fi
log "r6 finetune seed $SEED ($GENS generations, $W workers, refractory $($PY -c "import json;print(json.load(open('$OUT/train/baseline.json'))['prereg']['refractory_s'])") s)"
$PY -W ignore -m $M finetune --output "$OUT" --seed "$SEED" --generations "$GENS" --hours "$HOURS" --workers "$W" || exit 1
mkdir -p "$ART/finetune_s$SEED"
cp "$OUT"/train/finetune_s$SEED/{history.jsonl,best.json,spec_best.json,spec_last.json} "$OUT"/train/finetune_s$SEED/spec_val_g*.json "$ART/finetune_s$SEED/"
log "done seed $SEED; commit $ART/finetune_s$SEED and push (claude-ea selects + blind-tests)"
