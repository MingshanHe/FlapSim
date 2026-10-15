#!/usr/bin/env bash
# Plan v4 round 3 (gated descending layer + reflex envelope), any Linux machine: ES fine-tune of one seed on nod+dash with
# the penalised objective, then archive every validated centre. NO blind test here: selection (reselect_r3.py) and the
# blind test run on CloudPC only (the same spec scores differently across machines, measured in round 2).
# The round-3 directory (baseline with the fresh blind set, start specs, prereg) is copied from the committed artifacts,
# never re-prepared, so every machine uses the CloudPC opponents rows / normalisers.
# Usage (repo root, mujoco env active):  W=18 GENS=60 HOURS=9 PY=$PY bash artifacts/spike_steer/v4/handoff_s3/run_r3_seed.sh 2
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-python}
SEED=${1:?usage: run_r3_seed.sh SEED}
W=${W:?set W = cores minus 2}
GENS=${GENS:-60}
HOURS=${HOURS:-9}
OUT=runs/spike_steer/bench_r3
ART=artifacts/spike_steer/plan_v4/r3
M=experiments.algorithms.spike_steer.descend_r3
log() { echo "[$(date -Iseconds)] $*"; }
if [ ! -f "$OUT/train/baseline.json" ]; then
  log "copy round-3 directory from $ART"
  mkdir -p "$OUT/train/upset_v2" "$OUT/upset_v2"
  cp "$ART/baseline.json" "$OUT/" && cp "$ART/train/baseline.json" "$ART"/train/imitate_spec_s*.json "$OUT/train/" || exit 1
  cp "$ART/upset_v2/calibrate.json" "$OUT/upset_v2/" && cp "$ART/upset_v2/calibrate.json" "$OUT/train/upset_v2/" || exit 1
fi
[ -f "$OUT/train/imitate_spec_s$SEED.json" ] || { log "no start spec for seed $SEED"; exit 1; }
log "r3 finetune seed $SEED ($GENS generations, $W workers, envelope $($PY -c "import json;print(json.load(open('$OUT/train/baseline.json'))['prereg']['envelope_deg'])") deg)"
$PY -W ignore -m $M finetune --output "$OUT" --seed "$SEED" --generations "$GENS" --hours "$HOURS" --workers "$W" || exit 1
mkdir -p "$ART/finetune_s$SEED"
cp "$OUT"/train/finetune_s$SEED/{history.jsonl,best.json,spec_best.json,spec_last.json} "$OUT"/train/finetune_s$SEED/spec_val_g*.json "$ART/finetune_s$SEED/"
log "done seed $SEED; commit $ART/finetune_s$SEED and push (CloudPC selects + blind-tests)"
