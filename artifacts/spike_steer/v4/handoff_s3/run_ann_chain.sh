#!/usr/bin/env bash
# Hand-off runner (any machine, incl. macOS): H4 control = rate-coded ANN of the same size through the same pipeline.
#   collect teacher traces (deterministic seeds) -> ANN imitation x3 seeds -> blind test x3 -> ES fine-tune seed 1
#   (GENS generations, default 30 ~ what the SNN seed 1 reached in 4.5 h) -> blind test + 5 ablation variants.
# Usage (repo root, mujoco env active):  W=8 GENS=30 ANN_SEED=1 bash artifacts/spike_steer/v4/handoff_s3/run_ann_chain.sh
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-python}
W=${W:?set W = number of cores minus 2}
GENS=${GENS:-30}
S=${ANN_SEED:-1}        # which ANN seed this machine fine-tunes (imitation always covers seeds 1-3)
OUT=runs/spike_steer/bench
H=artifacts/spike_steer/v4/handoff_s3
M=experiments.algorithms.spike_steer.descend_ann
mkdir -p "$OUT/upset_v2"
cp -n "$H"/baseline.json "$H"/preregistration.json "$H"/dash_target.json "$OUT"/ 2>/dev/null
cp -n "$H"/upset_v2/calibrate.json "$OUT/upset_v2/" 2>/dev/null
log() { echo "[$(date -Is)] $*"; }
stage() {
  local name=$1; shift
  if [ -f "$OUT/.done_$name" ]; then log "skip $name"; return 0; fi
  log "start $name: $*"
  if "$@"; then touch "$OUT/.done_$name"; log "done $name"; return 0; fi
  log "FAILED $name"; return 1
}
stage collect $PY -W ignore -m experiments.algorithms.spike_steer.descend collect --output "$OUT" --workers "$W" || exit 1
if [ ! -f "$OUT/.done_train_ann" ]; then
  log "start train_ann (3 seeds in parallel)"
  for s in 1 2 3; do $PY -W ignore -m $M train --output "$OUT" --seed "$s" --epochs 40 > "$OUT/train_ann_s$s.log" 2>&1 & done
  wait
  for s in 1 2 3; do tail -1 "$OUT/train_ann_s$s.log"; done
  [ -f "$OUT/imitate_spec_s1_ann.json" ] && [ -f "$OUT/imitate_spec_s2_ann.json" ] && [ -f "$OUT/imitate_spec_s3_ann.json" ] && touch "$OUT/.done_train_ann" || { log "FAILED train_ann"; exit 1; }
fi
for s in 1 2 3; do
  stage "eval_imit_ann_s$s" $PY -W ignore -m $M evaluate --output "$OUT" --spec "$OUT/imitate_spec_s${s}_ann.json" --tag "imit_ann_s$s" --workers "$W" || exit 1
done
stage finetune_ann_s$S $PY -W ignore -m $M finetune --output "$OUT" --seed $S --generations "$GENS" --hours 12 --workers "$W" || exit 1
stage eval_ft_ann_s$S $PY -W ignore -m $M evaluate --output "$OUT" --spec "$OUT/finetune_s${S}_ann/spec_best.json" --tag ft_ann_s$S \
    --variants intact,clamp_fast,nodelay_all,noref,nopulse --workers "$W" || exit 1
ART=artifacts/spike_steer/v4; mkdir -p "$ART/finetune_s${S}_ann"
cp "$OUT"/evaluate_imit_ann_s?.json "$OUT"/evaluate_ft_ann_s$S.json "$OUT"/imitate_spec_s?_ann.json "$OUT"/imitate_history_s?_ann.json "$ART"/
cp "$OUT"/finetune_s${S}_ann/{history.jsonl,best.json,spec_best.json,spec_last.json} "$OUT"/finetune_s${S}_ann/spec_val_g*.json "$ART/finetune_s${S}_ann/"
log "done; commit artifacts/spike_steer/v4/*ann* and push"
