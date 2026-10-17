#!/usr/bin/env bash
# Research plan v4 overnight chain (32-core machine):
#   S0 T4 calibration (upset_v2) -> S1 classical opponents + preregistration (bench classical)
#   -> S2 teacher traces + imitation x3 seeds + blind evaluation -> S3 policy-search fine-tune (seed 1, then 2, 3)
#   -> S4 blind test + ablations. Stages are skipped when their .done marker exists (safe to re-run).
set -u
cd ~/FlapSim-repo/FlapSim || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/root/miniconda3/envs/mujoco/bin/python
OUT=${OUT:-runs/spike_steer/bench}
W=${W:-30}
Q=${QUICK:-}            # QUICK=--quick for the smoke chain
EPOCHS=${EPOCHS:-40}
GENS=${GENS:-40}
mkdir -p "$OUT"
log() { echo "[$(date -Is)] $*"; }
stage() {
  local name=$1; shift
  if [ -f "$OUT/.done_$name" ]; then log "skip $name"; return 0; fi
  log "start $name: $*"
  if "$@"; then touch "$OUT/.done_$name"; log "done $name"; return 0; fi
  log "FAILED $name"; return 1
}
[ -z "$Q" ] && while tmux has-session -t s0_upset 2>/dev/null; do sleep 60; done
stage upset $PY -W ignore -m experiments.algorithms.spike_steer.upset_v2 all --workers "$W" --output "$OUT/upset_v2" $Q || exit 1
stage classical $PY -W ignore -m experiments.algorithms.spike_steer.bench classical --output "$OUT" --workers "$W" $Q ${DASHM:+--dash-m $DASHM} || exit 1
stage collect $PY -W ignore -m experiments.algorithms.spike_steer.descend collect --output "$OUT" --workers "$W" $Q || exit 1
if [ ! -f "$OUT/.done_train" ]; then
  log "start train (3 seeds in parallel)"
  for s in 1 2 3; do
    $PY -W ignore -m experiments.algorithms.spike_steer.descend train --output "$OUT" --seed "$s" --epochs "$EPOCHS" $Q > "$OUT/train_s$s.log" 2>&1 &
  done
  wait
  for s in 1 2 3; do tail -2 "$OUT/train_s$s.log"; done
  [ -f "$OUT/imitate_spec_s1.json" ] && [ -f "$OUT/imitate_spec_s2.json" ] && [ -f "$OUT/imitate_spec_s3.json" ] && touch "$OUT/.done_train" || { log "FAILED train"; exit 1; }
  log "done train"
fi
for s in 1 2 3; do
  stage "eval_imit_s$s" $PY -W ignore -m experiments.algorithms.spike_steer.bench evaluate --output "$OUT" --spec "$OUT/imitate_spec_s$s.json" --tag "imit_s$s" --workers "$W" $Q || exit 1
done
for s in 1 2 3; do
  HOURS=4.5; [ "$s" != 1 ] && HOURS=3
  stage "finetune_s$s" $PY -W ignore -m experiments.algorithms.spike_steer.descend finetune --output "$OUT" --seed "$s" --generations "$GENS" --hours "$HOURS" --workers "$W" $Q || exit 1
  stage "eval_ft_s$s" $PY -W ignore -m experiments.algorithms.spike_steer.bench evaluate --output "$OUT" --spec "$OUT/finetune_s$s/spec_best.json" --tag "ft_s$s" \
      --variants intact,clamp_fast,nodelay_all,noref,nopulse --workers "$W" $Q || exit 1
done
[ -n "$Q" ] && { log "quick chain finished"; exit 0; }
ART=artifacts/spike_steer/v4
mkdir -p "$ART/upset_v2"
cp "$OUT"/preregistration.json "$OUT"/dash_target.json "$OUT"/baseline.json "$OUT"/evaluate_*.json "$OUT"/imitate_spec_s?.json "$OUT"/imitate_history_s?.json "$ART"/ 2>/dev/null
cp "$OUT"/upset_v2/calibrate.json "$OUT"/upset_v2/tune.json "$OUT"/upset_v2/test.json "$OUT"/upset_v2/preregistration.json "$ART/upset_v2/" 2>/dev/null
for s in 1 2 3; do [ -d "$OUT/finetune_s$s" ] && mkdir -p "$ART/finetune_s$s" && cp "$OUT/finetune_s$s"/history.jsonl "$OUT/finetune_s$s"/best.json "$OUT/finetune_s$s"/spec_best.json "$ART/finetune_s$s/" 2>/dev/null; done
cp runs/spike_steer/maneuver_v2/preregistration.json "$ART"/maneuver_v2_preregistration.json 2>/dev/null
$PY - <<'EOF'
import json
for t in ('nod','dash'):
    d=json.load(open(f'runs/spike_steer/maneuver_v2/{t}.json')); d.pop('test_rows',None)
    json.dump(d, open(f'artifacts/spike_steer/v4/maneuver_v2_{t}.json','w'), indent=1)
EOF
log "chain finished"
