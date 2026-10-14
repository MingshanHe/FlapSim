#!/usr/bin/env bash
# HomePC hand-off: ES fine-tune of imitation seed 3 + blind/ablation evaluation (research plan v4, stage S3/S4).
# Inputs (this directory) were produced on the 32-core machine: baseline.json (S1 opponents + test rows),
# preregistration.json, dash_target.json, upset_v2/calibrate.json (T4 levels), imitate_spec_s3.json (S2 imitation).
# Usage (repo root, mujoco env active):  W=12 bash artifacts/spike_steer/v4/handoff_s3/run_homepc_s3.sh
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-python}
W=${W:-$(( $(nproc) - 2 ))}
OUT=runs/spike_steer/bench
H=artifacts/spike_steer/v4/handoff_s3
mkdir -p "$OUT/upset_v2"
cp -n "$H"/baseline.json "$H"/preregistration.json "$H"/dash_target.json "$H"/imitate_spec_s3.json "$OUT"/
cp -n "$H"/upset_v2/calibrate.json "$OUT/upset_v2/"
log() { echo "[$(date -Is)] $*"; }
log "finetune seed 3 with $W workers"
$PY -W ignore -m experiments.algorithms.spike_steer.descend finetune --output "$OUT" --seed 3 --generations 20 --hours 8 --workers "$W" || exit 1
log "evaluate ft_s3 (5 variants)"
$PY -W ignore -m experiments.algorithms.spike_steer.bench evaluate --output "$OUT" --spec "$OUT/finetune_s3/spec_best.json" --tag ft_s3 \
    --variants intact,clamp_fast,nodelay_all,noref,nopulse --workers "$W" || exit 1
ART=artifacts/spike_steer/plan_v4; mkdir -p "$ART/finetune_s3"
cp "$OUT"/evaluate_ft_s3.json "$ART"/ && cp "$OUT"/finetune_s3/{history.jsonl,best.json,spec_best.json,spec_last.json} "$OUT"/finetune_s3/spec_val_g*.json "$ART/finetune_s3/"
log "done; commit artifacts/spike_steer/v4/{evaluate_ft_s3.json,finetune_s3} and push"
