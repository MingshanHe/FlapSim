#!/usr/bin/env bash
# Round 5 on claude-ea: selection + blind test + ablations for every seed as soon as its search is archived.
cd /root/flapsim && . .venv/bin/activate
log() { echo "[$(TZ=Asia/Tokyo date '+%m-%d %H:%M')] $*" >> logs/pipeline_r5.log; }
O=runs/spike_steer/bench_r5; A=artifacts/spike_steer/plan_v4/r5; M=experiments.algorithms.spike_steer.descend_r5
done_s=""
evalseed() {
  local S=$1
  rm -rf $O/train/sel_s$S && cp -r $A/finetune_s$S $O/train/sel_s$S
  python -W ignore -m $M reselect --output $O --seed $S --spec $O/train/sel_s$S --tag r5_s$S --workers 10 > logs/reselect_r5_s$S.log 2>&1 || { log "reselect s$S failed"; return; }
  local G=$(python -c "import json;print(json.load(open('$O/reselect_r5_s$S.json'))['selected']['spec'])")
  log "seed $S selected $G"
  python -W ignore -m $M evaluate --output $O --seed $S --spec $G --tag r5_s$S --variants intact --workers 10 > logs/blind_r5_s$S.log 2>&1
  log "seed $S blind: $(grep composite logs/blind_r5_s$S.log | tail -1)"
  python -W ignore -m $M evaluate --output $O --seed $S --spec $G --tag r5_s${S}_abl --variants clamp_fast,nodelay_all,noref,nopulse --workers 10 > logs/blind_r5_s${S}_abl.log 2>&1
  log "seed $S ablations done"
}
while [ $(echo $done_s | wc -w) -lt 5 ]; do
  for S in 1 2 3 4 5; do
    case " $done_s " in *" $S "*) continue;; esac
    if [ $S -ge 4 ]; then
      git fetch -q origin && git ls-tree -d origin/research/original-env-snn-neat $A/finetune_s$S | grep -q . || continue
      git pull -q --rebase --autostash origin research/original-env-snn-neat || { log "pull failed"; continue; }
    fi
    [ -f $A/finetune_s$S/spec_last.json ] || continue
    log "seed $S search archived"; evalseed $S; done_s="$done_s $S"
  done
  sleep 300
done
log "pipeline done"
