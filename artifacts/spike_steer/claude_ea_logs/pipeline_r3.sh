#!/usr/bin/env bash
# Round-3 scheduling after seed 1's search (claude-ea): seeds 3+4 searches in parallel (15 workers each);
# seeds 1 and 5: selection + blind test + ablations on the claude-ea baseline (8 workers).
cd /root/flapsim && . .venv/bin/activate
log() { echo "[$(TZ=Asia/Tokyo date '+%m-%d %H:%M')] $*" >> logs/pipeline_r3.log; }
L=runs/spike_steer/bench_r3_local; M=experiments.algorithms.spike_steer
while tmux has-session -t r3q 2>/dev/null; do sleep 60; done
log "queue ended: $(tr '\n' ' ' < logs/r3_queue.done)"
for s in 3 4; do mv runs/spike_steer/bench_r3/train/imitate_spec_s$s.json.queuehold runs/spike_steer/bench_r3/train/imitate_spec_s$s.json; done
for s in 3 4; do
  tmux new -d -s r3s$s "cd /root/flapsim && . .venv/bin/activate && W=15 GENS=60 HOURS=12 PY=python bash artifacts/spike_steer/v4/handoff_s3/run_r3_seed.sh $s > logs/r3_s$s.log 2>&1"
done
log "seeds 3, 4 searches started (15 workers each)"
evalseed() {  # $1 seed, $2 centres dir
  local S=$1
  rm -rf $L/train/finetune_s$S && cp -r $2 $L/train/finetune_s$S
  python -W ignore -m $M.reselect_r3 --seed $S --output $L --centres $L/train/finetune_s$S --tag r3_s$S --workers 8 > logs/reselect_r3_s$S.log 2>&1 || { log "reselect s$S failed"; return 1; }
  local G=$(python -c "import json;print(json.load(open('$L/reselect_r3_s$S.json'))['selected']['spec'])")
  log "seed $S selected $G"
  python -W ignore -m $M.descend_r3 evaluate --output $L --seed $S --spec $G --tag r3_s$S --variants intact --workers 8 > logs/blind_r3_s$S.log 2>&1
  log "seed $S blind: $(grep composite logs/blind_r3_s$S.log | tail -1)"
  python -W ignore -m $M.descend_r3 evaluate --output $L --seed $S --spec $G --tag r3_s${S}_abl --variants clamp_fast,nodelay_all,noref,nopulse --workers 8 > logs/blind_r3_s${S}_abl.log 2>&1
  log "seed $S ablations done"
}
grep -q "seed 1 exit 0" logs/r3_queue.done && evalseed 1 artifacts/spike_steer/plan_v4/r3/finetune_s1 || log "seed 1 search did not finish cleanly"
until git fetch -q origin && git ls-tree -d origin/research/original-env-snn-neat artifacts/spike_steer/plan_v4/r3/finetune_s5 | grep -q .; do sleep 300; done
git pull -q --rebase --autostash origin research/original-env-snn-neat && log "seed 5 pulled $(git log --oneline -1 | cut -c1-60)"
evalseed 5 artifacts/spike_steer/plan_v4/r3/finetune_s5
log "pipeline done"
