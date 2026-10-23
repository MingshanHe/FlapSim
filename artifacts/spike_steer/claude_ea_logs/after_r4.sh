#!/usr/bin/env bash
# claude-ea keep-busy queue after the round-4 searches of seeds 1-3 end (decided 2026-10-17 01:00 JST).
cd /root/flapsim && . .venv/bin/activate
log() { echo "[$(TZ=Asia/Tokyo date '+%m-%d %H:%M')] $*" >> logs/after_r4.log; }
while tmux has-session -t r4s1 2>/dev/null || tmux has-session -t r4s2 2>/dev/null || tmux has-session -t r4s3 2>/dev/null; do sleep 120; done
log "r4 searches 1-3 ended; fresh round-5 blind set + classical baseline"
python -W ignore -m experiments.algorithms.spike_steer._local.fresh_baseline --base 2_096_600_000 --round 5 --output runs/spike_steer/bench_r5 --workers 18 > logs/r5_baseline.log 2>&1
log "r5 baseline: $(grep normalisers logs/r5_baseline.log | head -1)"
until grep -q "seed 3 ablations done" logs/pipeline_r4.log && grep -q "seed 1 blind" logs/pipeline_r4.log && grep -q "seed 2 blind" logs/pipeline_r4.log; do sleep 120; done
EX=""
for S in 1 2 3 4; do G=$(python -c "import json;print(json.load(open('runs/spike_steer/bench_r4/reselect_r4_s$S.json'))['selected']['spec'])"); EX="$EX,r4_s${S}=$G"; done
EX=${EX#,}
log "head-to-head on claude-ea with final r4 seeds 1-4: $EX"
python -W ignore -m experiments.algorithms.spike_steer._local.compare_rounds --workers 18 --n 64 --extra "$EX" --out runs/spike_steer/compare_rounds_final > logs/compare_rounds_final.log 2>&1
log "head-to-head done"
