#!/usr/bin/env bash
# After round 6 completes: fresh blind set for M1, push the pre-registration, run the confirmatory re-test once.
cd /root/flapsim && . .venv/bin/activate
log() { echo "[$(TZ=Asia/Tokyo date '+%m-%d %H:%M')] $*" >> logs/chain_m1.log; }
# round 6 paused by the user (SIGSTOP) -- run now
log "round 6 paused; fresh blind set 2_096_900_000"
python -W ignore -m experiments.algorithms.spike_steer._local.fresh_baseline --base 2_096_900_000 --round 0 --output runs/spike_steer/bench_m1 --workers 30 > logs/m1_baseline.log 2>&1 || { log "baseline failed"; exit 1; }
mkdir -p artifacts/spike_steer/confirm_m1/claude_ea && cp runs/spike_steer/bench_m1/baseline.json logs/m1_baseline.log artifacts/spike_steer/confirm_m1/claude_ea/
python -W ignore -m experiments.algorithms.spike_steer.confirm_m1 selftest >> logs/chain_m1.log 2>&1 || { log "selftest failed"; exit 1; }
git add artifacts/spike_steer/confirm_m1 experiments/algorithms/spike_steer/confirm_m1.py experiments/algorithms/spike_steer/_local/fresh_baseline.py && git commit -q -m "confirmatory re-test M1 pre-registration, code and fresh blind-set baseline (before any learned controller is run on it)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XJCwSxuAndV8x1hbQ9Lmq2" && git pull -q --rebase origin research/original-env-snn-neat && git push -q origin HEAD && log "prereg pushed $(git log --oneline -1 | cut -c1-12)"
python -W ignore -m experiments.algorithms.spike_steer.confirm_m1 run --workers 30 > logs/m1_run.log 2>&1
log "M1 run done: $(tail -n 1 logs/m1_run.log)"
