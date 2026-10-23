#!/usr/bin/env bash
# Push new round-6 interim ES centres of claude-ea seeds 1-3 every 20 min (for the LabHPPC selection preview).
cd /root/flapsim
while true; do
  new=0
  for s in 1 2 3; do
    d=artifacts/spike_steer/plan_v4/r6/interim/s$s; mkdir -p $d
    for f in runs/spike_steer/bench_r6/train/finetune_s$s/spec_val_g*.json; do
      [ -f "$f" ] || continue; b=$(basename $f); [ -f $d/$b ] || { cp $f $d/$b; new=1; }
    done
  done
  if [ $new = 1 ]; then
    git add artifacts/spike_steer/plan_v4/r6/interim && git commit -q -m "r6 interim centres (claude-ea seeds 1-3) for the selection preview; not results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XJCwSxuAndV8x1hbQ9Lmq2" && git pull -q --rebase origin research/original-env-snn-neat && git push -q origin HEAD && echo "[$(TZ=Asia/Tokyo date '+%m-%d %H:%M')] pushed $(git log --oneline -1)" >> logs/push_r6_interim.log
  fi
  for s in 1 2 3; do tmux has-session -t r6s$s 2>/dev/null && break; [ $s = 3 ] && exit 0; done
  sleep 1200
done
