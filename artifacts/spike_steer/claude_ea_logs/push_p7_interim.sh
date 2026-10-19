#!/usr/bin/env bash
# Push new pilot-7 arm-B interim centres (seeds 1, 4) every 20 min for the LabPC preview; exits when both searches end.
cd /root/flapsim
while true; do
  new=0
  for s in 1 4; do
    d=artifacts/spike_steer/pilot7/interim/B/s$s; mkdir -p $d
    for f in runs/spike_steer/pilot7_B/train/finetune_s$s/spec_val_g*.json; do
      [ -f "$f" ] || continue; b=$(basename $f); [ -f $d/$b ] || { cp $f $d/$b; new=1; }
    done
  done
  if [ $new = 1 ]; then
    git add artifacts/spike_steer/pilot7/interim && git commit -q -m "pilot7 interim centres (arm B, claude-ea) for the LabPC preview; not results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XJCwSxuAndV8x1hbQ9Lmq2" && git pull -q --rebase origin research/original-env-snn-neat && git push -q origin HEAD && echo "[$(TZ=Asia/Tokyo date '+%m-%d %H:%M')] pushed $(git log --oneline -1 | cut -c1-8)" >> logs/push_p7_interim.log
  fi
  tmux has-session -t p7B1 2>/dev/null || tmux has-session -t p7B4 2>/dev/null || exit 0
  sleep 1200
done
