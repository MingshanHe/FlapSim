#!/usr/bin/env bash
# plan v11（v9 网络 + bench10 评分，descend12：每代每任务 8 例全覆盖、坠落感知排名、lr 每代衰减 0.85、16 代）种子 4、5 在 HomePC。
# 启动：tmux new-session -d -s v11 'bash runs/spike_steer/run_v11_homepc.sh > runs/spike_steer/v11_chain.log 2>&1; echo CHAIN_EXIT=$? >> runs/spike_steer/v11_chain.log'
set -u
cd /home/robot/FlapSim-repo/FlapSim-full || exit 1
source ~/miniconda3/etc/profile.d/conda.sh; conda activate mujoco
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
log() { echo "[$(date -Iseconds)] $*"; }
M="python -W ignore -m experiments.algorithms.spike_steer.descend12"
log "finetune seeds 4 5 in parallel (10 workers each)"
for s in 4 5; do $M finetune --seed $s --workers 10 > runs/spike_steer/v11/finetune_s${s}_v11.log 2>&1 & done; wait
log "finetune done"
for s in 4 5; do
  log "seed $s: reselect (64 per task)"
  $M reselect --seed $s --n-select 64 --workers 18 > runs/spike_steer/v11/reselect_s${s}_v11.log 2>&1 || { log "seed $s reselect FAILED"; exit 1; }
  log "seed $s: evaluate (blind 2_099_100_000, 64 per task)"
  $M evaluate --seed $s --n-test 64 --test-base 2099100000 --workers 18 --machine HomePC > runs/spike_steer/v11/evaluate_s${s}_v11.log 2>&1 || { log "seed $s evaluate FAILED"; exit 1; }
  log "seed $s: done"
done
log "all seeds done"
