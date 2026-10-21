#!/usr/bin/env bash
# plan v11 修正案 1：经典上界臂（descend_cls，28 个经典调度参数拿 v11 的 ES 预算，种子 2，全部核心），跑完后接着跑 v11 种子 4、5 链。
# 启动：tmux new-session -d -s v11cls 'bash runs/spike_steer/run_v11cls_homepc.sh > runs/spike_steer/v11cls_chain.log 2>&1; echo CHAIN_EXIT=$? >> runs/spike_steer/v11cls_chain.log'
set -u
cd /home/robot/FlapSim-repo/FlapSim-full || exit 1
source ~/miniconda3/etc/profile.d/conda.sh; conda activate mujoco
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
log() { echo "[$(date -Iseconds)] $*"; }
M="python -W ignore -m experiments.algorithms.spike_steer.descend_cls"
log "cls_es seed 2: finetune (18 workers)"
$M finetune --seed 2 --workers 18 > runs/spike_steer/v11/cls_es_s2.log 2>&1 || { log "cls_es finetune FAILED"; exit 1; }
log "cls_es seed 2: reselect"
$M reselect --seed 2 --workers 18 > runs/spike_steer/v11/cls_reselect_s2.log 2>&1 || { log "cls_es reselect FAILED"; exit 1; }
tail -n 1 runs/spike_steer/v11/cls_reselect_s2.log
log "cls_es seed 2: evaluate (blind 2_099_100_000, 64 per task)"
$M evaluate --seed 2 --workers 18 --machine HomePC > runs/spike_steer/v11/cls_evaluate_s2.log 2>&1 || { log "cls_es evaluate FAILED"; exit 1; }
tail -n 3 runs/spike_steer/v11/cls_evaluate_s2.log
log "cls_es seed 2: done"
log "starting v11 chain (seeds 4 5)"
bash runs/spike_steer/run_v11_homepc.sh > runs/spike_steer/v11_chain.log 2>&1
echo "CHAIN_EXIT=$?" >> runs/spike_steer/v11_chain.log
log "v11 chain finished: $(tail -n 1 runs/spike_steer/v11_chain.log)"
