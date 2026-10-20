#!/usr/bin/env bash
# plan v8（preregistration_v8.json）：冻结重测 v7 正式轮的 PPO 控制器（选中中心 mlp_ppo 与 bc 起点 mlp_bc，种子 1–5），
# 全新盲测集 cases9(task, 2_098_500_000, 64)；classical 对手行在本机生成（test_opponent_v8.json），之后可用 rereport 换成 CloudPC-2 的。
# 用法：bash experiments/algorithms/ppo_direct/run_v8ppo.sh 1 2 3 4 5
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MACHINE=${MACHINE:-HomePC} V7PPO_PROFILE=v8
PY=${PY:-python}; NENV=${NENV:-18}; FROM=${FROM:-runs/ppo_direct/v7full}
M=experiments.algorithms.ppo_direct.v7ppo
log() { echo "[$(date -Iseconds)] $*"; }
ART=artifacts/spike_steer/ppo_direct/v8; mkdir -p "$ART"
for S in "$@"; do
  log "seed $S: v8 evaluate (frozen v7-full controllers from $FROM; blind 2_098_500_000, 64 per task)"
  $PY -W ignore -m $M evaluate --seed "$S" --from-dir "$FROM" --workers "$NENV" || exit 1
  cp runs/ppo_direct/v8/{evaluate_mlp_ppo_s$S.json,evaluate_mlp_bc_s$S.json} "$ART"/
  cp runs/ppo_direct/v8/test_opponent_v8.json "$ART/test_opponent_v8_homepc.json"
  log "seed $S: done"
done
log "all seeds done"
