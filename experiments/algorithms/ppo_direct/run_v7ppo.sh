#!/usr/bin/env bash
# plan v7 快速验证的 PPO 参考臂，按种子依次：bc -> finetune（预算按回合数）-> reselect（16 例 / 任务）-> evaluate（盲测 16 例 / 任务，mlp_ppo + mlp_bc）。
# 用法（仓库根目录，mujoco 环境）：BUDGET=1020 NENV=18 bash experiments/algorithms/ppo_direct/run_v7ppo.sh 1 2
# 输入：runs/spike_steer/v7/{baseline.json, collect_train.npz}（= artifacts/spike_steer/plan_v7/ 的副本），可选 test_opponent.json。
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
PY=${PY:-python}
BUDGET=${BUDGET:-1020}        # SNN ES：20 代 × 17 × 12 = 4 080 回合；默认 1/4
NENV=${NENV:-18}
M=experiments.algorithms.ppo_direct.v7ppo
log() { echo "[$(date -Iseconds)] $*"; }
for S in "$@"; do
  log "seed $S: bc"
  $PY -W ignore -m $M bc --seed "$S" || exit 1
  log "seed $S: finetune budget $BUDGET episodes, $NENV envs"
  $PY -W ignore -m $M finetune --seed "$S" --budget-episodes "$BUDGET" --n-envs "$NENV" || exit 1
  log "seed $S: reselect (16 cases per task, 5 centres)"
  $PY -W ignore -m $M reselect --seed "$S" --workers "$NENV" || exit 1
  log "seed $S: evaluate (blind set 16 per task; arms mlp_ppo, mlp_bc)"
  $PY -W ignore -m $M evaluate --seed "$S" --workers "$NENV" || exit 1
  ART=artifacts/spike_steer/ppo_direct/v7ppo; mkdir -p "$ART/s$S/centres"
  cp runs/ppo_direct/v7ppo/{reselect_s$S.json,evaluate_mlp_ppo_s$S.json,evaluate_mlp_bc_s$S.json} "$ART"/
  cp runs/ppo_direct/v7ppo/s$S/{ckpt_g*.zip,history.jsonl} "$ART/s$S/centres/" 2>/dev/null
  [ -f runs/ppo_direct/v7ppo/test_opponent.json ] && cp runs/ppo_direct/v7ppo/test_opponent.json "$ART/test_opponent_homepc.json"
  log "seed $S: done"
done
log "all seeds done"
