#!/usr/bin/env bash
# plan v6 流程的 PPO 版，按种子依次：bc（模仿热启动）-> finetune（PPO，预算按回合数）-> reselect（64+64 选中心）-> evaluate（v6 盲测集 32+32）。
# 用法（仓库根目录，mujoco 环境）：BUDGET=7920 NENV=18 bash experiments/algorithms/ppo_direct/run_v6ppo.sh 1 2 3 4 5
# 其他机器：git clone -b ppo-full <repo>；依赖 mujoco 3.8、torch、stable-baselines3、gymnasium、tensorboard、g++；
#   模仿数据：runs/spike_steer/v6/collect_train.npz —— 若没有，先 `cp artifacts/spike_steer/plan_v6/baseline.json runs/spike_steer/v6/`
#   再 `python -m experiments.algorithms.spike_steer.descend8 collect --output runs/spike_steer/v6 --workers <N>`（约 2 分钟），
#   或直接用仓库里的副本：COLLECT_NPZ=artifacts/spike_steer/ppo_direct/collect_train_v6.npz。
# 修正案 1：选中心和盲测按 v6 规则只在 CloudPC-2 上跑；SKIP_EVAL=1 可只跑 bc + finetune，之后把 runs/ppo_direct/v6ppo/s<S>/ 交给 CloudPC-2。
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
PY=${PY:-python}
BUDGET=${BUDGET:-7920}        # ES 每种子 31 680 回合；默认 1/4
NENV=${NENV:-18}
M=experiments.algorithms.ppo_direct.v6ppo
log() { echo "[$(date -Iseconds)] $*"; }
for S in "$@"; do
  log "seed $S: bc"
  $PY -W ignore -m $M bc --seed "$S" || exit 1
  log "seed $S: finetune budget $BUDGET episodes, $NENV envs"
  $PY -W ignore -m $M finetune --seed "$S" --budget-episodes "$BUDGET" --n-envs "$NENV" || exit 1
  if [ "${SKIP_EVAL:-0}" = "1" ]; then log "seed $S: search done (selection + blind test left to CloudPC-2)"; continue; fi
  log "seed $S: reselect (64 + 64 cases, 13 centres)"
  $PY -W ignore -m $M reselect --seed "$S" --workers "$NENV" || exit 1
  log "seed $S: evaluate (v6 blind set 32 + 32)"
  $PY -W ignore -m $M evaluate --seed "$S" --workers "$NENV" || exit 1
  ART=artifacts/spike_steer/ppo_direct/v6ppo/s$S; mkdir -p "$ART"
  cp runs/ppo_direct/v6ppo/s$S/{reselect.json,evaluate.json,history.jsonl} "$ART"/ 2>/dev/null
  cp "$($PY -c "import json;print(json.load(open('runs/ppo_direct/v6ppo/s$S/reselect.json'))['selected']['spec'])")" "$ART/selected.zip"
  cp runs/ppo_direct/v6ppo/s$S/ckpt_g-1.zip "$ART/bc.zip"
  log "seed $S: done"
done
log "all seeds done"
