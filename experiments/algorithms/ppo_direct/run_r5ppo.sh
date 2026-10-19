#!/usr/bin/env bash
# 第五轮流程的 PPO 版，5 个种子依次：bc（模仿热启动）-> finetune（PPO，预算按回合数）-> reselect（128 例选中心）-> evaluate（第五轮盲测集 + 消融）。
# 用法（仓库根目录，mujoco 环境）：BUDGET=5940 NENV=18 bash experiments/algorithms/ppo_direct/run_r5ppo.sh 1 2 3 4 5
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
PY=${PY:-python}
BUDGET=${BUDGET:-5940}        # ES 每种子 23 760 回合；默认 1/4
NENV=${NENV:-18}
M=experiments.algorithms.ppo_direct.r5ppo
log() { echo "[$(date -Iseconds)] $*"; }
for S in "$@"; do
  log "seed $S: bc"
  $PY -W ignore -m $M bc --seed "$S" || exit 1
  log "seed $S: finetune budget $BUDGET episodes, $NENV envs"
  $PY -W ignore -m $M finetune --seed "$S" --budget-episodes "$BUDGET" --n-envs "$NENV" || exit 1
  log "seed $S: reselect (64 + 64 cases, 13 centres)"
  $PY -W ignore -m $M reselect --seed "$S" --workers "$NENV" || exit 1
  log "seed $S: evaluate (round-5 blind set, intact + clamp_fast + noref + nopulse)"
  $PY -W ignore -m $M evaluate --seed "$S" --workers "$NENV" || exit 1
  ART=artifacts/spike_steer/ppo_direct/r5ppo/s$S; mkdir -p "$ART"
  cp runs/ppo_direct/r5ppo/s$S/{reselect.json,evaluate.json,history.jsonl} "$ART"/ 2>/dev/null
  cp "$(python -c "import json;print(json.load(open('runs/ppo_direct/r5ppo/s$S/reselect.json'))['selected']['spec'])")" "$ART/selected.zip"
  cp runs/ppo_direct/r5ppo/s$S/ckpt_g-1.zip "$ART/bc.zip"
  log "seed $S: done"
done
log "all seeds done"
# 在其他机器上：git clone -b ppo-full <repo>；conda 环境需 mujoco 3.8、torch、stable-baselines3、gymnasium、tensorboard；
# 首次运行会自动编译 runs/neuro_gates/scratch/source（需要 g++）；NENV 取 核数-2；多个种子可并行启动（各自减小 NENV）。
