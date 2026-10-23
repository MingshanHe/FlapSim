# 上部 7 g＋下部 8 g：六维 IMU 的 NEAT / PPO 比较

实验结果见仓库根目录 [report.md](../../../report.md)。实际结果和“可用悬停”标准分开报告。

两种算法使用相同的六维 base_link IMU 输入、α/β 正负角度范围、电机配平基准、
物理模型、奖励与测试案例。NEAT 使用 128 个体、8 个初始隐藏节点以及可进化循环记忆；
PPO 使用 64×64 前馈网络。网络结构不同，因此这是具体方法配置的比较。

## 环境

本次实验在 WSL/Linux、Python 3.11 中运行，使用原生 C++ 仿真加速和 CPU 训练。
在仓库根目录安装：

```bash
python -m pip install -e '.[rl,video,test]'
python -m pip install -r experiments/algorithms/neat_imu/requirements-compare.txt
```

## 从头复现实验

使用新的输出目录；脚本会拒绝覆盖已有训练结果。先重新生成质量模型，
这样 meshdir 会指向本机的仓库位置；不要直接依赖历史 XML 中的原机绝对路径。

```bash
python -c "from experiments.algorithms.neat_imu.mass_model import make_variant; make_variant('artifacts/neat_ppo_reproduction/robot_upper7_lower8.xml', 7, 8)"

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.algorithms.neat_imu.trim \
  --model-path artifacts/neat_ppo_reproduction/robot_upper7_lower8.xml \
  --precise --workers 8 --output artifacts/neat_ppo_reproduction/calibration

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.algorithms.neat_imu.compare_prepare \
  --root artifacts/neat_ppo_reproduction

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.algorithms.neat_imu.compare_campaign \
  --root artifacts/neat_ppo_reproduction

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.algorithms.neat_imu.compare_results \
  --root artifacts/neat_ppo_reproduction

python -m experiments.algorithms.neat_imu.compare_report --root artifacts/neat_ppo_reproduction
```

每个算法两个训练种子。NEAT 每种子目标 524288 个实际控制步，并完成跨过预算的整代；
PPO 匹配对应实际步数，向上补齐至 2048 步 rollout。校验/测试消耗另计。
程序分别保存源码快照、进程状态、超时配置、种群或优化器检查点及冻结策略。

## 在其他目录回放本次保存的策略

下面的入口仅在临时 XML 中修改 meshdir，保留历史文件及其哈希，
并核对回放与记录的失效时刻及位置误差。视频为 4 倍慢放，末帧另停留 1 秒。

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.algorithms.neat_imu.compare_replay \
  --root artifacts/neat_ppo_7g8g_20261008 --label neat_seed835 --group nominal \
  --video runs/neat_seed835_replay.mp4
```

`--label` 可以是 neat_seed835、neat_seed836、ppo_seed835、ppo_seed836。
`--group` 可以是 nominal 或 stress。输出文件必须不存在。

## 检查

```bash
python -m pytest -q tests/test_neat_ppo_compare.py tests/test_neat_mass_model.py \
  tests/test_neat_hover_initial.py tests/test_neat_activation.py tests/test_neat_residual.py
```

`experiments/algorithms/neat/ppo_refine.py` 只保留本实验使用的 `TanhMeanPolicy` 定义，
类源码与训练时相同，兼容已保存的 SB3 模型；未包含无关的历史实验入口。
