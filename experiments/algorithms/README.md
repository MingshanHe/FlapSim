# 后续算法实验

此目录用于 PPO 之后的新算法开发。当前未预设算法，也没有开始新训练。
共享代码和 V2 物理模型沿用仓库现有实现；原 PPO 文件保持为可复现的基线。

建议每个算法单独放一个目录：

```text
experiments/algorithms/<algorithm>/
  train.py
  evaluate.py
  configs/
  README.md
runs/<algorithm>/<run_name>/      本地日志和临时模型
artifacts/<algorithm>/            验证后保留的成果
```

## 共享环境接口

现有 `TrajectoryTrackingEnv` 实现 Gymnasium 的 `reset` / `step`，动作是连续三维
`Box(-1, 1)`：电机输入轴转速命令、pitch 目标角、roll 目标角。
`CentroidalTrackingEnv` 增加整机角动量和机构方向特征，已用于 V2 基线。
算法自身的网络、优化器、经验池和观测归一化由新训练器负责。

```python
from flapsim.rl.env import TrackingConfig
from flapsim.rl.features import CentroidalTrackingEnv
from flapsim.rl.native_fast import accelerate_fast

env = CentroidalTrackingEnv(TrackingConfig(
    trajectories=('hover',),
    initial_speed=3400.,
    native_physics=True,
))
accelerate_fast(env.sim)  # Linux / WSL，需 g++
observation, info = env.reset(seed=42)
# action = agent.act(observation)
# observation, reward, terminated, truncated, info = env.step(action)
env.close()
```

比较算法时先使用相同的模型、质量、重力、控制周期、任务参数和评估种子。
已有基线是整机 15 g、正常重力、5 μs 物理步长、10 ms 控制周期。
这些条件和成功标准见 [V2 实验报告](../ppo_v2/README.md)。

`flapsim/rl/train.py`、`evaluate.py` 中的策略读写目前针对 SB3 PPO；新的算法需要
自己的策略保存/加载入口。环境、轨迹、原始状态日志和成功指标可以继续复用。
旧 PPO 权重只作为基线使用，不能直接当作其他算法的参数。
