# PPO 悬停任务

该目录把 FlapSim 的自由基座周期平均模型封装为 Gymnasium 环境，并使用
Stable-Baselines3 的 PPO 学习悬停。

## 接口

- 观测：目标位置误差、世界系线速度、机体 z 轴倾斜量、世界系角速度；
- 动作：归一化扑翼频率、`pitch_joint`、`roll_joint` 的残差修正；
- 目标：在随机位置和姿态扰动下保持在 `(0, 0, 0.5) m`；
- 失败条件：触地、飞出训练区域或倾斜超过 65°。

环境不会挂接原有的 `HoveringController`，而是在环境内部使用一个局部稳定器
产生名义动作；PPO 输出其上限为全动作范围 20% 的残差修正。这种 residual
PPO 可以避免初期随机探索立刻使轻量机器人翻转。
自由飞行仍使用 `AveragedHoverAero`，与 `SimCore(fix=False)` 一致。

## 训练

```bash
python -m flapsim.rl.train_ppo --timesteps 100000
```

模型默认保存到 `flapsim/rl/models/hover_ppo.zip`。快速检查训练管线可以使用：

```bash
python -m flapsim.rl.train_ppo --timesteps 2048 \
  --output flapsim/rl/models/smoke_hover_ppo
```

## 评估

```bash
python -m flapsim.rl.evaluate \
  --model flapsim/rl/models/hover_ppo.zip \
  --episodes 10
```

使用 `--no-randomize` 可以从完全水平、位于目标点的状态开始评估。

打开 MuJoCo Viewer 实时观察悬停：

```bash
python -m flapsim.rl.evaluate \
  --model flapsim/rl/models/hover_ppo.zip \
  --episodes 3 \
  --render
```

`--realtime-factor 0.5` 表示半速播放，`--realtime-factor 2` 表示二倍速。
评估过程中关闭 Viewer 会直接结束程序。
