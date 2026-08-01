# PPO 悬停任务

该目录把 FlapSim 的自由基座周期平均模型封装为 Gymnasium 环境，并使用
Stable-Baselines3 的 PPO 学习悬停。

## 接口

- 观测：目标位置误差、世界系线速度、机体 z 轴倾斜量、世界系角速度；
- 动作：归一化扑翼频率、`pitch_joint`、`roll_joint` 的残差修正；
- 目标：从 `(0, 0, 0.5) m` 附近出发，收敛到每回合随机生成的目标点；
- 失败条件：触地、飞出训练区域或倾斜超过 65°。

环境不会挂接原有的 `HoveringController`，而是在环境内部使用一个局部稳定器
产生名义动作；PPO 输出其上限为全动作范围 20% 的残差修正。这种 residual
PPO 可以避免初期随机探索立刻使轻量机器人翻转。
自由飞行仍使用 `AveragedHoverAero`，与 `SimCore(fix=False)` 一致。

## 训练

安装强化学习和 TensorBoard 依赖：

```bash
pip install -e '.[rl]'
```

```bash
python -m flapsim.rl.train_ppo --timesteps 100000
```

训练默认把 TensorBoard event 写到 `flapsim/rl/runs`。在另一个终端启动：

```bash
tensorboard --logdir flapsim/rl/runs
```

然后访问终端显示的地址（通常是 `http://localhost:6006`）。可观察：

- `rollout/ep_rew_mean`：平均回合奖励；
- `rollout/ep_len_mean`：平均回合长度；
- `rollout/success_rate`：随机目标收敛成功率；
- `train/value_loss`、`train/policy_gradient_loss`：网络损失；
- `train/approx_kl`、`train/entropy_loss`：PPO 更新幅度和探索程度。

指定实验名称和日志目录：

```bash
python -m flapsim.rl.train_ppo \
  --timesteps 100000 \
  --run-name random_target_seed7 \
  --tensorboard-log flapsim/rl/runs
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
  --episodes 10 \
  --random-target
```

随机目标范围默认为 `x/y in [-0.25, 0.25] m`、`z in [0.40, 0.65] m`。
使用 `--no-randomize` 可以关闭机器人初始位置和姿态的小扰动。

打开 MuJoCo Viewer 实时观察悬停：

```bash
python -m flapsim.rl.evaluate \
  --model flapsim/rl/models/hover_ppo.zip \
  --episodes 3 \
  --random-target \
  --render
```

Viewer 中的绿色球体表示当前回合的目标点。
`--realtime-factor 0.5` 表示半速播放，`--realtime-factor 2` 表示二倍速。
评估过程中关闭 Viewer 会直接结束程序。

## 位置与姿态轨迹跟踪

`TrajectoryTrackingEnv` 支持三种 8 秒周期的三维闭合轨迹，每条轨迹在
垂直方向都带有平滑起伏：

- `circle`：半径 `0.20 m` 的圆形；
- `figure8`：横向交叉的 8 字形；
- `ellipse`：长短轴不同的椭圆形。

每个 episode 持续 16 秒，即完整跟踪两圈。

姿态目标是与轨迹速度、加速度和平均气动阻力相容的 Roll/Pitch 倾角。
当前机器人没有独立 Yaw 执行通道，因此没有要求策略跟踪不可控的 Yaw
轨迹；Yaw 仅靠模型阻尼保持稳定。

轨迹观测包括：

- 三维位置误差；
- 三维速度误差；
- Roll/Pitch 姿态误差；
- 三维角速度；
- 轨迹相位的正弦和余弦。

训练轨迹 PPO：

```bash
python -m flapsim.rl.train_trajectory_ppo \
  --trajectory mixed \
  --timesteps 100000 \
  --run-name trajectory_tracking
```

`mixed` 会在每个 episode 从三种轨迹中随机选择一种，用来训练同一个通用
策略。也可以使用 `--trajectory figure8` 等参数只训练指定轨迹。

TensorBoard 与悬停训练共用 `flapsim/rl/runs`：

```bash
tensorboard --logdir flapsim/rl/runs
```

无界面评估：

```bash
python -m flapsim.rl.evaluate_trajectory \
  --model flapsim/rl/models/trajectory_ppo.zip \
  --trajectory figure8 \
  --episodes 20
```

可视化评估：

```bash
python -m flapsim.rl.evaluate_trajectory \
  --model flapsim/rl/models/trajectory_ppo.zip \
  --trajectory figure8 \
  --episodes 3 \
  --render
```

可视化中，蓝色点列表示完整位置轨迹，绿色球表示当前目标位置，黄色箭头
表示目标机体 z 轴，即参考 Roll/Pitch 姿态。
