# IST_robot-V2 的 PPO 轨迹训练

Gymnasium 环境通过 stable-baselines3 PPO 控制完整 MuJoCo 机构。
默认自由基座、整机 15 g、正常重力；气动力由翼片实际运动产生。
电机与气动参考见 [推力核对](../robot/IST_robot-V2/motor_reference.md)。
这是一套可训练、可复现和评估的流程；当前尚未提供收敛的飞行控制策略。

## 安装与检查

在项目根目录、已安装 MuJoCo 的 Python 环境中运行：

```bash
python -m pip install -e '.[rl]'
python -m flapsim.rl.preflight --voltages 4.0 4.2 --output runs/thrust_voltage.json
python -m flapsim.rl.preflight --speeds 4000 5000 --output runs/thrust_speed.json
```

普通仿真无需 RL 依赖。推力扫描固定基座并对完整周期积分，
输出平均升力、电压、电流、输入轴转速、实际扑翼频率和推重比。

## 轨迹与动作

| 名称 | 轨迹 |
| --- | --- |
| `hover` | 定点悬停 |
| `circle` | 水平圆 |
| `figure8` | 水平八字，x/y 基频比 1:2 |
| `line` | 平滑往返直线 |
| `helix` | 两圈水平环绕与正弦升降组成的闭合空间曲线 |
| `lissajous` | 三轴频率比 1:2:3 的空间曲线 |

默认中心 `(0,0,0.8) m`、半径 0.2 m、周期 8 s、垂直幅度 0.1 m。
每集随机轨迹、相位、半径/周期（±20%），并施加小幅初始状态扰动。
一次策略同时训练多条轨迹；观测中的目标位置、速度、加速度及未来目标点提供条件。

动作是三维 `[-1,1]`：输入轴目标转速、pitch、roll。电机动作 0 对应
`--trim-speed`，负向降至 0、正向增至配置上限；不是直接指定扑翼频率或推力。
两轴目标角度各限 ±0.3 rad，经有限力矩伺服驱动。动作以 20 ms 时间常数平滑。
控制周期默认 10 ms，内部物理时间步 5 μs；每个动作推进 2000 个物理步。

观测包含相对位置/速度、机体速度和姿态、角速度、参考加速度、未来 0.1/0.3/0.6 s
的相对位置，以及机构关节状态和上一动作。连续转动关节用 sin/cos 表示。
`VecNormalize` 统计必须与策略配套保存和恢复。

奖励鼓励位置和速度接近参考、延长飞行，惩罚倾斜、控制消耗和动作变化。
高度低于 0.1 m、位置误差超过 1 m、倾斜超过 1.4 rad 或数值异常时终止；
到达时限单独标记截断。成功定义为完整飞行且全程位置 RMSE < 0.1 m。
环境不模拟落地接触或碰撞着陆；它在接近地面时结束回合。

## 训练与续训

训练示例：

```bash
python -m flapsim.rl.train --trajectories hover circle figure8 line helix lissajous \
  --total-timesteps 1000000 --n-envs 4 --output runs/ist_v2_ppo
```

原生 MuJoCo 在 CPU 上计算，`--n-envs` 控制独立进程数；`--device cuda` 只影响
PPO 神经网络，不会把完整机构仿真移到 GPU。当前模型需要很多物理步，先参考短训的
实际吞吐量估算长训耗时。默认每次训练先测最大转速命令的推重比，若不大于 1 就退出。
`--skip-preflight` 仅供已有同一配置推力扫描的情况；不自动升压、降重力或改质量。

每个输出目录必须为空。输出包括 `policy.zip`、`vecnormalize.pkl`、
`simulation.json`、`run.json`、`progress.csv` 及配套中途检查点。
`run.json` 记录任务配置、实际步数、耗时、结束状态及模型/实现哈希。
Ctrl+C 会保存当前策略和归一化统计。中断时不足一个 rollout 的样本可能尚未更新网络。

```bash
python -m flapsim.rl.train --resume runs/ist_v2_ppo \
  --total-timesteps 1000000 --n-envs 4 --output runs/ist_v2_ppo_continued
```

续训的步数是追加量，沿用旧任务、物理和归一化统计。
更换物理代码、观测动作定义或机器人 XML 后需要新训，避免默默换掉策略所处环境。
示例中的 `--gravity-scale` 默认 1；若显式修改，会记录到产物中，不能与正常重力成绩混用。

## 独立评估

```bash
python -m flapsim.rl.evaluate --run runs/ist_v2_ppo --episodes 3 --plot
python -m flapsim.rl.evaluate --run runs/ist_v2_ppo --trajectories circle figure8 \
  --episodes 1 --render --output runs/ist_v2_view
```

评估冻结归一化统计，用固定轨迹尺寸和独立种子、不同初始相位运行。
记录各轨迹 CSV（实际/目标位置、误差、控制命令）和 `summary.json`，包含 RMSE、
最大误差、轨迹完成率、成功率和终止原因。`--plot` 保存平面轨迹与高度图，
`--render` 使用 MuJoCo 查看器，需要可用的图形显示环境。
学习曲线上的 reward 不能代替轨迹完成率和位置误差。

## 当前训练状态

截至 2026-10-04，训练、保存、续训和独立评估流程已运行验证。初始 2048 步策略
尚不能稳定飞行，六条轨迹均因倾斜超限提前结束，成功率为 0。
后续应先取得稳定姿态和悬停，再以完整轨迹评估跟踪能力。
`runs/` 中的本地日志、图表和策略文件不纳入源码仓库。

接口参考：[SB3 自定义环境](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html)、
[向量环境与归一化](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)。
