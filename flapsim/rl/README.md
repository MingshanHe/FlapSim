# IST_robot-V2 的 PPO 轨迹训练

Gymnasium 环境通过 stable-baselines3 PPO 控制完整 MuJoCo 机构。
默认自由基座、整机 15 g、正常重力；气动力由翼片实际运动产生。
电机与气动参考见 [推力核对](../robot/IST_robot-V2/motor_reference.md)。
配套的已训练策略保存在 `artifacts/ppo_v2/`，独立评估和复现记录见
[实验报告](../../experiments/ppo_v2/README.md)。
已验证的交付范围为悬停、圆、八字、椭圆和往返直线；快速三维轨迹仍属于实验范围。

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
| `ellipse` | 水平椭圆，半轴为 1.2r 和 0.7r |
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

### 本机加速与悬停课程

Linux / WSL 上可以用 `--native --split-step` 编译 C++ 物理循环（需要 `g++`）。
仍使用原来的 DC 电机、叶素力、云台伺服、MuJoCo 完整机构和 5 μs 时间步。
`--native` 保持原 `mj_step → mj_forward` 顺序；`--split-step` 使用
`mj_step1/mj_step2` 避免重复计算位置/速度动力学，动作结束后完成 forward。
不支持 RK4。编译缓存位于 `~/.cache/flapsim`，不写入安装包。

建议先训练悬停，并以独立评估决定是否进入轨迹阶段：

```bash
python -m flapsim.rl.train --native --split-step --features centroidal --trajectories hover \
  --trim-speed 3400 --gamma .995 --total-timesteps 600000 \
  --n-envs 16 --n-steps 256 --batch-size 256 --output runs/hover
python -m flapsim.rl.evaluate --run runs/hover --trajectories hover \
  --episodes 10 --randomize --best --plot

# 继承已训练的观测归一化、物理参数和策略，改变轨迹课程。
python -m flapsim.rl.train --warm-start runs/hover --from-best --trajectories hover circle figure8 \
  --radius .10 --learning-rate 1e-4 --total-timesteps 131072 --n-envs 16 --output runs/tracking_small
python -m flapsim.rl.train --warm-start runs/tracking_small \
  --from-best --trajectories hover circle figure8 ellipse line helix lissajous --radius .20 \
  --learning-rate 1e-4 --total-timesteps 262144 --n-envs 16 --output runs/tracking
```

3400 rad/s 是本次扫描的接近悬停的输入轴**命令**，对应实际约 2633 rad/s、26.2 Hz；
不能用它直接换算实际频率。默认策略初始 `log_std=-2`，沿用 master 的小幅探索思路。
上面是新训练的课程模板，不是已发布权重的逐步重演；实际阶段、检查点和步数见实验报告。
新建训练默认 `gamma=.995`；续训默认保留已有值，也可显式传入 `--gamma`。
`--warm-start` 只改变任务轨迹，保留原物理和控制定义；旧 master 策略的观测、动作含义
不同，不能通过此参数直接载入。

`--from-best` 可与 `--resume` 或 `--warm-start` 一起使用，继承已经评估过的最佳检查点。
续训可显式用 `--learning-rate 1e-4` 降低学习率。
可选 `--features centroidal` 追加整机角动量对应的刚体角速度，以及扑翼机构轴在支座
坐标中的方向。它们由原机构状态计算，不改变动力学。从普通观测升级时，保留已有
归一化统计和网络权重，新增输入权重置零，因此迁移前后的策略输出一致；Adam 状态
重新初始化，迁移方式写入 `run.json`。已经采用此观测的续训默认继续沿用。

每隔 `--eval-steps` 步在固定的独立种子序列上评估，`best/` 同时保存最佳策略与当时的
`vecnormalize.pkl`。`--best` 选择这一对文件；默认评估最终策略。
`--randomize` 用独立种子测试初始扰动和轨迹尺寸；`--zero-policy` 提供零动作对照。
`--reference-physics` 用原来的非分段物理顺序复核 split-step 策略。
`--seconds` 改变评估时长；`--period` 单独改变参考周期，并在结果中记录，不能将不同
周期的成绩当作同一任务合并。
可选 `--video` 生成 MP4，需要 imageio/imageio-ffmpeg；无桌面时可设置 `MUJOCO_GL=egl`。

加速回归与环境时限/种子/质量/重力检查：`python -m pytest -q tests`。
诊断和学习曲线工具见 `experiments/ppo_v2/`。

训练示例：

```bash
python -m flapsim.rl.train --trajectories hover circle figure8 ellipse line helix lissajous \
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
更换物理代码、观测动作定义或机器人 XML 后需要显式迁移或新训，避免默默换掉策略所处环境。
唯一登记的兼容源码扩展是增加椭圆参考；它锁定扩展前后两个完整哈希，并以固定样本
验证原有六条轨迹未变。追加 centroidal 观测则通过单独的行为保持迁移实现。
示例中的 `--gravity-scale` 默认 1；若显式修改，会记录到产物中，不能与正常重力成绩混用。

## 独立评估

```bash
python -m flapsim.rl.evaluate --run runs/ist_v2_ppo --episodes 3 --plot
python -m flapsim.rl.evaluate --run runs/ist_v2_ppo --trajectories circle figure8 \
  --episodes 1 --render --output runs/ist_v2_view
```

评估冻结归一化统计，默认使用固定轨迹尺寸、独立种子和等间隔初始相位。
`--randomize` 同时随机化初始状态、尺寸、周期和起始相位，并在结果中记录实际参考参数。
记录各轨迹 CSV（实际/目标位置、误差、控制命令）和 `summary.json`，包含 RMSE、
最大误差、轨迹完成率、成功率和终止原因。`--plot` 保存平面轨迹与高度图，
`--render` 使用 MuJoCo 查看器，需要可用的图形显示环境。
学习曲线上的 reward 不能代替轨迹完成率和位置误差。

## 导出与批量复核

```bash
python -m flapsim.rl.export_policy --run runs/tracking --best --output artifacts/my_tracking
python -m experiments.ppo_v2.evaluate_seeds --run artifacts/my_tracking \
  --trajectories hover circle figure8 ellipse line helix lissajous \
  --seeds 1000 1001 1002 1003 1004 --workers 4 --output runs/validation
python -m flapsim.rl.evaluate --run artifacts/my_tracking --trajectories circle figure8 ellipse \
  --episodes 3 --randomize --seconds 16 --reference-physics --plot --output runs/long_check
```

导出包含策略、配套归一化、仿真配置、训练元数据和文件哈希；加载时验证导出文件未变。
批量复核应使用训练结束后冻结的目录，不能直接读取仍在更新的 `best/`。
`runs/` 中的中途日志不纳入源码仓库，最终策略和关键评估证据单独保留。

接口参考：[SB3 自定义环境](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html)、
[向量环境与归一化](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)。
