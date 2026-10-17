# IST_robot-V2：两轴推力转向与自由飞行

在原 IST 机构下增加俯仰、横滚关节，使整个扑翼机构相对固定支座转动。
原 IST 的两级齿轮、16:1 减速、Scotch-yoke 闭环、运动部件质量与惯量、
左右翼被动羽化和气动参数均保留。固定底座和 U 形连接件采用 `SingleThrustFlapper`
的 `base_link.STL`、`link1.STL`，分别复制到本模型的 `meshes/mount_base.STL`、
`meshes/pitch_link.STL`。连接件保留源惯性；外层底座按用户指定的整机 15 g
质量预算调整，原始模型文件不修改。

两轴云台参考 `SingleThrustFlapper` 的 pitch/roll 串联结构及 ±0.3 rad 转角范围。

## 结构和坐标

```text
world
└─ base_link              SingleThrustFlapper 的固定底座
   └─ pitch_link          源 link1 的 U 形连接件；pitch_joint 绕支座 +Y
      └─ thrust_unit      roll_joint，绕俯仰后局部 +X；原 IST 的 base_link
         └─ 原 IST 传动、滑块和两翼
```

横滚轴位于原 IST 基座原点；俯仰轴在零位时位于其下方 3.25 mm，采用源模型的
两轴间距。`pitch_link` 和 `thrust_unit` 各有 −90° X 固定安装旋转，合成后
保留原 IST 的 180° X 零位安装姿态。新支座坐标系零姿态与世界坐标一致，
因此 V2 的 `scene.orientation` 使用 `[1,0,0,0]`，
不能直接沿用 V1 的 `[0,1,0,0]`。`fixed_base=true` 固定的是最外层支座，
两个转向关节仍可运动。V2 比 V1 增加两个独立关节和两个主动伺服通道。

底座 STL 保持原尺寸，绕 Z 旋转 +90°，再平移 `(0.1, 0, −41.5) mm`，
让源模型的俯仰安装点与 U 形连接件对齐，同时维持 V2 的 pitch/roll 方向定义。
底座质心和惯性坐标同步变换。U 形件使用源 CAD 局部坐标和原尺寸；
原 CAD 中近似 90° 的安装角规范为精确 90°。IST 机构与连接件横滚轴的连接
仍采用仿真装配假设，尚未提供匹配该接口的加工件 CAD。

角度以 rad 表示，API/JSON 顺序均为 **pitch、roll**。零位机构轴沿支座 +Z，
正 pitch 将其朝 +X 倾斜，正 roll 将其朝 −Y 倾斜。在支座坐标系中，该轴为：

```text
[sin(pitch)*cos(roll), -sin(roll), cos(pitch)*cos(roll)]
```

这是机构朝向；实际推力来自翼片运动和所选流体模型，含周期波动，
不强制等于上述轴向。观察平均推力时应对完整扑翼周期积分。
基座姿态变化时，上述方向还需乘以基座的世界旋转矩阵。

## 运行和控制

在项目根目录：

```bash
python -m flapsim --robot IST_robot-V2 --speed 100 --gimbal 0.15 -0.10
python -m flapsim --robot IST_robot-V2 --gimbal 0.15 -0.10 --headless --duration 0.2
```

Python 接口中，转速与转向命令独立：

```python
from flapsim import Simulator, load_config

sim = Simulator(load_config("flapsim/config/robots/IST_robot-V2.json"))
sim.set_speed(100.0)                       # 输入轴 rad/s
sim.set_gimbal(pitch=0.15, roll=-0.10)      # rad；约 +8.6°、−5.7°
sim.step(24000)                            # 默认时间步下 0.12 s
print(sim.data.joint("pitch_joint").qpos, sim.data.joint("roll_joint").qpos)
sim.set_gimbal(pitch=-0.10, roll=0.15)
sim.run(duration=0.2, viewer=True)
```

也可通过 `load_config("flapsim/config/robots/IST_robot-V2.json")` 加载完整预设。
`set_gimbal()` 只设置目标角度；实际运动由 MuJoCo 位置伺服产生的有限力矩推进。
其力矩为 `clip(kp*(target-q) - kd*qvel, -max_torque, max_torque)`，
存在动态跟踪误差。超限角度、NaN 和无穷值会报错。`reset()` 恢复配置中的
目标角度、零位关节状态及初始转速命令；非零初始目标会由伺服逐渐跟踪。

原 `set_speed()`、控制器和电机模型仍只控制 `motor_joint`。
位置伺服通过 MuJoCo 原生 actuator 接入 CPU/MJX 的步进，
加载器会移除模型文件中的演示执行器，然后按配置创建两个伺服。
直接用 MuJoCo 加载 XML 时，`ctrl` 顺序是 `[motor_speed, pitch, roll]`；
经 FlapSim 加载时，`data.ctrl` 只有 `[pitch, roll]`，电机力矩由 FlapSim 计算。

`--fix false` 可释放整个支座的六自由度，但这不提供自动悬停或完整姿态控制。
原来的六轴 PID 也不会自动分配新增云台角度。

## 仿真假设

| 参数 | 默认值 |
| --- | --- |
| pitch / roll 转角限位 | 各 ±0.3 rad，约 ±17.2° |
| 外层底座 | 源 STL 原尺寸约 27 × 11.5 × 39.4687 mm；仿真质量 2.01467 g |
| IST 机构与内层基座 | 保留原惯性，共 11.98533 g |
| 整机质量 | 15 g，包含外层底座和 1 g 连接件 |
| 固定底座质心 | 约 (−0.3339, 0, −25.9919) mm，相对于零位横滚轴原点 |
| U 形连接件 | 源 `link1.STL`，局部包围尺寸约 4.2 × 5.9909 × 2.5 mm，质量 1 g |
| 两轴间距 | 3.25 mm；零位俯仰轴在横滚轴下方 |
| 伺服比例增益 kp | 0.5 N·m/rad |
| 伺服速度反馈 kd | 0.005 N·m·s/rad |
| 每轴最大力矩 | 0.02 N·m |
| 转轴被动阻尼 | 1e-6 N·m·s/rad |

源底座的 19 g 不再用于 V2。生成脚本将它改为 `15 g - IST 机构 - 连接件`，
底座完整惯性张量同比缩放，质心位置保留。该分配满足总质量约束，但不代表已测得
每个部件质量；新增舵机、电池和支架的真实分布仍需称重后核实。
U 形连接件保留源 URDF 的质心和完整惯性张量。转向伺服尚无硬件带宽、力矩曲线
及新增舵机质量数据；默认控制参数用于验证转向功能，不能视为硬件选型结果。
原机构质量数据来自相邻 `IST_robot`；新支架不启用接触，与原模型一致。
调整控制参数可编辑 JSON 的 `gimbal` 段；调整结构、质量或限位需修改生成脚本。

## 模型再生成

```bash
python flapsim/robot/IST_robot-V2/build_model.py
python flapsim/robot/IST_robot/build_aerodynamics.py
```

生成脚本从相邻 `IST_robot/urdf/IST_robot.xml` 和 `.urdf` 构建 V2，
共享 `IST_robot/meshes`，并复制 `SingleThrustFlapper` 的底座、U 形件 STL 及上述惯性预算。
更新任一源模型的网格或惯性后应重新生成 V2。
URDF 用于结构交换，完整闭环动力学仍必须加载 MJCF。
自由飞行 PPO 使用 `flapsim/config/training/ist_v2_flight.json`；
电机诊断、参考绕组与限制见 [电机与推力核对](motor_reference.md)，
训练命令见 [PPO 说明](../../rl/README.md)。
