# FlapSim

扑翼机器人仿真 Python package。默认模型是 `IST_robot`，只负责仿真，
不包含 Gymnasium、PPO 或训练代码。所有接口使用 SI 单位。

固定基座实验见 [已执行的 notebook](experiments/fixbase/IST_robot_fixbase.ipynb)：
包含电机、扑翼与被动羽化曲线、六维气动/支架载荷和 headless MP4 回放。
IST_robot 的 MJCF 根刚体已绕 X 旋转 180°，使翼机构在世界 +Z 上方；
可通过 `SceneConfig.orientation`（wxyz 四元数）覆盖安装姿态。

## 安装与运行

```bash
pip install -e .
python -m flapsim --headless --duration 0.01
python -m flapsim --speed 100 --fluid mujoco
```

`--speed` 是输入轴角速度，单位 rad/s；不是扑翼频率。
IST_robot 的两级齿轮总减速比为 16:1，因此输入 100 rad/s 对应约
`100 / (16 * 2π) = 0.995 Hz` 的机构周期。
闭环机构时间步较小，仿真速度不保证达到实时。

```python
from flapsim import Simulator, SimulationConfig, FluidConfig

sim = Simulator(SimulationConfig(fluid=FluidConfig(kind="blade_element")))
sim.set_speed(100.0)
sim.step(2000)
data = sim.snapshot()          # 独立的 CPU MjData，便于记录/绘图
print(data.time, data.qpos)
sim.reset()
```

配置通过 dataclass 或单个 JSON 文件传入。`config/robots/` 提供三个机器人预设。
JSON 中的自定义 `model_path` 相对该 JSON 文件解析；内置资源不依赖当前工作目录。

```bash
flapsim --config flapsim/config/robots/IST_robot.json --headless --duration 0.01
flapsim --actuator bldcmotor --fluid mujoco --headless --duration 0.01
```

## 目录和调用关系

```text
flapsim/
  actuator/     Actuator -> DCMotor / BLDCMotor / DEA
  config/       参数、JSON、场景构建、模型加载
  controller/   Controller -> SpeedController / MoveController / HoveringController
  fluid/        FluidModel -> BladeElement / MuJoCoFluid / NoFluid
  robot/        URDF、MJCF、STL
  simulator.py  对外接口，组织组件
  backends.py   CPUBackend / MJXBackend
  kinematics.py 共享的坐标和速度计算
```

每步顺序是 `Controller.update → Actuator.forces → FluidModel.forces → physics step`。
控制器输出各驱动轴目标角速度，执行器转换为有界力矩，流体模型输出各刚体的气动力和力矩。
执行器不直接覆盖 `qpos/qvel`；齿轮、滑块和翼面由物理引擎推进。
`environment` 已移除，场景构建并入 `config/scene.py`。

执行器都继承 `Actuator`（别名 `BaseActuator`），可以通过构造函数注入自己的子类。
DC 是带电压、反电动势、电流限制的准静态电机；BLDC 是平均 q 轴电流控制模型，
忽略换相与电感；DEA 是一对拮抗介电执行器的等效 `V²` 力矩模型，包含弹性和阻尼。
这些是可替换的简化模型。默认参数用于演示，尤其 DEA 的电压、系数和传动需要自行标定。

## 六自由度命令与 PID

`SixDOFCommand` 支持世界坐标位置 `(x,y,z)` 和姿态 `(roll,pitch,yaw)`，
姿态采用 ZYX 欧拉角；也可附带世界坐标线速度和角速度。
`MoveController` 使用位置与 SO(3) 姿态误差构造局部 PID。
`HoveringController` 复用这套控制，持续保持给定目标，不自动切换航点。
这是一套局部调节器，不保证大姿态误差下全局收敛。

控制器通过 `LinearAllocator` 把机体坐标系的修正 wrench
`[Fx,Fy,Fz,Tx,Ty,Tz]` 转换为轴角速度：
`speed = trim_speed + allocation @ wrench`。矩阵尺寸是 `(驱动轴数, 6)`；
`ControllerConfig.speed` 在这里代表已标定的悬停配平转速，PID 增益输出力/力矩。
积分有界，输出按执行器转速限制裁剪；需要复杂饱和处理时可替换控制器。

**IST_robot 只有一个独立驱动电机，无法独立跟踪六个自由度。**
包不会自动虚构六轴分配矩阵或额外机体推力。
使用 hover/move 必须设置自由基座，并传入经过标定的分配矩阵或自定义 allocator；
秩不足 6 会给出提示。以下代码中的 `calibrated_matrix` 和 `trim_speeds`
需来自你自己的机构辨识：

```python
from flapsim import Simulator, SimulationConfig, SceneConfig, ControllerConfig, SixDOFCommand

# calibrated_matrix: (电机数, 6)；trim_speeds: 每个电机的配平角速度
config = SimulationConfig(
    scene=SceneConfig(fixed_base=False),
    controller=ControllerConfig(
        kind="move", allocation=calibrated_matrix, speed=trim_speeds,
    ),
)
sim = Simulator(config)
sim.set_command(SixDOFCommand(position=(0.1, 0, 0.5), orientation=(0, 0, 0.2)))
sim.step(200)
```

新增控制方法继承 `Controller` 并实现 `command()` 与
`update(data, state, command, xp)`，返回目标轴速度和新状态。
如需非线性控制分配，实现 `allocate(wrench_body, xp)` 并传给
`MoveController(config.controller, allocator=...)`。
然后以 `Simulator(config, controller=...)` 注入；执行器和流体组件同样可以注入。
支持 GPU 的扩展须用传入的 `xp` 做纯数组运算，不在步进函数中读文件、转 Python float 或修改对象状态。

## 气动力与 CPU/GPU

物理模型与运行设备是两项独立选择：

| 设置 | 选项 |
| --- | --- |
| `fluid.kind` | `blade_element`、`mujoco`、`none` |
| `backend` | `cpu`（原生 MuJoCo）、`mjx`（JAX） |
| `device` | `cpu`、`gpu`；GPU 需配合 `mjx` |

叶素模型沿翼展离散，使用实际翼片速度和风速计算准定常法向力、弦向阻力，
通过力臂生成力矩，并加入羽化旋转阻尼。翼长、弦长、体坐标轴、压力中心位置
在 `WingConfig` 中显式指定；默认矩形翼近似不等于实测气动标定。
`mujoco` 模式默认选择引擎内置的惯性盒流体模型（`builtin_model="inertia"`），
CPU 与 MJX 使用同一种模型。原生 CPU 还可以显式选择
`FluidConfig(kind="mujoco", builtin_model="ellipsoid")`，保留 MJCF 已给定的椭球系数。
CLI 对应 `--fluid mujoco --builtin-fluid ellipsoid`。
MJX-JAX 3.8 只支持惯性盒流体，不支持椭球流体；本包会拒绝该不支持的组合，
防止模型参数被引擎静默忽略。支持范围见
[MuJoCo 3.8 MJX 功能表](https://mujoco.readthedocs.io/en/3.8.0/mjx.html#feature-parity)。
切换到叶素模型会关闭内置流体，避免重复施力。
两者都不是 CFD，不包含尾迹演化、翼间流场耦合或流固耦合。

```bash
# 可在 CPU 上验证 MJX
pip install -e '.[mjx]'
python -m flapsim --backend mjx --device cpu --headless --duration 0.001

# Linux/WSL NVIDIA GPU；需要与驱动匹配的 JAX CUDA 运行时
pip install -e '.[gpu]'
python -m flapsim --backend mjx --device gpu --fluid blade_element --headless --duration 0.001
python -m flapsim --backend mjx --device gpu --fluid mujoco --headless --duration 0.001
```

MJX 将控制器、执行器、气动力和物理步进整体 JIT 到所选设备，
首次编译会有延迟。`step(n)` 在设备内循环 n 步；`snapshot()` 才传回 CPU。
显式选择 GPU 时，如果设备不可用会报错，不会悄悄退回 CPU。
JAX 默认采用 float32，原生 MuJoCo 采用 float64；极小惯量和强约束下应检查轨迹误差，
需要双精度时可在进程启动前设置 `JAX_ENABLE_X64=1`。
单个小模型的 GPU 运行未必快于 CPU；本版本没有封装批量仿真接口。

MJX 内置流体与 `implicitfast` 的组合不受支持，因此 `integrator="auto"`
在该组合下选择 Euler，其余默认 implicitfast；也可显式选择 Euler/RK4。
此限制也列在上述 MJX 功能表中。

## 模型与旧代码迁移

`robot/IST_robot/urdf/IST_robot.urdf` 和 `IST_robot.xml` 的文件名、
内部模型名称、相互引用已统一。仿真加载 MJCF，保留两级齿轮和 pin-in-slot
闭环约束；URDF 本身不能完整表达该非线性闭环。
浮动模式保留原始 CAD 质量/惯量，不再替换为 30 g 等效刚体。
翼面保持原模型的被动羽化，不添加主动羽化或镜像等式约束。
原模型的网格接触设置保持不变，IST_robot 默认不提供机器人与地面的接触响应。

旧的机器人专用控制器、周期平均悬停模型和 `flapsim.rl` 已删除。
`SimCore(robot, fix=...)` 仅保留构造入口；旧的 `environment`、
`set_thrust_command`、plate/yaw 等接口不再提供。
旧机器人的 URDF/STL 保留，预设为零轴速度；需要周期驱动时可自行编写
输出正弦目标角速度的 Controller，再通过电机力矩驱动。
`tutorial/`、`log/` 保留原始研究记录，旧运行接口的 notebook 需迁移。
旧 `fluid/aero.py` 及对应输入类型只用于保留纯公式教程，不参与新仿真链路。

重构前的全部工作文件（含未提交修改和训练模型）备份于仓库相邻的
`../FlapSim-backups/before-oop-20260921-203149.zip`，不进入安装包。

## 验证

```bash
pip install -e '.[test]'
pytest -q
FLAPSIM_TEST_MJX=1 FLAPSIM_TEST_DEVICE=cpu pytest -q tests/test_mjx.py
FLAPSIM_TEST_MJX=1 FLAPSIM_TEST_DEVICE=gpu pytest -q tests/test_mjx.py
```

测试检查模型资源、机构约束、电机限幅/制动、流体选择与耗散方向、
控制分配、重置和 CPU/MJX 短程一致性；它们不代表真实机器人的悬停能力已验证。
