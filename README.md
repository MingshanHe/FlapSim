# FlapSim

当前研究是 **PID 内环/网络残差与 SNN/ANN × NEAT/PPO 的闭环比较**，分支
`research/pid-residual-velocity-v3`。协作者和其他 AI 请先看 [当前研究交接](AI_HANDOFF.md)，
其中包含已做工作、最新计划、结果边界和正在运行的队列说明。
[研究报告](artifacts/robust_control_20261010/report.md)和
[独立阶段报告](artifacts/robust_control_20261010/interim_20261012/report.md)分别记录实际进展。

本分支 `research/wingbeat-synchronous-snn` 是独立的新研究线草案：[以翅拍为时钟的脉冲控制](research_proposal_wingbeat_snn.md)，
目前只有仿真预研，[probe.py](experiments/algorithms/wingbeat_feasibility/probe.py) 可复算。v3 队列仍以交接文档为准。

新增算法实验位于 [experiments/algorithms](experiments/algorithms/README.md)。
[旧工作目录记录](START_HERE.md)和下文的 PPO 自由飞行示例保留作历史资料；
旧 `ppo_v2` 因观测接口不同，不参与当前研究比较。

扑翼机器人仿真 Python package。默认模型是 `IST_robot`；可选的 `flapsim.rl`
提供 Gymnasium 环境和 stable-baselines3 PPO 自由飞行轨迹训练。所有接口使用 SI 单位。

IST_robot 的 MJCF 根刚体已绕 X 旋转 180°，使翼机构在世界 +Z 上方；
可通过 `SceneConfig.orientation`（wxyz 四元数）覆盖安装姿态。

## 安装与运行

需要 Python 3.10 或更新版本。基础仿真只依赖 MuJoCo 和 NumPy。
建议在独立的 Python 环境中安装：

```bash
python -m pip install -e .
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

配置通过 dataclass 或单个 JSON 文件传入。`config/robots/` 提供四个机器人预设。
JSON 中的自定义 `model_path` 相对该 JSON 文件解析；内置资源不依赖当前工作目录。

```bash
flapsim --config flapsim/config/robots/IST_robot.json --headless --duration 0.01
flapsim --actuator bldcmotor --fluid mujoco --headless --duration 0.01
```

## 目录和调用关系

```text
README.md       安装、配置和使用说明
pyproject.toml  包配置与运行依赖
.gitignore      缓存、虚拟环境和构建产物忽略规则
flapsim/
  actuator/     Actuator -> DCMotor / BLDCMotor / DEA
  config/       参数、JSON、场景构建、模型加载
  controller/   Controller -> SpeedController / MoveController / HoveringController
  fluid/        FluidModel -> BladeElement / MuJoCoFluid / NoFluid
  rl/           自由飞行环境、轨迹、推力扫描、PPO 训练与评估
  robot/        URDF、MJCF、STL
  simulator.py  对外接口，组织组件
  backends.py   CPUBackend / MJXBackend
  kinematics.py 共享的坐标和速度计算
  measurement.py 固定基座六维载荷测量
  cli.py        flapsim 命令行入口
  sim_core.py   SimCore 兼容构造入口
```

每步顺序是 `Controller.update → Actuator.forces → FluidModel.forces → physics step`。
控制器输出各驱动轴目标角速度，执行器转换为有界力矩，流体模型输出各刚体的气动力和力矩。
执行器不直接覆盖 `qpos/qvel`；齿轮、滑块和翼面由物理引擎推进。
场景构建由 `config/scene.py` 负责。

`IST_robot-V2` 在固定支座上增加俯仰、横滚两个关节，使整个 IST 机构转向。
电机转速控制扑翼，两个有限力矩位置伺服控制转向角：

```bash
python -m flapsim --robot IST_robot-V2 --speed 100 --gimbal 0.15 -0.10
```

```python
from flapsim import Simulator, load_config

sim = Simulator(load_config("flapsim/config/robots/IST_robot-V2.json"))
sim.set_speed(100.0)
sim.set_gimbal(pitch=0.15, roll=-0.10)  # rad，各轴限位 ±0.3
sim.step(24000)
```

V2 使用 `SingleThrustFlapper` 的底座和 U 形连接件网格；按整机 15 g 目标调整底座质量与惯量。
支座默认固定，转向关节仍能运动。IST 接口装配、质量分配和伺服参数为仿真假设，
详细坐标、参数和再生成方法见 [IST_robot-V2](flapsim/robot/IST_robot-V2/README.md)。

`measurement.FixedBaseWrenchProbe` 提供世界坐标系下的气动力、支架反力和重力载荷，
力矩参考点为基座原点。读取前需用当前外力执行 `mujoco.mj_forward`。

执行器都继承 `Actuator`（别名 `BaseActuator`），可以通过构造函数注入自己的子类。
DC 是带电压、反电动势、电流限制的准静态电机；BLDC 是平均 q 轴电流控制模型，
忽略换相与电感；DEA 是一对拮抗介电执行器的等效 `V²` 力矩模型，包含弹性和阻尼。
这些是可替换的简化模型。直接构造 `SimulationConfig()` 使用通用演示参数；
CLI 和 IST JSON 预设使用显式的朝力 CL-0720 参考电机参数。参考绕组仍待硬件确认，
不能只由“0720”型号确定。尤其 DEA 的电压、系数和传动需要自行标定。

## PPO 自由飞行轨迹训练

```bash
python -m pip install -e '.[rl]'
python -m flapsim.rl.evaluate --run artifacts/ppo_v2/hover --trajectories hover \
  --episodes 3 --randomize --seconds 16 --plot --output runs/hover_check
python -m flapsim.rl.evaluate --run artifacts/ppo_v2/tracking \
  --trajectories hover circle figure8 ellipse line --episodes 3 --randomize \
  --seconds 16 --reference-physics --plot --output runs/tracking_check
```

动作控制电机和 pitch/roll 目标角度，保留完整机构动力学。默认自由基座、正常重力、
整机 15 g；训练前检查平均升力能否超过重量。保存策略、归一化统计、物理与任务配置，
支持续训。圆、八字、椭圆、往返直线、闭合螺旋和 Lissajous 轨迹共用条件策略。
Linux/WSL 可用 C++ 循环加速完整机构仿真。已训练策略、独立评估和 master 迁移分析见
[实验报告](experiments/ppo_v2/README.md)，新训练和续训命令见
[训练与评估说明](flapsim/rl/README.md)。
当前交付通过了悬停和上述二维轨迹的独立检查；快速 helix / lissajous 仍有误差超限和失稳案例，
具体成功率与使用范围见报告。

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
在 `WingConfig` 中显式指定。IST 预设从曲面翼网格提取沿展向变化的弦长及中弦位置，
通用 `WingConfig` 不传 `planform` 时仍使用矩形近似。两者均不等于实测气动标定。
`mujoco` 模式默认选择引擎内置的惯性盒流体模型（`builtin_model="inertia"`），
CPU 与 MJX 使用同一种模型。原生 CPU 还可以显式选择
`FluidConfig(kind="mujoco", builtin_model="ellipsoid")`，保留 MJCF 已给定的椭球系数。
CLI 对应 `--fluid mujoco --builtin-fluid ellipsoid`。
本包目前仅开放 MJX-JAX 的惯性盒流体组合，并拒绝椭球流体组合，
防止模型参数被引擎静默忽略。实现依据见
[MuJoCo 3.8 MJX 功能表](https://mujoco.readthedocs.io/en/3.8.0/mjx.html#feature-parity)。
切换到叶素模型会关闭内置流体，避免重复施力。
两者都不是 CFD，不包含尾迹演化、翼间流场耦合或流固耦合。

MJX/JAX 的安装、版本指定和设备检查见下节。

MJX 将控制器、执行器、气动力和物理步进整体 JIT 到所选设备，
首次编译会有延迟。`step(n)` 在设备内循环 n 步；`snapshot()` 才传回 CPU。
显式选择 GPU 时，如果设备不可用会报错，不会悄悄退回 CPU。
JAX 默认采用 float32，原生 MuJoCo 采用 float64；极小惯量和强约束下应检查轨迹误差，
需要双精度时可在进程启动前设置 `JAX_ENABLE_X64=1`。
单个小模型的 GPU 运行未必快于 CPU；本版本没有封装批量仿真接口。

本包对 MJX 内置流体禁用 `implicitfast`，因此 `integrator="auto"`
在该组合下选择 Euler，其余默认 implicitfast；也可显式选择 Euler/RK4。
此限制也列在上述 MJX 功能表中。

## MJX 与 JAX 环境配置

FlapSim 的 `backend="mjx"` 使用 **MJX-JAX**，代码显式设置 `impl="jax"`。
安装包名是 `mujoco-mjx`，Python 导入方式是 `from mujoco import mjx`。
下面的命令在项目根目录、Linux 或 WSL2 的 Bash 中执行。

### 版本选择

以下是与当前 MuJoCo 版本一致的固定版本安装示例（2026-10-03 核对）：

| 组件 | 示例版本 | 配置关系 |
| --- | --- | --- |
| Python | 3.12 | 下述 JAX 0.11.2 要求 Python >=3.12 |
| `mujoco` | 3.14.0 | 原生物理引擎及模型编译 |
| `mujoco-mjx` | 3.14.0 | 与 MuJoCo 的主、次、修订版本一致 |
| `jax` / `jaxlib` | 0.11.2 / 0.11.2 | 本例由 JAX 的包依赖锁定相同版本 |
| NVIDIA GPU 运行时 | CUDA 12 | 与项目的 `gpu` 可选依赖一致 |

MJX 的版本规则见 [官方版本说明](https://github.com/google-deepmind/mujoco/blob/main/mjx/README.md#versioning)：
例如 `mujoco-mjx==3.14.0` 配 `mujoco==3.14.0`；若有 `.postN` 发布，
它仍对应同一个 MuJoCo 基础版本。JAX 使用独立的版本号，不设成 MuJoCo 的版本号。
发布依赖可查 [MJX 3.14.0](https://pypi.org/project/mujoco-mjx/3.14.0/) 和
[JAX 0.11.2](https://pypi.org/project/jax/0.11.2/)。

项目声明的 `mujoco-mjx>=3.8`、`jax>=0.6` 是安装下限，不是已验证的全部版本组合。
只运行原生 CPU 仿真仍可使用项目要求的 Python >=3.10；使用本节固定版本时选择 Python 3.12。
本例已通过 Python 3.12/Linux 下 CPU、CUDA 12 两种安装方案的 pip 依赖解析检查，
尚未完成这组版本的 MJX 实际步进和 GPU 验证；安装后需执行下面的运行检查。

### 创建环境

已有合适的 Python 环境时可直接激活；新环境可以使用 Conda：

```bash
conda create -n flapsim-mjx python=3.12 -y
conda activate flapsim-mjx
python -m pip install --upgrade pip
```

### 安装 CPU 版 MJX

无需显卡即可检查 MJX-JAX 路径。固定版本安装：

```bash
python -m pip install -e '.[mjx]' \
  'mujoco==3.14.0' 'mujoco-mjx==3.14.0' 'jax==0.11.2'
python -m pip check
```

### 安装 NVIDIA GPU 版 MJX

在 Linux 或 WSL2 中使用 GPU；原生 Windows 不支持 JAX CUDA，WSL2 的支持在 JAX
文档中标为实验性。CUDA 12 的 Linux 驱动下限为 525，建议使用适配显卡的新驱动。
本方案通过 pip 安装 CUDA/cuDNN 运行库，无需单独安装系统 CUDA Toolkit。
平台与运行库要求见 [JAX 安装文档](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu)。

WSL2 使用 Windows 主机的 NVIDIA 驱动，在 Windows 端安装或更新驱动，
不要在 WSL 内另装 Linux NVIDIA 显示驱动；见
[NVIDIA WSL 配置说明](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#step-1-install-nvidia-driver-for-gpu-support)。
先在 WSL/Linux 中确认 GPU 可见，再安装 GPU 方案：

```bash
nvidia-smi
python -m pip install -e '.[gpu]' \
  'mujoco==3.14.0' 'mujoco-mjx==3.14.0' 'jax[cuda12]==0.11.2'
python -m pip check
```

CPU 与 GPU 安装方案按设备选择一种即可。GPU 方案会同时解析 `jaxlib`、
`jax-cuda12-plugin`、`jax-cuda12-pjrt` 的匹配版本，避免手动混装不同版本。

### 检查版本和运行设备

先检查实际使用的解释器、包版本和 JAX 设备：

```bash
python - <<'PY'
import sys
from importlib.metadata import version
import jax
from mujoco import mjx

print("Python:", sys.executable)
for package in ("mujoco", "mujoco-mjx", "jax", "jaxlib"):
    print(f"{package}: {version(package)}")
print("JAX devices:", jax.devices())
PY

# GPU 配置必须能列出 GPU；若不可用，此命令会报错
python -c 'import jax; print(jax.devices("gpu"))'
```

CPU 安装只需运行第一个代码块，设备应为 CPU。GPU 安装还需执行第二条命令，
不能只凭 `import jax` 成功判断 GPU 已启用。
随后运行短程仿真；首次 JIT 编译可能明显慢于后续步进：

```bash
# 小惯量、闭环约束模型可先用双精度检查数值行为
export JAX_ENABLE_X64=1

# CPU 方案
python -m flapsim --backend mjx --device cpu --headless --duration 0.00001

# GPU 方案
python -m flapsim --backend mjx --device gpu --fluid blade_element --headless --duration 0.00001
python -m flapsim --backend mjx --device gpu --fluid mujoco --headless --duration 0.00001
```

成功时应打印 `mjx/cpu` 或 `mjx/gpu`，默认时间步下运行 2 步。
需要更长轨迹时增大 `--duration`。更换 MJX 版本时，在安装命令中同时修改
`mujoco` 和 `mujoco-mjx` 的版本；更换 JAX 版本时同时检查 Python 要求，
通过 `jax==版本` 或 `jax[cuda12]==版本` 让 pip 解析配套运行库，再重新检查设备与步进。

如果 GPU 不可见，先检查 `nvidia-smi` 和当前 Python 环境；如果提示 CUDA/cuDNN
版本冲突，检查 `LD_LIBRARY_PATH` 是否指向其他 CUDA 安装。使用 pip 运行库方案时，
可临时以 `env -u LD_LIBRARY_PATH python -c 'import jax; print(jax.devices("gpu"))'`
排查路径覆盖，参见 [JAX 的 CUDA 库检查说明](https://docs.jax.dev/en/latest/installation.html#pip-installation-nvidia-gpu-cuda-installed-via-pip-easier)。

## 机器人模型

`robot/IST_robot/urdf/IST_robot.urdf` 和 `IST_robot.xml` 的文件名、
内部模型名称、相互引用已统一。仿真加载 MJCF，保留两级齿轮和 pin-in-slot
闭环约束；URDF 本身不能完整表达该非线性闭环。
固定和浮动模式均保留模型文件中的质量与惯量；IST_robot 已按材料参数修正运动部件惯性。
翼面保持原模型的被动羽化，不添加主动羽化或镜像等式约束。
原模型的网格接触设置保持不变，IST_robot 默认不提供机器人与地面的接触响应。

内置 `IST_robot`、`IST_robot-V2`、`SingleThrustFlapper`、`SingleThrustFlapper-V2` 四个预设，
可通过 `--robot` 选择。后两个模型的预设为零轴速度；需要周期驱动时可自行编写
输出正弦目标角速度的 Controller，再通过电机力矩驱动。
`SimCore(robot, fix=...)` 保留为兼容构造入口，新代码建议使用 `Simulator(config)`。

IST_robot 同时提供曲面翼和矩形翼模型，以及生成模型所需的材料、惯性基准和脚本。
模型说明见 [IST_robot 材料与惯性参数](flapsim/robot/IST_robot/README.md)。
重新生成矩形翼网格时安装 `python -m pip install -e '.[model]'`；
使用内置模型运行仿真不需要这项可选依赖。
