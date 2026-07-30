# 气动力方向 Bug 修复记录

## 问题现象

在固定 `base_link` 的情况下提高 flapping frequency，左右翼法向力绝对值明显增加，但世界坐标系中的平均 `Fz` 始终为负：

```text
20 Hz: -0.067 mN
30 Hz: -0.300 mN
40 Hz: -0.823 mN
50 Hz: -1.685 mN
60 Hz: -2.774 mN
```

这意味着 frequency 越高，机器人受到的平均向下气动力越大。Hovering controller 因高度下降而继续提高 frequency，形成错误的正反馈。

需要注意：左右翼 `|F_N|` 的和大于机器人重量，并不代表产生了足够的世界坐标 `+Z` 升力。必须检查变换后的平均 `F_world[2]`。

## 对照 FlapX

直接运行上一级目录中的 FlapX Purdue Flapper，并使用：

```text
frequency = 30 Hz
amplitude = 1.5 rad
fluid = aero
torso = clamped
```

得到：

```text
L_stroke / R_stroke correlation = 1.0
L_pitch  / R_pitch  correlation = 1.0
mean Fz = +0.1029 N
```

FlapX 的关节轴为：

```text
L_stroke: +Z
R_stroke: -Z
L_pitch:  +Y
R_pitch:  +Y
```

左右 stroke actuator 接收相同的位置命令，镜像运动由相反的 joint axis 实现。左右 passive pitch 的 joint coordinate 同号。

## FlapSim 与 FlapX 的差异

### 1. Stroke joint coordinate

当前 URDF 经过 joint origin 旋转后，左右 stroke 的世界轴都接近 `+Z`。因此 FlapSim 必须使用相反的 joint coordinate 才能获得镜像 stroke：

```text
q_right_stroke = -q_left_stroke
```

在传入 FlapX 气动公式前，右翼 stroke coordinate 再通过 `stroke_sign=-1` 转回统一的 aerodynamic convention。

### 2. 错误的右翼 pitch sign

适配器中曾存在试验性的：

```python
pitch_sign=-1.0
```

但 FlapX 的 Purdue 模型左右 pitch axis 都是 `+Y`，左右 pitch coordinate 应使用相同符号。该反号已移除。

### 3. 左右翼零位 aero frame 不一致

FlapX 中，零位状态下 `L_wing` 和 `R_wing` 的 rotation matrix 都是单位矩阵，即左右 aero frame 完全一致。

URDF 导入后，左右 wing body 使用不同的局部 mesh frame。如果直接对左右使用同一个固定 body-to-aero 变换，右翼的 chord/normal 会指向错误方向。

修复方式是在模型零位时进行标定：

```python
neutral_aero_rotation = (
    left_body_rotation @ aero_from_body_left
)

aero_from_body_right = (
    right_body_rotation.T @ neutral_aero_rotation
)
```

标定后左右 neutral aero frame 的最大矩阵误差约为：

```text
1.3e-15
```

### 4. STL normal 与 FlapX 有向法向相反

逐相位比较 FlapX 和 FlapSim 后发现，stroke 和 feathering 相位已经一致，但世界 `Fz` 恰好反号。

示例相位数据：

```text
FlapX:   +0.256, -0.052, +0.254, -0.051 ... N
FlapSim: -0.210, +0.201, -0.212, +0.200 ... N
```

这说明 passive feathering phase 没有反，而是导入 STL 的 oriented normal 与 FlapX force convention 相反。

最终处理方式是在 `MuJoCoAeroAdapter.apply_forces()` 中，只反转施加到 wing body 的平移 normal force：

```python
left_wrench = result["xfrc_applied_L"].copy()
right_wrench = result["xfrc_applied_R"].copy()

left_wrench[:3] *= -1.0
right_wrench[:3] *= -1.0
```

已有的 aerodynamic torque 保持不变，因为它已经能正确产生 passive feathering 的换向相位。若将整个 wrench 直接乘以 `-1`，会同时翻转 feathering torque，破坏已经正确的被动翻转动力学。

诊断数据中的 `F_world` 也同步反转，保证 telemetry 表示实际施加的力：

```python
self.diagnostics["F_world"] = -np.asarray(
    result["diagnostics"]["F_world"]
)
```

## Feathering 镜像约束

URDF 左右 wing mesh 的局部坐标并非严格镜像。空间姿态检查表明，feathering joint coordinate 同号时，翼面才是世界坐标中的镜像。

因此模型中加入被动 joint equality：

```text
q_left_feathering = q_right_feathering
```

该 equality 不指定 feathering trajectory，也没有添加 feathering actuator。角度仍由气动力、惯性、弹簧、阻尼和 joint limit 决定。

## 质量和数值参数

机器人质量重新分配为：

```text
base_link + link1                         = 20 g
link2 + left/right arm + left/right wing = 10 g
total                                     = 30 g
```

每个 wing 的质量从约 `4.4 g` 降为 `0.0626 g`，并按质量比例缩放惯量。否则 FlapX 量级的气动力矩无法克服 wing 的重力矩和转动惯量，feathering 会长期贴在一个 limit 上。

仿真使用：

```text
timestep   = 5e-5 s
integrator = implicitfast
```

Passive feathering 参数：

```text
range        = ±0.785398 rad
stiffness    = 2e-7 N·m/rad
damping      = 1e-9 N·m·s/rad
frictionloss = 1e-8 N·m
armature     = 1e-10 kg·m²
```

## 修复后的升力验证

固定基座、amplitude `1.5 rad` 的扫频结果：

```text
20 Hz: mean Fz = +0.0662 N, L/W = 0.225
30 Hz: mean Fz = +0.1486 N, L/W = 0.505
40 Hz: mean Fz = +0.2638 N, L/W = 0.896
50 Hz: mean Fz = +0.4107 N, L/W = 1.396
```

现在满足：

```text
frequency 增大 -> mean +Z lift 增大
```

30 g 机器人的重量为：

```text
0.03 * 9.81 = 0.2943 N
```

由扫频结果估计 hover frequency 约为 `42 Hz`。

## Hovering controller 的后续状态

高度方向的气动力符号已经修复，floating-base 模型会在约 42 Hz 时产生接近重量的平均升力。

但姿态闭环仍可能在 0.1--0.2 秒内发生大角度翻转。当前剩余问题是：

- 42 Hz 周期气动力矩波动很大；
- 姿态控制直接使用瞬时状态，没有按 flap cycle 低通或周期平均；
- plate-to-body moment 增益随 frequency 改变；
- yaw 尚无独立控制通道；
- Euler angle 接近大姿态时存在奇异性。

因此气动力符号 Bug 已修复，但完整稳定 hovering 仍需要周期平均的姿态估计、基于实际 control effectiveness 的增益调度，以及更可靠的 quaternion attitude controller。

## 涉及文件

```text
flapsim/fluid/mujoco_adapter.py
flapsim/fluid/aero.py
flapsim/environment/environment.py
flapsim/actuator/SingleThrustFlapper/trajectory_controller.py
flapsim/controller/SingleThrustFlapper/hovering_controller.py
flapsim/robot/SingleThrustFlapper/urdf/SingleThrustFlapper.urdf
```
