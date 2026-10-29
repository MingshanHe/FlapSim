# PID 悬停与 Waypoint 控制实现记录

## 目标

对质量约 `30 g` 的 `SingleThrustFlapper` 实现自由基座悬停：

- 从 `z = 0.5 m` 起飞并保持目标高度；
- 稳定机体 Roll/Pitch；
- 控制水平位置 `x/y`；
- 每 `3 s` 切换一个 waypoint；
- 保留扑翼和 plate 关节的可视运动。

运行方式：

```bash
python flapsim.py --robot SingleThrustFlapper --fix false
```

`--fix true` 仍使用固定基座和逐时刻详细气动力，不启用悬停控制。

## 为什么没有直接在自由飞行中使用逐时刻气动力

固定基座扫频确认 `42 Hz` 左右的平均升力可以平衡重量，但逐时刻模型在自由基座下存在很大的周期气动力矩。与此同时，stroke trajectory 直接覆盖关节的 `qpos/qvel`，会向轻量多刚体系统注入能量。两者叠加后出现：

- 机体在很短时间内翻转；
- `QACC` 数值发散；
- PID 还没有来得及响应，姿态就已经超出线性控制范围。

因此浮动模式使用 `AveragedHoverAero` 周期平均模型。固定模式继续保留详细 `MuJoCoAeroAdapter`，用于检查单翼气动力和 passive feathering。

## 周期平均气动模型

固定基座实测：

```text
30 Hz -> mean Fz = 0.14858349 N
```

使用平方关系拟合升力：

```text
L = k_f * frequency^2
k_f = 0.14858349 / 30^2
```

30 g 机器人的重量为：

```text
W = 0.03 * 9.81 = 0.2943 N
```

所以理论悬停频率约为：

```text
frequency_hover = sqrt(W / k_f) ~= 42.2 Hz
```

Plate 指令同时改变推力方向：

```text
direction_body = normalize([-plate_roll, -plate_pitch, 1])
```

固定基座辨识得到 plate 到机体力矩的近似增益：

```text
roll moment  / plate_pitch = 0.01055 / 20 N*m/rad
pitch moment / plate_roll  = 0.01144 / 20 N*m/rad
```

增益随频率平方缩放：

```text
scale = (frequency / 42)^2
```

为了表示真实扑翼的周期平均阻力，并抑制自由飞行中没有耗散的水平摆动，还加入：

```text
angular damping    = [2e-4, 2e-4, 1e-4]
horizontal damping = 0.06 N/(m/s)
```

## 控制器结构

控制器是三级串级结构：

```text
waypoint (x, y, z)
        |
        +-- z 高度 PID --------> flapping frequency ------> lift
        |
        +-- x/y 位置 PD -------> desired body tilt
                                      |
                                      +-- Roll/Pitch PD --> plate angles --> moments
```

状态不是直接使用单个仿真步的瞬时值，而是在一个扑翼周期内累加并取平均。控制更新周期为：

```text
control_period = 1 / current_frequency
```

这样可以避免控制器追踪 42 Hz 的周期振动。

## 高度 PID

高度误差：

```text
e_z = target_z - z
```

控制律：

```text
frequency = 42
          + Kp_z * e_z
          - Kd_z * velocity_z
          + Ki_z * integral(e_z)
```

当前参数：

```text
Kp_z = 8.0
Kd_z = 4.0
Ki_z = 2.0
integral limit = +/-0.5
frequency limit = [20, 55] Hz
```

积分项负责消除气动拟合误差带来的高度静差，并通过积分限幅避免 windup。

## 水平位置外环

x/y 位置控制不直接产生水平力，而是产生目标机体倾角：

```text
desired_pitch_tilt = Kp_xy * (target_x - x) - Kd_xy * velocity_x
desired_roll_tilt  = -Kp_xy * (target_y - y) + Kd_xy * velocity_y
```

当前参数：

```text
Kp_xy = 0.10
Kd_xy = 0.50
tilt limit = +/-0.10 rad
```

Roll 的符号与 Pitch 不同，是因为小角度时机体 z 轴在世界坐标中的水平分量满足：

```text
acceleration_x proportional to +pitch_tilt
acceleration_y proportional to -roll_tilt
```

## 姿态内环

姿态控制没有直接使用 Euler Roll/Pitch 作为反馈，而是使用机体 z 轴相对世界 z 轴的倾斜量：

```text
roll_tilt  = -body_z_world[y]
pitch_tilt =  body_z_world[x]
```

这比直接用 Euler 角更适合小角度悬停，也避免部分角度绕序问题。

当前姿态 PD 参数：

```text
Kp_attitude = 0.80
Kd_attitude = 0.06
plate limit = +/-0.10 rad
```

早期版本使用 `Kp=0.04`、`Kd=0.003` 和未缩放的力矩增益，虽然机体
能够稳定，但 plate 最大只有约 `0.229 deg`，导致 thrust module 在 viewer
中看起来一直竖直。现在把关节控制量放大 20 倍，并把 plate-to-moment
增益除以 20，使控制力矩量级基本不变，而可见的 plate 最大偏转约为
`4.62 deg`。Plate 会在 waypoint 切换时明显偏转，接近目标后重新回到
接近竖直，这是位置 PID 收敛后的正常表现。

URDF 中两个 plate joint 的命名和物理作用轴是交叉的：

```text
pitch_joint -> 绕 +X -> 主要控制 body Roll
roll_joint  -> 绕 -Y -> 主要控制 body Pitch
```

因此控制律为：

```text
plate_pitch = -Kp * (roll_tilt  - desired_roll_tilt)  - Kd * roll_rate
plate_roll  =  Kp * (pitch_tilt - desired_pitch_tilt) + Kd * pitch_rate
```

## Waypoint 更新

默认 waypoint：

```text
0: (0.00, 0.00, 0.50)
1: (1.00, 0.00, 0.50)
2: (1.00, 1.00, 0.80)
3: (0.00, 1.00, 0.80)
```

每 `3 s` 切换一次，到最后一个点后循环回第一个点。也可以通过 Python 设置：

```python
simulation.set_hover_waypoints(
    [(0.0, 0.0, 0.5), (0.2, 0.0, 0.6)],
    interval=3.0,
)
```

## Plate 外观曾经不变化的原因

最初为了排除自由飞行数值发散，浮动模式断开了物理 `PositionController`。结果是：

- hovering controller 中的 plate 指令正常变化；
- 平均气动力也使用了这些指令；
- 但 MuJoCo 中可见的 `pitch_joint/roll_joint` 没有跟随指令；
- 关节受重力弯曲后，看起来保持在一个固定角度。

现在浮动模式使用 kinematic position mode：每步把 plate 的 `qpos` 同步为控制命令、把对应 `qvel` 设为 0，但不施加最高 `0.5 N*m` 的 PD 力矩。这样显示与控制量一致，同时不会重新向轻量内部关节注入能量。

正常启动没有人为初始姿态扰动。曾使用的：

```text
initial body Roll  = +10 deg
initial body Pitch = -8 deg
```

只存在于独立验证脚本中，是对 `floating_base` 四元数的扰动，不是 plate joint 初始角度。

## 验证结果

水平初始状态运行 20 s：

```text
position ~= (0.0000, 0.0000, 0.4999) m
body Roll/Pitch ~= 0 deg
frequency ~= 42.22 Hz
```

从 body Roll `+10 deg`、Pitch `-8 deg` 开始运行 20 s：

```text
position = (-0.0061, -0.0098, 0.4999) m
body Roll = -0.02 deg
body Pitch = +0.01 deg
frequency = 42.22 Hz
```

Waypoint 切换测试中，目标在 `3/6/9/12 s` 正常更新，高度和姿态保持稳定。

为便于在 viewer 中观察姿态控制，默认水平 waypoint 后来由
`0.10 m` 扩大到 `1.00 m`，高度变化由 `0.05 m` 扩大到 `0.30 m`。
一轮 15 s 闭环测试中：

```text
maximum |body Roll|  ~= 2.1 deg
maximum |body Pitch| ~= 2.1 deg
maximum |plate pitch_joint| = 4.62 deg
maximum |plate roll_joint|  = 4.62 deg
```

倾角命令仍受 `+/-0.10 rad`（约 `+/-5.7 deg`）限制，所以放大路径后
可以清楚看到倾斜，同时不会因为 waypoint 距离过大而要求危险姿态。

## 当前模型边界

- 浮动悬停使用的是周期平均气动模型，不是逐时刻翼面气动力；
- 目前没有独立 yaw actuator，yaw 主要依赖阻尼，不能跟踪指定 yaw；
- plate 指令通常只有约 `1e-4` 到 `1e-3 rad`，视觉变化很小，但关节状态已经与命令一致；
- 后续若恢复详细瞬时气动，需要对扑翼周期力矩进行系统辨识，并避免用直接覆盖 `qpos/qvel` 的方式向自由多刚体系统注入能量。

## 涉及文件

```text
flapsim/controller/SingleThrustFlapper/hovering_controller.py
flapsim/fluid/averaged_hover.py
flapsim/actuator/SingleThrustFlapper/position_controller.py
flapsim/actuator/SingleThrustFlapper/trajectory_controller.py
flapsim/environment/environment.py
flapsim/sim_core.py
flapsim.py
```
