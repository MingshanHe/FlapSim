# MJX（GPU）版 FlapSim 的可行性评估（2026-10-13 22:30）

用户提议：做 MJX/JAX 版本在 GPU 上训练，用 CPU 版的结果验证 GPU 版。本文件是据代码事实做的评估，供决定是否投入。

## 现状事实

| 事实 | 含义 |
|---|---|
| 现有环境**已经在用 C++ 原生步进**：`SensorBackend` 继承 `NativeCPUBackend`，400 个 5 µs 子步连同 IMU/AHRS 都在 `sensor_step.cpp`/`native_step.cpp` 里 | CPU 侧没有"免费"提速空间；10 ms/拍（25 µs/步）是这个模型在 MuJoCo 里的真实代价 |
| 模型：19 体、nv 17、**11 个等式约束**（齿轮传动）、**无接触**、Newton 求解器 100 次迭代、implicitfast、5 µs 步长、电机 33 000 rpm | 无接触对 MJX 有利；5 µs 步长来自刚性的齿轮等式约束与高转速——这是每个 episode 160 万步的根源 |
| 空气动力：flapsim 自己的**叶素模型**（`flapsim/fluid/blade_element.py`，C++ 版在 `native_step.cpp`），作用于机翼的 xfrc；XML 另有 `fluidshape="ellipsoid"`、`density=1.225` | MJX 版必须把叶素模型写成 JAX（可向量化，约 150 行）；MuJoCo 内置椭球流体 **MJX 3.8.0 不支持**（只有 inertia-box），且 `implicitfast not implemented for fluid drag` |
| 仓库已有 `MJXBackend`（`flapsim/backends.py`，`mjx.put_model(impl='jax')` + `fori_loop`），`config.backend='mjx'` 开关存在，但**没有任何测试覆盖** | 是脚手架，不是可用实现 |
| 本机 GPU：RTX 5060 Ti 16 GB，JAX 0.10.0 已识别 CUDA 设备 | 探针可以就地做 |
| JAX 默认 float32；消费级 GPU 的 FP64 吞吐约为 FP32 的 1/32–1/64 | 5 µs 步长 + 33 000 rpm 齿轮在 float32 下的数值稳定性是最大的未知 |

## 阻塞点（必须逐个解决）

1. **椭球流体**：要么确认 XML 的椭球流体在当前配置下实际贡献为零（叶素模型承担全部气动），要么在 JAX 里重写 MuJoCo 的椭球模型（附加质量、黏性阻力、Kutta 升力、Magnus，公式在 MuJoCo 文档里，约 200 行），并逐步对照 `mj_passive` 的输出做单元测试。
2. **implicitfast + 流体**：MJX 对带流体参数的模型拒绝 implicitfast。可把 MJX 模型的 density/viscosity 设 0、气动全部作为 xfrc 外力施加（CPU 版本的叶素模型本来就是这样做的），保留 implicitfast——但这改变了内置流体部分的隐式处理，需要验证。
3. **float32 稳定性**：齿轮等式约束在 5 µs、float32 下能否稳定 8 s（160 万步）。若不行，要么 x64（GPU 上慢几十倍，失去意义），要么重新设计传动约束（改模型，需重新验证整台机器）。
4. **控制回路向量化**：PID、PLL、LIF 网络、噪声/延迟传感器、扰动脉冲、终止判据都要写成 JAX 并在 N 个环境上并行；NEAT 的异构拓扑要填充成稠密掩码矩阵。梯度训练（模仿/RL）是最大受益者；NEAT 受益有限。
5. **验收**：不会逐位一致，只能统计等价。验收集 = 今天的预注册数字：law_tuned 旧/新 16 例 J 5.86/5.39、脉冲权限 4.1×、包线表、比较器 SNN 比值 1.02、全状态上限 7.8°。

## 收益估算（如果阻塞点都解决）

- 每个 8 s episode 160 万步。GPU 上 2048 个环境并行，每环境步约 0.5–2 µs（无接触、nv 17、Newton 解 11 个等式约束）→ 一批 2048 个 episode 约 15–50 分钟，即每 episode 0.4–1.5 s，对比 CPU 单核 45 s。
- 相对本机 18 核：约 **10–40 倍**吞吐；相对 AWS 64 核约 3–10 倍。
- 对 RL/模仿训练：还能用 MJX 的可微性做解析策略梯度——这是 CPU 上做不到的新能力。

## 建议：先做一天的可行性探针，再决定

| 步骤 | 内容 | 判据 |
|---|---|---|
| P1（2 h） | 在 MJX 里加载模型（流体参数清零），1024 个环境并行跑 2 s 开环（固定云台、固定转速），测 steps/s 与 float32 稳定性 | 不发散、齿轮相位与 CPU 版一致到 1%；吞吐 ≥ 1e7 环境步/s |
| P2（3 h） | 叶素模型的 JAX 实现；在 CPU 录下的状态序列上逐步对照 `native_step.cpp` 的气动力 | 力与力矩相对误差 < 1e-3 |
| P3（3 h） | 把 CPU 版一个 episode 的指令序列开环回放到 MJX，比较姿态/位置轨迹 | 2 s 内偏差 < 1°、< 2 cm；记录发散时间 |
| 决策 | P1–P3 通过 → 投入移植（传感器、控制回路、训练流水线，1–2 周）；任一失败 → 留在 CPU，用 AWS 多核扩展 |

成本对照：AWS c7a.16xlarge 把一次 B' 从 15 h 缩到 2 h，零工程风险、$6 一次；MJX 的收益是再快一个量级并解锁可微训练，代价是 1–2 周和上面的四个未知。两者不冲突：探针在本机 GPU 上做，期间 AWS 继续跑现有流水线。
