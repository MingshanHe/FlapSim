# 论文大纲（草案）：按翅拍时刻施力的单电机扑翼机器人抗阵风控制

状态：2026-10-12 草案。下文数字都来自**试跑**（每个模型 1 个种子，每档 8 例），投稿前必须用 [RUNBOOK](RUNBOOK.md) 中的正式实验替换。证据来源见：
- [P0 报告](../../artifacts/wingbeat_feasibility_20261012/P0_report.md)
- [P1 报告](../../artifacts/wingbeat_snn_p1/P1_pilot_report.md)
- [H5 报告](../../artifacts/wingbeat_snn_h5/H5_report.md)
- [相位估计报告](../../artifacts/wingbeat_snn_phase/phase_report.md)

## 题目与定位

- **暂定英文题目**：*Timing is control: wingbeat-synchronous drive modulation for gust rejection in a single-motor flapping-wing robot with IMU-only phase estimation*
- **一句话**：不加执行器，只改变单个电机在每个翅拍内的施力时刻，就能让 15 g 扑翼机器人可承受的前后阵风从不到 0.5 m/s 提高到 1.5 m/s；所需的翅拍相位只用 IMU 就能锁定。
- **投稿方向**：
  - 纯仿真可投：Bioinspiration & Biomimetics、Living Machines。
  - IROS、ICRA、RA-L：需要至少一项实机或台架验证（见文末"仿真之外"）。

## 贡献（按证据强度排序）

1. **C1 控制机制（主线）**：翅拍同步的驱动调制（split-cycle）是单电机扑翼机器人的一个新控制通道。
   - 自由飞行中，它带来的俯仰角速度权限是 v3 云台残差的 6–15 倍；电功率变化小于 3%，电流 ≤3.2 A。
   - 一条只用 IMU 的简单控制律，把可承受阵风从不到 0.5 m/s 提高到 1.5 m/s。
2. **C2 时序与相位**：这个通道对时序非常敏感。临界扰动下约 3 ms 的系统误差就会抵消收益；在可承受的扰动范围内，未知滞后可以容忍约 ±30–45°。
   - 俯仰陀螺仪 98% 的方差与翅拍锁相，所以只用 IMU 就能锁定相位。
   - 学习型估计器从原始 IMU 估计相位，比调参后的锁相环更准：替代梯度训练的 SNN 误差约为其一半，GRU 约为其 1/6。在强阵风下可再减小约 4° 俯仰峰值，但不能提高可承受的阵风上限。
3. **C3 否定结果（讨论部分）**：
   - NEAT 进化的残差控制器（SNN 或 ANN）不优于同等调参预算的手工控制律；
   - 未知滞后下没有观察到飞行中的适应；
   - NEAT 进化的 SNN 相位估计器比锁相环更差。
   - 教训：任何学习型方法都必须与"同预算重新调参的经典控制律"比较。

## 章节结构

| 节 | 内容 | 图表 | 证据状态 |
|---|---|---|---|
| 1 引言 | 欠驱动扑翼机器人的转向与抗扰难题；昆虫按翅拍组织飞行控制（转向肌的相位放电、平衡棒）；split-cycle 已有工作（手工推导，多为准定常分析）；本文贡献 | F1 概念图：曲柄相位、调制波形、昆虫类比 | 待画；文献需逐条核实 |
| 2 平台 | IST_robot-V2 闭链 MuJoCo 模型、叶素气动、电机模型、虚拟 IMU 与测距传感、5 μs 物理步长；"相位驱动"的 C++ 实现，以及不调制时逐位一致的检查 | T1 参数表 | 已有 |
| 3 控制通道 | 固定支座下的力与力矩随 ψ 的变化；自由飞行开环权限及代价 | F2 ΔFx、ΔMy 随 ψ 的变化；F3 自由飞行权限（E1） | F2 已有；**F3 需 E1 正式版（16 例）** |
| 4 只用 IMU 的相位 | 陀螺仪锁相比例；锁相环设计；把机体角速度与翅拍摇动分开的陷波（解释高度代价的来源） | F4 陀螺仪波形与锁相比例；T2 锁相环跟踪误差 | 已有（p0_imu_phase_check、P0-E4） |
| 5 控制律与抗阵风 | B2+ 控制律（俯仰 PD、高通、陷波、ψ 选择），只在训练集上调参 | **F5 生存率随阵风变化的曲线**（PID、B2+、调参 B2+，Wilson 区间）；T3 连续指标的配对差异 | **需 E2**（每档 32 例） |
| 6 时序敏感性与鲁棒性 | 未知滞后曲线；单因素失配（空气密度、阻尼、质量、电机、IMU 噪声与延迟）；横向阵风作为对照（split-cycle 对横滚无作用） | **F6 生存率随滞后变化**；**T4 鲁棒性表**；T5 消融（去掉高通、陷波；理想相位；调参锁相环） | **需 E3、E4、E5** |
| 7 学习型相位估计 | GRU、替代梯度 SNN、NEAT SNN 与默认及调参锁相环的对比；离线与闭环；SNN 的发放率与运算量 | **F7 离线误差与锁定时间**（5 个种子）；**T6 闭环表** | **需 E8**（5 个种子） |
| 8 学习型控制器的否定结果 | NEAT/固定拓扑 SNN 和 ANN 残差对比调参手工控制律；带滞后训练对比鲁棒调参 | **T7 按种子汇总，含符号检验**；T8 H5 | **需 E6、E7**（5 个种子） |
| 9 讨论 | 为什么权限而不是相位精度决定上限；准定常气动的局限；单电机只覆盖前后和俯仰；能耗主张的边界（不作）；对神经形态控制研究的启示（必须有调参基线） | — | 写作 |
| 10 结论 | — | — | 写作 |

## 摘要草稿（英文，[ ] 处待正式结果替换）

> Single-motor flapping-wing robots lack direct pitch and fore-aft authority. We show in a closed-chain MuJoCo model with blade-element aerodynamics that modulating the drive *within* each wingbeat (split-cycle timing) provides [6–15]× the pitch-rate authority of a gimbal residual at <3 % electrical-power cost. The wingbeat phase is recovered from the IMU alone: the pitch gyro is 98 % phase-locked to the stroke. A simple IMU-only law raises the survivable fore-aft gust from <0.5 m/s to [1.5] m/s ([n]/[n] held-out cases), and the benefit vanishes with a few milliseconds of timing error. Learned raw-IMU phase estimators reduce phase error relative to a tuned PLL (spiking network trained with surrogate gradients: [2×]; GRU: [6×]) and trim pitch excursions in strong gusts, but do not extend the gust limit. Neuroevolved controllers and estimators did not outperform tuned classical baselines.

## 每个主张的判据

| 主张 | 正式实验 | 成立的条件 | 不成立时的处理 |
|---|---|---|---|
| C1 可承受阵风提高 | E2 | 调参 B2+ 在 1.25 m/s 的 Wilson 下限高于 PID 在 0.5 m/s 的上限；1.5 m/s 生存率 ≥ 0.9 | 按实际曲线改写倍数 |
| C1 代价小 | E1、E2 | 电流 ≤ 上限的 60%；高度 RMSE 增加 ≤ 1 cm | 报告代价；讨论升力补偿 |
| C2 时序敏感 | E3 | 生存率在某个滞后 ± 范围内骤降，曲线可复现 | 改为报告容许范围 |
| C2 学习型估计更准 | E8 | 5 个种子的 SNN（梯度）相位 RMS 均低于调参锁相环（符号检验 5/5，p=0.06；若要 p<0.05 需 ≥6 个种子） | 只报告 GRU，SNN 列为未达标 |
| C3 否定结果 | E6、E7 | 按种子比较，学习型残差对调参 B2+ 没有一致优势 | 若某组别 5/5 个种子都更好，则改写结论 |

## 仿真之外（投稿前建议）

1. **台架验证 split-cycle**：在六维力传感器上，对真实 IST 机构用 ±15% 电压按相位调制，测量 ΔFx 和 ΔMy 随 ψ 的变化。这是对准定常气动最直接的检验。
2. **实测 IMU 的锁相比例**：把 IMU 装在真实机体上，测 gyro_y 在翅拍频率上的锁相方差和锁相环误差，检查安装共振的影响。
3. 第 1 项若做不了，主张需要相应降级为"仿真证据"，并在讨论中明确说明。
