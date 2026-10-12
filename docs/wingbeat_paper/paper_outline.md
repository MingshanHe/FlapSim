# 论文大纲（草案）：按翅拍时刻施力的单电机扑翼机器人抗阵风控制

状态：2026-10-12 草案。下文数字都来自**试跑**（每个模型 1 个种子，每档 8 例），投稿前必须用 [RUNBOOK](RUNBOOK.md) 中的正式实验替换。证据来源见：
- [P0 报告](../../artifacts/wingbeat_feasibility_20261012/P0_report.md)
- [P1 报告](../../artifacts/wingbeat_snn_p1/P1_pilot_report.md)
- [H5 报告](../../artifacts/wingbeat_snn_h5/H5_report.md)
- [相位估计报告](../../artifacts/wingbeat_snn_phase/phase_report.md)

## 正式实验进展（2026-10-12，另一台电脑运行，E1–E5 已完成）

结果在 `artifacts/wingbeat_paper/`，表格见 [report/tables.md](../../artifacts/wingbeat_paper/report/tables.md)。

| 实验 | 结果 | 对照 RUNBOOK 判据 |
|---|---|---|
| E1 控制权限（16 个配对案例） | 俯仰角速度为云台残差的 5.9–7.7 倍（A=150）、12.9–13.8 倍（A=300）；前向速度 4.7 倍、9.3 倍；电流峰值 ≤3.23 A（上限的 54%） | **通过** |
| E2 阵风生存（每档 32 例） | PID 只在 0.25 m/s 及以下全部飞满，0.5 m/s 时 12/32，0.75 m/s 起 0/32；原 B2+ 到 1.25 m/s 都是 32/32；**调参 B2+ 到 1.5 m/s 都是 32/32**，1.75 m/s 时 0/32。高度 RMSE 相对 PID 增加 ≤0.08 cm | **通过**。每档要么全部飞满、要么全部失败，上限在 1.5–1.75 m/s 之间 |
| E3 未知滞后（13 个滞后 × 每格 16 例） | 调参 B2+：1.0 m/s 时容许 −45° 至 +60°，1.25 m/s 时容许 ±45°（约 ±5 ms），±75° 以外全部失败；鲁棒调参把 1.0 m/s 下的 −60° 也救回来（16/16） | **通过**。P0 中"30° 即失效"出自临界扰动，正式结果改为容许约 ±45° |
| E4 单因素鲁棒性（每格 16 例） | 调参 B2+ 在气动阻尼 ×0.5/×1.5、密度 ×0.9、质量 ×1.1、IMU 延迟和噪声 ×10 下都是 16/16，PID 都是 0/16。**密度 ×1.1 时两者都失败**（第一次阵风前约 2.7 s 就高度越界）；电机不确定度 ±4% 时 11–12/16（部分高度越界）。横向阵风下 PID 和 B2+ 都 16/16 | **部分通过**。失败来自底层 PID 的电机下限（只比配平低约 2%），压不住多出的升力，与 split-cycle 无关；见下方待办 |
| E5 消融（每格 16 例） | 去掉陷波：1.25 和 1.5 m/s 都是 0/16，1.0 m/s 时俯仰峰值大 19°；去掉高通：无风时前后漂移 +9.9 cm；理想相位：生存不变，1.5 m/s 时俯仰峰值 −5.5°；调参锁相环：生存不变，峰值 ±2° 以内 | **通过**。陷波比预期更关键（决定能否飞满） |
| E8、E6、E7 | 运行中或未开始 | — |

**由此引出的待办**：E4 中"升力过剩"的失败需要处理。可选做法：
- (a) 补一个 E4b：把底层 PID 的电机下限放宽（例如 2900 rad/s），对两个控制器重跑这两个因素；
- (b) 在论文里把它写成底层控制器的局限。

推荐 (a)。

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
2. **C2 时序与相位**：这个通道需要正确的施力时刻。正式结果（E3）：阵风 1.25 m/s 时，调参后的控制律能容忍约 ±45°（约 ±5 ms）的未知滞后，超过 ±75° 就全部失败；鲁棒调参可以把容许范围再扩大一些。P0 中"约 3 ms 即失效"只出现在临界扰动下，不作为主要表述。
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
