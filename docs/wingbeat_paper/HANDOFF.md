# 交接：翅拍同步控制研究线（给接手的 Claude 或协作者）

最后更新：2026-10-12。分支 `research/wingbeat-synchronous-snn`。本文件概括原机器上那次长对话的全部要点。原始对话记录没有上传（其中包含个人信息），本文件就是它的替代。

## 1. 先读什么

1. 本文件；
2. [paper_outline.md](paper_outline.md)：论文的主张、章节、图表、判据；
3. [RUNBOOK.md](RUNBOOK.md)：投稿前所有正式实验的命令和验收标准；
4. 如需细节，再读各阶段报告：
   - [P0](../../artifacts/wingbeat_feasibility_20261012/P0_report.md)
   - [P1](../../artifacts/wingbeat_snn_p1/P1_pilot_report.md)
   - [H5](../../artifacts/wingbeat_snn_h5/H5_report.md)
   - [相位估计](../../artifacts/wingbeat_snn_phase/phase_report.md)
   - 研究方案 [research_proposal_wingbeat_snn.md](../../research_proposal_wingbeat_snn.md)

## 2. 这条研究线是怎么来的（按时间顺序）

1. **用户的出发点**：希望 SNN 与 NEAT 在仿生扑翼机器人上发挥作用，并在仿真里充分验证后投会议。
2. **对原有项目的评估**：v3（`robust_control`，PID 残差，SNN/ANN × NEAT/PPO）在另一台机器上运行。复盘已有数据发现：
   - PPO 在 v3 中几乎没有更新；
   - 拓扑进化的效应接近零；
   - SNN 与 ANN 看不出差别；
   - 成功率主要由 10° 倾角门槛决定。

   据此起草了 v4（分解各部分的贡献），在分支 `research/v4-contribution-decomposition`，**只在原机器本地，未推送**。
3. **用户要求另起一个新方向**，于是提出本研究线：把"翅拍内的施力时刻"当作控制变量（split-cycle），用相位锁定的脉冲回路，由 NEAT 进化延迟和结构。
4. **P0 预研**（用户同意）：
   - 自由飞行中控制权限是云台残差的 6–15 倍，决策点 G0 通过；
   - 闭环中阵风生存能力显著提高；
   - 时序很敏感，约 3 ms 误差就失效。
5. **用户确定感知边界 A**：
   - 新控制器只用 IMU，相位也从 IMU 估计；
   - 底层 PID 的高度环保留测距传感器，作为基础设施；
   - 前后位置不作目标；
   - 翼根传感不在本阶段范围内。

   P0-E4 证明只用 IMU 可行（俯仰陀螺仪 98% 锁相），陷波消除了高度代价。
6. **P1**（用户说"开始"）：
   - 实现了 C++ 相位驱动、延迟 SNN、带延迟基因的 NEAT；
   - 从零进化失败；
   - 在 B2+ 上学残差后验证集超过原 B2+，但**同预算重新调参的手工控制律同样能扛住 1.5 m/s，而且连续指标更好**。
7. **用户选择"从小到大"推进**：
   - 第 1 步 H5（未知执行器滞后）：否定结果，网络不比鲁棒调参的手工控制律更好；
   - 第 2 步改为"用 SNN 代替锁相环"：梯度训练的 SNN 误差约为调参锁相环的一半，闭环效果与 GRU 相同；NEAT SNN 失败；
   - 第 3 步：写论文大纲和 RUNBOOK，并实现全部正式实验脚本（QUICK 模式已端到端跑通）。
8. **用户计划在其他电脑上运行正式实验**，并于 2026-10-12 授权把本分支推送到 GitHub。

## 3. 关键结果（试跑，每个模型 1 个种子）

| 问题 | 结果 |
|---|---|
| 控制权限（自由飞行，0.1 s 脉冲） | 俯仰角速度为云台残差的 6–15 倍，前向速度 5–10 倍；电流 ≤3.2 A；电功率变化小于 3% |
| 阵风生存（测试集每档 8 例） | PID 在 0.5 m/s 时 3/8 飞满；原 B2+ 撑到 1.25 m/s；**调参 B2+** 在 1.5 m/s 时 8/8，1.75 m/s 时 0/8 |
| 只用 IMU 的相位 | 俯仰陀螺仪 98% 锁相；锁相环闭环误差约 5–6°；陷波后高度误差恢复到 PID 水平（6.0 cm） |
| 时序 | 临界扰动下 30° 系统误差（约 3 ms）就失效；未知滞后约 ±30–45° 内可容忍 |
| 学习型残差（P1-v2） | 与调参 B2+ 生存数相同，俯仰峰值大 1.8–7.2° |
| 带滞后训练（H5） | 与鲁棒调参（k_θ=8，k_q=0.4，ψ=250°）相当，没有观察到适应 |
| 相位估计（离线 RMS） | 锁相环 6.9°，调参后 4.7°；GRU 0.8°；梯度 SNN（48 个单元，τ=5/2.5 ms）2.4°；NEAT SNN 9.3° |
| 相位估计（闭环，1.5 m/s） | GRU 与梯度 SNN 都比锁相环的俯仰峰值小约 4°（8/8 配对）；所有相位来源在 1.75 m/s 时都失败 |

**论文定位**：主线是控制机制（C1）；第二个贡献是时序和学习型相位估计（C2）；NEAT 的否定结果作为讨论（C3）。

## 4. 代码地图

| 路径 | 作用 |
|---|---|
| `experiments/algorithms/wingbeat_feasibility/probe.py` | 固定支座预研（split-cycle 权限、翼根转动信号） |
| `.../wingbeat_feasibility/preliminary.py` | P0 自由飞行 E1–E4（Python 分段驱动，`WingbeatPhase` 锁相环） |
| `.../wingbeat_feasibility/imu_phase_check.py` | IMU 锁相比例的离线检查 |
| `experiments/algorithms/wingbeat_snn/phase_step.cpp` | C++ 相位驱动：每 5 μs 按估计相位计算电机目标；不调制时与原后端逐位一致 |
| `.../wingbeat_snn/plant.py` | 被控对象（v3 的 7 g＋8 g 机体）、`HandLaw`、阵风、未知滞后、气动失配、案例划分 |
| `.../wingbeat_snn/networks.py` | 延迟 SNN 和递归 ANN 的统一图格式；`GraphController`（可作为 B2+ 上的残差） |
| `.../wingbeat_snn/evolve.py` | NEAT（带延迟基因）、三个组别、残差训练 |
| `.../wingbeat_snn/evaluate.py` | 评估阶段：gust_scan、compare、retune、lag_scan、phase_closed_loop、robustness、ablation |
| `.../wingbeat_snn/phase_data.py` / `phase_train.py` | 相位数据录制；GRU、梯度 SNN、NEAT SNN 的训练；锁相环调参；`NetPhase`、`TruthPhase` |
| `.../wingbeat_snn/paper/*.sh`、`paper_stats.py` | 正式实验脚本（可续跑，有 QUICK 模式）和统计作图 |
| `tests/test_wingbeat_snn.py` | 9 个测试：逐位一致、延迟语义、NEAT 结构约束、残差等价、滞后、SNN 导出一致 |

## 4b. 正式实验进展（2026-10-12）

另一台电脑在跑 RUNBOOK，每完成一步自动提交到本分支。已完成：
- E1–E5，结论见 [paper_outline.md](paper_outline.md) 开头"正式实验进展"一节：
  - C1 成立：调参 B2+ 每档 32 例飞满到 1.5 m/s，PID 只到 0.25 m/s；
  - E3 的时序容许范围约为 ±45°；
  - 陷波是关键部件；
  - 密度 ×1.1 和电机 ±4% 的失败来自底层 PID 的电机下限，与 split-cycle 无关。
- E8 的数据录制和 NEAT 种子 1。

E8 的其余部分、E6、E7 仍在运行或未开始。

## 5. 下一步

1. 在新机器上按 RUNBOOK 运行：`check_env.sh` → `QUICK=1 run_all.sh` → 正式的 `run_all.sh`（约 27 h，可拆分到多台机器）。
2. 结果回来后：读 `artifacts/wingbeat_paper/report/tables.md`，逐项对照 paper_outline 的判据，更新数字，开始写正文。
3. 非计算事项：画 F1 概念图、逐条核实文献、尽量做台架验证（六维力传感器测 split-cycle、真实 IMU 测锁相比例）、确定投稿目标。
4. 原机器上还有 `research/v4-contribution-decomposition`（本地未推送），以及 v3 队列（在另一台机器上）。它们与本研究线互相独立。

## 6. 工作约定与教训（新会话请遵守）

- **结论要诚实**：失败的结果同样记录；任何学习型方法都必须与"同预算重新调参的经典控制律"比较，P1 就是因为补了这一对照才避免了误判；能做逐位一致性测试的地方就做。
- **用户偏好**：先给结论，再给依据；结果需要用户拍板的地方明确提问；推送到 GitHub 等对外操作先确认。
- **长任务**：
  - 用 `tmux new-session -d` 启动。用 `wsl.exe bash -c "cmd &"` 启动的进程，会在调用 shell 退出时被杀掉，P1 第一次试跑就因此没有真正开始。
  - **不要修改正在执行的 `.sh`**：bash 边执行边按位置读取脚本，改动会导致执行错位的命令。
  - 等待后台任务时，不要用长时间的 `sleep` 轮询（用户会以为卡住了）；用后台等待任务加完成通知，并先告诉用户预计耗时。
- **原机器的推送方式**：WSL 里的 git 没有配置 GitHub 凭据，会卡在输入密码。改用 Windows 端的 `E:\功能软件\Git\cmd\git.exe -c safe.directory=* -C \\wsl.localhost\Ubuntu\home\robot\FlapSim-repo\FlapSim push ...`（凭据管理器），不修改全局配置。
- **原机器环境**：`/home/robot/miniconda3/envs/mujoco/bin/python`（Python 3.11.11、MuJoCo 3.8.0、torch 2.6.0、neat-python 2.0.0）；20 个逻辑 CPU。
