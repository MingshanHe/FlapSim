# FlapSim 当前研究交接：PID、SNN/ANN、NEAT/PPO

本文件是其他 AI 和协作者的当前入口。**先核对实时状态，再开始操作；不要仅凭本文件的进度快照启动第二套训练。** 仓库中的旧 `START_HERE.md`、PPO 演示和旧机器路径属于历史记录。

## 1. 用户要解决的问题与已确认约束

目标是在扑翼机器人完整物理仿真中，比较以下问题，而不是预先宣布 SNN 或 NEAT 获胜：

1. SNN/ANN 学习 PID 后，经过 PPO 或 NEAT 闭环优化，能否可靠替代 PID 内环？
2. 保留 PID 内环、由网络提供有界动作残差，能否获得更稳健的控制？
3. SNN＋NEAT 若有改善，贡献来自结构进化，还是仅仅来自权重搜索？
4. 后续能否执行期望飞行方向、上升、下降、反向和停止，也就是速度指令跟踪？

用户已授权大型训练、测试、环境部署和本地分支提交。用户于 **2026-10-11 最新授权更新 GitHub**，覆盖更早“先不上传”的偏好；自动训练进程自身仍只做本地提交，人工/助手推送当前研究分支。不修改 `master`，不强推，不上传凭据或本地环境。

必须保持的研究条件：

- 神经计算采用**原逐时间步 Python/PyTorch 自动求导、float64**；用户明确要求停用自定义神经网络加速内核。现有原生物理后端保留，物理步长 5 μs，控制步长 2 ms（500 Hz）。
- 保高/姿态控制比较使用相同语义的 **11 维输入**。旧 `ppo_v2` 与当前接口不一致，排除在此次比较外。
- 速度控制另立 **16 维任务**：增加带噪声和时延的水平速度测量，以及三维目标速度；不能和 11 维任务混为同一比较。
- 保留全部种子与失败结果。不能用较低训练损失、选模验证高分或某一个最好种子，代替独立鲁棒性结论。
- “本结构、初始化、预算下未能替代 PID”可以由实验支持；“所有 ANN/SNN 都不可能替代 PID”不能由这些实验推出。

## 2. 当前机器、分支与状态快照

| 项目 | 当前值 |
|---|---|
| 工作目录 | `/home/hemingshan/FlapSim`（WSL/Linux） |
| 本地分支 | `research/pid-residual-velocity-v3` |
| GitHub | `https://github.com/MingshanHe/FlapSim.git` |
| 环境 | 仓库 `.venv`，是 Conda 前缀；不是全局 Python |
| 主要版本 | Python 3.11.11、PyTorch 2.6.0 CPU、NumPy 2.2.4、MuJoCo 3.8.0、NEAT-Python 2.0.0 |
| 资源配置 | 8 核 / 16 线程；两个训练任务并行，各 8 个仿真进程；评估最多 16 个进程 |
| 当前运行根目录 | `runs/robust_control/20261010_v3` |
| 主报告 | [artifacts/robust_control_20261010/report.md](artifacts/robust_control_20261010/report.md) |
| 阶段报告 | [interim_20261012/report.md](artifacts/robust_control_20261010/interim_20261012/report.md) |

<!-- LIVE_SNAPSHOT_START -->
截至 **2026-10-11 17:35 JST**：完成 **12/65** 组完整训练。 当前未完成训练：`hover_residual_ann_ppo_seed6203`：3,452,000/3,584,000 步（96.3%）。 阶段调度状态为 `waiting_for_active_runs_to_finish`；完成 0/18 个控制器的独立阶段评估。原最终测试尚未开始。
<!-- LIVE_SNAPSHOT_END -->

进度的权威来源是本机 `runs/.../status.json` 和每个任务的 `results.json`，不是 GitHub 上的静态快照。主状态的 `unix` 是上次阶段事件时间；判断进程活性应看 `heartbeat_unix`、子任务进度及实际进程。`artifacts/.../status.json` 是生成报告时复制的快照，不会每十秒刷新。

## 3. 已经做过什么、目前有哪些证据

### 3.1 历史实验与 PID 模仿失败定位

- 旧 `snn_hover` 的 2201 等训练记录保留；其未完成的 27 组继续暂停，不另开旧队列。符合 11 维接口的历史 SNN-NEAT、ANN-NEAT、循环 PPO 会在 v3 中单独复测，保留旧 100 Hz 推理频率，不计作新的 500 Hz 配对种子。
- `snn_pid_compare` v2.1 已完成 SNN/ANN 各三个种子的模仿训练及两轮 DAgger，但未通过进入其正式比较的门槛。不要重启其受门槛约束的后续队列来代替 v3。
- v2.1 最终常规验证：PID 16/16；SNN 三个种子为 7/16、5/16、6/16；ANN 均为 0/16。ANN 通常在约 1.2–1.3 s 因姿态问题终止。
- 已查出起飞动作误差、学生偏离教师后的恢复状态覆盖不足、持续闭环反馈问题。PID 控制最初 1 s 的干预帮助了一部分 SNN 案例，但 ANN 仍会后续失稳；不能将问题归为单一的冷启动原因。
- 六个最终策略的 PyTorch 与部署推理在连续 20 s 输入上最大差异 <6×10⁻¹⁶；未发现导出精度损失。

详见 [failure_analysis.md](artifacts/snn_pid_compare_20261010/failure_analysis.md)。它区分测量事实、干预证据和未证实解释，后续分析应沿用这些边界。

### 3.2 v3 已完成训练的选模验证

2026-10-11 17:00 的固定快照如下。这里每个模型使用同一组 **8 个常规＋8 个扰动、每例 12 s** 的选模验证，和上面的 v2.1 验证不同。

| 控制器 | 各已完成种子的成功数 /16 |
|---|---|
| 固定 PID / 零残差初始化 | 13 |
| PID＋SNN，NEAT | 16、14、16 |
| PID＋ANN，NEAT | 15、15、14 |
| PID＋SNN，PPO | 13、13、13 |
| PID＋ANN，PPO | 13、13；第三个仍在训练 |

这些选中策略的常规场景都是 8/8；改善主要发生在扰动场景。未达标的选模案例都飞满 12 s，但倾角超过门槛。PPO 尚未提高成功数，不等于连续误差或回报完全没有变化。NEAT 也不是所有指标都更好，例如部分种子的高度误差略有增加。

验证参与检查点选择，**不能把这些数字填进独立测试结果表**。完整快照及逐例证据见 [progress_20261011.md](artifacts/robust_control_20261010/progress_20261011.md)。

### 3.3 速度控制开发

已做开发辨识与固定速度外环试验，发现横向/前向倾角到速度的增益明显不同；对称控制公式会放大起飞摆动。冻结后的低速指令方案使用起飞阶段、平滑反向和停止；8 个开发案例都飞满 20 s，但只有 4 个严格通过。

这只是**固定外环的开发结果**，不是网络训练结果，也不是独立速度测试。正式速度训练仍在后续队列。详见 [velocity_development.md](artifacts/robust_control_20261010/velocity_development.md)。

## 4. 完整研究计划：65 组训练

详细方案见 [research_proposal_v3.md](research_proposal_v3.md)；技术定义见 [technical_details.md](artifacts/robust_control_20261010/technical_details.md)。

| 任务 | 方法 | 种子 | 运行数 |
|---|---|---|---:|
| PID 内环＋残差，姿态/保高 | SNN/ANN × NEAT/PPO | 6201–6205 | 20 |
| 网络直接控制，姿态/保高 | SNN/ANN × NEAT/PPO | 6201–6205 | 20 |
| PID 内环＋速度外环＋残差 | SNN/ANN × NEAT/PPO | 6201–6205 | 20 |
| 固定拓扑 SNN 权重进化对照 | SNN＋evo | 6201–6205 | 5 |

保高每组分配 3,584,000 动作步，速度每组 5,120,000 步。NEAT 为 16 个体、12 代；PPO 的分配步数与同任务 NEAT 一致。NEAT 提前终止造成实际消费低于分配预算时，两者分开记账。保高时长课程 4/8/16 s，速度 8/12/20 s。

残差动作是 `clip(PID + [0.035, 0.035, 100] × tanh(network), physical_limits)`。残差初始化输出为零，对应 PID；同种子的 PPO/NEAT 共用初始网络。直接组使用已选定的 SNN 5101 / ANN 5102 模仿先验，后续五个优化种子不是五个独立预训练种子。

完整训练结束后才封存模型并执行原最终测试：每个保高控制器 72 例，每个速度控制器 80 例；包含常规、扰动、60 s 长时、单因素变化、组合 OOD，速度另加指令泛化。按已测吞吐，10 月 11 日下午估计剩余完整研究约 **4–6 天**，不是保证完成时间。

## 5. 最新调整：明天下午前先形成独立阶段证据

用户要求 **2026-10-12 下午前**看到有研究价值的阶段结果，执行目标设为 **14:00 JST**。[阶段修订方案](artifacts/robust_control_20261010/interim_plan_20261012.md)已部署；修改的是执行顺序，原训练代码、预算、选模和判据保持不变。

实际顺序：

1. 当前 ANN＋PPO seed6203、SNN＋NEAT seed6204 完整结束，原调度器在任务边界释放锁。
2. 对 PID 和四种组合各前三个种子进行独立评估，共 13 个控制器；按预定种子顺序选取，不按验证名次选取。
3. 提前运行原队列中的 SNN 固定拓扑 evo 6201–6203，以及 SNN＋NEAT 6205，共四组训练，两个任务并行。
4. 评估这三个机制对照和 SNN＋NEAT 6204/6205；后两个仅作补充证据，主配对比较仍使用共同的前三个种子。
5. 自动续跑原 65 组队列，跳过已完成任务；直接控制和速度任务继续保留。

阶段评估共 **18 个控制器 ×48 例 =864 例**：每个控制器 16×20 s 常规、16×20 s 扰动、4×60 s 常规、4×60 s 扰动、8×20 s 组合 OOD。案例随机种子从 `1_700_000_000` 开始，与原训练、验证、最终测试隔离；同一案例在所有方法间配对。策略在访问该组测试前按验证结果复制封存。

预计阶段工作约 10–16 小时，今晚先出现核心比较，明天上午补充机制对照；实际以运行进度为准。每个控制器评估完即更新阶段报告，不等全组完成。

### 调度机制：接手时尤其注意

| 对象 | 职责 |
|---|---|
| tmux `flapsim-robust-v3` | 原队列；当前任务结束后正常退出 |
| tmux `flapsim-interim-20261012` | 已启动的阶段调度器，等待交接；阶段结束后在此进程下启动原完整队列 |
| `runs/.../pause_request.json` | 有意创建的交接请求，`owner=interim_20261012`；不是异常卡住的迹象 |
| `runs/.../campaign.lock` | 防止两个调度器同时运行训练/评估 |
| `runs/.../interim_20261012/plan.json`、`plan_seal.json` | 独立阶段协议及校验值 |
| `runs/.../interim_20261012/status.json` | 当前阶段；等待时另有 `waiting_heartbeat.json` |
| `runs/.../interim_20261012/evaluation_progress.json` | 正在评估的模型及已完成案例数 |
| `runs/.../interim_20261012/partial/` | 逐案例持久化记录，支持恢复 |

**正常交接期间不要删 `pause_request.json`，不要停止等待中的调度器，也不要手动启动第三个训练进程。** 原 tmux 会话退出后，工作可能已经转到新的 tmux 中，不能仅因旧会话消失判定训练中断。

## 6. 会交付什么结果，以及如何解释

| 产物 | 用途 |
|---|---|
| 主 `report.md`、`learning_curves.png` | 完成训练、选模验证与最终独立测试的实际记录 |
| 阶段 `interim_20261012/report.md` | 18 个控制器的独立阶段成功数、逐种子门槛和配对差异，随完成进度更新 |
| `analysis.json`、`evaluation/*.json` | 逐场景结果、未达判据、姿态/高度误差、配对比较 |
| `models/*.json`、`policies/*.json` | 按验证选中的策略副本、选择阶段与哈希 |
| `record_manifest.json` | 已归档科学记录的大小和 SHA-256；根主清单覆盖 `records/`，阶段清单覆盖阶段目录 |
| `technical_details.md`、`protocol.json` | 时钟、网络语义、观测、控制范围、预算、冻结实现及资产来源 |

阶段报告和最终报告采用的四项稳健门槛是：常规 ≥15/16、扰动 ≥14/16、60 s 常规 4/4、60 s 扰动 ≥3/4。OOD 单列，不能将长时失败用高总体成功率掩盖。

阶段结果可以初步回答：残差是否优于 PID、NEAT 与 PPO 的差别、SNN 与 ANN 的差别，以及结构进化的额外收益。主比较只有三个训练种子，双侧精确符号翻转检验最小 p=0.25；不能宣称统计显著，不能把 48 个场景当作 48 个独立训练重复。通过仅指已测试仿真包络，不代表实机结论。

阶段计划是在看到旧验证结果之后制定的，此事实已记录。阶段测试不能用于重新挑检查点或改原队列超参数；如由结果引出新方案，应另立实验和新的未访问测试集。**预期产物是可复核的证据，不保证得出 SNN-NEAT 优于所有方法。**

## 7. 代码入口、已经验证过的内容

| 文件 | 作用 |
|---|---|
| `experiments/algorithms/robust_control/environment.py` | 11/16 维环境、PID 残差、速度指令和外环 |
| `networks.py`、`evolution.py` | SNN/ANN 语义、NEAT 导入/导出与结构约束 |
| `train_ppo.py`、`train_neat.py` | 原计算方式的训练、验证选模和检查点 |
| `campaign.py` | 原 65 组队列、代码/资产封存、最终评估 |
| `interim.py`、`interim_evaluate.py` | 阶段交接、优先任务、独立测试、恢复和自动续跑 |
| `reporting.py` | 根据真实存在的记录生成主报告 |
| `experiments/algorithms/snn_pid_compare/diagnose.py` | 旧模仿失败的离线与回放诊断 |

训练子进程从 `runs/.../frozen_source/` 启动。该目录和物理资产按原 `protocol.json` 校验，**不能直接编辑它们**。阶段评估入口也已确认导入这套原封存计算；新的调度源码另存于 `interim_20261012/orchestration_source/`。只改工作区文件，不会自动改变已经启动/封存的实验。

v3 初始部署通过 12 个训练器接口 smoke 检查和既有回归检查。此次阶段调整又验证了 11 项 v3 控制测试及 4 项阶段边界测试；小规模开发案例检查验证了原封存实现导入、重复执行幂等、逐案例恢复及协议哈希变更拒绝。检查记录见 [interim_preflight.json](artifacts/robust_control_20261010/interim_preflight.json)。这些工程检查不是科学成功率。

## 8. 下一位 AI 应如何继续

先读取本文件、两份报告，再在**当前原机器**查看实时进度：

```bash
cd /home/hemingshan/FlapSim
git status --short --branch
tmux list-sessions
cat runs/robust_control/20261010_v3/status.json
cat runs/robust_control/20261010_v3/interim_20261012/status.json
tail -n 20 runs/robust_control/20261010_v3/interim_20261012/orchestrator.log
```

进程仍运行时，以观察和分析为主，无需重复启动。阶段尚未完成且调度器确实退出时，先查看异常、校验暂停请求归属和协议；保留已有目录，用原入口恢复：

```bash
bash experiments/algorithms/snn_pid_compare/run.sh \
  -m experiments.algorithms.robust_control.interim \
  --root runs/robust_control/20261010_v3 \
  --destination artifacts/robust_control_20261010
```

阶段已经完成、原完整队列也确实不在运行时，才使用 `experiments.algorithms.robust_control.campaign` 的原入口恢复。不要通过删除暂停/锁相关文件绕过未完成的交接。NEAT 恢复完整代检查点；PPO 保留优化器与 RNG，但显式重新开始未完成回合，不是整个 MuJoCo 状态的精确续接。

读取阶段结果后，先按种子对齐四种方法，再看长时/OOD和失败判据，最后解释结构消融。后续继续完成直接控制和速度任务；未经新实验定义，不把阶段测试转为调参集。

### GitHub 与跨机器接手

GitHub 包含代码、文档、紧凑策略与已归档科学结果。`.venv/` 和 `runs/` 被 Git 忽略，完整优化器检查点、种群、原始运行日志、当前活跃进程以及运行代码/资产快照不会随 `git push` 迁移。

因此，全新 clone 可以阅读与分析已发布结果，**不等于可以直接恢复当前队列**。精确接续需要从原机器另行迁移所需 `runs/`、依赖环境、资产和路径信息，并核对哈希；不要改写已经封存的协议来伪装相同实验。重新构建环境的说明见 [snn_pid_compare/README.md](experiments/algorithms/snn_pid_compare/README.md)，使用可移植 `requirements.txt`；历史 `requirements.lock.txt` 含本机来源条目，不宜原样在新机器直接安装。

GitHub 上的进度是推送时快照；未来完成结果仍需再次提交/推送。当前原机器的训练不依赖 GitHub 发布成功，认证问题不应中断训练。
