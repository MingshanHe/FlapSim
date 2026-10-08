# 扑翼机器人：NEAT 拓扑演化、PID 残差控制与 PPO 对照

**状态：正式评估已完成。**

上部 7 g、下部 8 g；yaw 和水平位置不控制。本轮目标是姿态稳定、保高及整机保持离地。不能把有限时长通过称为三维定点悬停。

**严格通过标准：** 跑满规定时长，整机保守离地间隙始终大于 3 cm；稳定阶段倾角 RMS 小于 10°；全程高度 RMS 小于 15 cm；最后 5 秒高度峰峰值小于 20 cm、线性漂移速度绝对值小于 3 cm/s。正式测试的稳定阶段从第 2 秒开始。提前终止均为失败。

## 阅读入口

- **先看代表策略（验证集选出，种子 1204）：** [结构如何演化](pilot/neat_seed1204/visuals/explorer.html) · [真实父代与权重](pilot/neat_seed1204/visuals/lineage_weights.png)。
- **再看飞行：** [扰动对照视频](videos/comparison_stress.mp4) · [翅膀慢动作](videos/neat_wings_stress_10x.mp4) · [逐拍输入与节点回放](flight/neat_stress.html)。
- [正式测试与结果](#正式测试) · [NEAT 演化可视化](#neat-演化可视化) · [测试视频与飞行激活](#测试视频与飞行激活)
- [训练与可视化总览](dashboard.html)：本地服务下显示实时进度；离线显示生成时的快照。
- [预先保存的实验协议](protocol.json) · [源文件哈希](source_hashes.json) · [传感器/PID 前置检查](preflight.json)
- 各个体、父代、物种、连接创新编号、启用状态、全部权重及节点偏置保存在每代 `generation_*.json.gz`。
- 下方动画显示各代冠军，冠军之间不一定有父子关系；祖先图和权重图另沿真实父代记录生成。

## 结果解读

本轮完成 **35 次训练、5 个独立训练种子、9,175,040 次训练交互**，并在冻结策略后开展独立测试。质量为上部 7 g、下部 8 g。共同 PID 内环负责基础姿态与保高，NEAT/PPO 学习三个有界残差：α、β 和电机角速度。主实验使用 **11 维输入：9 维 IMU＋测距高度误差＋估计垂向速度**，不是纯 IMU 定点悬停实验。

**实际做到的程度。** NEAT 的常规 20 秒测试严格通过 40/40；扰动场景撑满 40/40，但只有 15/40 达到全部稳定性标准，其余 25 段均因倾角 RMS 超过 10° 未通过。验证集选出的代表策略在 8 次 60 秒测试中都持续离地：常规 4/4 严格通过，高度 RMS 平均 2.19 cm；扰动 0/4 严格通过，高度 RMS 2.85 cm、倾角 RMS 10.63°，最低几何间隙 49.47 cm。常规和扰动场景 60 秒末水平位移平均分别为 1.38 m 和 2.49 m。因此它达到了有限场景下的持续离地与保高，尚未达到稳健的三维定点悬停。此处离地依据保守几何间隙，不是接触力或实机验证。

**NEAT 与 PPO 的比较。** 扰动 20 秒严格成功率分别为 NEAT 37.5%、固定结构进化 42.5%、循环 PPO 15.0%、前馈 PPO 22.5%；固定 PID 为 12.5%（8 个测试场景）。NEAT 相对循环 PPO 的配对成功率差为 +22.5 个百分点，训练种子 bootstrap 95% 区间为 [+2.5, +40.0]；相对固定结构进化则为 −5.0 个百分点，区间为 [−35.0, +15.0]。这些是当前预算和场景下的小样本、未校正多重比较的结果。常规场景中 PID 本身已全部通过，不能把基础飞行能力都归功于学习器。

**拓扑优势目前仍未证实。** 把 NEAT 选出的结构重新初始化并交给 PPO，与同节点数、边数及逐节点输入/循环入度的随机结构比较，成功率差为 +2.5 个百分点，95% 区间为 [−7.5, +15.0]；倾角差为 −0.71°，区间为 [−1.85, +0.09]。两个区间都包含零。该对照检查了特定小权重初始化及 PPO 重训条件下的结构可迁移性，未排除结构与权重共同适应的价值，也未将初始结构先验和后续变异完全分开。

随机结构组的一个扰动片段在 4.81 秒因倾角超限终止，对平均倾角差有明显影响。使用独立回放入口重跑后，9 个指标与原记录完全一致；同场景的 NEAT 结构移植策略跑满 20 秒，但仍未满足严格倾角标准。这个片段保留在全部统计中，没有事后删除或重新挑选策略。[失败回放核对](posthoc_transfer_failure_check/verification.json) · [失败过程曲线](posthoc_transfer_failure_check/failure_trace.png)。该补充检查仅验证可复现性，不增加训练或独立测试样本数。

**结构演化确实发生了，但需要看清来源。** 五个 NEAT 初始种群都是每个网络 55 个参数，最终选中策略平均为 42.6 个，减少 22.5%。其中一个种子最终保留第 0 代个体；其余种子的选中启用结构与初始群体不同。代表策略含 2 个隐藏节点、28 条启用连接和 33 个参数，只有 1 个隐藏节点仍有路径通向输出。相对 223 参数固定进化网络或 211 参数循环 PPO 的差距，还包含初始容量不同的贡献，不能全部归因于演化。当前 NumPy 图计算仍用密集矩阵，也没有证明实机更省电。

本轮每个 NEAT 种子仅记录 9 代，最后一代因相同交互预算缩短至约 0.48 秒；全部代次都只有 1 个物种。因此没有观察到多物种分化或充分的长时稳态演化。预定扩展门槛在验证集中为 0/5，未触发扩展到 10 个种子；这一决定在独立测试前完成。上述限制和负结果一并保留。

直接 PPO 在本轮预算下未学到可用控制；它同时失去 500 Hz PID 反馈、承担完整执行器范围，并使用额外行为克隆初始化，与残差组存在控制结构和先验差异。这个结果不能证明 PPO 必须使用 PID。组合 OOD 测试中各方法严格成功率均为零，也说明当前策略的稳健性还不够；由于同时改变质量、延迟和扰动，不能归因于质心这一单一因素。

**历史通路的实际作用。** 在另外 4 个扰动种子上对每个 NEAT 策略做结构干预，共 180 段。原网络与移除过去节点状态通路后的严格成功率都为 30.0%，平均倾角 RMS 分别为 10.366° 和 10.351°；同数量随机删边对照的成功率也均为 30.0%。移除指定直接交叉轴连接后的成功率为 35.0%，相应随机删边对照为 25.0%–30.0%。这些描述性结果没有显示明确的记忆通路依赖；不能仅凭循环连接的存在，声称网络已学会历史状态估计。删边也改变有效容量，且没有重新训练，所以该检查不能证明记忆永远无用。

**移除高度辅助后的诊断。** 每种方法使用相同的 4 个额外场景（2 个常规、2 个扰动），切换为纯惯性高度观察器；学习策略同时移除两个高度输入，没有重训。结果如下：

| 控制器 | 严格通过 | 平均持续时间 | 终止原因 |
|---|---:|---:|---|
| PID | 0/4 | 1.434 s | 4 次均为高度偏离范围 |
| NEAT 代表策略 | 0/4 | 1.447 s | 4 次均为高度偏离范围 |
| 循环 PPO 代表策略 | 0/4 | 1.440 s | 4 次均为高度偏离范围 |

这是当前传感器处理、惯性积分和未重训控制器的联合失效，不是坠地记录，也不是对所有纯 IMU 学习方法的否定。它确认本轮交付策略依赖高度辅助；不能把 11 维主实验包装成已完成的 9 维 IMU 训练。[完整传感器诊断](sensor_ablation.json)。

**下一轮最值得验证的问题。** 使用完全相同的初始网络群体，对比“允许结构变异”与“仅优化权重”，同时预先固定训练交互预算；增加每代场景覆盖和飞行时长，避免用短暂启动片段评价悬停；调整物种兼容阈值并记录是否真的形成多个物种。把“达到相同保高与姿态标准时，需要多少有效连接和循环状态”作为预先定义的主问题，并增加目标硬件测量。位置保持应作为独立任务加入可观测的水平反馈，不能仅凭当前 IMU 历史或更复杂网络宣称已解决漂移。这些是后续建议，没有根据本次测试分数追加训练。

**查看顺序：** 先打开下方代表策略演化浏览器，定位选中个体并沿父代查看权重变化；再播放扰动对照视频和翅膀慢动作；最后使用飞行节点回放，查看输入如何通过固定权重形成 α、β 和电机残差。图中的飞行节点状态随时间变化，权重只在训练代际间改变。

## 正式测试

**结论范围：** 当前测试尚未给出稳健的、独立于权重优化和网络规模的拓扑优势证据。不能仅凭某一段飞行视频或更小的参数量声称 NEAT 更好。

### 20 秒：nominal

| 方法 | 训练种子数 | 严格成功率（种子 bootstrap 95% 区间） | 撑满 20 秒 | 倾角 RMS | 高度 RMS | 最低几何间隙 |
|---|---:|---:|---:|---:|---:|---:|
| direct_ppo | 5 | 0.0% [0%, 0%] | 0.0% | 40.14° | 21.61 cm | 20.03 cm |
| ff_ppo | 5 | 100.0% [100%, 100%] | 100.0% | 3.50° | 3.81 cm | 51.86 cm |
| fixed_evo | 5 | 100.0% [100%, 100%] | 100.0% | 3.48° | 3.84 cm | 51.61 cm |
| graph_ppo | 5 | 100.0% [100%, 100%] | 100.0% | 3.53° | 3.81 cm | 51.74 cm |
| neat | 5 | 100.0% [100%, 100%] | 100.0% | 3.49° | 3.81 cm | 51.80 cm |
| pid | — | 100.0%（固定基线） | 100.0% | 3.53° | 3.81 cm | 51.90 cm |
| random_graph_ppo | 5 | 100.0% [100%, 100%] | 100.0% | 3.56° | 3.81 cm | 51.76 cm |
| rnn_ppo | 5 | 100.0% [100%, 100%] | 100.0% | 3.52° | 3.80 cm | 51.68 cm |

### 20 秒：stress

| 方法 | 训练种子数 | 严格成功率（种子 bootstrap 95% 区间） | 撑满 20 秒 | 倾角 RMS | 高度 RMS | 最低几何间隙 |
|---|---:|---:|---:|---:|---:|---:|
| direct_ppo | 5 | 0.0% [0%, 0%] | 0.0% | 41.37° | 22.15 cm | 7.53 cm |
| ff_ppo | 5 | 22.5% [15%, 30%] | 100.0% | 10.52° | 4.22 cm | 50.17 cm |
| fixed_evo | 5 | 42.5% [25%, 62%] | 100.0% | 9.99° | 4.29 cm | 49.97 cm |
| graph_ppo | 5 | 25.0% [18%, 32%] | 100.0% | 10.52° | 4.24 cm | 50.07 cm |
| neat | 5 | 37.5% [22%, 52%] | 100.0% | 10.11° | 4.23 cm | 50.18 cm |
| pid | — | 12.5%（固定基线） | 100.0% | 10.57° | 4.21 cm | 50.21 cm |
| random_graph_ppo | 5 | 22.5% [18%, 25%] | 97.5% | 11.23° | 4.42 cm | 50.06 cm |
| rnn_ppo | 5 | 15.0% [12%, 20%] | 100.0% | 10.54° | 4.21 cm | 50.00 cm |

### 20 秒：ood

| 方法 | 训练种子数 | 严格成功率（种子 bootstrap 95% 区间） | 撑满 20 秒 | 倾角 RMS | 高度 RMS | 最低几何间隙 |
|---|---:|---:|---:|---:|---:|---:|
| direct_ppo | 5 | 0.0% [0%, 0%] | 0.0% | 43.27° | 31.82 cm | 3.87 cm |
| ff_ppo | 5 | 0.0% [0%, 0%] | 100.0% | 16.51° | 11.78 cm | 36.71 cm |
| fixed_evo | 5 | 0.0% [0%, 0%] | 100.0% | 16.15° | 12.20 cm | 36.32 cm |
| graph_ppo | 5 | 0.0% [0%, 0%] | 100.0% | 16.46° | 12.01 cm | 36.57 cm |
| neat | 5 | 0.0% [0%, 0%] | 100.0% | 16.62° | 11.78 cm | 36.79 cm |
| pid | — | 0.0%（固定基线） | 100.0% | 16.63° | 11.87 cm | 36.88 cm |
| random_graph_ppo | 5 | 0.0% [0%, 0%] | 100.0% | 16.23° | 12.04 cm | 36.52 cm |
| rnn_ppo | 5 | 0.0% [0%, 0%] | 100.0% | 16.70° | 12.05 cm | 36.48 cm |

表中倾角与高度误差为各片段 RMS 的算术平均，最低几何间隙取所有片段中的最小值。撑满时长表示未触发离地间隙、倾角、高度或数值异常终止；严格成功还要求每个测试片段的姿态与高度误差分别达标。因此平均倾角小于 10°，仍可能有部分片段不通过。失败策略的 RMS 只覆盖终止前的数据，不能因为它较早结束而把较小 RMS 解释为更好。

相同测试场景在方法间配对；区间重采样的单位是独立训练种子，不是测试片段。样本量较小，区间仅描述当前训练与场景范围。OOD 同时改变下部质量、延迟、推力偏差与扰动幅度，不能把差异归因于某一个因素。

### 拓扑与算法的配对差异

| 对比（前者－后者） | 扰动成功率差 | 扰动倾角 RMS 差 |
|---|---:|---:|
| neat − fixed_evo | -5.0 pp [-35.0, +15.0] | +0.12° [-0.43, +0.94] |
| neat − rnn_ppo | +22.5 pp [+2.5, +40.0] | -0.43° [-0.99, -0.06] |
| neat − ff_ppo | +15.0 pp [-5.0, +35.0] | -0.41° [-0.93, -0.06] |
| graph_ppo − random_graph_ppo | +2.5 pp [-7.5, +15.0] | -0.71° [-1.85, +0.09] |

成功率差为正、倾角差为负才有利于前者。拓扑移植比较的是重新初始化并训练后的结构；原 NEAT 搜索开销不作为免费成本。多个探索性对比未做多重比较校正，不能只挑其中一个显著结果作结论。

左右图分别为 NEAT 选出的结构和匹配随机结构，均重新初始化后交给 PPO 训练。相同节点数、边数和逐节点入度约束下，连线对象仍然不同；图上画的是各自验证选中权重。下面依次链接全部训练种子，首图仅按种子编号展示。

[种子 1201](figures/topology_transfer_seed1201.png) · [种子 1202](figures/topology_transfer_seed1202.png) · [种子 1203](figures/topology_transfer_seed1203.png) · [种子 1204](figures/topology_transfer_seed1204.png) · [种子 1205](figures/topology_transfer_seed1205.png)

![拓扑移植与匹配随机结构](figures/topology_transfer_seed1201.png)

### NEAT 实际选中了哪一代

| 训练种子 | 验证选中的代次 | 个体出生代 | 隐藏节点（通向输出） | 启用连接 | 参数：初始均值→选中 | 启用结构是否不同于全部初始个体 |
|---|---:|---:|---:|---:|---:|---|
| 1201 | 8 | 8 | 4（4） | 32 | 55.0 → 39 | 是 |
| 1202 | 0 | 0 | 2（2） | 50 | 55.0 → 55 | 否 |
| 1203 | 4 | 4 | 4（4） | 47 | 55.0 → 54 | 是 |
| 1204 | 8 | 8 | 2（1） | 28 | 55.0 → 33 | 是 |
| 1205 | 8 | 8 | 2（1） | 27 | 55.0 → 32 | 是 |

括号内仅统计存在启用路径通向任一输出的隐藏节点；其余节点仍保存在基因和实际导出计算中，未在测试前额外剪枝。最后一列仅比较节点 ID 和启用连接集合，未做图同构判定，也不评价性能。如果选中初始个体，说明后续演化在本轮验证选择中没有改善该候选；这种结果不能作为“结构变异带来收益”的证据。

### 60 秒：验证集选出的代表策略

| 方法 | 场景 | 成功/总数 | 平均持续 | 倾角 RMS | 高度 RMS | 最低几何间隙 | 末端 XY 位移（均值） |
|---|---|---:|---:|---:|---:|---:|---:|
| direct_ppo | nominal | 0/4 | 1.00 s | 37.35° | 19.87 cm | 42.37 cm | 0.15 m |
| direct_ppo | stress | 0/4 | 1.19 s | 45.56° | 24.29 cm | 7.67 cm | 0.23 m |
| ff_ppo | nominal | 4/4 | 60.00 s | 3.30° | 2.19 cm | 51.48 cm | 1.35 m |
| ff_ppo | stress | 1/4 | 60.00 s | 10.66° | 2.72 cm | 49.66 cm | 2.16 m |
| fixed_evo | nominal | 4/4 | 60.00 s | 3.28° | 2.20 cm | 51.47 cm | 1.58 m |
| fixed_evo | stress | 2/4 | 60.00 s | 9.58° | 2.70 cm | 49.68 cm | 1.97 m |
| graph_ppo | nominal | 4/4 | 60.00 s | 3.28° | 2.21 cm | 51.37 cm | 0.61 m |
| graph_ppo | stress | 2/4 | 60.00 s | 10.58° | 2.72 cm | 49.51 cm | 1.90 m |
| neat | nominal | 4/4 | 60.00 s | 3.34° | 2.19 cm | 51.30 cm | 1.38 m |
| neat | stress | 0/4 | 60.00 s | 10.63° | 2.85 cm | 49.47 cm | 2.49 m |
| pid | nominal | 4/4 | 60.00 s | 3.30° | 2.20 cm | 51.53 cm | 1.44 m |
| pid | stress | 1/4 | 60.00 s | 10.70° | 2.71 cm | 49.71 cm | 2.17 m |
| random_graph_ppo | nominal | 4/4 | 60.00 s | 3.29° | 2.19 cm | 51.41 cm | 2.06 m |
| random_graph_ppo | stress | 0/4 | 60.00 s | 10.67° | 2.80 cm | 49.54 cm | 2.32 m |
| rnn_ppo | nominal | 4/4 | 60.00 s | 3.29° | 2.20 cm | 51.27 cm | 0.90 m |
| rnn_ppo | stress | 1/4 | 60.00 s | 10.65° | 2.89 cm | 49.41 cm | 2.00 m |

这里每种方法只检查验证集选出的一个训练种子；代表策略的长时表现不能替代全部训练种子的稳健性统计。XY 数值是每段实际终止时相对初始位置的水平距离，再对片段取平均；它不是位置 RMS，也没有计入本轮训练奖励或通过判据。

### 推理成本与训练成本

| 方法 | Actor 参数（均值） | 有记忆连接（均值） | CPU 推理中位数（均值） | 单种子训练步数（均值） |
|---|---:|---:|---:|---:|
| direct_ppo | 211.0 | 112.0 | 3.55 μs | 262,144 |
| ff_ppo | 1539.0 | 0.0 | 2.98 μs | 262,144 |
| fixed_evo | 223.0 | 91.0 | 3.47 μs | 262,144 |
| graph_ppo | 42.6 | 7.6 | 3.26 μs | 262,144 |
| neat | 42.6 | 7.6 | 3.27 μs | 262,144 |
| random_graph_ppo | 42.6 | 7.6 | 3.26 μs | 262,144 |
| rnn_ppo | 211.0 | 112.0 | 3.36 μs | 262,144 |

NEAT 从初始种群平均 55.0 个参数变为选中策略平均 42.6 个，减少 22.5%。相对较大的固定结构基线，其参数差距还包含初始隐藏节点数不同的贡献，不能全部归因于后续演化。参数数按启用连接权重与全部非输入节点偏置计，不包括禁用基因、critic 或训练用探索尺度，也不等于当前密集 NumPy 实现实际分配的内存。

移植组表中步数只列权重训练；完整流程还需加对应 NEAT 搜索步数。随机结构对照保留从 NEAT 获得的规模/入度，因此同样计入该搜索来源。直接 PPO 的演示交互在运行表中单列。CPU 推理在训练进程退出后使用同一导出路径统一复测，仍不能替代目标微控制器测量。

### 结构干预

| 干预 | 成功率 | 倾角 RMS | 高度 RMS |
|---|---:|---:|---:|
| remove_direct_cross_axis | 35.0% | 10.37° | 3.96 cm |
| remove_direct_cross_axis_random_control_0 | 30.0% | 10.36° | 3.96 cm |
| remove_direct_cross_axis_random_control_1 | 25.0% | 10.43° | 3.97 cm |
| remove_direct_cross_axis_random_control_2 | 25.0% | 10.53° | 3.96 cm |
| remove_past_state_paths | 30.0% | 10.35° | 3.96 cm |
| remove_past_state_paths_random_control_0 | 30.0% | 10.43° | 3.98 cm |
| remove_past_state_paths_random_control_1 | 30.0% | 10.36° | 3.94 cm |
| remove_past_state_paths_random_control_2 | 30.0% | 10.35° | 3.96 cm |
| unchanged | 30.0% | 10.37° | 3.96 cm |

去除过去节点状态通路也会减少有效计算路径和容量，因此不等于纯粹的“记忆开关”。交叉轴消融只去掉明确标识的直接 roll→alpha、pitch→beta 等输入连接。与同数量随机删边比较，再结合重新训练的拓扑移植结果判断。

[全部 20 秒测试](heldout_episodes.json) · [60 秒测试](long_episodes.json) · [结构干预](intervention_episodes.json) · [移除高度反馈的诊断](sensor_ablation.json)

各测试记录保存六项通过判据及末段高度峰峰值，方便区分提前终止、姿态超限和高度不稳定；同一次失败可能同时违反多项判据。

左图为全部方法的严格成功率；右图为共享固定 PID 内环的残差学习器，展示参数量与扰动倾角误差。

![方法性能与参数比较](method_comparison.png)


## 已确认的前置结果

| 传感器条件 | 场景 | 成功/总数 | 撑满时长 | 倾角 RMS | 高度 RMS |
|---|---|---:|---:|---:|---:|
| height | nominal | 8/8 | 8/8 | 3.54° | 3.81 cm |
| height | stress | 1/8 | 8/8 | 10.58° | 5.31 cm |
| imu | nominal | 0/2 | 0/2 | 6.88° | 37.15 cm |
| imu | stress | 0/2 | 0/2 | 27.00° | 36.77 cm |
| truth | nominal | 2/2 | 2/2 | 3.37° | 3.32 cm |
| truth | stress | 2/2 | 2/2 | 9.82° | 3.88 cm |

高度辅助在这里是 50 Hz、20 ms 延迟、2 mm 测距噪声的模拟测距仪，垂向速度由同一观察器估计。`truth` 是独立真值诊断；`imu` 为纯惯性积分。前置数据不用于最终算法排名。

加入 ±32 g 量程后，虚拟加速度计的少量高幅脉冲截断可产生明显积分误差；这揭示了现有模型与传感器接口的限制。它不是“实机一定会这样”的结论。姿态读数仍由抽样真值姿态加偏置/噪声模拟 AHRS，并未实现完整姿态解算。

## 控制和公平性

固定 PID 提供基准动作，学习器通过较低频率的有界残差补偿姿态与升力。以下为本轮残差组的共同结构：

```mermaid
flowchart LR
  Plant["扑翼机器人"] --> Sensor["9 维 IMU + 高度误差 / 垂向速度估计"]
  Sensor --> PID["固定 PID · 500 Hz"]
  Sensor --> Actor["NEAT / PPO · 100 Hz"]
  Actor --> Scale["有界残差 δα、δβ、δmotor"]
  PID --> Sum["相加与执行器限幅"]
  Scale --> Sum
  Sum -->|α、β、电机角速度命令| Plant
```

- 同一质量、质心、惯性、气动力、电机模型、软启动及执行器伺服。物理步长 5 μs；PID 500 Hz；所有学习策略 100 Hz。
- 残差动作为 `[δα, δβ, δω]`，三个输出均为 `[-1, 1]`，比例为 `[0.035 rad, 0.035 rad, 100 rad/s]`，再叠加固定 PID。实际命令仍受原有物理范围限制。
- 本轮训练的策略统一 11 维：9 维 IMU 加测距高度误差和估计垂向速度。另做移除测距及两个策略输入后的 9 维诊断，同时将 PID 高度估计切换为纯惯性积分；未单独重训 9 维策略，因此该诊断同时涉及传感器、估计器和策略输入的变化。训练/奖励可使用真值，部署策略不读取真实位置、速度、扑翼相位或外力。
- 所有网络先统一 tanh，避免把激活函数更换当成拓扑收益。NEAT 的节点来自非输入节点的连接均使用上一策略时刻状态。
- 输入按固定量纲归一化：roll/pitch 除以 0.35 rad、yaw 除以 π、角速度除以 10 rad/s、加速度除以 9.81 m/s²、高度误差除以 0.2 m、垂向速度除以 1 m/s；最后裁剪到 [-10, 10]。图中的权重作用于这些归一化值，不能只凭权重大小判断某个传感器的重要性。
- PPO 为本目录显式实现的 clipped PPO，包含 GAE、终止/时间截断区分、循环序列梯度与导出一致性检查。循环对照和拓扑移植使用同一图更新规则。
- 直接 PPO 的三个输出映射到 α/β 的 ±0.3 rad，以及电机 3461.36 ± 346.14 rad/s；它需要独立覆盖完整控制范围。另用 PID 演示初始化，演示数据和交互数单列。
- 直接 PPO 的控制决策为 100 Hz；混合方法还具有 500 Hz PID 反馈。因此这个对照同时改变控制结构、先验和反馈带宽，不是纯优化器对照。即使直接 PPO 失败，也不能推论 PPO 在更大预算或其他控制频率下必须依赖 PID。
- 训练预算按实际交互数记录，包括 NEAT 全种群评估；验证、演示和拓扑搜索成本分别记录。固定拓扑进化基线具有 8 个隐藏节点，完整 NEAT 从 2 个隐藏节点起步，故两者的初始容量并非完全一致；同规模随机拓扑移植提供额外结构对照。
- 固定循环 PPO 通过隐藏状态向输出传递观测，NEAT 初始结构允许输入直连输出。这些结构先验也可能影响表现；匹配随机结构对照保留各节点的输入入度和循环入度，以进一步区分连接关系与容量、通路类型的影响。
- CPU 推理时间使用相同 NumPy 导出路径，并在全部训练退出后统一重测；稀疏图当前以密集矩阵求值。部分结构移植任务与主队列并行执行，单次训练墙钟时间不能直接用于算法速度排名。连接少不等于已经证明微控制器更省电。

## 本轮训练参数

| 项目 | 配置 |
|---|---|
| NEAT 种群 | 128；每个训练种子约 262,144 次策略交互，统计整个种群的消耗 |
| 初始结构 | 2 个隐藏节点、稀疏输入和循环连接；输出含输入直连 |
| 结构变异 | 加连接 0.30、删连接 0.06、加节点 0.20、删节点 0.03、连接启用状态变异 0.015 |
| 参数变异 | 权重变异率 0.50、扰动标准差 0.04；偏置扰动标准差 0.015；完整配置见各运行 neat.ini |
| 激活 | 全部使用普通 tanh；本轮未进行 sigmoid/混合激活库搜索 |
| 固定结构进化 | 8 个隐藏节点，关闭结构和连接启用状态变异 |
| 前馈 PPO | 两层 32 节点；循环 PPO 为 8 个隐藏节点加 3 个输出状态 |
| PPO 优化 | 学习率 3e-4，clip 0.2，gamma 0.995，GAE lambda 0.95，32 步循环反传，最多 6 个 epoch |
| 独立重复 | 5 个 pilot 种子；只根据预定的验证集门槛决定是否扩展至 10 个 |
| 验证选择 | 每个种子 4 个固定 8 秒场景；按成功率、回报、倾角依次选择 |

主假设是演化出来的连接关系能否在同等规模和权重重新训练后继续带来收益。本轮先固定激活函数和种群规模隔离这个问题；256 种群、混合激活库及更大循环网络属于后续扩展，未被计作已完成实验。

NEAT 每代所有个体共享同一场景；前半程训练片段为 2 秒，后半程为 4 秒。最后一代为满足交互预算而统一缩短（当前约 0.48 秒），因此不能把记录的 9 代解释为 9 代完整稳态飞行训练。候选选择仍使用固定的 8 秒验证。

预定扩展门槛要求至少 3/5 个 NEAT 种子在配对验证中同时改善 PID 回报超过 1%、倾角 RMS 超过 5%。实际满足 0/5，保留预定的 5 个种子，未触发扩展。[门槛记录](expansion_decision.json)；该决定在独立测试前完成。

## 训练与验证进度

| 运行 | 状态 | 训练步数 | 验证步数 | 演示步数 | 选中阶段 | 验证成功率 | 验证倾角 RMS |
|---|---|---:|---:|---:|---|---:|---:|
| [pilot/direct_ppo_seed1201](pilot/direct_ppo_seed1201/status.json) | completed | 262,144 | 2,164 | 6,400 | 131,072 步 | 0% | 38.97° |
| [pilot/direct_ppo_seed1202](pilot/direct_ppo_seed1202/status.json) | completed | 262,144 | 2,196 | 6,400 | 262,144 步 | 0% | 44.72° |
| [pilot/direct_ppo_seed1203](pilot/direct_ppo_seed1203/status.json) | completed | 262,144 | 1,991 | 6,400 | 262,144 步 | 0% | 42.75° |
| [pilot/direct_ppo_seed1204](pilot/direct_ppo_seed1204/status.json) | completed | 262,144 | 2,283 | 6,400 | 0 步 | 0% | 40.19° |
| [pilot/direct_ppo_seed1205](pilot/direct_ppo_seed1205/status.json) | completed | 262,144 | 1,986 | 6,400 | 0 步 | 0% | 36.98° |
| [pilot/ff_ppo_seed1201](pilot/ff_ppo_seed1201/status.json) | completed | 262,144 | 16,000 | 0 | 65,536 步 | 75% | 7.20° |
| [pilot/ff_ppo_seed1202](pilot/ff_ppo_seed1202/status.json) | completed | 262,144 | 16,000 | 0 | 131,072 步 | 75% | 7.49° |
| [pilot/ff_ppo_seed1203](pilot/ff_ppo_seed1203/status.json) | completed | 262,144 | 16,000 | 0 | 0 步 | 75% | 7.26° |
| [pilot/ff_ppo_seed1204](pilot/ff_ppo_seed1204/status.json) | completed | 262,144 | 16,000 | 0 | 0 步 | 75% | 6.85° |
| [pilot/ff_ppo_seed1205](pilot/ff_ppo_seed1205/status.json) | completed | 262,144 | 16,000 | 0 | 262,144 步 | 50% | 7.80° |
| [pilot/fixed_evo_seed1201](pilot/fixed_evo_seed1201/status.json) | completed | 262,144 | 9,600 | 0 | 第 4 代 | 75% | 7.20° |
| [pilot/fixed_evo_seed1202](pilot/fixed_evo_seed1202/status.json) | completed | 262,144 | 9,600 | 0 | 第 4 代 | 75% | 7.31° |
| [pilot/fixed_evo_seed1203](pilot/fixed_evo_seed1203/status.json) | completed | 262,144 | 9,600 | 0 | 第 4 代 | 75% | 6.85° |
| [pilot/fixed_evo_seed1204](pilot/fixed_evo_seed1204/status.json) | completed | 262,144 | 9,600 | 0 | 第 0 代 | 75% | 6.67° |
| [pilot/fixed_evo_seed1205](pilot/fixed_evo_seed1205/status.json) | completed | 262,144 | 9,600 | 0 | 第 8 代 | 75% | 7.26° |
| [pilot/neat_seed1201](pilot/neat_seed1201/status.json) | completed | 262,144 | 9,600 | 0 | 第 8 代 | 75% | 6.90° |
| [pilot/neat_seed1202](pilot/neat_seed1202/status.json) | completed | 262,144 | 9,600 | 0 | 第 0 代 | 75% | 7.58° |
| [pilot/neat_seed1203](pilot/neat_seed1203/status.json) | completed | 262,144 | 9,600 | 0 | 第 4 代 | 75% | 7.05° |
| [pilot/neat_seed1204](pilot/neat_seed1204/status.json) | completed | 262,144 | 9,600 | 0 | 第 8 代 | 100% | 6.66° |
| [pilot/neat_seed1205](pilot/neat_seed1205/status.json) | completed | 262,144 | 9,600 | 0 | 第 8 代 | 50% | 7.68° |
| [pilot/rnn_ppo_seed1201](pilot/rnn_ppo_seed1201/status.json) | completed | 262,144 | 16,000 | 0 | 262,144 步 | 75% | 7.26° |
| [pilot/rnn_ppo_seed1202](pilot/rnn_ppo_seed1202/status.json) | completed | 262,144 | 16,000 | 0 | 262,144 步 | 75% | 7.46° |
| [pilot/rnn_ppo_seed1203](pilot/rnn_ppo_seed1203/status.json) | completed | 262,144 | 16,000 | 0 | 0 步 | 75% | 7.30° |
| [pilot/rnn_ppo_seed1204](pilot/rnn_ppo_seed1204/status.json) | completed | 262,144 | 16,000 | 0 | 131,072 步 | 75% | 6.84° |
| [pilot/rnn_ppo_seed1205](pilot/rnn_ppo_seed1205/status.json) | completed | 262,144 | 16,000 | 0 | 0 步 | 50% | 8.01° |
| [transfer/graph_ppo_seed1201](transfer/graph_ppo_seed1201/status.json) | completed | 262,144 | 16,000 | 0 | 131,072 步 | 75% | 7.22° |
| [transfer/graph_ppo_seed1202](transfer/graph_ppo_seed1202/status.json) | completed | 262,144 | 16,000 | 0 | 65,536 步 | 50% | 7.45° |
| [transfer/graph_ppo_seed1203](transfer/graph_ppo_seed1203/status.json) | completed | 262,144 | 16,000 | 0 | 0 步 | 75% | 7.34° |
| [transfer/graph_ppo_seed1204](transfer/graph_ppo_seed1204/status.json) | completed | 262,144 | 16,000 | 0 | 131,072 步 | 75% | 7.04° |
| [transfer/graph_ppo_seed1205](transfer/graph_ppo_seed1205/status.json) | completed | 262,144 | 16,000 | 0 | 0 步 | 50% | 7.89° |
| [transfer/random_graph_ppo_seed1201](transfer/random_graph_ppo_seed1201/status.json) | completed | 262,144 | 16,000 | 0 | 131,072 步 | 75% | 7.06° |
| [transfer/random_graph_ppo_seed1202](transfer/random_graph_ppo_seed1202/status.json) | completed | 262,144 | 16,000 | 0 | 131,072 步 | 75% | 7.47° |
| [transfer/random_graph_ppo_seed1203](transfer/random_graph_ppo_seed1203/status.json) | completed | 262,144 | 16,000 | 0 | 0 步 | 75% | 7.43° |
| [transfer/random_graph_ppo_seed1204](transfer/random_graph_ppo_seed1204/status.json) | completed | 262,144 | 16,000 | 0 | 131,072 步 | 75% | 7.00° |
| [transfer/random_graph_ppo_seed1205](transfer/random_graph_ppo_seed1205/status.json) | completed | 262,144 | 16,000 | 0 | 196,608 步 | 50% | 7.99° |

以上为验证集选择结果。不同训练种子的验证场景不同，同一种子的方法间保持配对。训练 fitness 随每代场景及训练时长变化，不应只看训练曲线单调性。

选中阶段为 PPO 的 0 步时，表示 PPO 更新前的模型最终保留；直接 PPO 在这个阶段已经做过行为克隆。NEAT 第 0 代仍包含整个初始种群的评估和选择，但尚未发生后续代际演化。这些情况均不能当作后续强化学习或结构变异改善了控制的证据。

## NEAT 演化可视化

每代将整个种群放进同一个训练场景评估。训练 fitness 为“累计回报 / 本代计划时长 − 0.00002 × 启用连接数”；提前失败仍除以计划时长。新网络通过父代交叉、权重/偏置扰动，以及增删节点和连接产生；NEAT 这部分不使用梯度下降。保留的精英个体沿用原 ID。场景和片段时长随代数更换，短片段中启动过程占比也更大，所以训练 fitness 上下波动不等于退化，固定验证集才用于选出交付策略。

共同的飞行回报在每个 2 ms 控制步累加 `dt × [2 + exp(-(倾角/0.15 rad)²) + exp(-(高度误差/0.12 m)²)]`，提前失败扣 1；每次策略决策再扣 `0.0002 × ||本次归一化动作 − 上次动作||²`。它鼓励持续飞行、姿态和高度稳定，并抑制动作突变；yaw 和水平位置没有跟踪奖励。

交互浏览器可以选择代数和任意个体、播放冠军序列、筛选连接权重、显示禁用连接、点击父代、查看全部连接权重及节点偏置。浏览器文件包含完整种群数据，可离线打开。

拓扑图中蓝色为正权重、红色为负权重，线宽表示权重绝对值；静态图仅显示启用连接。每个节点计算 `tanh(当前输入的加权和 + 上一时刻节点状态的加权和 + 偏置)`。因此图上从左到右的隐藏节点连接也可能带有一个控制周期的延迟，不能仅凭箭头方向把它当成普通前馈层。

家谱保留双亲；权重热图和曲线沿其中一条注明的祖先路径展示。相邻个体的权重差异可能同时来自交叉继承和参数变异；“变异事件”专门记录交叉完成后发生的修改，不将两者混为一谈。权重图中的负编号是输入通道，0/1/2 为 α/β/电机输出，其余编号为隐藏节点。

### neat_seed1201

[打开交互演化浏览器](pilot/neat_seed1201/visuals/explorer.html) · [完整父代关系](pilot/neat_seed1201/visuals/all_parent_links.json) · [图表说明与数据量](pilot/neat_seed1201/visuals/visualization_manifest.json)

记录 9 代；每代物种数范围 1–1。这些是当前有限预算下的演化记录，不能替代更长训练的收敛研究。

![演化过程与物种](pilot/neat_seed1201/visuals/evolution_summary.png)

![拓扑变化动画：各代冠军](pilot/neat_seed1201/visuals/topology_evolution.gif)

![验证选中网络](pilot/neat_seed1201/visuals/selected_topology.png)

![真实祖先路径的权重变化](pilot/neat_seed1201/visuals/lineage_weights.png)

![新增删除与种群权重分布](pilot/neat_seed1201/visuals/mutations_and_weights.png)

![祖先节点偏置变化](pilot/neat_seed1201/visuals/lineage_biases.png)

[完整种群 CSV](pilot/neat_seed1201/visuals/population.csv.gz) · [完整权重 CSV](pilot/neat_seed1201/visuals/weights.csv.gz) · [节点偏置 CSV](pilot/neat_seed1201/visuals/biases.csv.gz)

![选中网络的双亲家谱](pilot/neat_seed1201/visuals/selected_family_tree.png)

![祖先路径上变化最大的连接权重](pilot/neat_seed1201/visuals/lineage_weight_curves.png)

### neat_seed1202

[打开交互演化浏览器](pilot/neat_seed1202/visuals/explorer.html) · [完整父代关系](pilot/neat_seed1202/visuals/all_parent_links.json) · [图表说明与数据量](pilot/neat_seed1202/visuals/visualization_manifest.json)

记录 9 代；每代物种数范围 1–1。这些是当前有限预算下的演化记录，不能替代更长训练的收敛研究。

![演化过程与物种](pilot/neat_seed1202/visuals/evolution_summary.png)

![拓扑变化动画：各代冠军](pilot/neat_seed1202/visuals/topology_evolution.gif)

![验证选中网络](pilot/neat_seed1202/visuals/selected_topology.png)

![真实祖先路径的权重变化](pilot/neat_seed1202/visuals/lineage_weights.png)

![新增删除与种群权重分布](pilot/neat_seed1202/visuals/mutations_and_weights.png)

![祖先节点偏置变化](pilot/neat_seed1202/visuals/lineage_biases.png)

[完整种群 CSV](pilot/neat_seed1202/visuals/population.csv.gz) · [完整权重 CSV](pilot/neat_seed1202/visuals/weights.csv.gz) · [节点偏置 CSV](pilot/neat_seed1202/visuals/biases.csv.gz)

![选中网络的双亲家谱](pilot/neat_seed1202/visuals/selected_family_tree.png)

![祖先路径上变化最大的连接权重](pilot/neat_seed1202/visuals/lineage_weight_curves.png)

### neat_seed1203

[打开交互演化浏览器](pilot/neat_seed1203/visuals/explorer.html) · [完整父代关系](pilot/neat_seed1203/visuals/all_parent_links.json) · [图表说明与数据量](pilot/neat_seed1203/visuals/visualization_manifest.json)

记录 9 代；每代物种数范围 1–1。这些是当前有限预算下的演化记录，不能替代更长训练的收敛研究。

![演化过程与物种](pilot/neat_seed1203/visuals/evolution_summary.png)

![拓扑变化动画：各代冠军](pilot/neat_seed1203/visuals/topology_evolution.gif)

![验证选中网络](pilot/neat_seed1203/visuals/selected_topology.png)

![真实祖先路径的权重变化](pilot/neat_seed1203/visuals/lineage_weights.png)

![新增删除与种群权重分布](pilot/neat_seed1203/visuals/mutations_and_weights.png)

![祖先节点偏置变化](pilot/neat_seed1203/visuals/lineage_biases.png)

[完整种群 CSV](pilot/neat_seed1203/visuals/population.csv.gz) · [完整权重 CSV](pilot/neat_seed1203/visuals/weights.csv.gz) · [节点偏置 CSV](pilot/neat_seed1203/visuals/biases.csv.gz)

![选中网络的双亲家谱](pilot/neat_seed1203/visuals/selected_family_tree.png)

![祖先路径上变化最大的连接权重](pilot/neat_seed1203/visuals/lineage_weight_curves.png)

### neat_seed1204

[打开交互演化浏览器](pilot/neat_seed1204/visuals/explorer.html) · [完整父代关系](pilot/neat_seed1204/visuals/all_parent_links.json) · [图表说明与数据量](pilot/neat_seed1204/visuals/visualization_manifest.json)

记录 9 代；每代物种数范围 1–1。这些是当前有限预算下的演化记录，不能替代更长训练的收敛研究。

![演化过程与物种](pilot/neat_seed1204/visuals/evolution_summary.png)

![拓扑变化动画：各代冠军](pilot/neat_seed1204/visuals/topology_evolution.gif)

![验证选中网络](pilot/neat_seed1204/visuals/selected_topology.png)

![真实祖先路径的权重变化](pilot/neat_seed1204/visuals/lineage_weights.png)

![新增删除与种群权重分布](pilot/neat_seed1204/visuals/mutations_and_weights.png)

![祖先节点偏置变化](pilot/neat_seed1204/visuals/lineage_biases.png)

[完整种群 CSV](pilot/neat_seed1204/visuals/population.csv.gz) · [完整权重 CSV](pilot/neat_seed1204/visuals/weights.csv.gz) · [节点偏置 CSV](pilot/neat_seed1204/visuals/biases.csv.gz)

![选中网络的双亲家谱](pilot/neat_seed1204/visuals/selected_family_tree.png)

![祖先路径上变化最大的连接权重](pilot/neat_seed1204/visuals/lineage_weight_curves.png)

### neat_seed1205

[打开交互演化浏览器](pilot/neat_seed1205/visuals/explorer.html) · [完整父代关系](pilot/neat_seed1205/visuals/all_parent_links.json) · [图表说明与数据量](pilot/neat_seed1205/visuals/visualization_manifest.json)

记录 9 代；每代物种数范围 1–1。这些是当前有限预算下的演化记录，不能替代更长训练的收敛研究。

![演化过程与物种](pilot/neat_seed1205/visuals/evolution_summary.png)

![拓扑变化动画：各代冠军](pilot/neat_seed1205/visuals/topology_evolution.gif)

![验证选中网络](pilot/neat_seed1205/visuals/selected_topology.png)

![真实祖先路径的权重变化](pilot/neat_seed1205/visuals/lineage_weights.png)

![新增删除与种群权重分布](pilot/neat_seed1205/visuals/mutations_and_weights.png)

![祖先节点偏置变化](pilot/neat_seed1205/visuals/lineage_biases.png)

[完整种群 CSV](pilot/neat_seed1205/visuals/population.csv.gz) · [完整权重 CSV](pilot/neat_seed1205/visuals/weights.csv.gz) · [节点偏置 CSV](pilot/neat_seed1205/visuals/biases.csv.gz)

![选中网络的双亲家谱](pilot/neat_seed1205/visuals/selected_family_tree.png)

![祖先路径上变化最大的连接权重](pilot/neat_seed1205/visuals/lineage_weight_curves.png)

### fixed_evo_seed1201

[打开交互演化浏览器](pilot/fixed_evo_seed1201/visuals/explorer.html) · [完整父代关系](pilot/fixed_evo_seed1201/visuals/all_parent_links.json) · [图表说明与数据量](pilot/fixed_evo_seed1201/visuals/visualization_manifest.json)

记录 9 代；每代物种数范围 1–1。这些是当前有限预算下的演化记录，不能替代更长训练的收敛研究。

![演化过程与物种](pilot/fixed_evo_seed1201/visuals/evolution_summary.png)

![拓扑变化动画：各代冠军](pilot/fixed_evo_seed1201/visuals/topology_evolution.gif)

![验证选中网络](pilot/fixed_evo_seed1201/visuals/selected_topology.png)

![真实祖先路径的权重变化](pilot/fixed_evo_seed1201/visuals/lineage_weights.png)

![新增删除与种群权重分布](pilot/fixed_evo_seed1201/visuals/mutations_and_weights.png)

![祖先节点偏置变化](pilot/fixed_evo_seed1201/visuals/lineage_biases.png)

[完整种群 CSV](pilot/fixed_evo_seed1201/visuals/population.csv.gz) · [完整权重 CSV](pilot/fixed_evo_seed1201/visuals/weights.csv.gz) · [节点偏置 CSV](pilot/fixed_evo_seed1201/visuals/biases.csv.gz)

![选中网络的双亲家谱](pilot/fixed_evo_seed1201/visuals/selected_family_tree.png)

![祖先路径上变化最大的连接权重](pilot/fixed_evo_seed1201/visuals/lineage_weight_curves.png)

### fixed_evo_seed1202

[打开交互演化浏览器](pilot/fixed_evo_seed1202/visuals/explorer.html) · [完整父代关系](pilot/fixed_evo_seed1202/visuals/all_parent_links.json) · [图表说明与数据量](pilot/fixed_evo_seed1202/visuals/visualization_manifest.json)

记录 9 代；每代物种数范围 1–1。这些是当前有限预算下的演化记录，不能替代更长训练的收敛研究。

![演化过程与物种](pilot/fixed_evo_seed1202/visuals/evolution_summary.png)

![拓扑变化动画：各代冠军](pilot/fixed_evo_seed1202/visuals/topology_evolution.gif)

![验证选中网络](pilot/fixed_evo_seed1202/visuals/selected_topology.png)

![真实祖先路径的权重变化](pilot/fixed_evo_seed1202/visuals/lineage_weights.png)

![新增删除与种群权重分布](pilot/fixed_evo_seed1202/visuals/mutations_and_weights.png)

![祖先节点偏置变化](pilot/fixed_evo_seed1202/visuals/lineage_biases.png)

[完整种群 CSV](pilot/fixed_evo_seed1202/visuals/population.csv.gz) · [完整权重 CSV](pilot/fixed_evo_seed1202/visuals/weights.csv.gz) · [节点偏置 CSV](pilot/fixed_evo_seed1202/visuals/biases.csv.gz)

![选中网络的双亲家谱](pilot/fixed_evo_seed1202/visuals/selected_family_tree.png)

![祖先路径上变化最大的连接权重](pilot/fixed_evo_seed1202/visuals/lineage_weight_curves.png)

### fixed_evo_seed1203

[打开交互演化浏览器](pilot/fixed_evo_seed1203/visuals/explorer.html) · [完整父代关系](pilot/fixed_evo_seed1203/visuals/all_parent_links.json) · [图表说明与数据量](pilot/fixed_evo_seed1203/visuals/visualization_manifest.json)

记录 9 代；每代物种数范围 1–1。这些是当前有限预算下的演化记录，不能替代更长训练的收敛研究。

![演化过程与物种](pilot/fixed_evo_seed1203/visuals/evolution_summary.png)

![拓扑变化动画：各代冠军](pilot/fixed_evo_seed1203/visuals/topology_evolution.gif)

![验证选中网络](pilot/fixed_evo_seed1203/visuals/selected_topology.png)

![真实祖先路径的权重变化](pilot/fixed_evo_seed1203/visuals/lineage_weights.png)

![新增删除与种群权重分布](pilot/fixed_evo_seed1203/visuals/mutations_and_weights.png)

![祖先节点偏置变化](pilot/fixed_evo_seed1203/visuals/lineage_biases.png)

[完整种群 CSV](pilot/fixed_evo_seed1203/visuals/population.csv.gz) · [完整权重 CSV](pilot/fixed_evo_seed1203/visuals/weights.csv.gz) · [节点偏置 CSV](pilot/fixed_evo_seed1203/visuals/biases.csv.gz)

![选中网络的双亲家谱](pilot/fixed_evo_seed1203/visuals/selected_family_tree.png)

![祖先路径上变化最大的连接权重](pilot/fixed_evo_seed1203/visuals/lineage_weight_curves.png)

### fixed_evo_seed1204

[打开交互演化浏览器](pilot/fixed_evo_seed1204/visuals/explorer.html) · [完整父代关系](pilot/fixed_evo_seed1204/visuals/all_parent_links.json) · [图表说明与数据量](pilot/fixed_evo_seed1204/visuals/visualization_manifest.json)

记录 9 代；每代物种数范围 1–1。这些是当前有限预算下的演化记录，不能替代更长训练的收敛研究。

![演化过程与物种](pilot/fixed_evo_seed1204/visuals/evolution_summary.png)

![拓扑变化动画：各代冠军](pilot/fixed_evo_seed1204/visuals/topology_evolution.gif)

![验证选中网络](pilot/fixed_evo_seed1204/visuals/selected_topology.png)

![真实祖先路径的权重变化](pilot/fixed_evo_seed1204/visuals/lineage_weights.png)

![新增删除与种群权重分布](pilot/fixed_evo_seed1204/visuals/mutations_and_weights.png)

![祖先节点偏置变化](pilot/fixed_evo_seed1204/visuals/lineage_biases.png)

[完整种群 CSV](pilot/fixed_evo_seed1204/visuals/population.csv.gz) · [完整权重 CSV](pilot/fixed_evo_seed1204/visuals/weights.csv.gz) · [节点偏置 CSV](pilot/fixed_evo_seed1204/visuals/biases.csv.gz)

![选中网络的双亲家谱](pilot/fixed_evo_seed1204/visuals/selected_family_tree.png)

![祖先路径上变化最大的连接权重](pilot/fixed_evo_seed1204/visuals/lineage_weight_curves.png)

### fixed_evo_seed1205

[打开交互演化浏览器](pilot/fixed_evo_seed1205/visuals/explorer.html) · [完整父代关系](pilot/fixed_evo_seed1205/visuals/all_parent_links.json) · [图表说明与数据量](pilot/fixed_evo_seed1205/visuals/visualization_manifest.json)

记录 9 代；每代物种数范围 1–1。这些是当前有限预算下的演化记录，不能替代更长训练的收敛研究。

![演化过程与物种](pilot/fixed_evo_seed1205/visuals/evolution_summary.png)

![拓扑变化动画：各代冠军](pilot/fixed_evo_seed1205/visuals/topology_evolution.gif)

![验证选中网络](pilot/fixed_evo_seed1205/visuals/selected_topology.png)

![真实祖先路径的权重变化](pilot/fixed_evo_seed1205/visuals/lineage_weights.png)

![新增删除与种群权重分布](pilot/fixed_evo_seed1205/visuals/mutations_and_weights.png)

![祖先节点偏置变化](pilot/fixed_evo_seed1205/visuals/lineage_biases.png)

[完整种群 CSV](pilot/fixed_evo_seed1205/visuals/population.csv.gz) · [完整权重 CSV](pilot/fixed_evo_seed1205/visuals/weights.csv.gz) · [节点偏置 CSV](pilot/fixed_evo_seed1205/visuals/biases.csv.gz)

![选中网络的双亲家谱](pilot/fixed_evo_seed1205/visuals/selected_family_tree.png)

![祖先路径上变化最大的连接权重](pilot/fixed_evo_seed1205/visuals/lineage_weight_curves.png)

## 测试视频与飞行激活

视频使用预先指定的首个测试场景，不挑成功案例。并排比较 PID、NEAT 和循环 PPO，并附带机体近景；终止后的画面保持末帧并标记 STOPPED。PASS/FAIL 对应整段测试的严格判据，不是当前画面是否已经触地。策略由验证集选择，未按测试视频表现选择。

![常规测试预览](videos/comparison_nominal_preview.png)

[常规测试对照视频](videos/comparison_nominal.mp4) · [扰动测试对照视频](videos/comparison_stress.mp4)

[NEAT 翅膀 10 倍慢动作：常规](videos/neat_wings_nominal_10x.mp4) · [扰动](videos/neat_wings_stress_10x.mp4)

[交互飞行节点回放：常规](flight/neat_nominal.html) · [扰动](flight/neat_stress.html)：按时间查看归一化输入、节点状态、残差输出，以及各条连接的加权输入；包含上一拍状态和偏置。连接贡献大小本身不是因果重要性。

![NEAT 常规轨迹](figures/neat_nominal.png)

![NEAT 扰动轨迹](figures/neat_stress.png)

![策略实际收到的输入历史](figures/neat_stress_observations.png)

输入历史按策略实际使用的固定量纲归一化绘制，包含 9 维 IMU 与两个高度辅助通道。图中颜色在 ±1 饱和；原始数值完整保存在轨迹中，并可在交互回放中逐拍查看。

![飞行中的节点激活](figures/neat_stress_activations.png)

激活图显示神经元状态随时间变化，飞行期间网络权重保持固定；它不是在线重新训练。翅膀慢动作取自同一轨迹的 250 Hz 姿态记录，以 25 fps 播放，没有添加人工扑动或插值。常速 60 fps 视频对高频扑动可能产生频闪或混叠，因此检查翅膀运动时应结合慢动作及其关节角记录。

所有代表策略的轨迹都保留在下面，包括提前终止的直接 PPO；曲线只延伸到实际终止时刻。

| 方法 | 常规轨迹 | 扰动轨迹 | 原始记录 |
|---|---|---|---|
| direct_ppo | [查看](figures/direct_ppo_nominal.png) | [查看](figures/direct_ppo_stress.png) | [常规](traces/direct_ppo_nominal.npz) · [扰动](traces/direct_ppo_stress.npz) |
| ff_ppo | [查看](figures/ff_ppo_nominal.png) | [查看](figures/ff_ppo_stress.png) | [常规](traces/ff_ppo_nominal.npz) · [扰动](traces/ff_ppo_stress.npz) |
| fixed_evo | [查看](figures/fixed_evo_nominal.png) | [查看](figures/fixed_evo_stress.png) | [常规](traces/fixed_evo_nominal.npz) · [扰动](traces/fixed_evo_stress.npz) |
| graph_ppo | [查看](figures/graph_ppo_nominal.png) | [查看](figures/graph_ppo_stress.png) | [常规](traces/graph_ppo_nominal.npz) · [扰动](traces/graph_ppo_stress.npz) |
| neat | [查看](figures/neat_nominal.png) | [查看](figures/neat_stress.png) | [常规](traces/neat_nominal.npz) · [扰动](traces/neat_stress.npz) |
| pid | [查看](figures/pid_nominal.png) | [查看](figures/pid_stress.png) | [常规](traces/pid_nominal.npz) · [扰动](traces/pid_stress.npz) |
| random_graph_ppo | [查看](figures/random_graph_ppo_nominal.png) | [查看](figures/random_graph_ppo_stress.png) | [常规](traces/random_graph_ppo_nominal.npz) · [扰动](traces/random_graph_ppo_stress.npz) |
| rnn_ppo | [查看](figures/rnn_ppo_nominal.png) | [查看](figures/rnn_ppo_stress.png) | [常规](traces/rnn_ppo_nominal.npz) · [扰动](traces/rnn_ppo_stress.npz) |

## 策略文件与复现

[direct_ppo 代表策略](pilot/direct_ppo_seed1204/best_policy.json) · [ff_ppo 代表策略](pilot/ff_ppo_seed1204/best_policy.json) · [fixed_evo 代表策略](pilot/fixed_evo_seed1204/best_policy.json) · [graph_ppo 代表策略](transfer/graph_ppo_seed1204/best_policy.json) · [neat 代表策略](pilot/neat_seed1204/best_policy.json) · [random_graph_ppo 代表策略](transfer/random_graph_ppo_seed1201/best_policy.json) · [rnn_ppo 代表策略](pilot/rnn_ppo_seed1204/best_policy.json)

[全部策略的冻结哈希及测试清单](heldout_manifest.json) · [训练/验证/测试场景分离检查](split_verification.json)

上述 NEAT 和残差 PPO 文件需要配套本轮固定 PID、传感器处理及控制频率；其三个输出是归一化残差，不能直接当作执行器角度或电机转速。下面在仓库根目录使用已有 MuJoCo 环境回放验证集选出的 NEAT 代表策略，输出保存到一个新目录：

```sh
python -m experiments.algorithms.topology_hover.replay \
  --policy artifacts/topology_hover_20261008/pilot/neat_seed1204/best_policy.json \
  --output artifacts/manual_topology_neat_replay --condition stress --seconds 20
```

该命令使用另一个诊断种子，生成指标与轨迹，不会改写本报告的正式测试记录。下载或克隆完整实验目录后，可以离线打开演化浏览器和飞行节点回放。

## 证据边界

所有数据来自仿真。IMU 与倾角指标均对应下方机架 base_link；上部机构姿态可能与机架不同。模型原有机器人碰撞掩码关闭，离地标准采用全机器人几何包围盒下界，不代表接触力仿真或实机防碰撞验证。初始条件是约 0.8 m 静止释放和 100 ms 电机升速，不是地面自主起飞。初始高度作为准确释放参考；后续高度反馈才来自测距模型。

如果连接移除或随机化导致性能下降，只说明该已训练网络对连接敏感；只有重新训练和同规模拓扑对照，才更有力地支持结构价值。

## 参考依据

- [NEAT 原始论文与结构消融](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)。
- [Flappy Hummingbird：反馈控制上的 DDPG 附加控制](https://arxiv.org/html/1902.09628)。
- [扑翼悬停的行为克隆＋PPO，使用外部状态反馈](https://arxiv.org/html/2502.12355v1)。
- [结构演化与 actor–critic 结合的既有研究](https://arxiv.org/abs/1910.12824)。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent；当前任务内执行，未委派子代理。
- Mode: experiment execution, audit, scientific visualization。
- Verification: 仿真结果按文件记录；最终结论必须与正式评估状态一致。
- Version: topology_hover_20261008。
- [实际依赖版本](environment_versions.json)；[初始训练源码快照](initial_training_source.zip)。
- [最终实验与可视化源码](final_implementation_source.zip) · [对应文件哈希](final_implementation_hashes.json)。原始物理资产仍由本仓库的既有 PID 实验目录提供；此压缩包不是脱离仓库的独立模拟器。
- [实验产物完整性审查](artifact_verification.json) · [控制器测试记录](controller_tests.log)。
- [图表、浏览器与视频审阅记录](visual_review.json)。
- [完整实验文件 SHA-256 清单](artifact_manifest.json)，不包含清单自身。
