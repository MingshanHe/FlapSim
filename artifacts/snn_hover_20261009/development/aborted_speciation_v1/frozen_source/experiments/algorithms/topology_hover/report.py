"""Generate an honest live/final Markdown report from recorded artifacts."""
import argparse,json
from pathlib import Path
import numpy as np
from .common import write,summary

def generate(root,install_root=False):
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text())
    runs=sorted(root.glob('pilot/*/results.json'))+sorted(root.glob('transfer/*/results.json'))+sorted(root.glob('formal/*/results.json'))+sorted(root.glob('imu/*/results.json'))
    statuses=sorted(root.glob('pilot/*/status.json'))+sorted(root.glob('transfer/*/status.json'))+sorted(root.glob('formal/*/status.json'))+sorted(root.glob('imu/*/status.json'))
    completed=[json.loads(p.read_text()) for p in runs]
    final=(root/'final_results.json').exists()
    lines=['# 扑翼机器人：NEAT 拓扑演化、PID 残差控制与 PPO 对照','',
        '**状态：'+('正式评估已完成。' if final else '实验进行中；以下训练/验证结果不是最终结论。')+'**','',
        '上部 7 g、下部 8 g；yaw 和水平位置不控制。本轮目标是姿态稳定、保高及整机保持离地。不能把有限时长通过称为三维定点悬停。','',
        '**严格通过标准：** 跑满规定时长，整机保守离地间隙始终大于 3 cm；稳定阶段倾角 RMS 小于 10°；全程高度 RMS 小于 15 cm；最后 5 秒高度峰峰值小于 20 cm、线性漂移速度绝对值小于 3 cm/s。正式测试的稳定阶段从第 2 秒开始。提前终止均为失败。','',
        '## 阅读入口','',
        '- [正式测试与结果](#正式测试) · [NEAT 演化可视化](#neat-演化可视化) · [测试视频与飞行激活](#测试视频与飞行激活)',
        '- [训练与可视化总览](dashboard.html)：本地服务下显示实时进度；离线显示生成时的快照。',
        '- [预先保存的实验协议](protocol.json) · [源文件哈希](source_hashes.json) · [传感器/PID 前置检查](preflight.json)','- 各个体、父代、物种、连接创新编号、启用状态、全部权重及节点偏置保存在每代 `generation_*.json.gz`。',
        '- 下方动画显示各代冠军，冠军之间不一定有父子关系；祖先图和权重图另沿真实父代记录生成。','',
        '## 已确认的前置结果','']
    if (root/'preflight.json').exists():
        rows=json.loads((root/'preflight.json').read_text())['episodes']
        lines+=['| 传感器条件 | 场景 | 成功/总数 | 撑满时长 | 倾角 RMS | 高度 RMS |','|---|---|---:|---:|---:|---:|']
        for key in sorted({(r['sensor'],r['case']['condition']) for r in rows}):
            subset=[r for r in rows if (r['sensor'],r['case']['condition'])==key];s=summary(subset)
            lines.append(f'| {key[0]} | {key[1]} | {sum(r["success"] for r in subset)}/{len(subset)} | {sum(r["termination"]=="time_limit" for r in subset)}/{len(subset)} | {s["tilt_rms_deg"]:.2f}° | {s["height_rmse_m"]*100:.2f} cm |')
    lines+=['','高度辅助在这里是 50 Hz、20 ms 延迟、2 mm 测距噪声的模拟测距仪，垂向速度由同一观察器估计。`truth` 是独立真值诊断；`imu` 为纯惯性积分。前置数据不用于最终算法排名。',
        '','加入 ±32 g 量程后，虚拟加速度计的少量高幅脉冲截断可产生明显积分误差；这揭示了现有模型与传感器接口的限制。它不是“实机一定会这样”的结论。姿态读数仍由抽样真值姿态加偏置/噪声模拟 AHRS，并未实现完整姿态解算。',
        '','## 控制和公平性','',
        '固定 PID 提供基准动作，学习器通过较低频率的有界残差补偿姿态与升力。以下为本轮残差组的共同结构：','',
        '```mermaid',
        'flowchart LR',
        '  Plant["扑翼机器人"] --> Sensor["9 维 IMU + 高度误差 / 垂向速度估计"]',
        '  Sensor --> PID["固定 PID · 500 Hz"]',
        '  Sensor --> Actor["NEAT / PPO · 100 Hz"]',
        '  Actor --> Scale["有界残差 δα、δβ、δmotor"]',
        '  PID --> Sum["相加与执行器限幅"]',
        '  Scale --> Sum',
        '  Sum -->|α、β、电机角速度命令| Plant',
        '```','',
        '- 同一质量、质心、惯性、气动力、电机模型、软启动及执行器伺服。物理步长 5 μs；PID 500 Hz；所有学习策略 100 Hz。',
        '- 残差动作为 `[δα, δβ, δω]`，三个输出均为 `[-1, 1]`，比例为 `[0.035 rad, 0.035 rad, 100 rad/s]`，再叠加固定 PID。实际命令仍受原有物理范围限制。',
        '- 本轮训练的策略统一 11 维：9 维 IMU 加测距高度误差和估计垂向速度。另做移除测距及两个策略输入后的 9 维诊断，同时将 PID 高度估计切换为纯惯性积分；未单独重训 9 维策略，因此该诊断同时涉及传感器、估计器和策略输入的变化。训练/奖励可使用真值，部署策略不读取真实位置、速度、扑翼相位或外力。',
        '- 所有网络先统一 tanh，避免把激活函数更换当成拓扑收益。NEAT 的节点来自非输入节点的连接均使用上一策略时刻状态。',
        '- 输入按固定量纲归一化：roll/pitch 除以 0.35 rad、yaw 除以 π、角速度除以 10 rad/s、加速度除以 9.81 m/s²、高度误差除以 0.2 m、垂向速度除以 1 m/s；最后裁剪到 [-10, 10]。图中的权重作用于这些归一化值，不能只凭权重大小判断某个传感器的重要性。',
        '- PPO 为本目录显式实现的 clipped PPO，包含 GAE、终止/时间截断区分、循环序列梯度与导出一致性检查。循环对照和拓扑移植使用同一图更新规则。',
        '- 直接 PPO 的三个输出映射到 α/β 的 ±0.3 rad，以及电机 3461.36 ± 346.14 rad/s；它需要独立覆盖完整控制范围。另用 PID 演示初始化，演示数据和交互数单列。',
        '- 直接 PPO 的控制决策为 100 Hz；混合方法还具有 500 Hz PID 反馈。因此这个对照同时改变控制结构、先验和反馈带宽，不是纯优化器对照。即使直接 PPO 失败，也不能推论 PPO 在更大预算或其他控制频率下必须依赖 PID。',
        '- 训练预算按实际交互数记录，包括 NEAT 全种群评估；验证、演示和拓扑搜索成本分别记录。固定拓扑进化基线具有 8 个隐藏节点，完整 NEAT 从 2 个隐藏节点起步，故两者的初始容量并非完全一致；同规模随机拓扑移植提供额外结构对照。',
        '- 固定循环 PPO 通过隐藏状态向输出传递观测，NEAT 初始结构允许输入直连输出。这些结构先验也可能影响表现；匹配随机结构对照保留各节点的输入入度和循环入度，以进一步区分连接关系与容量、通路类型的影响。',
        '- CPU 推理时间使用相同 NumPy 导出路径，并在全部训练退出后统一重测；稀疏图当前以密集矩阵求值。部分结构移植任务与主队列并行执行，单次训练墙钟时间不能直接用于算法速度排名。连接少不等于已经证明微控制器更省电。',
        '','## 本轮训练参数','',
        '| 项目 | 配置 |','|---|---|',
        '| NEAT 种群 | 128；每个训练种子约 262,144 次策略交互，统计整个种群的消耗 |',
        '| 初始结构 | 2 个隐藏节点、稀疏输入和循环连接；输出含输入直连 |',
        '| 结构变异 | 加连接 0.30、删连接 0.06、加节点 0.20、删节点 0.03、连接启用状态变异 0.015 |',
        '| 参数变异 | 权重变异率 0.50、扰动标准差 0.04；偏置扰动标准差 0.015；完整配置见各运行 neat.ini |',
        '| 激活 | 全部使用普通 tanh；本轮未进行 sigmoid/混合激活库搜索 |',
        '| 固定结构进化 | 8 个隐藏节点，关闭结构和连接启用状态变异 |',
        '| 前馈 PPO | 两层 32 节点；循环 PPO 为 8 个隐藏节点加 3 个输出状态 |',
        '| PPO 优化 | 学习率 3e-4，clip 0.2，gamma 0.995，GAE lambda 0.95，32 步循环反传，最多 6 个 epoch |',
        '| 独立重复 | 5 个 pilot 种子；只根据预定的验证集门槛决定是否扩展至 10 个 |',
        '| 验证选择 | 每个种子 4 个固定 8 秒场景；按成功率、回报、倾角依次选择 |',
        '',
        '主假设是演化出来的连接关系能否在同等规模和权重重新训练后继续带来收益。本轮先固定激活函数和种群规模隔离这个问题；256 种群、混合激活库及更大循环网络属于后续扩展，未被计作已完成实验。','',
        'NEAT 每代所有个体共享同一场景；前半程训练片段为 2 秒，后半程为 4 秒。最后一代为满足交互预算而统一缩短（当前约 0.48 秒），因此不能把记录的 9 代解释为 9 代完整稳态飞行训练。候选选择仍使用固定的 8 秒验证。','',
        '## 训练与验证进度','',
        '| 运行 | 状态 | 训练步数 | 验证步数 | 演示步数 | 选中阶段 | 验证成功率 | 验证倾角 RMS |','|---|---|---:|---:|---:|---|---:|---:|']
    decision_path=root/'expansion_decision.json'
    if decision_path.exists():
        decision=json.loads(decision_path.read_text())
        message='已扩展到 10 个种子' if decision['expand_to_ten'] else '保留预定的 5 个种子，未触发扩展'
        position=lines.index('## 训练与验证进度')
        lines[position:position]=[f'预定扩展门槛要求至少 3/5 个 NEAT 种子在配对验证中同时改善 PID 回报超过 1%、倾角 RMS 超过 5%。实际满足 {sum(decision["benefits"])}/5，{message}。[门槛记录](expansion_decision.json)；该决定在独立测试前完成。','']
    for p in statuses:
        try:s=json.loads(p.read_text())
        except (ValueError,OSError):continue
        v=s.get('best_validation') or {};folder=p.parent.relative_to(root)
        chosen=json.loads((p.parent/'selection.json').read_text()) if (p.parent/'selection.json').exists() else {}
        checkpoint=f'第 {chosen["generation"]} 代' if 'generation' in chosen else f'{chosen["training_steps"]:,} 步' if 'training_steps' in chosen else '待选择'
        lines.append(f'| [{folder}]({folder}/status.json) | {s["status"]} | {s["training_steps"]:,} | {s["validation_steps"]:,} | {s.get("demonstration_steps",0):,} | {checkpoint} | {v.get("success_rate",0):.0%} | {v.get("tilt_rms_deg",0):.2f}° |')
    lines+=['','以上为验证集选择结果。不同训练种子的验证场景不同，同一种子的方法间保持配对。训练 fitness 随每代场景及训练时长变化，不应只看训练曲线单调性。','',
        '选中阶段为 PPO 的 0 步时，表示 PPO 更新前的模型最终保留；直接 PPO 在这个阶段已经做过行为克隆。NEAT 第 0 代仍包含整个初始种群的评估和选择，但尚未发生后续代际演化。这些情况均不能当作后续强化学习或结构变异改善了控制的证据。',
        '',
        '## NEAT 演化可视化','',
        '每代将整个种群放进同一个训练场景评估。训练 fitness 为“累计回报 / 本代计划时长 − 0.00002 × 启用连接数”；提前失败仍除以计划时长。新网络通过父代交叉、权重/偏置扰动，以及增删节点和连接产生；NEAT 这部分不使用梯度下降。保留的精英个体沿用原 ID。场景和片段时长随代数更换，短片段中启动过程占比也更大，所以训练 fitness 上下波动不等于退化，固定验证集才用于选出交付策略。',
        '',
        '共同的飞行回报在每个 2 ms 控制步累加 `dt × [2 + exp(-(倾角/0.15 rad)²) + exp(-(高度误差/0.12 m)²)]`，提前失败扣 1；每次策略决策再扣 `0.0002 × ||本次归一化动作 − 上次动作||²`。它鼓励持续飞行、姿态和高度稳定，并抑制动作突变；yaw 和水平位置没有跟踪奖励。',
        '',
        '交互浏览器可以选择代数和任意个体、播放冠军序列、筛选连接权重、显示禁用连接、点击父代、查看全部连接权重及节点偏置。浏览器文件包含完整种群数据，可离线打开。','']
    lines+=['拓扑图中蓝色为正权重、红色为负权重，线宽表示权重绝对值；静态图仅显示启用连接。每个节点计算 `tanh(当前输入的加权和 + 上一时刻节点状态的加权和 + 偏置)`。因此图上从左到右的隐藏节点连接也可能带有一个控制周期的延迟，不能仅凭箭头方向把它当成普通前馈层。','',
        '家谱保留双亲；权重热图和曲线沿其中一条注明的祖先路径展示。相邻个体的权重差异可能同时来自交叉继承和参数变异；“变异事件”专门记录交叉完成后发生的修改，不将两者混为一谈。权重图中的负编号是输入通道，0/1/2 为 α/β/电机输出，其余编号为隐藏节点。','']
    for folder in sorted(root.glob('*/*'),key=lambda p:(not p.name.startswith('neat_seed'),p.name)):
        vis=folder/'visuals'
        if not (vis/'explorer.html').exists():continue
        rel=vis.relative_to(root);name=folder.name
        recorded=json.loads((folder/'history.json').read_text())
        generations=len(recorded);species=sorted({r['species'] for r in recorded})
        lines +=[f'### {name}','',f'[打开交互演化浏览器]({rel}/explorer.html) · [完整父代关系]({rel}/all_parent_links.json) · [图表说明与数据量]({rel}/visualization_manifest.json)','',
            f'记录 {generations} 代；每代物种数范围 {min(species)}–{max(species)}。这些是当前有限预算下的演化记录，不能替代更长训练的收敛研究。','',
            f'![演化过程与物种]({rel}/evolution_summary.png)','',f'![拓扑变化动画：各代冠军]({rel}/topology_evolution.gif)','',
            f'![验证选中网络]({rel}/selected_topology.png)','',f'![真实祖先路径的权重变化]({rel}/lineage_weights.png)','',
            f'![新增删除与种群权重分布]({rel}/mutations_and_weights.png)','']
        if (vis/'lineage_biases.png').exists():lines +=[f'![祖先节点偏置变化]({rel}/lineage_biases.png)','',f'[完整种群 CSV]({rel}/population.csv.gz) · [完整权重 CSV]({rel}/weights.csv.gz) · [节点偏置 CSV]({rel}/biases.csv.gz)','']
        if (vis/'selected_family_tree.png').exists():lines +=[f'![选中网络的双亲家谱]({rel}/selected_family_tree.png)','',f'![祖先路径上变化最大的连接权重]({rel}/lineage_weight_curves.png)','']
    if final:
        result=json.loads((root/'final_results.json').read_text())
        position=lines.index('## 已确认的前置结果')
        lines[position:position]=['## 正式测试','',result['markdown'],'']
        representatives_path=root/'representatives.json'
        if representatives_path.exists():
            representative=json.loads(representatives_path.read_text())['neat']
            folder=Path(representative['path']).parent.as_posix()
            position=lines.index('## 阅读入口')+2
            lines[position:position]=[
                f'- **先看代表策略（验证集选出，种子 {representative["seed"]}）：** [结构如何演化]({folder}/visuals/explorer.html) · [真实父代与权重]({folder}/visuals/lineage_weights.png)。',
                '- **再看飞行：** [扰动对照视频](videos/comparison_stress.mp4) · [翅膀慢动作](videos/neat_wings_stress_10x.mp4) · [逐拍输入与节点回放](flight/neat_stress.html)。'] if (root/'videos/comparison_stress.mp4').exists() else [
                f'- **代表策略（验证集选出，种子 {representative["seed"]}）：** [打开结构演化浏览器]({folder}/visuals/explorer.html)。']
        interpretation=root/'interpretation.md'
        if interpretation.exists():
            position=lines.index('## 正式测试')
            lines[position:position]=['## 结果解读','',interpretation.read_text(encoding='utf8').strip(),'']
    else:lines+=['## 正式测试','', '尚未完成。没有把正在训练的分数写成 NEAT 或 PPO 胜出的结论。','']
    if (root/'videos/comparison_nominal.mp4').exists():
        lines+=['## 测试视频与飞行激活','',
            '视频使用预先指定的首个测试场景，不挑成功案例。并排比较 PID、NEAT 和循环 PPO，并附带机体近景；终止后的画面保持末帧并标记 STOPPED。PASS/FAIL 对应整段测试的严格判据，不是当前画面是否已经触地。策略由验证集选择，未按测试视频表现选择。','',
            '![常规测试预览](videos/comparison_nominal_preview.png)','',
            '[常规测试对照视频](videos/comparison_nominal.mp4) · [扰动测试对照视频](videos/comparison_stress.mp4)','',
            '[NEAT 翅膀 10 倍慢动作：常规](videos/neat_wings_nominal_10x.mp4) · [扰动](videos/neat_wings_stress_10x.mp4)','',
            '[交互飞行节点回放：常规](flight/neat_nominal.html) · [扰动](flight/neat_stress.html)：按时间查看归一化输入、节点状态、残差输出，以及各条连接的加权输入；包含上一拍状态和偏置。连接贡献大小本身不是因果重要性。','',
            '![NEAT 常规轨迹](figures/neat_nominal.png)','',
            '![NEAT 扰动轨迹](figures/neat_stress.png)','',
            '![策略实际收到的输入历史](figures/neat_stress_observations.png)','',
            '输入历史按策略实际使用的固定量纲归一化绘制，包含 9 维 IMU 与两个高度辅助通道。图中颜色在 ±1 饱和；原始数值完整保存在轨迹中，并可在交互回放中逐拍查看。','',
            '![飞行中的节点激活](figures/neat_stress_activations.png)','',
            '激活图显示神经元状态随时间变化，飞行期间网络权重保持固定；它不是在线重新训练。翅膀慢动作取自同一轨迹的 250 Hz 姿态记录，以 25 fps 播放，没有添加人工扑动或插值。常速 60 fps 视频对高频扑动可能产生频闪或混叠，因此检查翅膀运动时应结合慢动作及其关节角记录。','']
        lines+=['所有代表策略的轨迹都保留在下面，包括提前终止的直接 PPO；曲线只延伸到实际终止时刻。','',
            '| 方法 | 常规轨迹 | 扰动轨迹 | 原始记录 |','|---|---|---|---|']
        for path in sorted((root/'figures').glob('*_nominal.png')):
            method=path.stem.removesuffix('_nominal')
            if not (root/'figures'/f'{method}_stress.png').exists():continue
            lines.append(f'| {method} | [查看](figures/{method}_nominal.png) | [查看](figures/{method}_stress.png) | [常规](traces/{method}_nominal.npz) · [扰动](traces/{method}_stress.npz) |')
        lines+=['']
    else:
        lines+=['## 测试视频与飞行激活','','本轮视频尚未生成，将使用独立测试中预先指定的场景制作。','']
    if final and (root/'representatives.json').exists():
        representatives=json.loads((root/'representatives.json').read_text())
        neat_path=representatives['neat']['path']
        lines+=['## 策略文件与复现','',
            ' · '.join(f'[{method} 代表策略]({entry["path"]})' for method,entry in sorted(representatives.items())),
            '', '[全部策略的冻结哈希及测试清单](heldout_manifest.json) · [训练/验证/测试场景分离检查](split_verification.json)',
            '', '上述 NEAT 和残差 PPO 文件需要配套本轮固定 PID、传感器处理及控制频率；其三个输出是归一化残差，不能直接当作执行器角度或电机转速。下面在仓库根目录使用已有 MuJoCo 环境回放验证集选出的 NEAT 代表策略，输出保存到一个新目录：','',
            '```sh',
            'python -m experiments.algorithms.topology_hover.replay \\',
            f'  --policy {root.as_posix()}/{neat_path} \\',
            '  --output artifacts/manual_topology_neat_replay --condition stress --seconds 20',
            '```','',
            '该命令使用另一个诊断种子，生成指标与轨迹，不会改写本报告的正式测试记录。下载或克隆完整实验目录后，可以离线打开演化浏览器和飞行节点回放。','']
    lines+=['## 证据边界','',
        '所有数据来自仿真。IMU 与倾角指标均对应下方机架 base_link；上部机构姿态可能与机架不同。模型原有机器人碰撞掩码关闭，离地标准采用全机器人几何包围盒下界，不代表接触力仿真或实机防碰撞验证。初始条件是约 0.8 m 静止释放和 100 ms 电机升速，不是地面自主起飞。初始高度作为准确释放参考；后续高度反馈才来自测距模型。',
        '', '如果连接移除或随机化导致性能下降，只说明该已训练网络对连接敏感；只有重新训练和同规模拓扑对照，才更有力地支持结构价值。','',
        '## 参考依据','',
        '- [NEAT 原始论文与结构消融](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)。',
        '- [Flappy Hummingbird：反馈控制上的 DDPG 附加控制](https://arxiv.org/html/1902.09628)。',
        '- [扑翼悬停的行为克隆＋PPO，使用外部状态反馈](https://arxiv.org/html/2502.12355v1)。',
        '- [结构演化与 actor–critic 结合的既有研究](https://arxiv.org/abs/1910.12824)。',
        '', '## Material Passport','',
        '- Origin Skill: academic-research-suite / experiment-agent；当前任务内执行，未委派子代理。',
        '- Mode: experiment execution, audit, scientific visualization。',
        '- Verification: 仿真结果按文件记录；最终结论必须与正式评估状态一致。',
        '- Version: topology_hover_20261008。',
        '- [实际依赖版本](environment_versions.json)；[初始训练源码快照](initial_training_source.zip)。']
    if (root/'final_implementation_source.zip').exists():
        lines+=['- [最终实验与可视化源码](final_implementation_source.zip) · [对应文件哈希](final_implementation_hashes.json)。原始物理资产仍由本仓库的既有 PID 实验目录提供；此压缩包不是脱离仓库的独立模拟器。']
    if (root/'artifact_verification.json').exists():
        lines+=['- [实验产物完整性审查](artifact_verification.json) · [控制器测试记录](controller_tests.log)。']
    if (root/'visual_review.json').exists():
        lines+=['- [图表、浏览器与视频审阅记录](visual_review.json)。']
    if (root/'artifact_manifest.json').exists():
        lines+=['- [完整实验文件 SHA-256 清单](artifact_manifest.json)，不包含清单自身。']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    if install_root:
        target=Path('report.md');old=target.read_text() if target.exists() else ''
        marker='<!-- topology-hover-index -->'
        if marker not in old:
            target.write_text(f'{marker}\n# 最新实验：拓扑演化与 PPO 对照\n\n[打开完整 report.md]({root.as_posix()}/report.md)\n\n本轮含每代种群、拓扑动画、真实父代链、权重变化和交互浏览器。\n\n---\n\n'+old,encoding='utf8')
    return root/'report.md'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--install-root',action='store_true');a=p.parse_args();print(generate(a.root,a.install_root))
