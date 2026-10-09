"""Data-bound report; negative hypotheses and failed episodes remain visible."""
import argparse,json
from pathlib import Path
import numpy as np
from .common import write

LABELS={'pid':'固定 PID','snn_neat':'SNN 联合演化','snn_fixed_tau':'SNN 固定时间常数','snn_fixed_topology':'SNN 固定拓扑','leaky_ann_neat':'Leaky ANN-NEAT','ann_neat':'ANN-NEAT','rnn_ppo':'RNN-PPO','ff_ppo':'FF-PPO'}

def report(root):
    p=json.loads((root/'protocol.json').read_text());d=json.loads((root/'analysis.json').read_text());cap=p['final_budget'];methods=p['methods'];g=d['groups'];reps=json.loads((root/'representatives.json').read_text())
    def entry(m,group,condition,budget=None):return g[f'{m}/{0 if m=="pid" else cap if budget is None else budget}/{group}/{condition}']
    def count(x,recovery=False):return f'{x["recovery_count" if recovery else "success_count"]}/{x["episodes"]} ({100*x["recovery_count" if recovery else "success_count"]/x["episodes"]:.1f}%)'
    def pctinterval(x):return f'[{x[0]*100:+.1f}, {x[1]*100:+.1f}] pp'
    def table(headers,rows):return ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows]+['']
    costs=d['costs'];cost=lambda m,k:float(np.mean([r[k] for r in costs if r['method']==m]))
    rank=sorted(methods,key=lambda m:entry(m,'core','stress')['success_rate'],reverse=True);best=rank[0];gate=[LABELS[m] for m in methods if d['engineering'][m]['meets_proposed_gate']]
    lines=['# 扑翼悬停：拓扑、时间常数与脉冲机制的配对比较','',
        '**状态：35 次正式训练和全部独立评估已完成。以下结论只适用于冻结的仿真平台、传感器接口与执行器范围。**','',
        '[原 proposal](research_proposal_approved.md) · [执行协议](protocol.json) · [全部分析数据](analysis.json) · [逐回合原始结果](evaluation/episodes.jsonl) · [完整演化浏览器](evolution_index.html)','',
        '## 结果与贡献','',
        f'在这组共同扰动测试中，描述性成功率最高的是 **{LABELS[best]}：{count(entry(best,"core","stress"))}**；联合 SNN 为 **{count(entry("snn_neat","core","stress"))}**，固定 PID 为 **{count(entry("pid","core","stress"))}**。方法间比较使用五个独立训练种子的配对差异；此处的最高均值不是统计优胜声明。','',
        f'达到预设“可用仿真候选”工程门槛的方法：**{"、".join(gate) if gate else "没有方法全部达到"}**。该门槛同时要求多数训练种子的20秒稳健性，以及验证选中代表的60秒常规／扰动成功，不能只靠一次视频判定。','',
        '本研究可明确交付的贡献，是把有实验约束的扑翼模型、同一 PID 和11维观测接口下的拓扑／时间尺度搜索，与固定拓扑、固定时间常数、连续泄漏网络和 PPO 做可重放的匹配比较；同时给出参数规模、长时失效和大角度经验恢复范围。NEAT 与 SNN 的组合已有先例，组合本身不作为新颖性声明。','']
    for r in d['hypothesis_checks'][:2]:
        name='时间常数搜索' if r['hypothesis'].startswith('H1') else '拓扑搜索'
        c=next(x for x in d['contrasts'] if x['b']==('snn_fixed_tau' if name.startswith('时间') else 'snn_fixed_topology'))
        lines += [f'- **{name}**：联合 SNN 相对对应消融组的扰动成功率差为 {c["success_difference"]*100:+.1f} 个百分点，97.5% 种子自助区间 {pctinterval(c["success_interval"])}。'+('达到预设的探索性大效应判据。' if r['exploratory_large_positive_effect'] else '未达到预设的探索性大效应判据。')+f' Holm 校正精确检验 p={c["holm_adjusted_p"]:.4f}。']
    z=d['hypothesis_checks'][2]
    lines += [f'- **规模收益**：联合 SNN 对固定拓扑的平均有效连接减少 {100*z["effective_connection_reduction"]:.1f}%；'+('同时达到预设探索性非劣与20%压缩判据。' if z['exploratory_noninferiority'] else '未同时达到预设探索性非劣与20%压缩判据，不能宣称“更小且性能相当”。'),'',
        '**统计限度：**只有五个训练种子，两侧精确符号翻转检验的最小 p 值是2/32=0.0625，因此本设计无法在0.05水平给出精确显著性结论。自助区间可能更乐观，需与每个种子的原始结果一起读；区间不包含零也不等于稳定性证明。区间以训练种子为重采样单位，条件于此组固定测试场景，不把80个回合当80次独立训练。','',
        '## 公平比较的范围','',
        '所有方法使用上部7 g、下部8 g的同一模型；IMU位于 base_link，策略输入为姿态3维、角速度3维、加速度3维、测高误差和估计垂直速度，共11维。**这轮依赖测高信息；不是纯 IMU 的9维定高证明，也不是水平定点悬停。** yaw 和 XY没有闭环目标，水平漂移完整报告。','',
        '输出均为[-1,1]内的 α、β、电机残差，分别乘0.035 rad、0.035 rad、100 rad/s叠加同一固定 PID。执行器最终受 α/β ±0.3 rad、电机3115.224–3807.496 rad/s限制。PID为500 Hz，策略100 Hz，物理步长5 μs。SNN和Leaky ANN每次决策内部积分10次，每步1 ms；没有因此获得更多观测或控制机会。','',
        'SNN采用连续电流输入、隐藏CUBA-LIF神经元和连续tanh读出，是混合脉冲控制器。隐藏背景电流1.2、阈值1、减法重置1，脉冲幅值20；膜时间常数允许5–200 ms，突触时间常数2–100 ms，初值20/10 ms。连续泄漏对照使用同一双时间滤波和图结构，但背景为0，因此它检验的是阈值／重置／背景的组合机制，不能隔离单一“脉冲阈值”效应。','']
    lines+=table(['方法','图结构','时间常数','初始学习参数','优化'],[
        ['SNN 联合演化','同一4隐藏节点初图，允许增删','进化',81,'NEAT'],['SNN 固定时间常数','相同初图，允许增删','固定20/10 ms',67,'NEAT'],['SNN 固定拓扑','相同初图，固定60条连接','进化',81,'权重/偏置/τ演化'],['Leaky ANN-NEAT','相同初图，允许增删','进化',81,'NEAT'],['ANN-NEAT','相同初图，允许增删','无连续时间常数',67,'NEAT'],['RNN-PPO','固定8隐藏节点循环图','10 ms离散状态',211,'PPO'],['FF-PPO','固定32×32前馈图','无记忆',1539,'PPO']])
    lines+=['PPO两组用于共同观测／动作／物理预算下的工程比较，初始容量和优化器不同；它们不能单独证明拓扑因果收益。严格的拓扑对照是同初图的两组SNN。所有进化组使用同一连接复杂度惩罚。','',
        '## 训练、选模和收敛','',
        '每组五个种子2201–2205。NEAT种群64，最多32隐藏节点；兼容阈值1.0、每物种保留2精英、停滞15代，连接添加/删除0.30/0.06，节点添加/删除0.20/0.03；权重、偏置和对数时间常数变异见每次训练的 neat.ini。所有演化组从匹配初始种群出发，第一只个体为零残差 PID。','',
        '训练课程2/4/8秒；每次预算1,024,000个分配交互，256,000和512,000处保存桥接检查点。预算较proposal的262,144/1,048,576统一向下对齐2.34375%，以便完整种群代数与PPO采样批次对齐。进化共40代；若提前失败，实际交互少于分配预算。PPO按实际交互计数，不宣称二者实际样本数完全相等。','',
        '每51,200步在固定8个8秒验证场景上选模，按严格成功率、平均回报、负倾角RMS依次排序，初始策略亦可被选中。最终独立测试开始前冻结全部策略和哈希。停止原因为预先固定预算；“经验平台期”只表示最近验证选中结果变化较小，不是全局收敛或动力学稳定性证明。','',
        '**保留的协议修订：**v1的兼容阈值0.25将64个体过度分为28个物种，压缩了繁殖机会；在89,600个已记录训练步和12,800验证步后停止，另有最多19,200步中断工作。旧数据全部在[开发归档](development/aborted_speciation_v1/)。v2的1.0阈值由独立随机适应度繁殖检查确定，不使用飞行奖励或测试成绩；[完整修订记录](protocol_revision.json)和[筛查结果](development/speciation_screen.json)均保留。','',
        '![验证曲线](figures/validation_curves.png)','']
    trainrows=[]
    for m in methods:
        for seed in p['seeds']:
            folder=root/f'training/{m}_seed{seed}';r=json.loads((folder/'results.json').read_text());s=json.loads((folder/'selection.json').read_text());v=r['best_validation']
            trainrows.append([LABELS[m],seed,r['training_steps'],r['validation_steps'],s.get('generation','PPO'),f'{v["success_rate"]:.1%}',f'{v["tilt_rms_deg"]:.2f}°',r['validation_plateau']])
    lines+=table(['方法','种子','实际训练步','验证步','选中代','验证通过','验证倾角 RMS','经验平台期'],trainrows)
    lines+=['## 独立20秒测试','',
        '每方法五个训练种子×16共同场景=80回合；PID每条件16回合。常规、扰动和组合OOD全部保留。严格成功需完整时长、最低几何间隙>3 cm、稳定段倾角RMS<10°、全程高度RMS<15 cm、最后5秒高度峰峰值<20 cm、漂移斜率绝对值<3 cm/s。未飞满的RMS可能因提前终止而变小，不能单独用低RMS判优。','']
    lines+=table(['方法','常规严格成功','扰动严格成功','组合OOD严格成功','扰动飞满','扰动倾角 RMS','扰动高度 RMS'],[
        [LABELS[m],count(entry(m,'core','nominal')),count(entry(m,'core','stress')),count(entry(m,'core','ood')),f'{entry(m,"core","stress")["full_count"]}/{entry(m,"core","stress")["episodes"]}',f'{entry(m,"core","stress")["tilt_rms_deg"]:.2f}°',f'{entry(m,"core","stress")["height_rmse_m"]:.3f} m'] for m in ['pid']+methods])
    lines+=['![性能与规模](figures/performance_complexity.png)','',
        '扰动条件每个训练种子的严格成功率：','']
    lines+=table(['方法']+list(map(str,p['seeds'])),[[LABELS[m]]+[f'{x["success_rate"]:.1%}' for x in entry(m,'core','stress')['per_seed']] for m in methods])
    lines+=table(['联合SNN 减去对照','成功率差','区间','倾角RMS差','比较性质'],[[LABELS[r['b']],f'{100*r["success_difference"]:+.1f} pp',pctinterval(r['success_interval']),f'{r["tilt_difference_deg"]:+.2f}°','主要97.5%' if r['primary'] else '次要95%'] for r in d['contrasts']])
    lines+=['![主要比较的逐种子配对](figures/primary_seed_pairs.png)','']
    lines+=['所有21对方法间扰动成功率差均在 analysis.json 的 all_pairwise_exploratory 中公开；未作多重校正的次要区间不用于确认性优胜结论。','',
        '## 与上一轮的衔接','',
        '当前256,000步检查点和上一轮262,144步冻结策略，在本轮相同的新场景上重新测试。上一轮种群128、课程/初始化不同，所以这是跨轮工程比较；严格消融仍看本轮内部。旧策略没有因新测试结果重新训练或重新选模。','']
    lines+=table(['策略来源','分配训练预算','常规','扰动','OOD'],[[LABELS[m],p['bridge_budget']]+[count(entry(m,'core',c,p['bridge_budget'])) for c in ['nominal','stress','ood']] for m in methods]+[[m,262144]+[count(g[f'{m}/262144/core/{c}']) for c in ['nominal','stress','ood']] for m in ['old_neat','old_fixed_evo','old_rnn_ppo','old_ff_ppo']])
    lines+=['## 60秒持续悬停与工程门槛','',
        '本轮对每个训练种子都运行8个常规、8个扰动长回合，避免只展示最佳策略。下面总体结果每方法每条件40回合；代表策略由训练验证结果选定，独立测试不参与选择。','']
    lines+=table(['方法','常规严格成功','扰动严格成功','常规飞满','扰动飞满','常规/扰动末端XY漂移'],[[LABELS[m],count(entry(m,'long','nominal')),count(entry(m,'long','stress')),f'{entry(m,"long","nominal")["full_count"]}/{entry(m,"long","nominal")["episodes"]}',f'{entry(m,"long","stress")["full_count"]}/{entry(m,"long","stress")["episodes"]}',f'{entry(m,"long","nominal")["xy_end_mean_m"]:.2f} / {entry(m,"long","stress")["xy_end_mean_m"]:.2f} m'] for m in ['pid']+methods])
    lines+=table(['方法','达到短时门槛的种子数','代表种子','代表常规/扰动长时成功','整体门槛'],[[LABELS[m],f'{d["engineering"][m]["qualifying_training_seeds"]}/5',d['engineering'][m]['representative_seed'],str(d['engineering'][m]['representative_long_success']),d['engineering'][m]['meets_proposed_gate']] for m in methods])
    lines+=table(['方法','种子','60秒常规','60秒扰动'],[[LABELS[m],seed]+[f'{next(x for x in entry(m,"long",c)["per_seed"] if x["seed"]==seed)["success_rate"]:.1%}' for c in ['nominal','stress']] for m in methods for seed in p['seeds']])
    lines+=['## 大角度与飞行中扰动恢复','',
        '初始真实倾角分别5°、10°、15°、20°、30°、45°，8个水平轴方向，每方向2个噪声实现；两个实现复用预定方向对应相位。初始角速度为零，释放高度、电机启动过程和神经状态重置规则相同。45°属于探索极限档。独立脉冲在第5秒开始，持续0.1秒，三个力矩档16/32/64 μN·m。脉冲通过外力自然改变运动，不重写姿态或清空神经状态。','',
        '恢复成功：事件结束后5秒内，过去0.5秒倾角RMS低于10°并继续保持1秒，随后飞满20秒且满足原高度和间隙标准。下表为此新增指标；原严格成功、恢复时间、最大倾角、下降高度、执行器饱和及测高角度失效比例仍完整保存在分析及逐回合文件。','']
    lines+=table(['方法']+[f'{x}°' for x in p['recovery']['angles']],[[LABELS[m]]+[count(entry(m,'angle',str(a)),True) for a in p['recovery']['angles']] for m in ['pid']+methods])
    lines+=table(['方法','16 μN·m','32 μN·m','64 μN·m'],[[LABELS[m]]+[count(entry(m,'pulse',str(t)),True) for t in p['recovery']['pulse_torques']] for m in ['pid']+methods])
    lines+=['![经验恢复范围](figures/large_angle_recovery.png)','',
        '该曲线只测定给定状态、传感器和控制权限下的经验恢复范围，不是吸引域证明。以合力近似沿机体倾斜方向估算，30°/45°定高分别需要约15.5%/41.4%额外推力；实际α/β偏转和扑翼非定常载荷会改变这一近似。测高有效性受机体倾角限制，失败不能直接归于某种网络缺乏记忆。','',
        '## 单因素诊断与机制干预','',
        '单因素实验从同一扰动基线分别提高观测延迟、电机增益不确定性、下部质量、外力与力矩强度；不与组合OOD混为一组。','']
    lines+=table(['方法']+p['factors']['names'],[[LABELS[m]]+[count(entry(m,'factor',x)) for x in p['factors']['names']] for m in ['pid']+methods])
    lines+=table(['联合SNN干预','严格成功','倾角RMS','高度RMS','飞满'],[[x,count(entry('snn_neat','intervention',x)),f'{entry("snn_neat","intervention",x)["tilt_rms_deg"]:.2f}°',f'{entry("snn_neat","intervention",x)["height_rmse_m"]:.3f} m',f'{entry("snn_neat","intervention",x)["full_count"]}/40'] for x in p['interventions']['variants']])
    lines+=['tau_initial恢复20/10 ms；tau_shuffle在隐藏／读出节点内分别打乱成对时间常数；reset_each_tick每个100 Hz决策前清零动态状态；remove_recurrent移除隐藏节点之间的连接（含自环），三个随机删边对照删去相同数量。干预可能破坏共同适应或控制容量，应结合重新训练的消融组解释，不能单凭破坏后性能下降证明某条边的独立因果贡献。','',
        '## 参数、计算成本及活动','',
        '参数计数包括权重、偏置和可进化时间常数；固定常数另外保存但不计为学习参数。“有效连接”是输入可达且能到输出的启用边数，不等于被证明必要的边。背景或偏置可驱动的独立分支仍可能产生动作，因此同时报告全部启用连接和参数。推理时间为本机NumPy CPU实现、训练结束后的计时，含SNN的全部10个内部步骤；它不是MCU或神经形态硬件能耗测量。','']
    lines+=table(['方法','学习参数均值','启用连接均值','状态浮点数','推理中位μs均值','总训练墙钟h'],[[LABELS[m],f'{cost(m,"parameters"):.1f}',f'{cost(m,"connections"):.1f}',f'{cost(m,"state_floats"):.1f}',f'{cost(m,"numpy_inference_us_median"):.1f}',f'{sum(x["wall_seconds"] for x in costs if x["method"]==m)/3600:.2f}'] for m in methods])
    lines+=table(['方法','种子','隐藏节点平均脉冲Hz','不放电隐藏节点比例','接近时间常数边界的节点'],[[LABELS[r['method']],r['seed'],f'{r["mean_hidden_spike_hz"]:.2f}',f'{r["mean_silent_hidden_fraction"]:.1%}',r['near_boundary_nodes']] for r in d['temporal']])
    lines+=['连续泄漏网络本身不产生脉冲，表中的零脉冲是定义，不是故障。接近边界按上下界的1%乘性范围判定；时间常数聚集或脉冲相位关系不自动意味着找到了物理共振。','', '![全部选中隐藏节点的时间尺度](figures/time_constants.png)','']
    formal_steps=sum(x['actual_training_steps'] for x in costs);validation_steps=sum(x['validation_steps'] for x in costs)
    lines += [f'正式实际训练共 **{formal_steps:,}** 步；验证与选模 **{validation_steps:,}** 步；独立评估 **{d["total_evaluation_episodes"]:,}** 回合、**{d["actual_evaluation_steps"]:,}** 步。开发PID包线29,806步、v1已记录训练89,600与验证12,800步以及中断上界19,200步均不计入正式性能，但属于总项目成本。小规模软件／媒体检查另存开发目录。','',
        '## 可视化、模型与视频','',
        '[打开所有25次演化运行](evolution_index.html)。每个浏览器包含全部64个体×40代、真实父代、出生时变异、权重、偏置、时间常数；冠军动画不是一条父子链。家谱显示双亲，热图仅沿编号较小父代的确定性分支，不暗示该父代贡献较大。','']
    for m in methods:
        e=reps[m];path=Path(e['path']);rel=path.relative_to(root).as_posix();folder=path.parents[1].relative_to(root).as_posix()
        lines += [f'- {LABELS[m]}：验证代表种子{e["training_seed"]}，[策略]({rel})'+(f' · [演化浏览器]({folder}/visuals/explorer.html)' if not m.endswith('_ppo') else '')]
    sf=Path(reps['snn_neat']['path']).parents[1].relative_to(root).as_posix()
    lines += ['',f'![联合SNN最终结构]({sf}/visuals/selected_network.png)',f'![联合SNN时间常数祖先变化]({sf}/visuals/lineage_tau_m.png)','',
        '[SNN脉冲、电压、电流交互回放](flight/snn_neat_stress.html) · [全部原始轨迹](traces/)','',
        '![SNN内部活动](figures/snn_neat_stress_spikes.png)','',
        '[常规：PID / SNN / ANN](videos/comparison_nominal_ann_neat.mp4) · [扰动：PID / SNN / ANN](videos/comparison_stress_ann_neat.mp4) · [扰动：PID / SNN / RNN-PPO](videos/comparison_stress_rnn_ppo.mp4) · [30°释放](videos/comparison_angle30_rnn_ppo.mp4) · [翅膀10倍慢放](videos/snn_wings_stress_10x.mp4)','',
        '![扰动视频预览](videos/comparison_stress_ann_neat_preview.png)','',
        '视频统一使用验证选中的代表和第一个预定测试种子，未按测试效果挑选。所有姿态及翅膀位置来自真实仿真qpos，重建机体位置误差检查为1e−9 m；飞行对比60 fps正常速度，翅膀近景250状态样本/s以25 fps播放，明确为10倍慢放。失效面板冻结并标注终止原因。跟随XY的镜头用于看清机器人，不代表无水平漂移。','',
        '## 物理依据与适用边界','',
        '模型依据包含36条10–15 Hz单翼测力记录；平均升力系数用10–12 Hz拟合、13–15 Hz留出，所存两种模型留出MAPE分别约2.64%和3.57%。这是实际数据约束，不应写成完全没有实测依据。另一方面，它未直接验证双翼约26.7 Hz自由飞行的周期气动力、力矩和传感器误差；从该频段到当前控制工况仍有外推。来源原字节与哈希保存在[证据目录](../../docs/research/snn_neat_20261009/evidence/manifest.json)。','',
        '15 g静态校准约3461.36 rad/s、26.75 Hz、15.00 gf，但力矩容差下 converged=false；当前自由飞行PID前馈为3190 rad/s，二者不能视为同一个已验证平衡点。候选0720电机绕组尚未唯一识别。IMU姿态由带噪真值和滤波近似，并非实机AHRS；测高有角度与量程限制。地面采用保守几何间隙判据、碰撞掩码关闭，因此本研究没有验证真实接触力。','',
        '控制器可以抑制姿态和高度误差，却不能从有限的IMU历史永久恢复绝对水平位置。SNN稀疏脉冲数只是活动代理，不能直接写成节能；本次也不证明系统辨识“天然适合SNN”。','',
        '## 文献定位','',
        '- [Qiu等，2018/2019，NEAT与循环SNN控制](https://arxiv.org/abs/1903.01180)：该组合已有控制先例。',
        '- [Fang等，ICCV2021，可学习膜时间常数](https://arxiv.org/abs/2007.05785)：支持时间尺度作为可学习变量，原任务不是扑翼悬停。',
        '- [Burgers等，IMAV2024](https://pure.tudelft.nl/ws/portalfiles/portal/226969121/8.pdf)：进化SNN模仿飞行PID，优化器为CMA-ES，非NEAT。',
        '- [Stroobants等，2025，500 Hz无人机SNN控制](https://arxiv.org/html/2411.13945v2)：控制器时延重要，实验使用外部位置信息，不能证明纯IMU定点悬停。',
        '- [van Breukelen Castillo等，IMAV2025](https://www.imavs.org/papers/2025/17.pdf)：SNN-PPO飞行应用，不是本研究的扑翼平台。',
        '- [El Filali等，2026 v3，Spiking Neural Network Control of a Flapping-Wing Robot on Resource-Constrained Hardware](https://arxiv.org/html/2605.19430v3)：已有26 g扑翼硬件、SNN模仿及CPG控制，不能把本研究写成首次扑翼SNN；其结果也不等于NEAT结构/时间常数共同演化与悬停消融。',
        '- [Loyola-Jara等，2026，NEAT神经元与编码比较](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1697163/full)：神经元与编码应作为可检验因素。',
        '- [Lee、Leok、McClamroch，2010](https://arxiv.org/abs/1003.2005)：大姿态恢复有专门的几何控制理论；其四旋翼假设不能直接移植为本模型稳定性保证。','',
        '## 复现与完整性','',
        '运行入口及依赖见[实验说明](../../experiments/algorithms/snn_hover/README.md)。protocol.json保存训练前版本、包版本、全部核心源码哈希；frozen_source保存冻结训练实现，test_manifest记录独立评估前策略哈希。final_implementation_source.zip保存评估与可视化的最终实现。verification.json及manifest.json由完整性检查生成；未通过检查不能交付为完成。','',
        '所有终止原因、逐项失败标准、完整种子结果和成本均保留。未通过统计／工程门槛时，贡献应表述为该控制接口下的定量边界与可重放对照证据；不能把“某个种子飞得好”改写成“拓扑或SNN普遍更优”。','']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    return dict(descriptive_stress_leader=best,engineering_gate_passes=gate,actual_training_steps=formal_steps,validation_steps=validation_steps)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();write(a.root/'report_summary.json',report(a.root))
if __name__=='__main__':main()
