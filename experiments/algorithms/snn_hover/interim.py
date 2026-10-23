"""Seed2201 interim report from existing selection validation only; no new tests."""
import argparse,gzip,hashlib,json,posixpath,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import write
from .networks import complexity
from .report import LABELS


def read(path):return json.loads(Path(path).read_text())
def table(headers,rows):
    return ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,x))+' |' for x in rows]+['']


def generate(root):
    protocol=read(root/'protocol.json');methods=protocol['methods'];rows=[]
    for method in methods:
        folder=root/f'training/{method}_seed2201';r=read(folder/'results.json');s=read(folder/'selection.json');h=read(folder/'history.json')
        assert r['status']=='completed' and r['budget']==1024000
        filename=f'validation_{s["allocated_steps"]:07d}.json' if 'generation' in s else f'validation_{s["update"]:04d}.json'
        episodes=read(folder/filename);assert len(episodes)==8 and len({x['seed'] for x in episodes})==8
        assert abs(np.mean([x['success'] for x in episodes])-r['best_validation']['success_rate'])<1e-12
        spec=read(folder/'best_policy.json');c=complexity(spec);initial=h[0]['validation'];last=[x for x in h if 'validation' in x][-1]['validation']
        by_condition={name:dict(passed=sum(x['success'] for x in episodes if x['case']['condition']==name),
            tilt_rms_deg=float(np.mean([x['tilt_rms_deg'] for x in episodes if x['case']['condition']==name]))) for name in ['nominal','stress']}
        max_hidden=None
        graph_paths=None
        if not method.endswith('_ppo'):
            maxima=[]
            for path in folder.glob('generation_*.json.gz'):
                with gzip.open(path,'rt') as f:g=json.load(f)
                maxima.append(max(len(x['policy']['nodes'])-3 for x in g['individuals']))
            max_hidden=max(maxima)
            ancestors=set(spec['outputs']);previous=None
            while previous!=ancestors:
                previous=ancestors.copy()
                ancestors.update(e['source'] for e in spec['edges'] if e['enabled'] and e['target'] in ancestors)
            names=['roll','pitch','yaw','gyro_x','gyro_y','gyro_z','accel_x','accel_y','accel_z','height_error','vertical_speed']
            graph_paths=dict(input_names_with_output_path=[names[-i-1] for i in sorted(ancestors,reverse=True) if i<0],
                hidden_nodes_with_output_path=sorted(i for i in ancestors if i>=0 and i not in spec['outputs']),
                bias_only_outputs=[i for i in spec['outputs'] if not any(e['enabled'] and e['target']==i for e in spec['edges'])],
                changed_time_constants=sum(abs(n[k]-v)>1e-12 for n in spec['nodes'] for k,v in [('tau_m',.02),('tau_s',.01)]),
                changed_hidden_time_constants=sum(abs(n[k]-v)>1e-12 for n in spec['nodes'] if n['id'] not in spec['outputs'] for k,v in [('tau_m',.02),('tau_s',.01)]))
        rows.append(dict(method=method,result=r,selection=s,complexity=c,initial_validation=initial,last_candidate_validation=last,
            selected_validation=episodes,by_condition=by_condition,max_explored_hidden=max_hidden,
            graph_paths=graph_paths,
            policy_sha256=hashlib.sha256((folder/'best_policy.json').read_bytes()).hexdigest()))
    output=root/'interim_2201';output.mkdir(exist_ok=True);figures=output/'figures';figures.mkdir(exist_ok=True)
    write(output/'analysis.json',dict(scope='One training seed; existing8s model-selection validation; no independent test accessed',methods=rows))
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True);x=np.arange(len(rows));colors=plt.get_cmap('tab10')(x)
    values=[r['result']['best_validation']['success_rate']*100 for r in rows]
    axes[0].bar(x,values,color=colors)
    for i,v in enumerate(values):axes[0].text(i,v+2,f'{int(round(v/100*8))}/8',ha='center',fontsize=8)
    axes[0].set(ylabel='Selected validation strict success (%)',ylim=(0,112))
    for offset,condition in [(-.18,'nominal'),(.18,'stress')]:
        axes[1].bar(x+offset,[r['by_condition'][condition]['tilt_rms_deg'] for r in rows],width=.36,label=condition)
    axes[1].axhline(10,color='#aa3333',ls=':',label='10 deg criterion');axes[1].set(ylabel='Validation tilt RMS (deg)');axes[1].legend(fontsize=8)
    for ax in axes:ax.set_xticks(x,methods,rotation=30,ha='right',fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Seed2201 / 8-second model-selection validation only / NOT held-out performance')
    fig.savefig(figures/'validation_comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for i,r in enumerate(rows):
        history=read(root/f'training/{r["method"]}_seed2201/history.json');history=[x for x in history if 'best_validation' in x]
        xs=[x.get('allocated_steps',x['training_steps'])/1e6 for x in history]
        axes[0].plot(xs,[x['best_validation']['success_rate'] for x in history],label=r['method'],color=colors[i])
        axes[1].plot(xs,[x['best_validation']['tilt_rms_deg'] for x in history],color=colors[i])
    axes[0].legend(fontsize=7);axes[0].set(ylabel='Selected validation success');axes[1].set(ylabel='Selected validation tilt RMS (deg)')
    for ax in axes:ax.set_xlabel('Allocated training steps (million)');ax.grid(alpha=.2)
    fig.suptitle('Validation selection history / flat curves do not establish convergence')
    fig.savefig(figures/'selection_history.png',dpi=150);plt.close(fig)
    evo=[r for r in rows if not r['method'].endswith('_ppo')];x=np.arange(len(evo))
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    axes[0].bar(x-.18,[67 if r['method'] in ['ann_neat','snn_fixed_tau'] else 81 for r in evo],width=.36,label='Initial')
    axes[0].bar(x+.18,[r['complexity']['parameters'] for r in evo],width=.36,label='Selected')
    axes[1].bar(x-.18,[60]*len(evo),width=.36,label='Initial');axes[1].bar(x+.18,[r['complexity']['connections'] for r in evo],width=.36,label='Selected')
    axes[0].set(ylabel='Learned scalar parameters');axes[1].set(ylabel='Enabled connections')
    for ax in axes:ax.set_xticks(x,[r['method'] for r in evo],rotation=25,ha='right',fontsize=8);ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('Seed2201 complexity / parameter count includes evolved time constants')
    fig.savefig(figures/'complexity.png',dpi=150);plt.close(fig)
    state=read(root/'campaign_status.json');paused=state['status']=='paused'
    total_steps=sum(x['result']['training_steps'] for x in rows);validation_steps=sum(x['result']['validation_steps'] for x in rows)
    lines=['# 阶段性研究报告：seed 2201 的 NEAT、SNN 与 PPO','',
        '**这是阶段性报告，不是 report_final.md。七种方法各完成一个训练种子；本报告只分析训练中已经使用的8秒验证场景，尚未执行正式独立测试。**','',
        f'暂停状态：{("已在 snn_neat_seed2202 完成后暂停；共完成8/35次训练，剩余27次。" if paused else "已停止后续队列；snn_neat_seed2202 正在跑完，之后暂停。")} 2202 的结果只用于保存续跑边界，不混入本页的2201算法比较。','',
        '[迁移与续跑 exec.md](../../../exec.md) · [冻结协议](../protocol.json) · [原始阶段分析](analysis.json) · [全部已生成演化浏览器](../evolution_index.html)','',
        '## 当前能支持的研究结果','',
        '五个演化组的验证选中策略均通过8/8回合；两个PPO组均为5/8。其中常规场景所有方法都为4/4，扰动场景演化组为4/4、PPO为1/4。这说明当前进化实现能在固定PID基础上改善这组短时验证场景的姿态控制；不能由一个种子和重复选模场景推导出泛化优势。','',
        '**当前较清晰的候选贡献是稀疏控制图的规模收益，证据仍是阶段性的。** ANN-NEAT选中模型从60条连接减至12条、学习参数从67减至19，分别减少80.0%和71.6%，仍通过8/8验证回合，倾角RMS为5.47°。固定时间常数SNN为31个学习参数、22条连接，也通过8/8，倾角RMS为5.36°。二者值得在后续独立场景验证，但当前不能证明删边本身造成改善或性能非劣。','',
        '**联合演化时间常数暂未体现额外优势。** 联合SNN通过8/8，但倾角RMS为6.25°、80个学习参数、59条连接；固定时间常数SNN在同组验证上的RMS更低且更小。联合SNN相对固定拓扑SNN仅少1条连接，不能据此主张明显拓扑压缩，更不能称SNN优于ANN。','',
        '**PPO的“选中模型”是训练前初始化模型。** 两组都实际完成1,024,000步训练，但按预先规定的验证排序，没有后续检查点优于初始策略。因此不能把5/8描述为PPO学到的提升，也不能把这个配置的一次失败归结为PPO算法普遍无效。','',
        '## 同一输入、输出和物理条件','',
        '上部7 g、下部8 g；base_link提供姿态、角速度、加速度9维，另加测高误差和估计垂直速度，共11维。各方法输出α、β、电机转速的有符号残差，加到同一个固定PID。残差幅度0.035 rad、0.035 rad、100 rad/s；最终角度±0.3 rad，电机3115.224–3807.496 rad/s。内环500 Hz、策略100 Hz；SNN每次策略决策内部积分10个1 ms步骤。','',
        '严格成功需飞满、几何间隙>3 cm、稳定段倾角RMS<10°、高度RMS<15 cm及晚段高度波动和斜率达标。yaw与水平位置不闭环；当前结论不是纯IMU定高或水平定点悬停证明。','',
        '## 2201 已完成结果','']
    lines+=table(['方法','常规/扰动通过','总通过','倾角RMS','高度RMS','最小间隙','参数/连接'],[
        [LABELS[r['method']],f'{r["by_condition"]["nominal"]["passed"]}/4；{r["by_condition"]["stress"]["passed"]}/4',
         f'{round(r["result"]["best_validation"]["success_rate"]*8)}/8',f'{r["result"]["best_validation"]["tilt_rms_deg"]:.3f}°',
         f'{r["result"]["best_validation"]["height_rmse_m"]*100:.3f} cm',f'{r["result"]["best_validation"]["minimum_clearance_m"]*100:.2f} cm',
         f'{r["complexity"]["parameters"]} / {r["complexity"]["connections"]}'] for r in rows])
    lines+=['倾角与高度列是8个回合各自RMS的均值，最小间隙取全部回合的最小值。成功率属于反复用于选模的验证集；不计算跨训练种子置信区间。','',
        '![验证对比](figures/validation_comparison.png)','',
        '## 学习过程与收敛判断','',
        '联合SNN的最终选中策略来自第3代、51,200个分配训练步；继续训练到预算结束没有替换它。最后一代冠军本身只通过5/8，说明保留验证最佳模型很重要，也说明“最佳曲线不再变化”不能解释为种群已经稳健收敛。','']
    lines+=table(['方法','选中代/更新','选中时分配步','初始→选中→最后候选通过','实际训练步','训练耗时'],[
        [LABELS[r['method']],str(r['selection'].get('generation','update '+str(r['selection'].get('update')))),
         r['selection'].get('allocated_steps',r['selection']['training_steps']),
         ' → '.join(f'{round(v["success_rate"]*8)}/8' for v in [r['initial_validation'],r['result']['best_validation'],r['last_candidate_validation']]),
         r['result']['training_steps'],f'{r["result"]["wall_seconds"]/60:.1f} min'] for r in rows])
    lines += [f'2201合计实际训练 **{total_steps:,}** 步、反复验证 **{validation_steps:,}** 步。分配预算为7×1,024,000步；提前失败会使演化实际步数略少。停止依据是预先预算，不是证明全局收敛。','',
        '![选模过程](figures/selection_history.png)','',
        '## 结构、权重与时间常数','',
        '选中模型不一定是最后一代冠军。以下浏览器保存每个演化组40代×64个体、双亲、出生变异、权重、偏置与时间常数；结构动画展示每代冠军，不冒充单条祖先链。家谱保留双亲，热图沿编号较小父代的确定分支。','']
    lines+=table(['方法','选中隐藏节点','本次搜索最多隐藏节点','启用连接','有效连接','模型/演化'],[
        [LABELS[r['method']],r['complexity']['hidden'],r['max_explored_hidden'],r['complexity']['connections'],r['complexity']['effective_connections'],
         f'[模型](../training/{r["method"]}_seed2201/best_policy.json) · [浏览器](../training/{r["method"]}_seed2201/visuals/explorer.html)'] for r in evo])
    lines+=['![规模变化](figures/complexity.png)','',
        'ANN选中图的隐藏节点为3、106、183、281，初图为3–6；因此最终虽仍有4个隐藏节点，内部节点和连接确实经历了新增、替换与删除，并非只调整了相同图的权重。可通到输出的有效隐藏节点为3、281；另两支仍存于导出图中，所以报告保守计入全部19个参数，没有把未经部署简化的14参数图当作实测结果。','',
        '从有向图可达性看，ANN的动态学习残差只依赖pitch、gyro_x、gyro_z、accel_x和估计垂直速度；β残差只有一个常数偏置。固定PID仍动态控制全部执行器，不能据此说整个机器人的β固定不动，也不能说这些输入在其他扰动下已被证明足够。该结构提供了后续可验证的简化线索。','',
        '联合SNN选中模型仅4个时间常数偏离20/10 ms初值，其中3个属于连续读出、1个属于隐藏神经元；其选中隐藏节点仍为初始3–6。这不足以支撑“演化找到了丰富的脉冲时间尺度”的表述。','',
        '![ANN最终结构](../training/ann_neat_seed2201/visuals/selected_network.png)',
        '![联合SNN最终结构](../training/snn_neat_seed2201/visuals/selected_network.png)',
        '![联合SNN祖先时间常数](../training/snn_neat_seed2201/visuals/lineage_tau_m.png)',
        '![ANN权重变化](../training/ann_neat_seed2201/visuals/lineage_weights.png)','',
        'SNN的学习参数包括可变时间常数；固定τ不计为学习参数但仍需存储。ANN节点中的τ字段仅为公共导出格式，ANN运行时不用这些连续时间常数。CPU推理计时暂不作为本阶段结论，因为不同任务运行时的机器负载不相同，最终流程会空闲重测。','',
        '[档案核查](archive_verification.json)逐代验证了五个演化组的40代、12,800次个体评估、真实父代、基因变异记录、选择模型和可加载检查点。迁移用的两个已有验证场景另重放1,600步，未加入本表成绩或训练预算。','',
        '## 目前不能得出的结论与后续计划','',
        '尚未运行20秒独立测试、60秒持续测试、大角度恢复、单因素扰动和时间常数/记忆干预，因此不能确认长期hovering可用、SNN时间尺度有益、拓扑具有因果优势或某算法稳健优于其他算法。没有根据验证均值调整冻结参数，也没有提前读取保留测试集。','',
        '后续从 snn_fixed_tau_seed2202 开始完成剩余27次训练，保持原协议、种群64、16个环境/工作进程、同一预算、同一输入输出及选模规则。所有训练完成后再执行冻结的11,888回合测试和正式分析。若要改变PPO超参数或SNN机制，须另列新实验版本，不能覆盖当前比较集。','',
        '物理依据仍来自此前单翼测力辨识；10–15 Hz的实测约束不等于已经验证双翼约26.7 Hz自由飞行。地面判断为保守几何间隙，模型未启用真实接触动力学；SNN放电稀疏度也不是硬件能耗证据。','',
        '## 暂停与迁移文件','',
        '[exec.md](../../../exec.md)给出依赖、绝对网格路径处理、迁移哈希检查、已用验证场景回放一致性检查、显式续跑命令和最终收尾步骤。暂停在完整训练运行边界，保留全部导出模型、预算检查点、种群和日志；不要求恢复某个回合中途的神经状态。','',
        '本阶段报告不替代最终报告；完整比较结束并检查通过后才生成 report_final.md。','']
    text='\n'.join(lines).rstrip()+'\n';(output/'report.md').write_text(text,encoding='utf8')
    def rootlink(match):
        target=match.group(1)
        return match.group(0) if '://' in target or target.startswith('#') else ']('+posixpath.normpath((output/target).as_posix())+')'
    Path('report_stage_2201.md').write_text(re.sub(r'\]\(([^)]+)\)',rootlink,text),encoding='utf8')
    (root/'report.md').write_text('# 实验已按用户要求暂停\n\n[2201阶段性研究报告](interim_2201/report.md) · [迁移与续跑说明](../../exec.md)\n\n'+('8/35次训练完成，剩余27次。' if paused else '2201全部七组完成，2202联合SNN跑完后暂停。')+'正式独立测试尚未开始，最终报告尚未生成。\n',encoding='utf8')
    index=Path('report.md');previous=index.read_text();marker='<!-- topology-hover-index -->'
    assert marker in previous
    index.write_text('<!-- snn-stage-index -->\n# 本轮阶段结果与迁移\n\n[2201阶段性报告](report_stage_2201.md) · [exec.md：部署与续跑](exec.md)\n\n五个演化组在8秒选模验证中均为8/8，两个PPO组为5/8且选中初始化模型。ANN选中图为19个学习参数、12条连接；联合SNN暂未体现额外收益。单种子验证结果不代表独立泛化结论。按要求在snn_neat_seed2202完成后暂停，保留剩余27次训练。\n\n以下历史结果保留原文。\n\n---\n\n'+previous[previous.index(marker):],encoding='utf8')
    return dict(scope='seed2201_validation_only',training_steps=total_steps,validation_steps=validation_steps,methods=len(rows),paused=paused)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();print(json.dumps(generate(a.root)))
if __name__=='__main__':main()
