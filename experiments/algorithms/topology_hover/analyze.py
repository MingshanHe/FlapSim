"""Seed-level summaries and paired bootstrap; no episode pseudoreplication."""
import argparse,gzip,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import write,summary
from .report import generate
from .networks import complexity
from .visualize import draw_graph


def topology_signature(spec):
    return (tuple(sorted(n['id'] for n in spec['nodes'])),
            tuple(sorted((e['source'],e['target']) for e in spec['edges'] if e['enabled'])))


def ci(values,seed=431):
    values=np.asarray(values,dtype=float)
    if len(values)<2:return [float(values.mean())]*2
    rng=np.random.default_rng(seed);means=values[rng.integers(len(values),size=(10000,len(values)))].mean(1)
    return np.quantile(means,[.025,.975]).tolist()


def transfer_figure(root,seed):
    paths=[root/'transfer'/f'{method}_seed{seed}'/'best_policy.json'
           for method in ['graph_ppo','random_graph_ppo']]
    if not all((path.parent/'results.json').exists() for path in paths):
        raise RuntimeError('Transfer diagrams require two completed runs')
    specs=[json.loads(path.read_text()) for path in paths]
    fig,axes=plt.subplots(1,2,figsize=(16,6),constrained_layout=True)
    for ax,spec,title in zip(axes,specs,['NEAT-selected structure + PPO','Matched random structure + PPO']):
        size=complexity(spec)
        draw_graph(ax,spec,f'{title}\nSeed {seed} | {size["parameters"]} parameters | {size["connections"]} enabled edges')
    fig.text(.5,.01,'Same node count and per-node input/recurrent indegree; independently trained weights. Blue: positive | Red: negative | Width: magnitude',ha='center',fontsize=9)
    output=root/'figures'/f'topology_transfer_seed{seed}.png';output.parent.mkdir(exist_ok=True)
    fig.savefig(output,dpi=170);plt.close(fig)
    return output


def analyze(root):
    manifest=json.loads((root/'heldout_manifest.json').read_text());index=manifest['policies']
    idle_path=root/'idle_inference_benchmark.json'
    if idle_path.exists():
        idle=json.loads(idle_path.read_text())
        for entry in index:
            entry['inference']=idle.get(entry['stage']+'/'+entry['name'],entry['inference'])
    rows=json.loads((root/'heldout_episodes.json').read_text());long=json.loads((root/'long_episodes.json').read_text())
    interventions=json.loads((root/'intervention_episodes.json').read_text())
    if json.loads((root/'evaluation_status.json').read_text())['stage']!='completed':raise RuntimeError('Evaluation incomplete')
    methods=sorted({r['method'] for r in rows});groups={};lines=[]
    for condition in ['nominal','stress','ood']:
        lines += [f'### 20 秒：{condition}','', '| 方法 | 训练种子数 | 严格成功率（种子 bootstrap 95% 区间） | 撑满 20 秒 | 倾角 RMS | 高度 RMS | 最低几何间隙 |','|---|---:|---:|---:|---:|---:|---:|']
        for method in methods:
            subset=[r for r in rows if r['method']==method and r['case']['condition']==condition]
            seeds=sorted({r['training_seed'] for r in subset})
            per=[dict(seed=s,**summary([r for r in subset if r['training_seed']==s])) for s in seeds]
            key=f'{method}/{condition}';groups[key]=dict(per_seed=per,summary=summary(subset),success_ci=ci([x['success_rate'] for x in per]))
            groups[key]['completed_horizon_rate']=float(np.mean([r['termination']=='time_limit' and r['elapsed']>=19.99 for r in subset]))
            groups[key]['termination_counts']={reason:sum(r['termination']==reason for r in subset) for reason in sorted({r['termination'] for r in subset})}
            groups[key]['failed_criteria_counts']={criterion:sum(not r['criterion_pass'][criterion] for r in subset) for criterion in subset[0]['criterion_pass']}
            s=groups[key]['summary'];lo,hi=groups[key]['success_ci']
            interval=f' [{lo:.0%}, {hi:.0%}]' if method!='pid' else '（固定基线）'
            lines.append(f'| {method} | {len(seeds) if method!="pid" else "—"} | {s["success_rate"]:.1%}{interval} | {groups[key]["completed_horizon_rate"]:.1%} | {s["tilt_rms_deg"]:.2f}° | {s["height_rmse_m"]*100:.2f} cm | {s["minimum_clearance_m"]*100:.2f} cm |')
        lines+=['']
    lines+=['表中倾角与高度误差为各片段 RMS 的算术平均，最低几何间隙取所有片段中的最小值。撑满时长表示未触发离地间隙、倾角、高度或数值异常终止；严格成功还要求每个测试片段的姿态与高度误差分别达标。因此平均倾角小于 10°，仍可能有部分片段不通过。失败策略的 RMS 只覆盖终止前的数据，不能因为它较早结束而把较小 RMS 解释为更好。',
        '', '相同测试场景在方法间配对；区间重采样的单位是独立训练种子，不是测试片段。样本量较小，区间仅描述当前训练与场景范围。OOD 同时改变下部质量、延迟、推力偏差与扰动幅度，不能把差异归因于某一个因素。','',
        '### 拓扑与算法的配对差异','', '| 对比（前者－后者） | 扰动成功率差 | 扰动倾角 RMS 差 |','|---|---:|---:|']
    contrasts=[]
    for a,b in [('neat','fixed_evo'),('neat','rnn_ppo'),('neat','ff_ppo'),('graph_ppo','random_graph_ppo')]:
        ga={x['seed']:x for x in groups[a+'/stress']['per_seed']};gb={x['seed']:x for x in groups[b+'/stress']['per_seed']};seeds=sorted(ga.keys()&gb.keys())
        ds=[ga[s]['success_rate']-gb[s]['success_rate'] for s in seeds];dt=[ga[s]['tilt_rms_deg']-gb[s]['tilt_rms_deg'] for s in seeds]
        sc=ci(ds);tc=ci(dt);record=dict(a=a,b=b,seeds=seeds,success_difference=float(np.mean(ds)),success_ci=sc,tilt_difference_deg=float(np.mean(dt)),tilt_ci_deg=tc);contrasts.append(record)
        lines.append(f'| {a} − {b} | {np.mean(ds)*100:+.1f} pp [{sc[0]*100:+.1f}, {sc[1]*100:+.1f}] | {np.mean(dt):+.2f}° [{tc[0]:+.2f}, {tc[1]:+.2f}] |')
    lines+=['','成功率差为正、倾角差为负才有利于前者。拓扑移植比较的是重新初始化并训练后的结构；原 NEAT 搜索开销不作为免费成本。多个探索性对比未做多重比较校正，不能只挑其中一个显著结果作结论。','']
    transfer_seeds=sorted(e['seed'] for e in index if e['method']=='graph_ppo')
    transfer_figures=[transfer_figure(root,seed) for seed in transfer_seeds]
    lines+=['左右图分别为 NEAT 选出的结构和匹配随机结构，均重新初始化后交给 PPO 训练。相同节点数、边数和逐节点入度约束下，连线对象仍然不同；图上画的是各自验证选中权重。下面依次链接全部训练种子，首图仅按种子编号展示。','',
        ' · '.join(f'[种子 {seed}]({path.relative_to(root).as_posix()})' for seed,path in zip(transfer_seeds,transfer_figures)),
        '',f'![拓扑移植与匹配随机结构]({transfer_figures[0].relative_to(root).as_posix()})','',
        '### NEAT 实际选中了哪一代','',
        '| 训练种子 | 验证选中的代次 | 个体出生代 | 隐藏节点（通向输出） | 启用连接 | 参数：初始均值→选中 | 启用结构是否不同于全部初始个体 |',
        '|---|---:|---:|---:|---:|---:|---|']
    evolution_selection=[]
    for entry in [e for e in index if e['method']=='neat']:
        folder=(root/entry['path']).parent;chosen=json.loads((folder/'selection.json').read_text())
        spec=json.loads((folder/'best_policy.json').read_text());size=complexity(spec)
        output_paths=set(spec['outputs'])
        for _ in spec['nodes']:
            output_paths.update(edge['source'] for edge in spec['edges'] if edge['enabled'] and edge['target'] in output_paths)
        hidden_on_output_paths=sum(node['id'] in output_paths and node['id'] not in spec['outputs'] for node in spec['nodes'])
        lineage=json.loads((folder/'visuals/selected_ancestry.json').read_text())
        with gzip.open(folder/'generation_0000.json.gz','rt') as stream:initial=json.load(stream)
        changed=all(topology_signature(spec)!=topology_signature(r['policy']) for r in initial['individuals'])
        record=dict(seed=entry['seed'],selection_generation=chosen['generation'],birth_generation=lineage[-1]['generation'],
                    differs_from_initial_population=changed,hidden=size['hidden'],hidden_on_output_paths=hidden_on_output_paths,connections=size['connections'],
                    initial_parameter_mean=float(np.mean([complexity(r['policy'])['parameters'] for r in initial['individuals']])),
                    selected_parameters=size['parameters'])
        evolution_selection.append(record)
        lines.append(f'| {entry["seed"]} | {record["selection_generation"]} | {record["birth_generation"]} | {size["hidden"]}（{hidden_on_output_paths}） | {size["connections"]} | {record["initial_parameter_mean"]:.1f} → {size["parameters"]} | {"是" if changed else "否"} |')
    lines+=['','括号内仅统计存在启用路径通向任一输出的隐藏节点；其余节点仍保存在基因和实际导出计算中，未在测试前额外剪枝。最后一列仅比较节点 ID 和启用连接集合，未做图同构判定，也不评价性能。如果选中初始个体，说明后续演化在本轮验证选择中没有改善该候选；这种结果不能作为“结构变异带来收益”的证据。','',
        '### 60 秒：验证集选出的代表策略','',
        '| 方法 | 场景 | 成功/总数 | 平均持续 | 倾角 RMS | 高度 RMS | 最低几何间隙 | 末端 XY 位移（均值） |','|---|---|---:|---:|---:|---:|---:|---:|']
    long_groups={}
    for method in methods:
        for condition in ['nominal','stress']:
            subset=[r for r in long if r['method']==method and r['case']['condition']==condition];s=summary(subset)
            s.update(xy_end_mean_m=float(np.mean([r['xy_offset_m'] for r in subset])),
                     xy_end_max_m=max(r['xy_offset_m'] for r in subset),
                     completed_horizon_rate=float(np.mean([r['criterion_pass']['full_horizon'] for r in subset])),
                     failed_criteria_counts={k:sum(not r['criterion_pass'][k] for r in subset) for k in subset[0]['criterion_pass']})
            long_groups[f'{method}/{condition}']=s
            lines.append(f'| {method} | {condition} | {sum(r["success"] for r in subset)}/{len(subset)} | {s["mean_elapsed"]:.2f} s | {s["tilt_rms_deg"]:.2f}° | {s["height_rmse_m"]*100:.2f} cm | {s["minimum_clearance_m"]*100:.2f} cm | {s["xy_end_mean_m"]:.2f} m |')
    lines+=['','这里每种方法只检查验证集选出的一个训练种子；代表策略的长时表现不能替代全部训练种子的稳健性统计。XY 数值是每段实际终止时相对初始位置的水平距离，再对片段取平均；它不是位置 RMS，也没有计入本轮训练奖励或通过判据。','',
        '### 推理成本与训练成本','', '| 方法 | Actor 参数（均值） | 有记忆连接（均值） | CPU 推理中位数（均值） | 单种子训练步数（均值） |','|---|---:|---:|---:|---:|']
    cost={}
    for method in methods:
        entries=[e for e in index if e['method']==method]
        if not entries:continue
        c={k:float(np.mean([e['inference'][k] for e in entries])) for k in ['parameters','recurrent_edges','numpy_inference_us_median']}
        c['training_steps']=float(np.mean([e['training_steps'] for e in entries]));cost[method]=c
        lines.append(f'| {method} | {c["parameters"]:.1f} | {c["recurrent_edges"]:.1f} | {c["numpy_inference_us_median"]:.2f} μs | {c["training_steps"]:,.0f} |')
    initial_size=float(np.mean([r['initial_parameter_mean'] for r in evolution_selection]));selected_size=float(np.mean([r['selected_parameters'] for r in evolution_selection]))
    lines+=['',f'NEAT 从初始种群平均 {initial_size:.1f} 个参数变为选中策略平均 {selected_size:.1f} 个，减少 {(1-selected_size/initial_size)*100:.1f}%。相对较大的固定结构基线，其参数差距还包含初始隐藏节点数不同的贡献，不能全部归因于后续演化。参数数按启用连接权重与全部非输入节点偏置计，不包括禁用基因、critic 或训练用探索尺度，也不等于当前密集 NumPy 实现实际分配的内存。','',
        '移植组表中步数只列权重训练；完整流程还需加对应 NEAT 搜索步数。随机结构对照保留从 NEAT 获得的规模/入度，因此同样计入该搜索来源。直接 PPO 的演示交互在运行表中单列。CPU 推理在训练进程退出后使用同一导出路径统一复测，仍不能替代目标微控制器测量。','',
        '### 结构干预','', '| 干预 | 成功率 | 倾角 RMS | 高度 RMS |','|---|---:|---:|---:|']
    ig={}
    for label in sorted({r['intervention'] for r in interventions}):
        s=summary([r for r in interventions if r['intervention']==label]);ig[label]=s
        lines.append(f'| {label} | {s["success_rate"]:.1%} | {s["tilt_rms_deg"]:.2f}° | {s["height_rmse_m"]*100:.2f} cm |')
    lines+=['','去除过去节点状态通路也会减少有效计算路径和容量，因此不等于纯粹的“记忆开关”。交叉轴消融只去掉明确标识的直接 roll→alpha、pitch→beta 等输入连接。与同数量随机删边比较，再结合重新训练的拓扑移植结果判断。','',
        '[全部 20 秒测试](heldout_episodes.json) · [60 秒测试](long_episodes.json) · [结构干预](intervention_episodes.json) · [移除高度反馈的诊断](sensor_ablation.json)','',
        '各测试记录保存六项通过判据及末段高度峰峰值，方便区分提前终止、姿态超限和高度不稳定；同一次失败可能同时违反多项判据。','']
    transfer=next(c for c in contrasts if c['a']=='graph_ppo')
    if transfer['success_ci'][0]>0 or (transfer['success_ci'][0]>=-.05 and transfer['tilt_ci_deg'][1]<0):
        conclusion='当前配对测试为 NEAT 流程选出的连接结构相对同规模随机结构的收益提供了初步证据。它尚不能单独区分初始结构先验、候选选择和结构变异各自的贡献，也不能推论 NEAT 普遍优于 PPO。'
    else:conclusion='当前测试尚未给出稳健的、独立于权重优化和网络规模的拓扑优势证据。不能仅凭某一段飞行视频或更小的参数量声称 NEAT 更好。'
    lines=['**结论范围：** '+conclusion,'']+lines
    fig,ax=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    colors=plt.get_cmap('tab10')
    for i,method in enumerate(methods):
        y=[groups[f'{method}/{c}']['summary']['success_rate']*100 for c in ['nominal','stress','ood']]
        ax[0].plot(range(3),y,'o-',label=method,color=colors(i))
        if method in cost and method!='direct_ppo':
            x=cost[method]['parameters'];v=groups[method+'/stress']['summary']['tilt_rms_deg']
            ax[1].scatter(x,v,color=colors(i),s=65,label=method)
    ax[0].set(xticks=range(3),xticklabels=['Nominal','Stress','Combined OOD'],ylabel='Success (%)',ylim=(-3,103));ax[0].legend(fontsize=8)
    ax[1].set(xscale='log',xlabel='Mean actor parameters',ylabel='Mean stress tilt RMS (deg)',title='Residual actors: performance / parameters')
    ax[1].legend(fontsize=8)
    for a in ax:a.grid(alpha=.25)
    fig.savefig(root/'method_comparison.png',dpi=170);plt.close(fig)
    lines += ['左图为全部方法的严格成功率；右图为共享固定 PID 内环的残差学习器，展示参数量与扰动倾角误差。','',
              '![方法性能与参数比较](method_comparison.png)','']
    result=dict(conclusion=conclusion,groups=groups,paired_contrasts=contrasts,evolution_selection=evolution_selection,long_groups=long_groups,cost=cost,interventions=ig,markdown='\n'.join(lines))
    write(root/'final_results.json',result);generate(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();analyze(a.root)
