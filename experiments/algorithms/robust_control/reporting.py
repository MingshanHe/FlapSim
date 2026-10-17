"""Report actual completed work; never infer success from a finished process."""
from collections import Counter
from pathlib import Path
import shutil
import json
import numpy as np
from ..snn_pid_compare.common import read, write, sha
from ..snn_pid_compare.analysis import paired_effect


def generate(root, destination):
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    protocol = read(root/'protocol.json') if (root/'protocol.json').exists() else {}
    status = read(root/'status.json') if (root/'status.json').exists() else {}
    jobs = protocol.get('jobs', [])
    completed = [j for j in jobs if (root/'training'/j['name']/'results.json').exists()]
    evaluated = [p for p in (root/'evaluation').glob('*.json')] if (root/'evaluation').exists() else []
    lines = ['# PID 内环、神经网络残差与速度控制研究', '',
             f"当前阶段：`{status.get('stage', 'preparing')}`。训练完成 {len(completed)}/{len(jobs) or status.get('planned_training_runs', 65)}。", '',
             '**只有下面存在的记录才代表已经执行；队列、预算和测试计划不代表完成。**', '',
             '[协议与技术说明](technical_details.md) · [v2.1 失败原因](../snn_pid_compare_20261010/failure_analysis.md)', '',
             '所有训练沿用逐时间步 Python/PyTorch float64；不使用自定义神经网络加速内核。', '',
             '## 已完成训练的验证结果', '',
             '以下为选模验证，不能代替最终独立测试。零残差初始化也参与选择；若仍选中初始化，不声称学习改善。', '',
             '| 运行 | 实际训练步数 | 分配步数 | 验证成功/例数 | 选中阶段 |',
             '|---|---:|---:|---:|---:|']
    for j in completed:
        directory = root/'training'/j['name']
        result, selected = read(directory/'results.json'), read(directory/'selection.json')
        s = selected['summary']
        lines.append(f"| {j['name']} | {result['training_steps']} | {result['allocated_steps']} | "
                     f"{s['successes']}/{s['episodes']} | {selected.get('generation', selected.get('update'))} |")
    lines += ['', '## 独立测试', '']
    if (destination/'interim_20261012/report.md').exists():
        lines += ['另行提前执行的阶段独立评估见 [SNN＋NEAT 阶段报告](interim_20261012/report.md)。'
                  '它使用独立的新案例，下面仍仅记录原协议的最终测试。', '']
    groups = {}
    manifest = {}
    if not evaluated:
        lines += ['原协议的最终独立测试尚未运行。选模验证不能代替独立测试或判定某种训练方法获胜。']
    else:
        lines += ['下表逐训练种子汇总，测试场景在所有方法间配对。', '',
                  '| 控制器 | 工况 | 严格成功 | 倾角 RMS ° | 高度 RMS m | 速度误差 m/s |',
                  '|---|---|---:|---:|---:|---:|']
        for p in sorted(evaluated):
            result = read(p)
            families = {}
            for family in sorted({r['case']['test_family'] for r in result['episodes']}):
                rows = [r for r in result['episodes'] if r['case']['test_family'] == family]
                families[family] = dict(success_rate=float(np.mean([r['success'] for r in rows])),
                    successes=sum(r['success'] for r in rows), episodes=len(rows),
                    tilt_rms_deg=float(np.mean([r['tilt_rms_deg'] for r in rows])),
                    height_rmse_m=float(np.mean([r.get('reference_height_rmse_m',r['height_rmse_m']) for r in rows])),
                    velocity_rmse_m_s=float(np.mean([r.get('velocity_vector_rmse_m_s',0.) for r in rows])),
                    failures=dict(Counter(r['termination'] for r in rows if not r['success'])),
                    unmet_criteria=dict(Counter(k for r in rows for k,v in r.get('success_checks',{}).items() if not v)))
                f = families[family]
                lines.append(f"| {p.stem} | {family} | {f['successes']}/{f['episodes']} | "
                             f"{f['tilt_rms_deg']:.3f} | {f['height_rmse_m']:.4f} | {f['velocity_rmse_m_s']:.4f} |")
            groups[p.stem] = families
        write(root/'analysis.json', groups)
        contrasts = {}
        seeds = protocol.get('seeds', [])
        for task in ('hover', 'velocity'):
            for mode in ('residual','direct') if task == 'hover' else ('residual',):
                pairs = [(f'{task}_{mode}_{k}_neat', f'{task}_{mode}_{k}_ppo') for k in ('snn','ann')]
                pairs += [(f'{task}_{mode}_snn_{m}', f'{task}_{mode}_ann_{m}') for m in ('neat','ppo')]
                if task == 'hover' and mode == 'residual':
                    pairs += [('hover_residual_snn_neat', 'hover_residual_snn_evo')]
                for a, b in pairs:
                    if not all(f'{x}_seed{s}' in groups for x in (a,b) for s in seeds):
                        continue
                    effect = paired_effect(
                        [groups[f'{a}_seed{s}']['stress']['success_rate'] for s in seeds],
                        [groups[f'{b}_seed{s}']['stress']['success_rate'] for s in seeds])
                    difference = np.array(effect['seed_differences'])
                    draws = np.random.default_rng(9182).choice(difference,(10000,len(difference)),replace=True).mean(1)
                    effect['seed_bootstrap_95_interval'] = np.quantile(draws,[.025,.975]).tolist()
                    contrasts[a+'__minus__'+b] = effect
        extra_pairs=[]
        for k in ('snn','ann'):
            for m in ('neat','ppo'):
                extra_pairs.append((f'hover_residual_{k}_{m}',f'hover_direct_{k}_{m}',False))
                extra_pairs.append((f'hover_direct_{k}_{m}',f'hover_bc_{k}',True))
                for t in ('hover','velocity'):
                    extra_pairs.append((f'{t}_residual_{k}_{m}',f'{t}_pid',True))
        for a,b,fixed in extra_pairs:
            left=[f'{a}_seed{s}' for s in seeds]
            right=[b]*len(seeds) if fixed else [f'{b}_seed{s}' for s in seeds]
            if all(n in groups for n in left+right):
                effect=paired_effect([groups[n]['stress']['success_rate'] for n in left],
                                     [groups[n]['stress']['success_rate'] for n in right])
                difference=np.array(effect['seed_differences'])
                draws=np.random.default_rng(9182).choice(difference,(10000,len(difference)),replace=True).mean(1)
                effect['seed_bootstrap_95_interval']=np.quantile(draws,[.025,.975]).tolist()
                effect['reference_is_fixed_baseline']=fixed
                contrasts[a+'__minus__'+b]=effect
        write(root/'paired_contrasts.json', contrasts)
        lines += ['', '配对差异见 `paired_contrasts.json`；重采样单位是训练种子，不能用测试片段数代替独立训练数。', '',
                  '稳健门槛预先固定为：每种子的常规 ≥15/16、扰动 ≥14/16、60 s 常规 4/4、60 s 扰动 ≥3/4。',
                  '单因素和组合 OOD 单列；通过上述有限仿真门槛不构成实机安全或普遍稳健性保证。']
        verdicts = {}
        for name, g in groups.items():
            needed = ('nominal','stress','long_nominal','long_stress')
            if not all(f in g for f in needed):
                continue
            checks = dict(nominal=g['nominal']['success_rate']>=15/16,
                          stress=g['stress']['success_rate']>=14/16,
                          long_nominal=g['long_nominal']['success_rate']==1.,
                          long_stress=g['long_stress']['success_rate']>=.75)
            verdicts[name] = dict(checks=checks, robust_within_tested_envelope=all(checks.values()))
        write(root/'verdicts.json', verdicts)
        passing = [name for name,v in verdicts.items() if v['robust_within_tested_envelope']]
        lines[6:6] = [f"目前完成评估的控制器中，通过预定四项稳健门槛：{', '.join(passing) or '无'}。", '']
        lines += ['', '逐种子是否达标见 `verdicts.json`。直接组若未达标，结论限定为本网络、初始化与预算下不能可靠替代 PID，不能推出所有 ANN/SNN 都不可能。']
    lines += ['', '## 可复核记录', '',
              '完整训练日志、检查点、NEAT 每代个体及父代、场景和失败记录保留在运行目录。',
              f'运行目录：`{root.resolve()}`。代码快照与哈希见 `protocol.json`；本地保存，不推送 GitHub。', '']
    if completed:
        lines += ['![验证学习曲线](learning_curves.png)', '']
    (destination/'report.md').write_text('\n'.join(lines))
    (root/'report.md').write_text('\n'.join(lines))
    costs = {}
    for j in completed:
        from .networks import complexity
        p = root/'training'/j['name']
        spec, result, selected = read(p/'best_policy.json'),read(p/'results.json'),read(p/'selection.json')
        costs[j['name']] = dict(**complexity(spec), allocated_steps=result['allocated_steps'],
            consumed_steps=result['training_steps'],validation_steps=result['validation_steps'],
            training_wall_seconds=result['wall_seconds'],controller_has_pid=j['mode']=='residual',
            selected_initialization=selected.get('generation',selected.get('update'))==0,
            resume_count=result.get('resume_count',0))
        key=f"{j['task']}_{j['mode']}_{j['kind']}"+(f"_{j['seed']}" if j['mode']=='residual' else '')
        initial=read(protocol['initializations'][key]['path'])
        def structure(s):
            return (set(n['id'] for n in s['nodes']),
                    set((e['source'],e['target']) for e in s['edges'] if e['enabled']))
        costs[j['name']]['selected_topology_changed_from_initial']=structure(spec)!=structure(initial)
        ev=root/'evaluation'/f"{j['name']}.json"
        if ev.exists():
            episodes=read(ev)['episodes']
            steps=sum(r['steps'] for r in episodes)
            costs[j['name']]['controller_us_per_500hz_tick']=1e6*sum(r['complete_controller_seconds'] for r in episodes)/steps
            costs[j['name']]['actor_us_per_call']=1e6*sum(r['actor_inference_seconds'] for r in episodes)/max(1,sum(r['actor_calls'] for r in episodes))
    write(root/'costs.json',costs)
    for relative in ('protocol.json','status.json','analysis.json','paired_contrasts.json','verdicts.json','selection_frozen.json','costs.json'):
        if (root/relative).exists():
            shutil.copyfile(root/relative, destination/relative)
    # Compact scientific evidence only; large checkpoints/populations remain in runs/.
    selected_paths = [p for p in (root/'training').glob('*/*') if p.name in
                      ('results.json','selection.json','best_policy.json','history.jsonl','resume_events.jsonl')]
    selected_paths += evaluated
    for folder in ('development','development_filtered','direction_probes','velocity_outer_development',
                   'velocity_ramp_development','velocity_confirmation','smoke','smoke_final'):
        selected_paths += list((root/folder).glob('*.json'))
    for p in selected_paths:
        relative = p.relative_to(root)
        target = destination/'records'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
        manifest[str(relative)] = dict(sha256=sha(target), bytes=target.stat().st_size)
    write(destination/'record_manifest.json', manifest)
    if completed:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
        panels=[('hover','residual'),('hover','direct'),('velocity','residual')]
        styles={('snn','neat'):'tab:blue',('snn','ppo'):'tab:orange',
                ('ann','neat'):'tab:green',('ann','ppo'):'tab:red',('snn','evo'):'tab:purple'}
        for ax,(task,mode) in zip(axes,panels):
            seen=set()
            for j in completed:
                if (j['task'],j['mode']) != (task,mode):
                    continue
                path=root/'training'/j['name']/'history.jsonl'
                rows=[json.loads(line) for line in path.read_text().splitlines()]
                tag=j['kind']+' '+j['method']
                ax.plot([r['training_steps'] for r in rows],
                        [r['best_validation']['successes']/r['best_validation']['episodes'] for r in rows],
                        color=styles[(j['kind'],j['method'])],alpha=.6,label=tag if tag not in seen else None)
                seen.add(tag)
            ax.set(title=f'{task} / {mode}',xlabel='Consumed action steps',ylabel='Best validation success',ylim=(-.02,1.02))
            if seen:
                ax.legend()
        fig.savefig(destination/'learning_curves.png',dpi=160)
        plt.close(fig)
    return destination
