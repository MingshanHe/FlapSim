"""Generate the completed, bounded-budget NEAT/PPO comparison report."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from . import compare_env as common
from .train import write_json


def main(root):
    root=Path(root).resolve()
    read=lambda p:json.loads((root/p).read_text())
    protocol=read('protocol.json'); campaign=read('campaign_status.json'); tests=read('final_test/results.json')
    assert campaign['status']==tests['status']=='completed'
    labels=[f'{method}_seed{seed}' for method in ('neat','ppo') for seed in protocol['seeds']]
    runs={label:read(label+'/results.json') for label in labels}
    frozen=read('final_test/frozen_selection.json')
    for label in labels:
        assert hashlib.sha256((root/label/'best_policy.json').read_bytes()).hexdigest()==frozen['policy_sha256'][label]
        assert hashlib.sha256((root/'final_test'/(label+'_frozen.json')).read_bytes()).hexdigest()==frozen['policy_sha256'][label]
    repository=Path(__file__).resolve().parents[3]
    recorded_root=Path(read('calibration/trim.json')['model_path']).parent
    def local_source(name):
        path=Path(name)
        if path.is_relative_to(recorded_root):
            return root/path.relative_to(recorded_root)
        if not path.is_absolute():
            return repository/path
        for marker in ('experiments','flapsim','tests'):
            if marker in path.parts:
                return repository/Path(*path.parts[path.parts.index(marker):])
        raise ValueError(f'Unrecognized recorded source path: {name}')
    changed=[name for name,value in read('training_source_manifest.json').items()
             if hashlib.sha256(local_source(name).read_bytes()).hexdigest()!=value]
    assert not changed,changed
    selected={method:max([f'{method}_seed{s}' for s in protocol['seeds']],
                       key=lambda label:common.selection_key(runs[label]['best_validation']))
              for method in ('neat','ppo')}
    comparisons=[]
    for label in labels:
        pair=tests['summary'][label]
        comparisons.append(dict(label=label,nominal=pair['nominal'],stress=pair['stress'],
            usable=bool(pair['nominal']['successes']>=30 and pair['stress']['successes']>=28)))
    usable=[row['label'] for row in comparisons if row['usable']]
    verdict=('达到预注册可用仿真门槛的策略：'+', '.join(usable)) if usable else '两种方法都没有达到预注册的可用悬停门槛。'
    write_json(root/'comparison_summary.json',dict(status='completed',verdict=verdict,
        validation_selected_video_policies=selected,comparisons=comparisons,
        training_control_steps=sum(r['training_steps'] for r in runs.values()),
        validation_control_steps=sum(r['validation_steps'] for r in runs.values()),
        final_test_control_steps=tests['test_control_steps'],source_hash_check='passed'))

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    colors={'neat':'#1c7c85','ppo':'#bd7534'}
    for label in labels:
        method,seed=label.split('_seed')
        history=read(label+'/validation.json')
        x=[h['training_steps'] for h in history]
        y=[h['summary']['mean_elapsed'] for h in history]
        axes[0,0].plot(x,y,label=f'{method.upper()} seed {seed}',color=colors[method],ls='-' if seed==str(protocol['seeds'][0]) else '--')
    axes[0,0].set(title='Validation flight duration',xlabel='Training control steps',ylabel='Mean duration (s)')
    axes[0,0].legend(fontsize=8,frameon=False)
    x=np.arange(len(labels)); short=[s.replace('_seed','\n') for s in labels]
    for offset,group,color in [(-.18,'nominal','#467b9e'),(.18,'stress','#d3974b')]:
        axes[0,1].bar(x+offset,[tests['summary'][s][group]['successes']/32 for s in labels],.36,label=group,color=color)
        for i,label in enumerate(labels):
            count=tests['summary'][label][group]['successes']
            axes[0,1].text(i+offset,count/32+.025,f'{count}/32',ha='center',fontsize=8)
        axes[1,0].bar(x+offset,[tests['summary'][s][group]['mean_elapsed'] for s in labels],.36,label=group,color=color)
        axes[1,1].bar(x+offset,[tests['summary'][s][group]['mean_tilt_rmse_deg'] for s in labels],.36,label=group,color=color)
    axes[0,1].set(title='Held-out hover success',xticks=x,xticklabels=short,ylim=(0,1.18),ylabel='Success fraction')
    axes[0,1].legend(fontsize=8,frameon=False)
    axes[1,0].set(title='Held-out flight duration',xticks=x,xticklabels=short,ylabel='Mean duration (s)',ylim=(0,8.5))
    axes[1,1].set(title='Held-out tilt RMS',xticks=x,xticklabels=short,ylabel='Degrees')
    fig.suptitle('Upper 7 g + lower 8 g | same IMU 6 -> alpha, beta, motor controls',fontsize=14)
    fig.savefig(root/'comparison.png',dpi=170); plt.close(fig)

    fig,axes=plt.subplots(4,2,figsize=(12,10),layout='constrained')
    for column,method in enumerate(('neat','ppo')):
        label=selected[method]; episode=read('final_test/'+label+'_nominal.json')[0]
        data=np.asarray(episode['trace']); t=data[:,0]
        axes[0,column].plot(t,data[:,1:3]); axes[0,column].set(title=f'{label}: first declared nominal test',ylabel='Roll / pitch (deg)')
        axes[0,column].legend(['roll','pitch'],fontsize=8,frameon=False)
        axes[1,column].plot(t,data[:,4:6]*180/np.pi); axes[1,column].set(ylabel='Alpha / beta (deg)',ylim=(-18,18))
        axes[1,column].legend(['alpha','beta'],fontsize=8,frameon=False)
        axes[2,column].plot(t,data[:,8]); axes[2,column].axhline(.8,color='grey',ls='--'); axes[2,column].set(ylabel='Base height (m)')
        axes[3,column].plot(t,data[:,9]); axes[3,column].set(ylabel='Motor command (rad/s)',xlabel='Simulation time (s)')
    fig.suptitle('Validation-selected policies; case is not selected for good performance',fontsize=13)
    fig.savefig(root/'test_actions.png',dpi=160); plt.close(fig)

    lines=['## Material Passport','',
           '- Origin Skill: academic-research-suite / experiment-agent',
           '- Origin Mode: run','- Origin Date: 2026-10-08',
           '- Verification Status: ANALYZED; deterministic first test trajectories replayed exactly; full training not re-run',
           '- Version Label: neat_ppo_7g8g_v1','',
           '# 上 7 g＋下 8 g：六维 IMU 的 NEAT / PPO 对照实验','',
           '**实验执行已完成。'+verdict+'**','',
           '“训练程序完成”与“学到了可用策略”分别报告。以下所有策略先按验证集选择并冻结，再进入未见过的最终测试。没有使用最终测试结果调参或挑选演示种子。','',
           '## 实验条件','',
           '| 项目 | 两种方法共用的配置 |','|---|---|',
           '| 质量 | 完整上部 7 g；下部 8 g；整机 15 g |',
           '| 观测 | base_link 的 roll/pitch/yaw＋三轴角速度，只有 6 维 |',
           '| 归一化 | 姿态除以 0.35 rad；角速度除以 10 rad/s |',
           '| 输出 | α、β 各 ±0.3 rad（±17.19°）；电机基准上下 10% |',
           f"| 电机基准 | {protocol['motor_center_rad_s']:.3f} rad/s 指令；标定扑翼 {read('calibration/trim.json')['flapping_hz']:.3f} Hz |",
           f"| 标定平均升力 | 固定机架标定 {read('calibration/trim.json')['lift_gf']:.4f} gf；不是自由飞行稳定性的保证 |",
           '| 动力学 | 9.81 m/s² 重力、完整自由飞行、5 µs 物理步长、100 Hz 控制、20 ms 动作平滑 |',
           '| 初态 | 8 个带电扑翼相位随机释放；只在重置时补偿机构初始动量 |',
           '| 奖励 | 相同姿态、位置、速度、动作变化和失败奖励；位置/速度只用于评分，不进入网络 |',
           '| 课程 | 随交互预算逐步从 2 s、±2°发展到 4/8 s 与常规/扰动初态 |',
           '| 随机种子 | 每种算法两个独立训练种子：835、836 |',
           '| 最终测试 | 每个策略 32 个常规＋32 个扰动案例，同一组配对测试种子 |','',
           '常规测试姿态初扰动为各轴 ±5°；扰动测试为 ±10°，另有各轴 ±0.25 rad/s 角速度扰动、0.25° / 0.03 rad/s 的 IMU 噪声、一个控制步延迟和 ±3% 电机指令增益误差。重置时还共同叠加各轴标准差 1 cm 的位置噪声和 0.01 m/s 的平移速度噪声，再执行已说明的相位动量补偿。','',
           '最终单例成功要求：完整飞行 8 s、位置 RMSE＜10 cm、倾角 RMS＜10°。预注册可用仿真门槛：常规至少 30/32、扰动至少 28/32。','',
           '回合提前终止条件：base_link 高度＜0.1 m（ground，非接触检测）、位置误差＞1 m、倾角＞1.4 rad（约 80.2°）或数值异常。文中的飞行时长指到这些终止条件或 8 s 时限为止的仿真时间。','',
           f"整机质心相对 base_link 原点为 {np.round(protocol['properties']['total']['com_base_mm'], 4).tolist()} mm。上下部分按用户给定总质量设置，移动部件质量不变，通过调整固定结构质量实现 7 g / 8 g。",
           '', 'CAD 质心位置、固定结构惯量按质量缩放仍是模型假设。配平标定的严格力矩容差没有完全收敛，使用的是近似配平种子。初态的动量补偿抵消内部运动对整机动量的贡献，并不保证 base_link 在每个扑翼相位的瞬时速度为零，也不等于已求得周期悬停轨道。这里不声称完成了实物验证。','',
           '## 算法配置与公平性边界','',
           'NEAT：neat-python 2.0.0，种群 128，初始新增 8 个隐藏节点，演化连接、节点、响应和激活函数；保留九个 IMU 滤波记忆节点，时间常数 10–150 ms。初始群体包含近似配平、随机小网络与不同符号/增益的 IMU 反馈图，未使用旧质量模型的赢家或教师策略。','',
           '隐藏激活候选为 tanh、sigmoid、softsign、clamped、sin、gauss；输出限定 tanh、clamped、中心化 sigmoid、softsign，确保覆盖正负控制。此库的中心化 sigmoid 与 tanh 在当前缩放下数学等价，所以不会单独增加表达能力。','',
           'NEAT 基础变异概率：新增/删除节点 0.10/0.01，新增/删除连接 0.25/0.015，隐藏/输出激活 0.20/0.15，权重/偏置 0.80/0.50。权重和偏置扰动尺度分别为 0.18、0.05，训练预算超过 70% 后减半；记忆节点仍受稳定性修复约束。','',
           'PPO：Stable-Baselines3 2.6.0，actor/critic 均为 64×64 tanh MLP，actor 均值经 tanh 限制；学习率 3e−4、gamma 0.995、GAE 0.95、clip 0.2、10 epochs、熵系数 0.003、初始 log_std −1.8。初始输出偏置设在同一个近似配平角附近；没有行为克隆。','',
           '**这不是只替换优化器的比较：** NEAT 能演化循环记忆，PPO 基线是前馈网络；网络大小、结构先验和探索方式不同。两者的物理环境、当前输入、动作范围、奖励和测试条件相同，比较的是两套具体方法配置。不能据此概括所有低维问题中的 NEAT/PPO 优劣。','',
           '配置依据：[PPO 2.6 官方说明](https://stable-baselines3.readthedocs.io/en/v2.6.0/modules/ppo.html)、[NEAT 激活函数说明](https://neat-python.readthedocs.io/en/latest/activation.html)，并核对本机安装源码。','',
           '## 预算与运行结果','',
           'NEAT 完成跨过 524288 步的整代；对应 PPO 匹配该实际交互数并向上补齐至 2048 步 rollout。验证额外消耗单独列出，不并入训练预算。','',
           '| 方法/种子 | 实际训练控制步 | 验证控制步 | 运行秒数 | NEAT 代数 |','|---|---:|---:|---:|---:|']
    for label,run in runs.items():
        lines.append(f"| {label} | {run['training_steps']} | {run['validation_steps']} | {run['wall_seconds']:.1f} | {run.get('generations','—')} |")
    lines+=['','### 最终选定网络','','| 策略 | 选中时训练步数 | 导出网络结构 | 激活函数 |','|---|---:|---|---|']
    structures={}
    initial_selected=[]
    for label in labels:
        policy=read('final_test/'+label+'_frozen.json')
        chosen=max(read(label+'/validation.json'),key=lambda row:common.selection_key(row['summary']))
        selected_steps=chosen['training_steps']
        if selected_steps==0: initial_selected.append(label)
        if label.startswith('neat'):
            nodes=policy['nodes']; links=sum(len(node['links']) for node in nodes)
            activations=dict(sorted(Counter(node['activation'] for node in nodes).items()))
            memory_keys={int(key) for key in policy.get('memory_time_constants_s',{})}
            memory_count=sum(node['key'] in memory_keys for node in nodes)
            detail=f'{len(nodes)} 个执行节点（其中 EMA 记忆节点 {memory_count} 个），{links} 条连接'
            activation_text=', '.join(f'{name}: {count}' for name,count in activations.items())
            structures[label]=dict(executed_nodes=len(nodes),executed_ema_nodes=memory_count,links=links,activations=activations,
                                   memory_time_constants_s=policy.get('memory_time_constants_s',{}))
        else:
            dimensions=[6]+[len(layer['bias']) for layer in policy['layers']]
            parameters=sum(np.size(layer['weight'])+len(layer['bias']) for layer in policy['layers'])
            detail=' → '.join(map(str,dimensions))+f'，均值网络共 {parameters} 个参数'
            activation_text='隐藏层和输出均值均为 tanh'
            structures[label]=dict(actor_dimensions=dimensions,mean_network_parameters=parameters)
        structures[label]['selected_at_training_steps']=selected_steps
        lines.append(f'| {label} | {selected_steps} | {detail} | {activation_text} |')
    if initial_selected:
        lines+=['', '按验证集选择规则，'+', '.join(initial_selected)+' 最终保留的是训练前的初始策略；这些运行不能被描述为训练带来了策略改善。']
    write_json(root/'policy_structure.json',structures)
    lines+=['','## 冻结后的独立测试','','| 策略 | 组别 | 成功/32 | 平均飞行 s | 位置 RMSE m | 倾角 RMS ° | 动作饱和占比 |',
            '|---|---|---:|---:|---:|---:|---:|']
    for label in labels+['constant_trim']:
        for group,row in tests['summary'][label].items():
            lines.append(f"| {label} | {group} | {row['successes']}/32 | {row['mean_elapsed']:.3f} | {row['mean_tracking_rmse_m']:.3f} | {row['mean_tilt_rmse_deg']:.2f} | {row['mean_saturation_fraction']:.1%} |")
    lines+=['','### 失效原因','','| 策略 | 组别 | 回合终止统计 |','|---|---|---|']
    for label in labels+['constant_trim']:
        for group,row in tests['summary'][label].items():
            counts=', '.join(f'{name}: {count}' for name,count in row['termination_counts'].items())
            lines.append(f'| {label} | {group} | {counts} |')
    lines+=['',
            'RMSE 计算至该回合结束；提早坠落/翻转的回合持续时间更短，因此不能只看更小的 RMSE 判断更好。主要判断依据是完整 8 s 的成功率，同时报告持续时间和失效原因。两个训练种子仅支持描述性结论，没有做“算法普遍优越”的显著性声明。','',
            '六维 IMU 不直接观测高度、水平位置或平移速度。网络可能稳定姿态，但对持续升力偏差引起的高度漂移缺少直接反馈；本实验没有擅自增加观测维度，也没有偷偷按测试场景调整电机基准。','',
            '本轮没有分别消融质心位置、种群规模或激活函数，因此不能将训练结果归因于其中某一个因素。结论限定为这份 7 g / 8 g 模型、六维观测和已记录的训练预算。','',
            '后续诊断应先核对自由飞行中 α/β 到机体力矩的符号、幅值和时延，以及周期配平状态的稳定性；再在同样六维输入下比较明确使用滤波记忆的 NEAT 与循环 PPO。这些是尚未验证的后续实验建议，不是本轮结果。若目标包含长期定点与定高，还需要单独处理平移状态的估计或观测问题；增加高度信息本身也不能保证解决姿态失稳。','',
            '## 保存的策略与视频','',
            '每个运行目录保存最佳 JSON 策略；PPO 另保存 SB3 `.zip`，NEAT 另有种群检查点。两个演示策略按验证集选取，视频均使用最终测试预先声明的第一个常规/扰动种子，4 倍慢放。','']
    for method,label in selected.items():
        lines.append(f'- {method.upper()} 演示策略：`{label}`；[常规测试视频](videos/{label}_nominal.mp4)、[扰动测试视频](videos/{label}_stress.mp4)。')
        example=read('final_test/'+label+'_nominal.json')[0]
        trace=np.asarray(example['trace'])
        lines.append(f"  常规示例实际飞行 {example['elapsed']:.2f} s，倾角 RMS {example['tilt_rmse_deg']:.2f}°，位置 RMSE {100*example['tracking_rmse_m']:.2f} cm；电机指令中位数 {np.median(trace[:,9]):.1f} rad/s，终止原因为 `{example['termination']}`。")
    lines+=['','这些视频是预先指定案例的回放，不能替代上面的全部测试统计；局部姿态较稳也不代表维持了高度或位置。']
    lines+=['','## 完整性与复现','','- 正式训练源文件及模型哈希校验通过，见 `training_source_manifest.json` 和 `training_source_snapshot.zip`。',
            '- 19 项相关预检测试通过；两种算法的冒烟训练、相同种子环境轨迹、物理动作范围与六维观测检查通过。',
            '- 首次冒烟运行出现 `root` 与继承环境关节编号重名，已改为 `experiment_root` 后重新验证；失败日志保留。正式训练前冻结了修复后的源码。',
            '- PPO 导出的 NumPy actor 与 SB3 确定性输出按 2e−6 容差核对；所有最终策略第一例常规测试轨迹都进行完全一致的重复回放。',
            '- 训练种子范围与验证/最终测试种子范围分开；最终测试在全部四个策略冻结后才运行。',
            '- `protocol.json` 是正式训练前的配置；`campaign_status.json` 记录进程、退出码及时间。单次训练硬超时 4 小时。',
            '', '[实验协议](protocol.json) · [最终测试汇总](final_test/results.json) · [逐回合轨迹](final_test/) · [冻结源码归档](training_source_snapshot.zip) · [复现说明](../../experiments/algorithms/neat_imu/COMPARE_README.md)',
            '',f"总训练控制步：{sum(r['training_steps'] for r in runs.values())}；验证控制步：{sum(r['validation_steps'] for r in runs.values())}；最终测试控制步：{tests['test_control_steps']}。不含标定、预热、冒烟、重复回放和视频渲染。",
            '', '运行入口：','',
            '```bash',
            'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.algorithms.neat_imu.compare_campaign --root artifacts/neat_ppo_7g8g_20261008',
            'python -m experiments.algorithms.neat_imu.compare_results --root artifacts/neat_ppo_7g8g_20261008',
            '```',
            '', '以上入口防止覆盖已有结果；复现实验需复制配置与模型到新的输出目录。',
            '', '![对照结果](comparison.png)','', '![预声明测试的控制动作](test_actions.png)','']
    (root/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    write_json(root/'report_verification.json',dict(training_source_unchanged=True,frozen_policy_hashes_match=True,
        final_test_replay_checks=tests['replay_verification'],report_sha256=hashlib.sha256((root/'report.md').read_bytes()).hexdigest()))
    print(json.dumps(dict(verdict=verdict,video_policies=selected,report=str(root/'report.md')),ensure_ascii=False,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
