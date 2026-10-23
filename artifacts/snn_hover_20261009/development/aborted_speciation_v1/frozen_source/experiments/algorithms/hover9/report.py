"""Build the report and figures from frozen measured results, without reselection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):return json.loads(Path(path).read_text())


def figures(root,results,labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#247ca4','#6caac4','#cd713c','#e5ac66','#777777']
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    metrics=[('mean_elapsed','Time before termination (s)',1),('mean_tracking_rmse_m','Position RMSE until termination (cm)',100),
             ('mean_tilt_rmse_deg','Tilt RMS until termination (deg)',1),('mean_observer_final_error_m','Final inertial estimate error (cm)',100)]
    for ax,(metric,title,scale) in zip(axes.flat,metrics):
        x=np.arange(len(labels));w=.37
        ax.bar(x-w/2,[results['summary'][k]['nominal'][metric]*scale for k in labels],w,color=colors,label='Nominal')
        ax.bar(x+w/2,[results['summary'][k]['stress'][metric]*scale for k in labels],w,color=colors,alpha=.4,hatch='//',label='Stress')
        ax.set_xticks(x,[s.replace('_seed','\n').replace('shared_prior','Shared\nprior') for s in labels]);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0,0].axhline(8,ls=':',color='#555555',label='8 s target');axes[0,0].legend(fontsize=8)
    fig.suptitle('IMU history residual control: all frozen policies, 24 cases per group\nErrors use each flight duration; early crashes are not accurate hovering.',fontsize=13)
    fig.savefig(root/'comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,label in zip(axes.flat,labels[:4]):
        row=read(root/'final_test'/f'{label}_nominal.json')[0];t=np.array(row['trace'])
        for axis,color in enumerate(['#2864aa','#cf7042','#378257']):
            ax.plot(t[:,0],100*t[:,10+axis],color=color,label=f'True {"xyz"[axis]}')
            ax.plot(t[:,0],100*t[:,13+axis],color=color,ls='--',label=f'Estimated {"xyz"[axis]}')
        ax.set_title(f'{label}: {row["elapsed"]:.2f} s, {row["termination"]}');ax.set_xlabel('Simulation time (s)');ax.set_ylabel('Relative displacement (cm)');ax.grid(alpha=.2)
    axes[0,0].legend(ncol=2,fontsize=8);fig.suptitle('First predeclared nominal case: truth is used only for evaluation')
    fig.savefig(root/'position_estimates.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    for label,color in zip(labels[:4],colors):
        rows=read(root/label/'validation.json');best=-np.inf;xs=[];ys=[]
        for r in rows:
            best=max(best,r['summary']['mean_discounted_return']);xs.append(r['training_steps']);ys.append(best)
        ax.step(xs,ys,where='post',label=label,color=color)
    ax.set_xlabel('Training transitions (validation excluded)');ax.set_ylabel('Best validation discounted return');ax.legend();ax.grid(alpha=.2)
    ax.set_title('Common reward; NEAT/PPO environment budgets matched per training seed')
    fig.savefig(root/'learning_curves.png',dpi=170);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    protocol=read(root/'protocol.json');result=read(root/'final_test/results.json')
    labels=[f'{m}_seed{s}' for m in ('neat','ppo') for s in protocol['seeds']]+['shared_prior']
    assert result['status']=='completed'
    figures(root,result,labels)
    total_success=sum(result['summary'][label][g]['successes'] for label in labels[:4] for g in ('nominal','stress'))
    total_cases=sum(result['summary'][label][g]['episodes'] for label in labels[:4] for g in ('nominal','stress'))
    conclusion=(f'本轮四个策略在 {total_cases} 次正式常规/扰动测试中，满足 8 秒悬停标准的次数为 **{total_success}**。'
                + ('尚未训练出可用的定点悬停策略。' if total_success==0 else '部分测试达到标准，仍需按全部测试成功率判断稳健性。'))
    mean_neat=np.mean([result['summary'][label]['nominal']['mean_elapsed'] for label in labels[:2]])
    mean_ppo=np.mean([result['summary'][label]['nominal']['mean_elapsed'] for label in labels[2:4]])
    history_deltas=[result['summary'][label]['nominal']['mean_elapsed']-result['summary'][label]['zero_history']['mean_elapsed'] for label in labels[:2]]
    interpretation=(f'常规测试中，NEAT 两个种子的平均持续时间为 {mean_neat:.3f} s，PPO 为 {mean_ppo:.3f} s，共有零残差基线为 {result["summary"]["shared_prior"]["nominal"]["mean_elapsed"]:.3f} s。'
        f'保留历史估计相对于关闭历史，两个 NEAT 种子的平均持续时间变化分别是 {history_deltas[0]:+.3f} s 和 {history_deltas[1]:+.3f} s，效果不一致。'
        '因此这轮不能宣称 IMU 历史已经可靠改善了定点保持。NEAT 的短时位移估计末端误差约 2.5–3.3 cm，而倾角 RMS 仍约 38–39°；主要问题仍包括控制稳定性，而不只是缺少位移信息。')
    lines=['# 九维 IMU 历史：NEAT / PPO 残差控制实验报告','', '日期：2026-10-08。上部 7 g、下部 8 g，IMU 位于 base_link。','',conclusion,'',
        interpretation,'',
        '这轮完成了 PID 参数筛选、虚拟 IMU 历史估计器、两个随机种子的 NEAT 与 PPO 训练、冻结后的独立测试、历史特征消融和测试视频。PID 筛选没有得到稳定内环，因此正式比较改为两种算法共用上一轮的六维 NEAT 姿态控制器，学习附加的执行器修正。它是并联残差控制实验，不能写成“PID 内环已经成功”。','',
        '## 1. 输入、输出和状态估计','',
        '- 原始信息源为九通道：三姿态、三轴角速度、三轴加速度；没有位置、速度、关节角或扑翼相位传感器输入。',
        '- 固定的惯性估计器维护三维相对位移和速度。学习器实际接收 **15 个特征 = 九通道当前读数 + 六个历史估计量**，不是仅九个瞬时数值。NEAT 允许循环连接，PPO 为 64×64 的前馈网络；二者都有相同的积分历史。',
        '- 起点定义为复位时真实 base_link 位置，只用于奖励和评分。估计器的位移、速度均从零开始；常规测试将 base_link 原点初速度置零，保留扑翼机构运动相位及原有角动量初始化。飞行过程中不使用真值修正积分。',
        '- 加速度取机体系比力，在每个 5 µs 物理步用姿态转换到世界系、加重力、经 2 ms 低通后积分；控制器以 100 Hz 读取结果。机体系加速度也先滤波再降采样。姿态/角速度六通道保留旧策略的采样接口；只有 PID 滤波筛选使用另一个高频姿态/角速度前端。',
        '- 姿态旋转使用理想高频虚拟 IMU，扰动测试额外加入固定姿态误差和加速度偏差。积分前端使用降采样前的数据，并非仅对网络收到的 100 Hz 九个数做事后积分；网络姿态测量噪声与积分前端的固定姿态误差分别建模。没有实现真实 IMU 的量程、饱和、温漂、姿态融合或物理带宽标定。因此结果是乐观的仿真前端实验，不是硬件就绪的纯惯性定位。','',
        '归一化尺度依次为姿态 0.35 rad、角速度 10 rad/s、比力 9.81 m/s²、位移 0.5 m、速度 2 m/s；输入裁剪至 ±10。位移/速度历史来自固定的物理积分公式，这轮没有训练一个独立的神经惯性里程计。','',
        '共享姿态控制器固定为上一轮 NEAT seed836 的图，只使用它的 alpha/beta 输出，忽略其电机输出。若姿态控制器输出为 u，学习器输出为 a，则最终执行器目标为：','',
        '```text\nalpha = 0.3 × clip(u_alpha + 2 a_alpha, -1, 1) rad\nbeta  = 0.3 × clip(u_beta  + 2 a_beta,  -1, 1) rad\nmotor = 3461.360088 × (1 + 0.1 a_motor) rad/s\n```','',
        '最终目标仍经过原有 20 ms 平滑。alpha/beta 保留完整 ±0.3 rad（±17.19°），两个算法输出含义和执行器限制完全相同。零残差时电机处于约 15 gf 的原标定基准；这个标定值不保证飞行中任意倾角和运动状态下都有 15 gf 向上推力。','',
        '## 2. PID 筛选为什么没有作为最终内环','',
        '试验覆盖普通两轴姿态 PID 的正负增益、积分、姿态/角速度滤波和 trim；又增加了依据自身舵机指令历史估计机架反作用的补偿版本，以及先滤波再抽样的版本。筛选使用开发种子，最终参数用各自独立的验证种子选择。这里只说明这些候选方案在本轮未成功，不能推断所有 PID 都不可行。','',
        '| 方案 | 参数验证回合 | 平均持续时间 s | 最长持续时间 s |','|---|---:|---:|---:|']
    for filename,name in [('pid.json','普通 PID'),('compensated_pid.json','指令历史补偿 PID'),('filtered_pid.json','高频滤波 PID')]:
        data=read(root/filename);rows=data['validation']['episodes']
        lines.append(f'| {name} | {len(rows)} | {np.mean([r["elapsed"] for r in rows]):.3f} | {max(r["elapsed"] for r in rows):.2f} |')
    lines += ['', '这些验证集的种子不同，表格不是相同回合上的算法优劣检验。开发阶段合计 3,733 次参数候选评估（包含重复候选），每个候选两个回合；另有验证筛选。普通 PID 的开发控制步数为 213,777，补偿搜索和其他筛选见各自记录；高频筛选没有记录全部候选的精确控制步总和，不补造这个数字。','',
        '短阶跃检查表明，两轴正向 gimbal 指令在 0.3 s 的平均响应方向均对应机架相应姿态正向变化，但存在明显瞬态与耦合。另用四个独立开发种子检查释放时机架角速度清零：普通 PID 平均持续时间从 0.623 s 变为 0.593 s，没有明显改善；该诊断没有改变正式训练初始条件，见 release_diagnostic.json。单凭方向不能得出稳定增益。后续更适合先辨识周期平均的输入—姿态/推力动态及相位延迟，而不是继续盲目增大 population。','',
        '## 3. 训练与公平性边界','',
        'NEAT population=96，初始 12 个同源隐藏节点，允许拓扑与权重进化。隐藏激活候选为 tanh、bipolar sigmoid、softsign、sin、gauss、clamped；输出仅允许有符号的 tanh、bipolar sigmoid、softsign、clamped。当前 bipolar sigmoid 与 NEAT 的缩放 tanh 等价，不能当成两种独立表达能力。','',
        'PPO 使用相同 15 特征、同一个共享姿态控制器、动作组合、环境奖励和测试条件。网络为 64×64 tanh，学习率 3e-4、gamma=0.995、GAE=0.95、clip=0.2、每批 10 次更新。零残差基线参与两个算法的验证选择；初始 NEAT 群体还包含小扰动，PPO 的均值输出精确初始化为零并以随机动作探索。NEAT 候选适应度为 0.7×回合平均折扣回报 + 0.3×最低回合折扣回报；PPO 优化标准 PPO 目标。这一额外稳健性权重也是算法配置差异。','',
        '每个 NEAT 种子至少训练 262,144 个控制步并完成当前代；对应 PPO 再按实际 NEAT 步数向上取整到完整的 1,024 步 rollout。NEAT 以同代共同回合评价候选，前期每代一个回合，后期两个；PPO 用交替常规/轻微扰动的回合采样。因此是同环境与近似同预算的工程对照，采样顺序、优化方式和架构并不相同。两个训练种子不足以支持普遍的算法优劣结论。','',
        '| 运行 | 训练控制步 | 验证控制步 | 运行秒数 |','|---|---:|---:|---:|']
    training=validation=0
    for label in labels[:4]:
        data=read(root/label/'results.json');training+=data['training_steps'];validation+=data['validation_steps']
        lines.append(f'| {label} | {data["training_steps"]:,} | {data["validation_steps"]:,} | {data["wall_seconds"]:.1f} |')
    lines += ['',f'总训练 {training:,} 步，验证 {validation:,} 步；最终测试 {result["test_control_steps"]:,} 步。不含 PID 开发、标定、冒烟、重复回放和视频。','',
        '奖励沿用上一轮：存活、倾角、位置误差和速度的连续奖励，加角速度/动作变化惩罚及终止惩罚。位置/速度真值仅进入训练奖励、终止判断和评估，不进入 actor 特征。所有四个策略先按验证成功次数、平均折扣回报、持续时间选定并保存哈希，再开始最终测试。','',
        '![训练曲线](learning_curves.png)','',
        '## 4. 独立测试结果','',
        '每个策略使用 24 个常规回合和 24 个扰动回合。成功标准：完整 8 s、不触发失败终止、位置 RMSE <10 cm、倾角 RMS <10°。扰动组含原有初始姿态/角速度、噪声、一个控制步延迟和电机不确定性，另外加入每轴 ±0.03 m/s² 固定加速度偏差、±0.3° 姿态误差及 ±0.05 m/s 未知初速度。训练只见过较轻的偏差；详见 protocol.json。','',
        '| 策略 | 条件 | 成功 | 平均持续 s | 位置 RMS cm | 倾角 RMS ° | 估计末端误差 cm | 动作饱和 % |','|---|---|---:|---:|---:|---:|---:|---:|']
    for label in labels:
        for group in ('nominal','stress'):
            s=result['summary'][label][group]
            lines.append(f'| {label} | {group} | {s["successes"]}/{s["episodes"]} | {s["mean_elapsed"]:.3f} | {100*s["mean_tracking_rmse_m"]:.2f} | {s["mean_tilt_rmse_deg"]:.2f} | {100*s["mean_observer_final_error_m"]:.2f} | {100*s["mean_saturation_fraction"]:.1f} |')
    lines += ['', '**位置和倾角误差统计到各次终止时刻；更早摔落会缩短误差累计时间，所以不能凭较低 RMSE 判断悬停更好。** 动作饱和表示三个最终归一化指令中至少一个绝对值超过 0.98，不能单凭此指标判定具体是哪一个通道。shared_prior 是相同初始条件下零残差的共有控制基线。上一轮起点和初速度定义不同，不能直接拿上一轮平均值作这一轮的配对改善量。','',
        '![全部策略对照](comparison.png)','', '## 5. 历史是否被用到','',
        '在冻结策略后，将六个积分位移/速度特征置零，其余九通道、内部循环状态、物理环境和测试种子相同。这里是输入消融，不重新训练；下降或改善都说明策略对这些特征存在依赖，不能据此认定独立的神经网络学会了无漂移定位。','',
        '| 策略 | 有历史平均持续 s | 去历史平均持续 s | 配对平均差 s | 有历史/去历史成功数 |','|---|---:|---:|---:|---:|']
    for label in labels[:4]:
        full=result['summary'][label]['nominal'];zero=result['summary'][label]['zero_history']
        lines.append(f'| {label} | {full["mean_elapsed"]:.3f} | {zero["mean_elapsed"]:.3f} | {full["mean_elapsed"]-zero["mean_elapsed"]:+.3f} | {full["successes"]}/{zero["successes"]} |')
    lines += ['', '下面展示预声明的第一例常规测试，实线为用于评估的真实相对位移，虚线为控制器可见的估计值。没有根据效果选择轨迹。','',
        '![第一例的位置估计](position_estimates.png)','',
        '即使短时间估计准确，也不能消除惯性误差随时间累计的问题。例如静态条件下每轴 0.03 m/s² 恒定偏差在 8 s 后对应 0.96 m 位置误差；未知 0.05 m/s 初速度对应 0.4 m。这些是解析量级，不是本轮飞行测试结果。外部速度/位置参考仍是长期定点保持的重要可选路线。','',
        '## 6. 测试视频','',
        '每个视频均为第一例常规测试的完整轨迹，4 倍慢放，最后画面停留 1 个播放秒；叠字显示真实仿真时间和失败原因。','']
    for label in labels:lines.append(f'- [{label}](videos/{label}_nominal.mp4) · [对应数据](videos/{label}_nominal.json)')
    lines += ['', '## 7. 可复现性与资料来源','',
        '新增传感器前后动力学一致、观测不含绝对位置、历史复位、动作正负全范围与短时积分检查通过。与此前相关回归测试合计 25 项通过。PPO 导出 actor 在随机输入上与 SB3 确定性预测按 2e-6 容差核对；五个策略的第一例测试均进行逐点完全一致的重复回放。具体记录见 verification.json。','',
        '- [预声明协议](protocol.json) · [训练启动源码哈希](training_manifest.json) · [冻结策略哈希](final_test/frozen_selection.json) · [种子无重叠检查](seed_audit.json)',
        '- [全部最终结果](final_test/results.json) · [逐回合轨迹](final_test/) · [PID 普通](pid.json) / [补偿](compensated_pid.json) / [滤波](filtered_pid.json)',
        '- [完整前轮文献核对](../imu_history_20261008/report.md)：其中区分了真正的 NEAT、PPO、DDPG，以及仿真/实机和是否使用位置测量。','',
        '本轮借鉴的是已有反馈控制器加学习残差的思路：[Flappy Hummingbird](https://arxiv.org/abs/1902.09628) 使用 DDPG 残差及位置/姿态反馈；[Przychodzki 与 Niewiadomski 的 NEAT 四旋翼研究](https://czasopisma.uph.edu.pl/studiainformatica/article/download/3616/3407/8080) 使用 Betaflight PID；[Hsiao 等的 PPO 扑翼悬停](https://arxiv.org/html/2502.12355v1) 使用专家初始化及位置测量。这些工作不是“仅九维 IMU 就能长期无漂移定位”的证据。积分误差原理见 [Woodman 技术报告](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-696.html)。','',
        '### Material Passport','',
        '- 材料：既有 CAD 多体/气动仿真，7 g/8 g 质量变体、冻结六维策略及本轮生成的模拟轨迹；没有真实飞行数据。',
        '- 处理：固定 IMU 预积分、参数搜索、NEAT/PPO 残差训练和独立评估。报告表格/图从保存的 JSON 自动生成。',
        '- 用途：判断这组配置的仿真可行性和失效环节；不能据此声明实机可用、完整 PID 控制失败或 NEAT/PPO 的普遍优劣。',
        '- 追溯：源码、配置、种子、策略、结果和验证记录随本报告保存；依赖版本见 training_manifest.json。','',
        '### 复现','',
        '在仓库根目录、具有 mujoco / neat-python / stable-baselines3 / g++ 的环境中执行。先运行 `python -m experiments.algorithms.hover9.setup --root NEW_ROOT` 生成新的实验目录，并自动设置本地 mesh 路径。训练入口拒绝覆盖已有输出；完整说明见 [代码目录 README](../../experiments/algorithms/hover9/README.md)。下面重现学习对照，PID 开发和单独诊断另有各自记录。','',
        '```bash\n# 顺序示例；每个 seed 的 PPO 自动匹配 NEAT 实际训练步数\npython -m experiments.algorithms.hover9.train --root NEW_ROOT --method neat --seed 941 --workers 8\npython -m experiments.algorithms.hover9.train --root NEW_ROOT --method ppo --seed 941 --workers 8\n# 对 seed 942 重复以上两个命令，随后评估\npython -m experiments.algorithms.hover9.results --root NEW_ROOT\n# 已有结果在不同 checkout 路径下的便携视频复现\nMUJOCO_GL=egl python -m experiments.algorithms.hover9.replay \\\n  --root artifacts/hover9_pid_20261008 --label neat_seed941 --video /tmp/hover9_replay.mp4\npython -m pytest --import-mode=importlib experiments/algorithms/hover9/test_history.py -q\n```','',
        '新目录若位于不同 checkout，需像便携 replay 一样只重定位 XML 的 meshdir，不能改变质量、惯量和执行器参数。新一轮训练更应优先检查稳定姿态/足够升力的共同工作区，以及虚拟 IMU 前端的物理可信度；单纯加长 IMU 历史不能替代这两项。','']
    (root/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(conclusion)


if __name__=='__main__':main()
