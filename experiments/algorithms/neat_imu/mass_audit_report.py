"""Render the mass audit, its limitations, and the next training protocol."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .mass_audit_run import ROOT
from .train import write_json


def read(relative):
    return json.loads((ROOT / relative).read_text())


def main():
    models = read('models.json')
    coarse = read('results.json')
    fine = read('refinement/results.json')
    authority = read('refinement/authority_fine.json')
    lifts = read('refinement/lift_refined.json')
    trim = read('nominal_trim/trim.json')
    releases = read('trim_release/results.json')
    source = Path(models[0]['path'])
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_hash == 'c3fcec3057491e3c1d91545d19ef8a436b7536d30aec954a91ce7b02152f7751'
    episode_sets = [read('episodes.json'), read('refinement/episodes.json')]
    episode_sets.append([e for r in releases['results'] for e in r['episodes']])
    episodes = [e for group in episode_sets for e in group]
    assert len(episodes) == 224 and not any(e['success'] for e in episodes)
    assert all(e['case']['seconds'] == 8. for e in episodes)
    smoke = read('interface_smoke/run.json')
    assert smoke['physics_model_sha256'] == trim['model_sha256']
    assert smoke['environment']['model_path'] == trim['model_path']
    assert smoke['inputs'] == ['roll', 'pitch', 'yaw', 'gyro_x', 'gyro_y', 'gyro_z']

    protocol = dict(
        status='proposed_not_executed',
        objective='Find a usable IMU-only attitude controller and assess finite-duration hover separately',
        nominal_model=models[2]['path'], nominal_mass_g=16.,
        observation_names=smoke['inputs'], imu_body='base_link',
        action=dict(alpha_rad=[-.3,.3], beta_rad=[-.3,.3],
                    motor_center_rad_s=trim['command'][0], motor_fraction=[-.1,.1],
                    seed_angles_rad=trim['command'][1:],
                    note='Angle bias initializes the network; physical action ranges stay centered on zero'),
        prerequisites=[
            'Measure cycle-averaged free-flight base-link angular response near the new approximate trim',
            'Check release-phase sensitivity; fixed-base mean-force trim is not a free-flight periodic orbit',
            'Identify feedback signs on this model, not from the old-model policy or COM moment alone'],
        pilot=dict(arms=[dict(population=64, hidden=0, generations=16),
                         dict(population=128, hidden=0, generations=8),
                         dict(population=128, hidden=8, generations=8)],
                   individual_evaluations_per_arm=1024,
                   training_cases_per_individual=3, horizon_s=2.,
                   replicate_seeds=[824,825,826], initial_angle_degrees=2.,
                   note='Compare matched individual-evaluation budgets and also report actual physics/control steps; validation work is accounted separately'),
        network=dict(recurrent=True, additional_hidden_layer='6 -> 8 -> 3 with direct paths retained',
                     initial_hidden_activation='tanh', evolvable_hidden_activations=['tanh','clamped','sin','gauss'],
                     signed_output_activations=['tanh','clamped'],
                     protected_ema_time_constant_s=[.01,.15],
                     sensor_noise_and_delay='Progressively include IMU noise and one control-step delay'),
        search=dict(profile='exploratory', weight_mutate_rate=.8, weight_mutate_power=.18,
                    bias_mutate_rate=.5, bias_mutate_power=.05,
                    node_add_prob=.10, conn_add_prob=.25, survival_threshold=.35,
                    species_cap=8, species_elitism=2,
                    late_numeric_mutation_scale=.5),
        initialization=dict(seeds='New-model constant-trim and newly identified low-gain IMU feedback graphs; old-model NEAT only a comparison',
                            nominal='Powered approximate-trim release',
                            phase='Randomize settled release phase after local stabilization; do not cherry-pick the best release time'),
        proposed_fitness_changes=[
            'First emphasize tilt, low-frequency angular rate and action smoothness; retain real gravity and free-flight termination',
            'Then include vertical speed, height and horizontal drift in the reward only; policy remains six IMU values',
            'Report survival, tilt RMS, position RMSE and actuator saturation separately; survival time is not hovering'],
        full_training=dict(population_candidate=128, hidden_candidate=8,
                           generations_budget=60, decision='Only after the controlled pilot demonstrates 2 s local attitude stability',
                           horizons_s=[2,4,8], initial_degrees=[2,5,10]),
        robustness=dict(lower_mass_g=[8,9,10], base_link_com_z_uncertainty_mm=[-5,0,5],
                        uncertain_mass_test='Keep the nominal motor mapping fixed; never silently rebase it using hidden true mass',
                        known_configuration_calibration='Separate per-airframe calibration may use its measured total weight; label it explicitly'),
        acceptance=dict(validation_base=904810000, final_nominal_test_base=1043810000,
                        final_stress_test_base=1043820000, cases_per_test_group=32,
                        final_tests_status='Reserved, not evaluated',
                        attitude_gate='Complete 2 s with tilt RMS < 10 degrees in at least 10/12 validation cases',
                        hover_gate='After selection is frozen: 8 s with position RMSE < 0.1 m and tilt RMS < 10 degrees; report all nominal and disturbed successes',
                        usable_simulation_target='At least 30/32 nominal and 28/32 disturbed successes; no hardware claim'),
        observability='Six attitude/gyro inputs do not directly observe height or vertical drift; robust altitude hold under unknown sustained lift errors is not assured. Adding height/vertical speed would require a later input-scope decision.',
        implementation_status='Custom-model training, replay, hash recording and trim binding are implemented and smoke-tested. New staged fitness, phase/mass randomization and campaign are proposed, not implemented or run.')
    write_json(ROOT / 'proposed_training.json', protocol)

    # A scientific summary figure; no synthetic robot imagery.
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    figure, axes = plt.subplots(2,2,figsize=(12,8),layout='constrained')
    labels = ['Original\n15 g','7 + 8\n15 g','7 + 9\n16 g','7 + 10\n17 g']
    x = np.arange(4)
    upper = [m['properties']['upper']['mass_g'] for m in models]
    lower = [m['properties']['lower']['mass_g'] for m in models]
    ax=axes[0,0]
    ax.bar(x,lower,color='#4886ad',label='Lower assembly')
    ax.bar(x,upper,bottom=lower,color='#e3a24a',label='Complete upper mechanism')
    for i,(a,b) in enumerate(zip(lower,upper)):
        ax.text(i,a/2,f'{a:.2f}',ha='center',va='center',color='white')
        ax.text(i,a+b/2,f'{b:.2f}',ha='center',va='center')
    ax.set(xticks=x,xticklabels=labels,ylabel='Mass (g)',ylim=(0,21),title='The old 15 g total hid the wrong mass split')
    ax.legend(loc='upper left',fontsize=8,ncol=2,frameon=False)
    ax=axes[0,1]
    z=[m['properties']['total']['com_base_mm'][2] for m in models]
    ax.bar(x,z,color=['#a8afb7','#4886ad','#4886ad','#4886ad'])
    for i,v in enumerate(z):
        ax.text(i,v+.6 if v>=0 else -4.3,f'{v:+.2f}',ha='center',va='bottom' if v>=0 else 'top')
    ax.axhline(0,color='#343a40',ls='--',lw=1,label='Roll pivot: 0 mm')
    ax.axhline(-3.25,color='#a56536',ls=':',lw=1.5,label='Pitch pivot: -3.25 mm')
    ax.set(xticks=x,xticklabels=labels,ylabel='Whole-system COM z (mm)',ylim=(-9,23),title='CAD-based COM moves near the gimbal axes')
    ax.legend(loc='upper right',fontsize=8,frameon=False)
    ax=axes[1,0]
    matrices=[np.asarray(r['roll_pitch_authority_Nm_per_rad'])*1000 for r in authority]
    ax.bar(np.arange(2)-.17,[m[0,1] for m in matrices],width=.34,label='Roll moment / beta',color='#a56536')
    ax.bar(np.arange(2)+.17,[m[1,0] for m in matrices],width=.34,label='Pitch moment / alpha',color='#4886ad')
    ax.axhline(0,color='#343a40',lw=1)
    ax.set(xticks=[0,1],xticklabels=['Original 15 g','Upper 7 + lower 9 g'],ylabel='Aerodynamic moment slope (mN m / rad)',
           title='Fine-sampled roll-moment response reverses sign',ylim=(-.65,2.2))
    ax.legend(loc='upper left',fontsize=8,frameon=False)
    ax.text(.98,.05,'Fixed base, about whole-system COM\nSame motor command; 50 us sampling',transform=ax.transAxes,ha='right',fontsize=8)
    ax=axes[1,1]
    for offset,policy,label,color in [(-.17,'constant_neutral','Neutral angles','#a8afb7'),(.17,'old_neat_rebased','Old NEAT, motor rebased','#4886ad')]:
        vals=[next(r['summary']['mean_elapsed'] for r in fine['summaries'] if r['model']==m['label'] and r['policy']==policy) for m in models]
        ax.bar(x+offset,vals,width=.34,label=label,color=color)
    ax.set(xticks=x,xticklabels=labels,ylabel='Mean time to failure (s)',ylim=(0,1.7),title='Weight-matched release: every trial still fails')
    ax.legend(loc='upper right',fontsize=8,frameon=False)
    ax.text(.98,.79,'Each bar: 0 / 8 successful 8 s flights',transform=ax.transAxes,ha='right',fontsize=8)
    figure.suptitle('Mass / COM audit — simulated models, not measured hardware COM',fontsize=15)
    figure.savefig(ROOT / 'mass_audit.png',dpi=170)
    plt.close(figure)

    lines=[
        '# 扑翼机器人质量、质心检查与 NEAT 重训方案',
        '',
        '结论：现有模型存在明确的上下质量分配错误，显著改变了质心、惯量和气动力矩。它是需要先修正的重要因素，但本轮没有证明它是悬停失败的唯一原因，也没有得到可用悬停策略。',
        '',
        '用户确认：上部约 7 g 包含电机、传动和双翼；下部为 8–10 g。IMU 继续取 base_link，控制器仍为姿态 3 维＋角速度 3 维。',
        '',
        '## 实际质量分配',
        '',
        '原构建器 `flapsim/robot/IST_robot-V2/build_model.py` 保留原机构约 11.99 g 的惯性数据，再用下部质量补足整机 15 g。因此整机总质量正确，分配却与实物描述不符。',
        '',
        '| 模型 | 上部 g | 下部 g | 总质量 g | 总质心相对 roll 轴 mm | 相对 pitch 轴 mm |',
        '|---|---:|---:|---:|---:|---:|']
    for m in models:
        p=m['properties']; z=p['total']['com_base_mm'][2]
        lines.append(f"| {m['label']} | {p['upper']['mass_g']:.3f} | {p['lower']['mass_g']:.3f} | {p['total']['mass_g']:.1f} | {z:+.3f} | {z+3.25:+.3f} |")
    lines += [
        '',
        '正值表示在轴上方。roll 轴位于 base 原点；pitch 轴在其下方 3.25 mm。7＋9 g 的总质心虽然位于 roll 轴下方约 0.73 mm，仍位于 pitch 轴上方约 2.52 mm。不能把两个转轴混为一个。',
        '',
        '独立 XML 保留所有运动件（翼、齿轮、转子等）的质量与惯量，仅把 thrust_unit 固定部分从 10.913 g 调为 5.928 g，令完整上部为 7 g；保留下部 pitch_link 连接件 1 g，再把 base_link 设为 7/8/9 g。改变的刚体惯量按质量同比缩放，局部质心沿用原 CAD。',
        '',
        '**限制：这些是按现有几何推算的仿真质心，不是实物测量。** 7＋9 g 模型的 base_link 局部质心上下移动 5 mm，整机质心相对 roll 轴就从 −3.232 mm 变到 +1.768 mm。电池、控制板等在机架上的位置仍需实测或核对 CAD；仅知道上下重量不能唯一确定实物质心。',
        '',
        '## 控制力矩和电机基准',
        '',
        '固定 base_link，保持相同电机指令 3377.71 rad/s，以 α/β＝±0.05 rad 作中心差分。精细复核采用 5 µs 物理步长、50 µs 采样、0.8 s 等待和约 1.2 s 整周期平均：',
        '',
        '| 模型 | β→滚转力矩 N·m/rad | α→俯仰力矩 N·m/rad |',
        '|---|---:|---:|']
    for row in authority:
        matrix=row['roll_pitch_authority_Nm_per_rad']
        lines.append(f"| {row['model']} | {matrix[0][1]:+.7f} | {matrix[1][0]:+.7f} |")
    lines += [
        '',
        'β 对整机质心的平均滚转气动力矩变号，α 的俯仰力矩斜率约增至 3.6 倍。这里测的是整机质心处的气动力矩；它不是自由飞行中 base_link 角加速度的完整传递关系，不能只凭这张表机械地翻转反馈符号。',
        '',
        '下表为 α＝β＝0 时的竖直升力标定；转速指令不等于负载下实际电机转速，也不等于扑翼频率。',
        '',
        '| 模型 | 电机指令 rad/s | 实际扑翼 Hz | 平均升力 gf | 相对重力误差 |',
        '|---|---:|---:|---:|---:|']
    for row in lifts:
        lines.append(f"| {row['model']} | {row['command'][0]:.1f} | {row['flapping_hz']:.2f} | {row['lift_gf']:.3f} | {100*row['relative_vertical_lift_error']:+.3f}% |")
    lines += [
        '',
        '整机 16/17 g 时需要约 16/17 gf 的平均升力，原来的 15 gf 不足。把电机输出设计成“新标定基准上的正负修正”仍合理，但基准须跟已知实物重量一起重新标定。',
        '',
        '对 7＋9 g 另做了同时优化升力与滚转/俯仰平均力矩的数值配平：',
        f"- 电机指令 {trim['command'][0]:.2f} rad/s，实际电机 {trim['actual_motor_rad_s']:.2f} rad/s，扑翼 {trim['flapping_hz']:.3f} Hz。",
        f"- α＝{np.rad2deg(trim['command'][1]):.3f}°，β＝{np.rad2deg(trim['command'][2]):.3f}°；合力大小误差 {100*trim['normalized_residual'][0]:+.4f}%。",
        f"- 残余滚转/俯仰力矩范数 {np.linalg.norm(trim['torque_com_Nm'][:2]):.3g} N·m，高于预设 1e−7 N·m；`converged=false`。这是近似初始种子，不能称为已求得稳定悬停平衡。",
        '',
        '## 自由飞行检验',
        '',
        '初始姿态 ±5°、同一组 8 个种子、100 Hz 控制、真实重力、0.1 s 通电预热后自由释放；每例目标为 8 s 且位置 RMSE＜10 cm。旧 NEAT 只重设电机中心，没有重新训练。下表使用精细升力标定值：',
        '',
        '| 模型 | 零角度、恒定电机：平均失效时间 s | 旧 NEAT：平均失效时间 s | 各组成功数 |',
        '|---|---:|---:|---|']
    for m in models:
        group=[next(r['summary'] for r in fine['summaries'] if r['model']==m['label'] and r['policy']==policy) for policy in ('constant_neutral','old_neat_rebased')]
        lines.append(f"| {m['label']} | {group[0]['mean_elapsed']:.3f} | {group[1]['mean_elapsed']:.3f} | 0/8、0/8 |")
    lines += [
        '',
        '为区分总质量与分配，在相同 15 g 总质量、相同电机指令下，原模型与 7＋8 g 的恒定中立控制平均失效时间为 0.565 s 和 0.851 s，均因倾角过大结束。增加下部质量后沿用旧指令出现的延时失效不能解释为悬停改善，因为升力不足会引入混淆。',
        '',
        '对 7＋9 g 的近似力矩配平种子另测释放方式：',
        '',
        '| 释放方式 | 预热 s | 平均失效时间 s | 成功数 |',
        '|---|---:|---:|---:|']
    for row in releases['results']:
        lines.append(f"| {row['initialization']} | {row['warmup_seconds']:.1f} | {row['summary']['mean_elapsed']:.3f} | 0/8 |")
    lines += [
        '',
        '`trim` 使用带电的近似配平关节状态和合力对齐姿态；`momentum` 还只在初始时刻补偿内部机构动量，后续没有外力或重置。延长预热会同时改变机构阶段和释放相位，所以不能把 0.79 s 单独归因于更好的稳态。',
        '',
        '**128 次初查＋64 次精细升力对照＋32 次近似配平释放，共 224 次诊断飞行，悬停成功为 0。** 这些复用了诊断种子，包含重复对照，不是 224 个独立随机种子或最终泛化测试。时间变长不等于稳定悬停。',
        '',
        '## 重新提出的 NEAT 训练',
        '',
        '1. 以 7＋9 g 的独立模型为暂定名义模型，7＋8/7＋10 g 和机架质心位置作为模型不确定性。先辨识新模型在自由飞行中的小幅 α/β 响应和扑翼周期内机架运动，继续改进近似配平与释放状态。不要把旧模型策略作为主要种群种子。',
        '2. 保留 base_link 的六维 IMU。角度输出继续为 α＝0.3uα、β＝0.3uβ，u∈[−1,1]；初始网络的偏置产生约 0.79°/0.04° 配平角，物理范围仍完整覆盖 ±17.19°。电机为新基准约 3667 rad/s 上下 10% 修正。',
        '3. 做同预算对照：64 个体×16 代、128 个体×8 代、128 个体＋8 隐藏节点×8 代，每组 1024 次个体评估，各用 3 个进化随机种子。固定 2 s 早期场景和 ±2° 扰动；同时记录实际控制/物理步数。比较 population 与初始中间层的效果，不预设越大越好。',
        '4. 新增隐藏层从 tanh 开始并保留直接通路；输出限于能给正负值的 tanh/clamped，其他隐藏节点允许 tanh/clamped/sin/gauss。维持 10–150 ms 的可进化 IMU 滤波记忆，避免网络追逐每次扑翼尖峰。先验证反馈方向，再用小增益反馈种子与近似配平种子开始进化。',
        '5. 先以倾角、低频角速度和动作平滑训练局部姿态稳定，再逐步加入高度、竖直速度与水平漂移的奖励，延长为 2→4→8 s。奖励可读仿真状态，网络仍只读六维 IMU。达到局部姿态门槛后，再考虑 128 个体、8 隐藏节点、60 代预算；目前没有启动这轮长训练。',
        '6. 冻结策略后，以保留的 32 例常规＋32 例扰动测试评估；每例要求 8 s、位置 RMSE＜10 cm、倾角 RMS＜10°。暂定可用仿真门槛为常规至少 30/32、扰动至少 28/32，发布全部成功率、失效原因与测试视频。没有达到就继续查问题，不能只报告最佳种子。',
        '',
        '**质量鲁棒性与可观测性：** 对“未知质量误差”的测试必须固定名义电机映射，不可偷偷按每个测试模型的真实质量改基准。实物已称重、逐台重新标定是另一种已知配置评估，应单独报告。六维姿态/角速度不直接观测高度或纯竖直漂移，因此不能保证抵抗持续升力误差的定高能力；本轮不擅自增加高度输入。',
        '',
        '以上超参数是待验证的设计选择。NEAT 参数与激活函数定义参考 [官方配置文档](https://neat-python.readthedocs.io/en/latest/config_file.html) 和 [激活函数文档](https://neat-python.readthedocs.io/en/latest/activation.html)，实际运行按本机 neat-python 2.0.0 与已保存的源码；不使用文档中新增于 2.1 的原生种群自适应功能。',
        '',
        '## 已完成的实现与复现文件',
        '',
        '- `models.json`、`upper7_lower8.xml`、`upper7_lower9.xml`、`upper7_lower10.xml`：独立质量模型和逐刚体信息。',
        '- `refinement/authority_fine.json`、`refinement/lift_refined.json`、`nominal_trim/trim.json`：精细力矩复核、升力标定与未严格收敛的近似配平。',
        '- `episodes.json`、`refinement/episodes.json`、`trim_release/results.json`：所有诊断场景与结果；`proposed_training.json`：明确标记未执行的新方案。',
        '- `trim`、`robust_train` 新增 `--model-path`，训练记录模型哈希并校验配平模型，回放使用相同模型并核对哈希。新分阶段奖励、随机释放相位和质量随机化尚未实现。',
        '- `interface_smoke/`：8 个体×1 代、50 ms 的接口冒烟检查及回放，**只验证数据流和模型绑定，不能作为悬停成功证据**。',
        '- 30 个相关测试通过；共享机器人 XML 原文件哈希保持不变。旧质量模型上的 population/hidden-layer 长实验已停止。',
        '',
        f'原模型 SHA256：`{source_hash}`。',
        '',
        f"诊断控制步数：{sum(e['steps'] for e in episodes)}；标定物理步数：{coarse['fixed_physics_steps']+fine['fixed_physics_steps']+trim['total_physics_steps']}。不含环境预热、单元测试、接口冒烟和回放。",
        '',
        '![质量、质心、力矩和飞行结果](mass_audit.png)',
        '']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    source_files=list(Path(__file__).parent.glob('*.py'))+list(Path('flapsim').rglob('*.py'))+list(Path('flapsim').rglob('*.cpp'))
    source_files += list(Path('tests').glob('test_neat_mass_model.py'))+[source]
    source_files += list(ROOT.glob('upper7_lower*.xml'))
    manifest={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    write_json(ROOT / 'final_source_manifest.json',manifest)
    with zipfile.ZipFile(ROOT/'final_source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in source_files: archive.write(path,str(path))
    write_json(ROOT/'verification.json',dict(source_model_unchanged=True,source_model_sha256=source_hash,
        diagnostic_episodes=len(episodes),diagnostic_hover_successes=0,
        custom_model_training_smoke='passed; 50 ms only',custom_model_replay='passed; 50 ms only',
        related_pytest_passes=30,neat_python_version=importlib.metadata.version('neat-python'),
        new_long_training_started=False))
    print(json.dumps(dict(report=str(ROOT/'REPORT.md'),figure=str(ROOT/'mass_audit.png'),episodes=len(episodes)),indent=2))


if __name__ == '__main__':
    main()
