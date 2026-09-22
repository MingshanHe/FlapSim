"""Reproduce the user's archived Flappy baseline and isolate motion differences.

Source dataset is read-only. Requires the native Purdue bridge from flappy_audit.
"""
from pathlib import Path
import json
import hashlib
import csv
import sys
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flappy_audit_experiment import OUT as AUDIT, BASE, original, config_for, fluid_for, mean_between
from flappy_audit_models import NativeWing
from flapsim import Simulator

DATA=Path('/mnt/c/Users/hemin/OneDrive/Desktop/DataSet')
SOURCE=DATA/'RA-L/force_experiment_analysis_20260915'
UNSTEADY=DATA/'FSI-Model/V_final/unsteady'
ARCHIVE=DATA/'force experiment/six_wing_unsteady_physics_validation'
OUT=AUDIT/'historical_trajectory'


def table(path):
    return np.genfromtxt(path,delimiter=',',names=True,encoding='utf-8',dtype=None)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dump(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def native_force(geometry,phi,phidot,theta,thetadot):
    # Common convention is the archived MATLAB/Python frame. Our IST joint
    # coordinates map to phi=-q_flap, theta=-q_pitch. Purdue input is -theta.
    wing=NativeWing(geometry,1.225)
    force=[]
    try:
        for p,pd,t,td in zip(phi,phidot,theta,thetadot):
            force.append(wing.evaluate(p,pd,-t,-td)[0]*np.sin(t))
    finally:
        wing.close()
    return np.array(force)


def write_csv(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def zero_crossings(phase,values):
    p=np.r_[phase,1.];v=np.r_[values,values[0]]
    ii=np.flatnonzero(v[:-1]*v[1:]<0)
    return np.array([p[i]-v[i]*(p[i+1]-p[i])/(v[i+1]-v[i]) for i in ii])


def rotation_timing(phase,trajectory):
    reversals=zero_crossings(phase,trajectory['phi_dot'])
    crossings=zero_crossings(phase,trajectory['theta'])
    rows=[]
    for crossing in crossings:
        delta=(crossing-reversals+.5)%1-.5
        i=int(np.argmin(abs(delta)))
        rows.append(dict(pitch_zero_phase=float(crossing),stroke_reversal_phase=float(reversals[i]),
                         lag_cycles=float(delta[i]),lag_ms_at_15Hz=float(delta[i]/15*1000)))
    return rows


def write_report(metrics,similarity,matrix,diagnostics,timing,force_rows):
    old,new=metrics
    def lift(stroke,pitch,geometry='current_CAD'):
        return next(r['mean_lift_mN'] for r in matrix if (r['geometry'],r['stroke'],r['pitch'])==(geometry,stroke,pitch))
    rows='\n'.join(f"|{r['hz']}|{r['archive_raw_cycle_mN']:.3f}|{r['window_change_mN']:+.3f}|{r['tare_change_mN']:+.3f}|{r['current_tared_mN']:.3f}|" for r in force_rows)
    report=f'''# 旧 Flappy baseline 与当前 IST 的运动轨迹核对

本报告更新此前气动力核对的归因。**先前把用户旧结果归因于 Flappy 默认 ±45° 的判断不适用于这组 baseline，已撤回。** 找到的旧代码直接向原版 C++ 输入规定角度和角速度，不读取机器人默认关节限位，也不求解被动羽化。实际输入来自 `FSI-Model/V7/data/rectangle_kapton_periodic_kinematics.csv` 的标称 20 Hz 周期，按频率缩放角速度/加速度后用于 10–15 Hz。

## 结论与量级

**运动轨迹是关键差异，扑翼和羽化两者都有影响，不能全部归因于气动力公式或 feathering 求解错误。** 原版 C++ 使用旧几何和原始 96 点轨迹，可以重现旧 Flappy 的全部六档结果；15 Hz 为 **60.143936 mN**，与用户图表一致。

|15 Hz，统一坐标定义|旧 baseline 规定轨迹|当前被动仿真轨迹|
|---|---|---|
|扑翼角范围|{old['flap_min_deg']:.2f}°～{old['flap_max_deg']:.2f}°|约 −79.33°～79.33°|
|羽化角范围，从竖直量起|{old['pitch_min_deg']:.2f}°～{old['pitch_max_deg']:.2f}°|约 −26.1°～26.1°|
|扑翼角速度 RMS|{old['stroke_velocity_rms_rad_s']:.2f} rad/s|{new['stroke_velocity_rms_rad_s']:.2f} rad/s|
|羽化过零相对本侧扑翼换向|提前 1.44、3.42 ms|滞后 1.28、1.25 ms|

角速度统一归一至 15 Hz。叠加图只用扑翼角确定一个整体相位平移，没有独立拟合羽化或升力相位。该对齐下羽化角 RMSE 为 {similarity['pitch_RMSE_deg']:.2f}°、相关系数 {similarity['pitch_correlation']:.3f}；这是两条轨迹的差异量，不是真实翼根角的测量误差。

![角度与角速度](trajectory_comparison.png)

当前羽化角约 {100*new['fraction_pitch_near_26p1']:.1f}% 的周期处于限位幅度的 98% 以上；旧轨迹约 {100*old['fraction_pitch_outside_26p1_with_0p2deg_tolerance']:.1f}% 的周期超出 ±26.1°（另留 0.2° 数值容差）。因此，两者不可能在同一严格翼根限位下被直接视为同一种关节运动。

## 分别替换扑翼与羽化轨迹

以下全部使用 **当前 CAD 几何 + 同一个 Purdue C++ 气动力核心 + 1.225 kg/m³ + 15 Hz**，逐点重放运动；只替换表中指定的运动变量。

|扑翼轨迹|羽化轨迹|平均升力 mN|
|---|---|---:|
|当前|当前|{lift('current','current'):.2f}|
|当前|旧 baseline|{lift('current','archive'):.2f}|
|旧 baseline|当前|{lift('archive','current'):.2f}|
|旧 baseline|旧 baseline|{lift('archive','archive'):.2f}|

单换羽化轨迹增加 {lift('current','archive')-lift('current','current'):.2f} mN；单换扑翼轨迹增加 {lift('archive','current')-lift('current','current'):.2f} mN。两项有交互作用，不能简单相加。扑翼角速度 RMS 增加约 {(old['stroke_velocity_rms_rad_s']/new['stroke_velocity_rms_rad_s']-1)*100:.1f}%，其平方增幅约 {((old['stroke_velocity_rms_rad_s']/new['stroke_velocity_rms_rad_s'])**2-1)*100:.1f}%，符合准定常力对角速度平方的敏感性。

![分别替换轨迹](trajectory_swap_lift.png)

42.70 mN 是在原模型保存的运动轨迹上重算、统一 15 Hz 并做周期对齐后的力；此前 43.28 mN 是 Purdue 核心与机构重新耦合后的结果，轨迹略有变化。这两种检查回答不同问题，不能混用。

旧 baseline 的几何也不同：理想 **80 × 28 mm**，展向从扑翼轴起算，没有当前 CAD 的膜根位置和 7 mm 安装偏移。二阶展向面积矩为 {diagnostics['archive_geometry_area_second_moment_m4']:.9g} m⁴，当前 CAD 为 {diagnostics['current_geometry_area_second_moment_m4']:.9g} m⁴。使用旧几何后，“旧扑翼 + 旧羽化”的 77.91 mN 变为 {lift('archive','archive','archive_80x28'):.2f} mN；原始 96 点直接执行为 60.143936 mN。60.09 与 60.14 的约 0.053 mN 差别来自为叠加/交叉替换使用的周期线性插值，而非重新拟合系数。

## 是否是当前模型把羽化算错了

已证实 **当前模型没有重现旧输入羽化波形**：幅度较小、翻转较晚、长期贴限位，旧输入还包含半行程内回摆。但旧 Flappy baseline 的羽化是外部输入，不能用它本身证明被动关节模型的正确性。

当前模型 `left_feathering_joint` 明确设置 `range=±0.455530934771 rad`，即用户确认的 ±26.1°；弹簧刚度 0、黏性阻尼 1e-8 N·m·s/rad、摩擦损失 0。它把整片翼视为刚体，用单个翼根转角代表所有翼段。硬限位决定幅度，惯性、气动力矩和碰撞决定翻转时机。若真实膜/翼架在根部限位之外仍有扭转，当前刚体角无法表示这种局部翼面运动；这是待验证的结构近似，不是已经找到的确定程序错误。

代码坐标检查表明，两边的 `theta=0` 都对应翼弦与升力轴平行，即竖直翼面。IST 采用 `phi=-q_flap, theta=-q_pitch` 映射到旧 baseline，不能再随意作 `90°-theta` 来让数值接近。旧 CSV 的三视角重建角是否代表翼根轴、整体拟合翼面，或含有安装/重建偏差，现有输入记录不足以独立确认；旧报告也注明视频运动向力实验迁移尚未验证。

为了检查“只要放开限位就能得到旧轨迹”这一可能性，另跑了 **±66° 诊断**：当前机构、质量和 Purdue 气动力都保持一致，频率校准到 {diagnostics['expanded_stop_actual_frequency_Hz']:.5f} Hz。实际转角达到约 ±66°，平均升力仅 **{diagnostics['expanded_stop_66deg_lift_mN']:.2f} mN**，羽化波形相对旧轨迹 RMSE 仍为 {diagnostics['expanded_stop_pitch_rmse_to_archive_deg']:.2f}°。因此仅改限位不能修复旧波形差异，正式 URDF 保留 ±26.1°。

![放宽限位后的波形](passive_pitch_stop_diagnostic.png)

把旧羽化波形整体缩小至 ±26.1°，保留其相位/形状并使用当前扑翼，升力反而为 {diagnostics['archive_pitch_scaled_to_26p1_current_stroke_mN']:.2f} mN。这也表明“旧相位一定更好”不成立；幅度、相位和停留时间必须一起比较。

**可执行的判断：** 先取得与本次力采集相对应、能区分翼根和翼面扭转的真实轨迹。若翼根确实保持 ±26.1°，就应保留关节限位，检查柔性翼面角及翻转动力学；若旧视频重建值被当成翼根角使用，则需要先核对它为何达到 ±65°、扑翼为何接近 ±100°。不能仅凭旧均值更接近实验就调关节或气动力系数。

## 同一批原始数据，但均值处理不同

矩形和曲边 Kapton 共 36 个原始力文件逐一 SHA-256 匹配，确实是同一批数据。两个 baseline 为 Sane–Dickinson 几何适配版和 Purdue Flappy 气动力核，另有 U1；本次没有改动它们的数据或源文件。

旧图用 1–4 s 内完整力信号周期的 **raw Fz**，未扣尾段零偏；本次用 0.5–4 s 平均再扣 5.5 s 至末尾前 0.1 s 的尾段均值。

|Hz|旧 raw 周期均值 mN|改平均窗口的影响 mN|减尾段零偏的影响 mN|当前均值 mN|
|---|---:|---:|---:|---:|
{rows}

15 Hz 的 55.522 → 59.315 mN 差别几乎全部来自减去 −3.869 mN 尾段均值，窗口影响只有 −0.076 mN。这是统计口径差异，不是原始数据不同。这里只分解差异；没有额外空载标定，不能据此宣称任一零偏处理绝对正确。

## 证据、复现与限制

- [用户提供的矩形 Kapton 图表数据](C:/Users/hemin/OneDrive/Desktop/DataSet/RA-L/force_experiment_analysis_20260915/extracted/Rect_K_trend.csv)
- [旧 baseline 生成代码](C:/Users/hemin/OneDrive/Desktop/DataSet/FSI-Model/V_final/unsteady/literature_baselines.py)，`compute()` 调用规定运动并编译 C++；没有机器人关节积分。
- [旧运动输入函数](C:/Users/hemin/OneDrive/Desktop/DataSet/FSI-Model/V_final/unsteady/unsteady_aerodynamics.py)，`kinematics()` 读 V7 周期文件并按 f/20 缩放。
- [实际输入给 C++ 的归档轨迹](C:/Users/hemin/OneDrive/Desktop/DataSet/FSI-Model/V_final/unsteady/baseline_sources/flappy_inputs.csv)
- [旧实验处理协议](C:/Users/hemin/OneDrive/Desktop/DataSet/force%20experiment/six_wing_unsteady_physics_validation/figure_manifest.json)
- 本目录 `baseline_reproduction.json`、`raw_file_identity.json`、`force_processing_comparison.csv`、`factorial_lift.csv`、`aligned_trajectories.csv`、`reversal_timing.json` 可核对全部量值。`source_manifest.json` 保存输入哈希。

```bash
python experiments/fixbase_curved_real/historical_trajectory_audit.py
```

脚本断言六档旧 baseline 重现、36 份力文件哈希一致、旧周期均值重算一致、新均值等于原始窗口减尾段零偏，并生成图表。源文件只读，没有启动会改写旧实验输出的原生成脚本。原始 baseline 是规定运动模型；本报告的轨迹交叉替换也是诊断，不是新被动动力学仿真。±66° 案例是单独重新耦合的动力学诊断，已保存频率校准和原始轨迹，不能当成实际硬件设置。

状态：ANALYZED。已明确旧结果与本次结果的运动、几何和统计差异；尚不能独立判定哪条视频轨迹等于本次实物翼根运动。
'''
    (OUT/'report.md').write_text(report,encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=True)
    files=[SOURCE/'extracted/Rect_K_trend.csv',UNSTEADY/'baseline_sources/flappy_inputs.csv',
           UNSTEADY/'literature_baselines.py',UNSTEADY/'unsteady_aerodynamics.py',
           UNSTEADY/'baseline_sources/run_flappy.cpp',DATA/'FSI-Model/V7/data/rectangle_kapton_periodic_kinematics.csv',
           ARCHIVE/'data/experiment_metadata.json',ARCHIVE/'figure_manifest.json',
           BASE/'source_manifest.json',AUDIT/'baseline_states.npz']
    manifest=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
    dump('source_manifest.json',manifest)
    raw=table(files[1]);old=raw[(raw['shape']=='rectangular')&(raw['frequency_Hz']==15)]
    geometry=[float(old[k][0]) for k in ('R','cbar','r33','r22','r11')]+[1.]+[float(old[k][0]) for k in ('zcp2','zcp1','zcp0','zrd')]
    trends=table(files[0]);reproduction=[]
    for row in trends:
        d=raw[(raw['shape']=='rectangular')&(raw['frequency_Hz']==row['frequency_Hz'])]
        f=native_force(geometry,d['phi'],d['phi_dot'],d['theta'],d['theta_dot'])
        value=float(f.mean()*1000)
        np.testing.assert_allclose(value,row['Flappy_baseline_mN'],rtol=1e-12,atol=1e-10)
        reproduction.append(dict(frequency_Hz=float(row['frequency_Hz']),archive_mN=float(row['Flappy_baseline_mN']),reproduced_mN=value))
    dump('baseline_reproduction.json',reproduction)
    normalized_equal=(UNSTEADY/'baseline_sources/purdue_flappy/flappy/envs/Wing.cpp').read_text()==(AUDIT/'reference/Wing.cpp').read_text()
    assert normalized_equal

    # Establish raw-record identity and account for means/tare separately.
    raw_metadata=read(ARCHIVE/'data/experiment_metadata.json')
    current_manifest={r['path']:r['sha256'] for r in read(BASE/'source_manifest.json')}
    current_runs=read(BASE/'real_runs.json')
    identities=[];force_rows=[]
    for m in raw_metadata:
        if m['shape'] not in ('rectangular','curve') or m['material']!='kapton':continue
        kind='rect' if m['shape']=='rectangular' else 'curve'
        for r in m['runs']:
            rel=r['force_file'].replace('\\','/')
            current_rel='/'.join(rel.split('/')[-3:])
            p=DATA/rel
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            identical=digest==r['sha256']==current_manifest[current_rel]
            assert identical
            identities.append(dict(path=rel,current_path=current_rel,sha256=digest,identical=identical))
        if kind!='rect':continue
        f=m['frequency_Hz'];group=[r for r in current_runs if r['kind']=='rect' and r['hz']==f]
        raw_new=[];old_cycle=[]
        for r in m['runs']:
            a=np.loadtxt(DATA/r['force_file'].replace('\\','/'),delimiter=',',skiprows=4,usecols=(0,3),encoding='latin1')
            mask=(a[:,0]>=.5)&(a[:,0]<4)
            raw_new.append(a[mask,1].mean()*1000)
            # Reproduce the archived full-cycle mean with its stored force clock.
            rate=r['force_clock_Hz'];edges=1+np.arange(int(np.floor(3*rate))+1)/rate
            cycle=np.mean([mean_between(a[:,0],a[:,1],x,y) for x,y in zip(edges[:-1],edges[1:])])*1000
            np.testing.assert_allclose(cycle,r['cycle_mean_N']*1000,atol=1e-10)
            old_cycle.append(cycle)
        archive_mean=float(trends[trends['frequency_Hz']==f]['experiment_mN'][0])
        np.testing.assert_allclose(np.mean(old_cycle),archive_mean,atol=1e-10)
        new_mean=float(np.mean([r['Fz_mean'] for r in group])*1000)
        tare=float(np.mean([r['Fz_baseline'] for r in group])*1000)
        np.testing.assert_allclose(np.mean(raw_new)-tare,new_mean,atol=1e-10)
        force_rows.append(dict(hz=f,archive_raw_cycle_mN=archive_mean,current_raw_window_mN=float(np.mean(raw_new)),
                               current_tail_tare_mN=tare,current_tared_mN=new_mean,
                               window_change_mN=float(np.mean(raw_new))-archive_mean,
                               tare_change_mN=-tare,total_difference_mN=new_mean-archive_mean))
    dump('raw_file_identity.json',identities);write_csv('force_processing_comparison.csv',force_rows)

    # Average current trajectories over complete time cycles and normalize only
    # their clock to 15 Hz. Fit ONE phase using flapping angle alone; no pitch fit.
    sim=Simulator(config_for(),fluid=fluid_for('flappy_cpp',config_for()))
    native=sim.fluid;new_geometry=native.geometry
    data=np.load(AUDIT/'baseline_states.npz');r=original();ph=np.arange(1536)/1536
    fields={'phi':-data['qpos'][:,native.stroke_q], 'theta':-data['qpos'][:,native.pitch_q],
            'phi_dot':-data['qvel'][:,native.stroke_v], 'theta_dot':-data['qvel'][:,native.pitch_v]}
    current={k:[] for k in fields}
    for a,b in zip(r['cycle_boundaries_s'][:-1],r['cycle_boundaries_s'][1:]):
        query=a+ph*(b-a)
        for k,v in fields.items():
            value=np.interp(query,data['time_s'],v)
            if k.endswith('dot'):value=value*15*(b-a)
            current[k].append(value)
    current={k:np.mean(v,axis=0) for k,v in current.items()}
    archive={k:np.interp(ph,old['phase'],old[k],period=1) for k in fields}
    def objective(shift):
        a=archive['phi'];b=np.interp((ph+shift)%1,ph,current['phi'],period=1)
        return np.mean(((a-a.mean())/np.std(a)-(b-b.mean())/np.std(b))**2)
    candidates=np.arange(720)/720;best=candidates[np.argmin([objective(x) for x in candidates])]
    opt=minimize_scalar(objective,bounds=(best-1/720,best+1/720),method='bounded')
    phase_shift=float(opt.x%1)
    current={k:np.interp((ph+phase_shift)%1,ph,v,period=1) for k,v in current.items()}
    metrics=[]
    steady=data['time_s']>.6
    for label,d in [('archive_prescribed',archive),('current_passive',current)]:
        metrics.append(dict(name=label,flap_min_deg=float(np.degrees(d['phi'].min())),flap_max_deg=float(np.degrees(d['phi'].max())),
                            pitch_min_deg=float(np.degrees(d['theta'].min())),pitch_max_deg=float(np.degrees(d['theta'].max())),
                            stroke_velocity_rms_rad_s=float(np.sqrt(np.mean(d['phi_dot']**2))),
                            pitch_velocity_rms_rad_s=float(np.sqrt(np.mean(d['theta_dot']**2))),
                            fraction_pitch_outside_26p1_with_0p2deg_tolerance=float(np.mean(abs(d['theta'])>np.radians(26.3))),
                            fraction_pitch_near_26p1=float(np.mean(abs(d['theta'])>=.98*np.radians(26.1)))))
    similarity=dict(phase_alignment='Single shared shift fitted to centered and normalized flapping angle only; no pitch/force fitting',
                    current_phase_shift_cycles=phase_shift,
                    pitch_RMSE_deg=float(np.sqrt(np.mean(np.degrees(archive['theta']-current['theta'])**2))),
                    flap_RMSE_deg=float(np.sqrt(np.mean(np.degrees(archive['phi']-current['phi'])**2))),
                    pitch_correlation=float(np.corrcoef(archive['theta'],current['theta'])[0,1]))
    timing={label:rotation_timing(ph,d) for label,d in [('archive',archive),('current',current)]}
    dump('reversal_timing.json',timing)
    # Full 2x2x2 factorial makes geometry/stroke/pitch interactions visible.
    matrix=[];forces={}
    for geometry_name,g in [('current_CAD',new_geometry),('archive_80x28',geometry)]:
        for stroke_name,s in [('current',current),('archive',archive)]:
            for pitch_name,p in [('current',current),('archive',archive)]:
                force=native_force(g,s['phi'],s['phi_dot'],p['theta'],p['theta_dot'])
                name=f'{geometry_name}__{stroke_name}__{pitch_name}'
                forces[name]=force
                matrix.append(dict(geometry=geometry_name,stroke=stroke_name,pitch=pitch_name,mean_lift_mN=float(force.mean()*1000)))
    # Reduce only pitch range in the archived shape, without inventing a 90-deg
    # complement convention or changing the turn phase.
    lim=np.radians(26.1)
    pitch_scale=lim/max(abs(archive['theta']))
    scaled=native_force(new_geometry,current['phi'],current['phi_dot'],archive['theta']*pitch_scale,archive['theta_dot']*pitch_scale)
    diagnostics=dict(archive_pitch_scaled_to_26p1_current_stroke_mN=float(scaled.mean()*1000),
                     archive_pitch_scale=pitch_scale,
                     current_geometry_area_second_moment_m4=native.area_r2,
                     archive_geometry_area_second_moment_m4=geometry[1]*geometry[0]**3*geometry[3],
                     archive_baseline_exact_96_sample_mN=reproduction[-1]['reproduced_mN'],
                     interpolation_note='Factorial uses 1536-point piecewise-linear periodic interpolation; exact archive reproduction uses original 96 samples.',
                     current_raw_pitch_minmax_deg=np.degrees([fields['theta'][steady].min(),fields['theta'][steady].max()]).tolist(),
                     current_angle_definition='theta=0 is a vertical wing: chord parallel to lift axis; same as old rotation(phi,theta) at theta=0',
                     native_source_matches_ignoring_CRLF=normalized_equal)
    # Same mechanics and native force, only allowing a much wider pitch range.
    expanded_result=read(AUDIT/'coupled/flappy_cpp_66deg/15hz/result.json')
    expanded=np.load(AUDIT/expanded_result['source'])
    expanded_curves={k:[] for k in ('phi','theta')}
    for a,b in zip(expanded_result['cycle_boundaries_s'][:-1],expanded_result['cycle_boundaries_s'][1:]):
        for k,column in [('phi','flap_rad'),('theta','pitch_rad')]:
            expanded_curves[k].append(-np.interp(a+ph*(b-a),expanded['time_s'],expanded[column]))
    expanded_curves={k:np.mean(v,axis=0) for k,v in expanded_curves.items()}
    def expanded_objective(shift):
        a=archive['phi'];b=np.interp((ph+shift)%1,ph,expanded_curves['phi'],period=1)
        return np.mean(((a-a.mean())/np.std(a)-(b-b.mean())/np.std(b))**2)
    best=candidates[np.argmin([expanded_objective(x) for x in candidates])]
    shift=minimize_scalar(expanded_objective,bounds=(best-1/720,best+1/720),method='bounded').x
    expanded_curves={k:np.interp((ph+shift)%1,ph,v,period=1) for k,v in expanded_curves.items()}
    diagnostics.update(expanded_stop_66deg_lift_mN=expanded_result['mean_lift_mN'],
                       expanded_stop_actual_frequency_Hz=expanded_result['measured_wing_frequency_hz'],
                       expanded_stop_pitch_minmax_deg=expanded_result['pitch_minmax_deg'],
                       expanded_stop_pitch_rmse_to_archive_deg=float(np.sqrt(np.mean(np.degrees(expanded_curves['theta']-archive['theta'])**2))))
    native.native.close()
    dump('trajectory_metrics.json',metrics);dump('alignment.json',similarity);dump('factorial_lift.json',matrix);dump('diagnostics.json',diagnostics)
    write_csv('factorial_lift.csv',matrix)
    traj=[dict(phase=float(phase),**{label+'_'+k:float(d[k][i]) for label,d in [('archive',archive),('current',current)] for k in fields}) for i,phase in enumerate(ph)]
    write_csv('aligned_trajectories.csv',traj)
    np.savez_compressed(OUT/'counterfactual_forces.npz',phase=ph,**forces)

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,constrained_layout=True)
    for ax,k,title,unit in zip(axes.flat,fields,['Flapping angle','Feathering angle from vertical','Flapping angular velocity','Feathering angular velocity'],['deg','deg','rad/s','rad/s']):
        for label,d,color in [('Archived prescribed trajectory',archive,'#7045a4'),('Current passive simulation',current,'#16877e')]:
            ax.plot(ph,d[k]*(180/np.pi if unit=='deg' else 1),label=label,color=color)
        ax.set(title=title,ylabel=unit,xlabel='Cycle phase');ax.grid(alpha=.18)
        if k=='theta':
            ax.axhline(26.1,color='gray',ls=':',lw=.9);ax.axhline(-26.1,color='gray',ls=':',lw=.9)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('15 Hz comparison | common coordinate convention | aligned using flapping angle only',fontsize=12)
    fig.savefig(OUT/'trajectory_comparison.png',dpi=180);fig.savefig(OUT/'trajectory_comparison.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,4.5),constrained_layout=True)
    labels=['Current stroke\nCurrent pitch','Current stroke\nArchived pitch','Archived stroke\nCurrent pitch','Archived stroke\nArchived pitch']
    vals=[r['mean_lift_mN'] for r in matrix if r['geometry']=='current_CAD']
    bars=ax.bar(range(4),vals,color=['#16877e','#73a9a3','#a593bf','#7045a4']);ax.bar_label(bars,fmt='%.2f',padding=3)
    ax.set_xticks(range(4),labels);ax.set(ylabel='Mean lift [mN]',title='Same current CAD geometry + actual Purdue C++ core',ylim=(0,max(vals)*1.19))
    fig.savefig(OUT/'trajectory_swap_lift.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.5),constrained_layout=True)
    for label,d,color in [('Archived prescribed trajectory',archive,'#7045a4'),('Passive, confirmed +/-26.1 deg',current,'#16877e'),('Passive, +/-66 deg (diagnostic)',expanded_curves,'#dd8b26')]:
        ax.plot(ph,np.degrees(d['theta']),label=label,color=color)
    ax.set(xlabel='Cycle phase, aligned by flapping angle only',ylabel='Feathering angle from vertical [deg]',title='A larger pitch stop does not reproduce the archived trajectory')
    ax.legend(fontsize=8);ax.grid(alpha=.18)
    fig.savefig(OUT/'passive_pitch_stop_diagnostic.png',dpi=180);plt.close(fig)
    write_report(metrics,similarity,matrix,diagnostics,timing,force_rows)
    print(json.dumps(dict(metrics=metrics,alignment=similarity,matrix=matrix,diagnostics=diagnostics,force_processing=force_rows,raw_files_identical=len(identities)),indent=2))


if __name__=='__main__':main()
