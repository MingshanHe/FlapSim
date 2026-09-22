"""Rectangular.STL + PET/Kapton: rerun the existing single-wing lift protocol.

python experiments/fixbase_curved_real/rect_material_experiment.py --prepare --run --report
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace, asdict
import argparse
import hashlib
import json
import platform
import sys
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[1]
ROBOT = ROOT/'flapsim/robot/IST_robot'
OUTPUT = FOLDER/'rect_material_validation'
sys.path[:0] = [str(ROOT), str(ROBOT), str(ROOT/'experiments/fixbase')]
from build_rectangular import build
from inspect_planform import planform
from single_wing_sweep import single_config
from rect_simulation import run_case, ProfileBladeElement
from analyze_comparison import write_csv
from flapsim import Simulator


def config_for(strips=12, timestep=5e-6):
    config = single_config(OUTPUT/'model/IST_robot_rectangular_single_left.xml')
    meta = json.loads((ROBOT/'rectangular_inertials.json').read_text())
    profile = np.load(OUTPUT/'rect_profile.npz')
    span = float(profile['span'])
    wing = replace(config.fluid.wings[0], span=span, chord=meta['membrane_area_m2']/span,
                   origin=(0., 0., 0.), chord_offset=.019)
    return replace(config, scene=replace(config.scene, timestep=timestep),
                   fluid=replace(config.fluid, strips=strips, wings=(wing,)))


def prepare():
    meta = build(retain_root=True)
    folder = OUTPUT/'model'
    folder.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(ROBOT/'urdf/IST_robot_rectangular.xml')
    root = tree.getroot()
    right = root.find(".//body[@name='right_flapping']")
    right.remove(right.find("body[@name='right_wing']"))
    for sensor in list(root.find('sensor')):
        if sensor.get('joint') == 'right_feathering_joint':
            root.find('sensor').remove(sensor)
    assets = root.find('asset')
    assets.remove(assets.find("mesh[@name='right_wing_mesh']"))
    root.find('compiler').set('meshdir', Path(os.path.relpath(ROBOT/'meshes', folder)).as_posix())
    root.set('model', 'IST_robot_rectangular_single_left')
    ET.indent(tree, space='  ')
    tree.write(folder/'IST_robot_rectangular_single_left.xml', encoding='utf-8', xml_declaration=True)
    # Actual membrane boundary; unlike the old 80 x 26 mm approximation, this
    # includes the small notch and CAD spanwise offset in the supplied STL.
    r, c, z, dr = planform(ROBOT/'meshes/rectangular_membrane.stl', bins=2400)
    np.savez(OUTPUT/'rect_profile.npz', radius=r, width=c, center=z, span=r[-1]+dr/2)
    assert abs(c.sum()*dr/meta['membrane_area_m2']-1) < .001
    sim = Simulator(config_for(), fluid=ProfileBladeElement(config_for().fluid, OUTPUT/'rect_profile.npz'))
    assert sim.model.body('right_flapping').id > 0
    assert mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, 'right_wing') == -1
    # Reuse exactly the prior force/encoder parsing, window and tare protocol.
    import rect_real_data
    previous = rect_real_data.OUTPUT
    try:
        rect_real_data.OUTPUT = OUTPUT
        rect_real_data.analyze_real()
    finally:
        rect_real_data.OUTPUT = previous
    meta.update(aerodynamic_profile='STL exposed membrane projected into blade strips',
                aerodynamic_area_m2=float(c.sum()*dr), experiment='Single left wing; both flapping drives retained',
                python=platform.python_version(), numpy=np.__version__, mujoco=mujoco.__version__)
    (OUTPUT/'model_assumptions.json').write_text(json.dumps(meta, indent=2))
    (OUTPUT/'simulation_config.json').write_text(json.dumps(asdict(config_for()), indent=2))
    print('Prepared rectangular model; mass mg:', meta['mass_kg']*1e6, flush=True)


def run_one(row, strips=12, label=None, timestep=5e-6):
    return run_case('rect', row['hz'], row['encoder_hz'], strips=strips, label=label,
                    timestep=timestep, config=config_for(strips, timestep),
                    profile_path=OUTPUT/'rect_profile.npz', output=OUTPUT)


def run():
    groups = [r for r in json.loads((OUTPUT/'real_grouped.json').read_text()) if r['kind']=='rect']
    results = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(run_one, row) for row in groups]
        for future in as_completed(jobs):
            results.append(future.result())
            results.sort(key=lambda r:r['nominal_hz'])
            (OUTPUT/'simulation_summary.json').write_text(json.dumps(results, indent=2))


def convergence():
    row = next(r for r in json.loads((OUTPUT/'real_grouped.json').read_text()) if r['kind']=='rect' and r['hz']==15)
    # Planned check: double strip count and halve time step, separately.
    results = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(run_one, row, 24, 'rect_strips24', 5e-6),
                pool.submit(run_one, row, 12, 'rect_dt2p5us', 2.5e-6)]
        for future in as_completed(jobs):
            results.append(future.result())
            (OUTPUT/'convergence_summary.json').write_text(json.dumps(results, indent=2))


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sims = json.loads((OUTPUT/'simulation_summary.json').read_text())
    real = {r['hz']:r for r in json.loads((OUTPUT/'real_grouped.json').read_text()) if r['kind']=='rect'}
    assert sorted(s['nominal_hz'] for s in sims) == list(range(10,16))
    old = {r['hz']:r for r in json.loads((FOLDER/'rect_validation/comparison_summary.json').read_text())['lift'] if r['kind']=='rect'}
    rows = []
    for s in sorted(sims, key=lambda s:s['nominal_hz']):
        hz = s['nominal_hz']; r = real[hz]
        actual, measured = s['measured_wing_frequency_hz'], r['Fz_mean']*1000
        assert abs(actual-r['encoder_hz']) < .01 and not any(s['numerical_warnings'])
        assert s['max_base_pose_error'] == 0
        rows.append(dict(hz=hz, encoder_hz=r['encoder_hz'], simulation_hz=actual,
                         real_lift_mN=measured, real_sd_mN=r['Fz_sd']*1000,
                         simulation_lift_mN=s['mean_lift_mN'], error_mN=s['mean_lift_mN']-measured,
                         relative_error_pct=100*(s['mean_lift_mN']/measured-1),
                         prior_simulation_lift_mN=old[hz]['sim_lift_mN'], repeats=r['n'],
                         complete_cycles=s['complete_cycles'], source=s['source']))
    write_csv(OUTPUT/'lift_comparison.csv', rows)
    (OUTPUT/'lift_comparison.json').write_text(json.dumps(rows, indent=2))
    fig, axes = plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    hz = [r['hz'] for r in rows]
    axes[0].errorbar(hz,[r['real_lift_mN'] for r in rows],yerr=[r['real_sd_mN'] for r in rows],fmt='o-',capsize=4,label='Real: mean +/- SD, n=3',color='#202b3b')
    axes[0].plot(hz,[r['simulation_lift_mN'] for r in rows],'s-',label='New STL + PET/Kapton',color='#1678a5')
    axes[0].plot(hz,[r['prior_simulation_lift_mN'] for r in rows],'--',label='Previous assumed-material model',color='#b4835e')
    axes[0].set(xlabel='Nominal frequency [Hz]',ylabel='Mean lift Fz [mN]',title='Rectangular wing: lift comparison')
    axes[0].legend(fontsize=8)
    axes[1].plot(hz,[r['relative_error_pct'] for r in rows],'o-',color='#1678a5')
    axes[1].axhline(0,color='gray',lw=1)
    axes[1].set(xlabel='Nominal frequency [Hz]',ylabel='(Simulation / real - 1) [%]',title='Relative lift error')
    for a in axes: a.grid(alpha=.2);a.set_xticks(hz)
    fig.savefig(OUTPUT/'lift_comparison.png',dpi=180)
    fig.savefig(OUTPUT/'lift_comparison.svg')
    plt.close(fig)
    fig, axes = plt.subplots(3,1,figsize=(11,8),layout='constrained')
    for s in (sims[0],sims[-1]):
        d=np.load(OUTPUT/s['source']); keep=d['time_s']>=1.6
        for ax,key,scale in zip(axes,('aero_Fz_N','pitch_rad','flap_rad'),(1000,180/np.pi,180/np.pi)):
            ax.plot(d['time_s'][keep],d[key][keep]*scale,label=f"{s['nominal_hz']} Hz",lw=.9)
    for ax,title in zip(axes,('Instantaneous lift [mN]','Passive feathering [deg]','Flapping angle [deg]')):
        ax.set(xlabel='Time [s]',ylabel=title);ax.grid(alpha=.2);ax.legend()
    fig.savefig(OUTPUT/'steady_waveforms.png',dpi=160);plt.close(fig)
    make_report(rows)
    print(json.dumps(rows,indent=2),flush=True)


def make_report(rows):
    meta=json.loads((OUTPUT/'model_assumptions.json').read_text())
    table=['|档位 Hz|编码器 Hz|仿真 Hz|实测均值 ± SD (mN)|本次仿真 (mN)|误差|旧仿真 (mN)|',
           '|---:|---:|---:|---:|---:|---:|---:|']
    table += [f"|{r['hz']}|{r['encoder_hz']:.4f}|{r['simulation_hz']:.4f}|{r['real_lift_mN']:.2f} ± {r['real_sd_mN']:.2f}|{r['simulation_lift_mN']:.2f}|{r['relative_error_pct']:+.1f}%|{r['prior_simulation_lift_mN']:.2f}|" for r in rows]
    numerical='未执行额外离散化检查。'
    if (OUTPUT/'convergence_summary.json').exists():
        checks=json.loads((OUTPUT/'convergence_summary.json').read_text())
        numerical='\n'.join(f"- 15 Hz：{s['simulation']['fluid']['strips']} 叶素、步长 {s['simulation']['scene']['timestep']*1e6:g} μs，实际 {s['measured_wing_frequency_hz']:.4f} Hz，升力 {s['mean_lift_mN']:.3f} mN（相对主实验 {100*(s['mean_lift_mN']/rows[-1]['simulation_lift_mN']-1):+.2f}%）。" for s in checks)
    text=f'''# 新矩形 STL：PET / Kapton 升力实验

## Material Passport
- 模式：run + 描述性 validate；Verification Status: ANALYZED。
- 输入：用户新增 Rectangular.STL，PEEK 传动、PET frame/翼根、10 μm Kapton；矩形翼 18 次实测记录。
- 状态：六档仿真实际完成，原始测量未改动；本报告不是气动模型已被验证的声明。

## 升力结果

{chr(10).join(table)}

![升力对比](lift_comparison.png)

## 模型与质量
- 新增双翼模型 `flapsim/robot/IST_robot/urdf/IST_robot_rectangular.urdf` 与配套 XML；实验使用从它派生的单左翼 XML，两侧传动均保留。
- base_link 和 PEEK 传动参数继承上次材料修正，羽化限位 ±26.1°、弹簧刚度 0；阻尼和电机参数保持原设置。
- 新 STL 为毫米单位，含薄膜实体；通过共同的 2 × 2 mm 根部特征对齐旧装配坐标，再保留用户确认的 PET 翼根连接件。安装对应关系未由实测坐标再次标定。
- frame **{meta['frame_mass_kg']*1e6:.3f} mg**，原翼根 **{meta['root_connector_mass_kg']*1e6:.3f} mg**，Kapton **{meta['membrane_mass_kg']*1e6:.3f} mg**，单翼总质量 **{meta['mass_kg']*1e6:.3f} mg**。
- 直接对新封闭 STL 计算体积、质心和惯性。先按 PET 积分整体，再把已包含的 10 μm 暴露膜层修正为 Kapton 密度；没有重复叠加膜质量。被翼骨覆盖的膜层无法从融合网格独立识别，归入剩余 PET 实体，胶层未建模。
- 膜面积 **{meta['membrane_area_m2']*1e4:.5f} cm²**，气动轮廓直接取新 STL 的膜面，包括根部偏移与小缺口，不再假设完整 80 × 26 mm 矩形。旧仿真列同时使用了旧几何/质量假设，差异不能全归因于材料。
- PET 1390、Kapton 1420、PEEK 1300 kg/m³，沿用已记录的厂家典型密度；采购牌号与真实总质量仍未测定。来源在机器人目录的 materials.json。

## 实验和比较口径
- 每档 2 s；步长 5 μs；12 叶素；记录 2 kHz；0.15 s 电机速度渐升。剔除前 0.6 s，以完整输出轴周期积分世界 +Z 气动力。
- 各档目标频率是同档三次实测编码器平均频率；电机指令只补偿负载造成的转速下降，最终频率误差均 <0.01 Hz。没有按实测升力调整气动系数或乘力倍率。
- 实测沿用旧流程：0.5–4.0 s 力均值，减去 5.5 s 至记录末尾前 0.1 s 的尾段均值。该尾段是静态基线的假设仍适用；若实际未停机，需重新定义去皮口径。
- 三次独立记录的样本 SD 用作误差棒；不是 95% CI，也不是模拟的置信区间。保留全部 18 次矩形翼记录。编码器和力记录无共同时间基准，本次只比周期均值，不逐点对齐。
- 保存六维气动力、动态支座反力、总线动量、扑翼/羽化角、完整周期统计及每次转速校准。主升力使用周期平均气动力；直接采样的支座反力受尖峰采样影响，不当作相同精度的均值。
- 固定基座位移误差为 0，所有最终运行有限数检查通过，MuJoCo 无数值警告。零数值警告不代表物理模型已验证。

## 数值检查
{numerical}

![稳态波形](steady_waveforms.png)

## 统计解释检查（11/11）
1. Simpson：逐频率比较，不混合翼型汇总回归。
2. 生态谬误：统计单位为三次记录，不把采样点当独立重复。
3. Berkson：未按匹配程度筛选实测。
4. Collider：未按残差控制或删除样本。
5. 基率：不涉及诊断概率。
6. 均值回归：未挑选极端频率或重复。
7. 幸存者：全部 18 次矩形实测与六档最终仿真均报告，校准过程保留。
8. 多处寻找：无显著性检验，报告所有指定频率。
9. 分析路径：去皮、窗口、材料、安装与离散化明示；沿用既有实验口径。
10. 相关与因果：升力残差不能单独证明膜柔性、阻尼或某个气动机制是原因。
11. 反向因果：不从力匹配反推唯一运动规律，不按实测力拟合参数。

## 复现
```bash
python experiments/fixbase_curved_real/rect_material_experiment.py --prepare --run --convergence --report
```
依赖 NumPy、MuJoCo、SciPy、Matplotlib、trimesh。`source_manifest.json` 保存原始数据哈希，`model_assumptions.json` 保存几何与材料输入哈希，`simulation_summary.json` 保存最终模型和气动轮廓哈希。历史 rect_validation 结果保持原样。
'''
    (OUTPUT/'report.md').write_text(text,encoding='utf-8')
    sources=[Path(__file__),FOLDER/'rect_simulation.py',FOLDER/'rect_real_data.py',FOLDER/'inspect_planform.py',
             ROBOT/'build_rectangular.py',ROOT/'flapsim/backends.py',ROOT/'flapsim/fluid/blade_element.py']
    (OUTPUT/'code_manifest.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prepare','run','convergence','report'):
        parser.add_argument('--'+name,action='store_true')
    args=parser.parse_args()
    if args.prepare: prepare()
    if args.run: run()
    if args.convergence: convergence()
    if args.report: report()
