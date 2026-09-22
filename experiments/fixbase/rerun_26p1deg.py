"""Repeat the single-wing comparison using the corrected +/-26.1 degree stops."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import sys
import numpy as np

from single_wing_sweep import build_single_wing_model, single_config
from sweep_frequency import measure, saved_run, plot_sweep, FREQUENCIES
from experiment import plot_results
from run_15hz import cycle_statistics

FOLDER=Path(__file__).resolve().parent
REAL=FOLDER.parent/'fixbase_curved_real'
sys.path.insert(0,str(REAL))
from geometry_sensitivity import run as geometry_run
OUTPUT=FOLDER/'results_single_wing_26p1deg_10_15hz'
GEOMETRY=REAL/'analysis/geometry_sensitivity_26p1deg'
OLD=FOLDER/'results_single_wing_10_15hz'


def report(results,diagnostics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    config=single_config(OUTPUT/'model/IST_robot_single_left_wing.xml')
    old={r['target_wing_frequency_hz']:r for r in json.loads((OLD/'sweep.json').read_text())}
    measured={r['target_hz']:r for r in json.loads((REAL/'analysis/analysis.json').read_text())['summary']}
    rows=[]
    for r in results:
        f=r['target_wing_frequency_hz']
        data,summary=saved_run(FOLDER/r['source_folder'],config)
        assert summary['wing_bodies']==['left_wing']
        np.testing.assert_allclose(summary['pitch_limits_deg']['left_feathering_joint'],[-26.1,26.1],atol=1e-8)
        keep=data['time_s']>.6
        angles=np.degrees(data['left_feathering_joint_angle_rad'][keep])
        assert angles.min() < -25 and angles.max() > 25 and np.max(np.abs(angles)) < 26.2
        recomputed=cycle_statistics(data,target_hz=f)
        np.testing.assert_allclose(recomputed['mean_lift_N'],r['mean_lift_N'],atol=1e-12)
        assert abs(recomputed['measured_wing_frequency_hz']-f)<.01
        rows.append(dict(target_hz=f,actual_hz=r['measured_wing_frequency_hz'],
                         new_26p1deg_Fz_mN=r['mean_lift_mN'],old_50deg_Fz_mN=old[f]['mean_lift_mN'],
                         real_baseline_corrected_Fz_mN=measured[f]['real_mean_Fz_mN'],
                         error_vs_real_percent=100*(r['mean_lift_mN']/measured[f]['real_mean_Fz_mN']-1),
                         pitch_min_deg=float(angles.min()),pitch_max_deg=float(angles.max()),
                         complete_cycles=r['complete_cycles'],source_folder=r['source_folder']))
        figs=plot_results(data,OUTPUT/f'{f:02d}hz',pitch_limits_deg=(-26.1,26.1))
        for fig in figs: plt.close(fig)
    with (OUTPUT/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plot_sweep(results,OUTPUT,title='Single-wing mean lift | Feathering stops +/-26.1 deg',
               subtitle='Both flapping drives retained | Original aerodynamic parameters')
    fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
    f=[r['target_hz'] for r in rows]
    ax.plot(f,[r['old_50deg_Fz_mN'] for r in rows],'s--',color='#94a3b8',label='Previous run: +/-50 deg (incorrect stop setting)')
    ax.plot(f,[r['new_26p1deg_Fz_mN'] for r in rows],'o-',color='#2563eb',label='Rerun: +/-26.1 deg, original aerodynamic geometry')
    ax.errorbar(f,[r['real_baseline_corrected_Fz_mN'] for r in rows],
                yerr=[measured[h]['real_between_run_sd_mN'] for h in f],fmt='o-',color='#111827',
                capsize=4,label='Measured on-minus-off estimate, mean +/- run SD')
    ax.plot([r['target_wing_frequency_hz'] for r in diagnostics],
            [r['mean_lift_mN'] for r in diagnostics],'*',color='#16a34a',markersize=12,
            label='CAD-planform diagnostic, +/-26.1 deg (10/15 Hz only)')
    for r in rows:
        ax.annotate(f"{r['new_26p1deg_Fz_mN']:.2f}",(r['target_hz'],r['new_26p1deg_Fz_mN']),
                    xytext=(0,10),textcoords='offset points',ha='center',color='#1d4ed8')
    ax.set(xlabel='Nominal flapping frequency [Hz]',ylabel='Single-wing mean Fz [mN]',xticks=f,
           title='Corrected feathering limit: separate stop-angle and geometry effects')
    ax.set_ylim(0,max(max(r['old_50deg_Fz_mN'],r['new_26p1deg_Fz_mN']) for r in rows)*1.3)
    ax.legend(fontsize=8,loc='upper left');ax.grid(alpha=.2)
    fig.savefig(OUTPUT/'comparison_26p1deg.png',dpi=190)
    fig.savefig(OUTPUT/'comparison_26p1deg.svg');plt.close(fig)
    table='| Hz | 实际 Hz | ±26.1° Fz/mN | 历史 ±50° Fz/mN | 实测基线校正估计/mN |\n|---:|---:|---:|---:|---:|\n'
    table+='\n'.join(f"| {r['target_hz']} | {r['actual_hz']:.5f} | {r['new_26p1deg_Fz_mN']:.2f} | {r['old_50deg_Fz_mN']:.2f} | {r['real_baseline_corrected_Fz_mN']:.2f} |" for r in rows)
    text=f'''# 单翼羽化限位修正：±26.1°

用户明确纠正实际羽化限位为 ±26.1°。此前 ±50° 的记录是错误限位下的历史结果，不应继续用于实物验证。
MJCF 与 URDF 左右羽化关节均已改为 ±0.455530934771 rad；未增加羽化驱动或弹簧。

单翼装配仍为保留两侧扑翼传动、拆除右翼板。每组运行 2 s，步长 5 µs，采样 2 kHz。
前 0.6 s 排除，对完整输出轴周期积分；实际频率误差小于 0.01 Hz。
主扫描仅修正限位，保留原来的矩形条带气动参数，保证能单独观察角度设置的影响。

{table}

![角度修正对比](comparison_26p1deg.png)

## CAD 几何端点复查

为了复查上一次几何归因，又对 CAD 条带面积、作用点、旋转阻尼诊断模型应用了 ±26.1° 限位。
该诊断没有根据实测力拟合系数：
'''
    for r in diagnostics:
        text+=f"\n- {r['target_wing_frequency_hz']} Hz：实际 {r['measured_wing_frequency_hz']:.5f} Hz，平均 Fz {r['mean_lift_mN']:.3f} mN。"
    text+='''

上一轮“CAD 几何诊断的 15 Hz 与实测接近”是在错误的 ±50° 限位下得到的，不能继续据此认定模型已准确或几何是唯一主因。
矩形轮廓与真实 CAD 不匹配这个代码事实仍然成立，但必须在正确限位下联合验证几何和羽化响应。
真实数据仍沿用先前运行减末段基线的估计；静止清零、Fz 正向和末段状态尚需确认。
本次不调整原始实测数据，不调整经验气动系数来贴合结果。

## 复现与文件

运行 `python experiments/fixbase/rerun_26p1deg.py`。
新扫描保存于 `results_single_wing_26p1deg_10_15hz/`，旧 ±50° 记录保留。
每个频率子目录包含调速记录、CSV/NPZ、六维载荷图与关节运动图；`model/` 保存本次模型副本。
CAD 端点诊断保存于 `experiments/fixbase_curved_real/analysis/geometry_sensitivity_26p1deg/`。
'''
    (OUTPUT/'report.md').write_text(text,encoding='utf-8')
    write_notebook(text)
    print(json.dumps({'sweep':rows,'CAD_diagnostics':diagnostics},indent=2),flush=True)


def write_notebook(text):
    import nbformat as nbf
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    nb=nbf.v4.new_notebook()
    nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    nb.cells=[nbf.v4.new_markdown_cell(text.replace('![角度修正对比](comparison_26p1deg.png)','')),
      nbf.v4.new_code_cell('''from pathlib import Path
import sys,json
import numpy as np
from IPython.display import display,Image
FOLDER=Path.cwd().resolve()
sys.path.insert(0,str(FOLDER))
from rerun_26p1deg import OUTPUT
from single_wing_sweep import single_config
from sweep_frequency import saved_run
from run_15hz import cycle_statistics
config=single_config(OUTPUT/'model/IST_robot_single_left_wing.xml')
results=json.loads((OUTPUT/'sweep.json').read_text())
assert len(results)==6
for r in results:
    d,s=saved_run(FOLDER/r['source_folder'],config)
    np.testing.assert_allclose(s['pitch_limits_deg']['left_feathering_joint'],[-26.1,26.1],atol=1e-8)
    assert np.max(np.abs(np.degrees(d['left_feathering_joint_angle_rad'])))<26.2
    assert abs(r['measured_wing_frequency_hz']-r['target_wing_frequency_hz'])<.01
    np.testing.assert_allclose(cycle_statistics(d)['mean_lift_N'],r['mean_lift_N'],atol=1e-12)
print('通过：六组真实记录、±26.1° 限位、实际频率及完整周期升力复核。')
display(Image(filename=str(OUTPUT/'comparison_26p1deg.png')))
display(Image(filename=str(OUTPUT/'15hz/joint_motion.png')))
display(Image(filename=str(OUTPUT/'15hz/wrench_6d.png')))''')]
    km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    NotebookClient(nb,timeout=120,km=km,resources={'metadata':{'path':str(FOLDER)}}).execute()
    nbf.write(nb,FOLDER/'IST_robot_single_wing_26p1deg.ipynb')


def main():
    path=build_single_wing_model(OUTPUT/'model');config=single_config(path)
    results=[];diagnostics=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        jobs={pool.submit(measure,hz,sim_config=config,output=OUTPUT,load_coefficient=.68):'sweep'
              for hz in FREQUENCIES}
        jobs.update({pool.submit(geometry_run,hz,'full_planform',model_path=path,output=GEOMETRY):'geometry'
                     for hz in (10,15)})
        for job in as_completed(jobs):
            if jobs[job]=='sweep':
                results.append(job.result());results.sort(key=lambda r:r['target_wing_frequency_hz'])
                (OUTPUT/'sweep.json').write_text(json.dumps(results,indent=2))
            else:
                diagnostics.append(job.result());diagnostics.sort(key=lambda r:r['target_wing_frequency_hz'])
                (GEOMETRY/'summary.json').write_text(json.dumps(diagnostics,indent=2))
    with (OUTPUT/'sweep.csv').open('w',newline='') as stream:
        fields=['target_wing_frequency_hz','measured_wing_frequency_hz','mean_lift_N','mean_lift_mN',
                'complete_cycles','motor_command_rad_s','source_folder']
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(results)
    report(results,diagnostics)


if __name__=='__main__':
    if '--report-only' in sys.argv:
        report(json.loads((OUTPUT/'sweep.json').read_text()),
               json.loads((GEOMETRY/'summary.json').read_text()))
    else:
        main()
