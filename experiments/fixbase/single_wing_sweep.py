"""Single left wing; right wing plate removed, both flapping drives retained."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from experiment import SimulationConfig, ExperimentConfig, run_experiment, plot_results
from sweep_frequency import current_simulation, measure, plot_sweep, saved_run, FREQUENCIES
from run_15hz import cycle_statistics

FOLDER = Path(__file__).resolve().parent
OUTPUT = FOLDER / 'results_single_wing_10_15hz'


def build_single_wing_model(folder=OUTPUT / 'model'):
    """Derive a separate MJCF; preserve both flapping bodies and their coupling."""
    folder.mkdir(parents=True, exist_ok=True)
    source = SimulationConfig().path
    tree = ET.parse(source)
    root = tree.getroot()
    right_drive = root.find(".//body[@name='right_flapping']")
    right_wing = right_drive.find("body[@name='right_wing']")
    removed_mass = float(right_wing.find('inertial').get('mass'))
    right_drive.remove(right_wing)
    # Only the removed passive pitch joint's sensors become invalid.
    sensors = root.find('sensor')
    for sensor in list(sensors):
        if sensor.get('joint') == 'right_feathering_joint':
            sensors.remove(sensor)
    assets = root.find('asset')
    assets.remove(assets.find("mesh[@name='right_wing_mesh']"))
    compiler = root.find('compiler')
    meshdir = (source.parent / compiler.get('meshdir')).resolve()
    compiler.set('meshdir', Path(os.path.relpath(meshdir, folder)).as_posix())
    root.set('model', 'IST_robot_single_left_wing')
    path = folder / 'IST_robot_single_left_wing.xml'
    ET.indent(tree, space='  ')
    tree.write(path, encoding='utf-8', xml_declaration=True)
    metadata = {
        'configuration': 'Left wing installed; right wing plate removed; both flapping drives retained',
        'source_model': str(source),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'removed_body': 'right_wing', 'removed_mass_kg': removed_mass,
        'retained_drives': ['left_flapping', 'right_flapping'],
        'note': 'The complete CAD right_wing body is treated as the removable wing plate.',
    }
    (folder / 'configuration.json').write_text(json.dumps(metadata, indent=2))
    return path


def single_config(path):
    base = current_simulation()
    return replace(base, model_path=str(path.resolve()),
                   fluid=replace(base.fluid, wings=tuple(w for w in base.fluid.wings if w.body == 'left_wing')))


def plot(results):
    plot_sweep(results, OUTPUT, title='IST_robot | Single-wing mean lift',
               subtitle='Left wing only | Both flapping drives retained | Full-cycle mean')


def write_notebook():
    import nbformat as nbf
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nb.cells = [
        nbf.v4.new_markdown_cell('''# 单翼 10–15 Hz：保留两侧扑翼传动

按实测配置移除右侧翼膜/翼板，只保留左翼；左右扑翼传动机构及减速、滑块机构全部保留。
具体建模为移除 CAD 的 `right_wing` 刚体（含其质量、惯量、被动羽化关节和气动力），
保留 `right_flapping` 刚体、关节和滑块耦合。原双翼模型文件没有修改。
左翼的质量、几何和气动参数继承模型；羽化限位以运行时模型和 summary.json 的 pitch_limits_deg 为准。

每隔 1 Hz 独立运行，并校准实际频率误差至 0.01 Hz 内。
每组 2 s、步长 5 µs、采样 2 kHz；排除前 0.6 s 后按完整周期时间积分平均。
图中 Fz 是**单片左翼的气动升力**，正向为世界 +Z；不是支架总反力。
误差棒为各周期均值的最小/最大值，不是置信区间。
尚未读入实测数据，也未据此拟合气动系数。'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import sys, json
import numpy as np
from IPython.display import display, Image, Markdown
here = Path.cwd().resolve()
ROOT = next(p for p in (here, *here.parents) if (p / 'flapsim' / 'simulator.py').is_file())
FOLDER = ROOT / 'experiments' / 'fixbase'
sys.path.insert(0, str(FOLDER))
from single_wing_sweep import OUTPUT, single_config, plot
from sweep_frequency import saved_run
from run_15hz import cycle_statistics
config = single_config(OUTPUT / 'model' / 'IST_robot_single_left_wing.xml')
results = json.loads((OUTPUT / 'sweep.json').read_text())
assert len(results) == 6
for r in results:
    data, summary = saved_run(FOLDER / r['source_folder'], config)
    assert summary['wing_bodies'] == ['left_wing']
    assert 'right_flapping_joint_angle_rad' in data
    assert 'right_feathering_joint_angle_rad' not in data
    np.testing.assert_allclose(data['aero_Fz_N'], data['left_aero_Fz_N'], atol=1e-12)
    measured = cycle_statistics(data, target_hz=r['target_wing_frequency_hz'])
    assert abs(measured['measured_wing_frequency_hz']-r['target_wing_frequency_hz']) <= .01
    assert np.isclose(measured['mean_lift_N'], r['mean_lift_N'])
plot(results)
display(Image(filename=str(OUTPUT / 'mean_Fz_vs_frequency.png')))
rows = ['| 目标 Hz | 实际 Hz | 单翼平均 Fz / mN | 周期数 |', '|---:|---:|---:|---:|']
rows += [f"| {r['target_wing_frequency_hz']} | {r['measured_wing_frequency_hz']:.4f} | {r['mean_lift_mN']:.3f} | {r['complete_cycles']} |" for r in results]
display(Markdown(chr(10).join(rows)))'''),
        nbf.v4.new_code_cell("display(Image(filename=str(OUTPUT / '15hz' / 'wrench_6d.png')))\ndisplay(Image(filename=str(OUTPUT / '15hz' / 'joint_motion.png')))"),
        nbf.v4.new_markdown_cell('''复现：`python experiments/fixbase/single_wing_sweep.py`。
`results_single_wing_10_15hz/sweep.csv` 与 `sweep.json` 为汇总结果。
每组 `source_folder` 指向真实 CSV/NPZ、仿真参数及调速记录，包含单翼六维气动力和支座载荷。
`model/configuration.json` 记录拆除质量、保留机构及原模型哈希。'''),
    ]
    km = KernelManager(kernel_name='python3')
    km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    NotebookClient(nb, timeout=120, km=km, resources={'metadata': {'path': str(FOLDER)}}).execute()
    path = FOLDER / 'IST_robot_single_wing_10_15hz.ipynb'
    nbf.write(nb, path)
    print('Executed notebook:', path, flush=True)


def main():
    path = build_single_wing_model()
    config = single_config(path)
    results = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(measure, f, sim_config=config, output=OUTPUT,
                            load_coefficient=.68) for f in FREQUENCIES]
        for job in as_completed(jobs):
            result = job.result()
            result['configuration'] = 'Left wing only; both flapping drives retained'
            results.append(result)
            results.sort(key=lambda r: r['target_wing_frequency_hz'])
            (OUTPUT / 'sweep.json').write_text(json.dumps(results, indent=2))
    fields = ['target_wing_frequency_hz', 'measured_wing_frequency_hz', 'mean_lift_N',
              'mean_lift_mN', 'complete_cycles', 'motor_command_rad_s', 'source_folder', 'configuration']
    with (OUTPUT / 'sweep.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    for result in results:
        data, _ = saved_run(FOLDER / result['source_folder'], config)
        figures = plot_results(data, OUTPUT / f"{result['target_wing_frequency_hz']:02d}hz")
        import matplotlib.pyplot as plt
        for figure in figures:
            plt.close(figure)
    plot(results)
    write_notebook()


if __name__ == '__main__':
    main()
