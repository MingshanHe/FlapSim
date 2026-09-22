"""Measured 10--15 Hz wing-frequency sweep with unchanged motor and fluid models."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from experiment import ExperimentConfig, run_experiment, SimulationConfig, SceneConfig, ControllerConfig
from run_15hz import cycle_statistics

FOLDER = Path(__file__).resolve().parent
OUTPUT = FOLDER / 'results_10_15hz'
FREQUENCIES = (10, 11, 12, 13, 14, 15)
TOLERANCE_HZ = .01


def current_simulation():
    return SimulationConfig(scene=SceneConfig(timestep=5e-6, fixed_base=True),
                            controller=ControllerConfig(speed=(0.0,)))


def check_run(data, summary, config=None):
    config = config or current_simulation()
    assert summary['model_sha256'] == hashlib.sha256(config.path.read_bytes()).hexdigest()
    assert json.dumps(summary['simulation'], sort_keys=True) == json.dumps(asdict(config), sort_keys=True)
    assert summary['experiment']['duration'] == 2.0
    assert summary['experiment']['ramp_time'] == .15
    assert summary['experiment']['sample_hz'] == 2000
    assert summary['max_base_pose_error'] == 0
    assert not any(summary['numerical_warnings'])
    assert all(np.isfinite(values).all() for values in data.values())


def saved_run(folder, config=None):
    with np.load(folder / 'measurements.npz') as saved:
        data = {key: saved[key] for key in saved.files}
    summary = json.loads((folder / 'summary.json').read_text())
    check_run(data, summary, config)
    return data, summary


def measure(target, *, sim_config=None, output=OUTPUT, load_coefficient=1.36):
    folder = output / f'{target:02d}hz'
    folder.mkdir(parents=True, exist_ok=True)
    # Existing 15 Hz measurement has the identical 2 s protocol and model.
    # Reuse actual samples, and recompute the statistic instead of extrapolating.
    cached_summary = FOLDER / 'results_15hz' / 'summary.json'
    reuse = sim_config is None and target == 15 and cached_summary.is_file()
    if reuse:
        cached = json.loads(cached_summary.read_text())
        current = current_simulation()
        reuse = (cached.get('model_sha256') == hashlib.sha256(current.path.read_bytes()).hexdigest()
                 and json.dumps(cached['simulation'], sort_keys=True) == json.dumps(asdict(current), sort_keys=True))
    if reuse:
        source = FOLDER / 'results_15hz'
        data, summary = saved_run(source)
        stats = cycle_statistics(data, target_hz=target)
        assert abs(stats['measured_wing_frequency_hz'] - target) <= TOLERANCE_HZ
        result = {**stats, 'motor_command_rad_s': summary['experiment']['motor_speed'],
                  'source_folder': str(source.relative_to(FOLDER)), 'reused_previous_run': True}
    else:
        # Approximate load compensation only seeds calibration; reported points
        # always come from simulated measurements at the verified frequency.
        command = 2*np.pi*16*target + load_coefficient*target**2
        trials = []
        for attempt in range(5):
            source = folder / f'run_{attempt+1}'
            data, summary = run_experiment(source, ExperimentConfig(duration=2.0,
                                    motor_speed=command, ramp_time=.15), video=False, sim_config=sim_config)
            check_run(data, summary, sim_config)
            stats = cycle_statistics(data, target_hz=target)
            measured = stats['measured_wing_frequency_hz']
            trials.append({'command_rad_s': float(command), **stats})
            (folder / 'calibration.json').write_text(json.dumps(trials, indent=2))
            print(f'{target} Hz trial {attempt+1}: {measured:.5f} Hz, '
                  f'{stats["mean_lift_mN"]:.3f} mN', flush=True)
            if abs(measured-target) <= TOLERANCE_HZ:
                break
            slope = 1/(2*np.pi*16 + 2*load_coefficient*target)
            if len(trials) > 1:
                previous = trials[-2]
                slope = (measured-previous['measured_wing_frequency_hz']) / (command-previous['command_rad_s'])
            if slope <= 0:
                raise RuntimeError(f'{target} Hz calibration stalled')
            command += (target-measured)/slope
            if not 0 < command <= 2000:
                raise RuntimeError('Frequency requires a command outside the existing motor limits')
        else:
            raise RuntimeError(f'{target} Hz did not converge')
        result = {**stats, 'motor_command_rad_s': float(command),
                  'source_folder': str(source.relative_to(FOLDER)), 'reused_previous_run': False}
    (folder / 'lift_summary.json').write_text(json.dumps(result, indent=2))
    print(f'COMPLETE {target} Hz: {result["mean_lift_mN"]:.3f} mN', flush=True)
    return result


def plot_sweep(results, output=OUTPUT, *, title='IST_robot | Mean lift vs. flapping frequency',
               subtitle='Fixed base | Both wings | Full cycles after 0.6 s'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = np.array([r['measured_wing_frequency_hz'] for r in results])
    y = np.array([r['mean_lift_mN'] for r in results])
    bounds = np.array([r['cycle_lift_min_max_N'] for r in results])*1000
    fig, ax = plt.subplots(figsize=(9, 5.7), layout='constrained')
    ax.errorbar(x, y, yerr=np.array([y-bounds[:, 0], bounds[:, 1]-y]),
                color='#2563eb', marker='o', markersize=7, linewidth=2,
                capsize=4, label='Time-weighted mean; bars = cycle min/max')
    for f, lift in zip(x, y):
        ax.annotate(f'{lift:.2f}', (f, lift), xytext=(0, 11), textcoords='offset points',
                    ha='center', fontsize=11, color='#1e3a8a')
    ax.set(xlabel='Measured flapping frequency [Hz]', ylabel='Mean aerodynamic Fz [mN]',
           xlim=(9.7, 15.3), ylim=(0, max(y)*1.18), xticks=FREQUENCIES)
    ax.set_title(title+'\n'+subtitle, pad=14)
    ax.grid(alpha=.22)
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend(loc='upper left', fontsize=9, frameon=False)
    fig.savefig(output / 'mean_Fz_vs_frequency.png', dpi=220)
    fig.savefig(output / 'mean_Fz_vs_frequency.svg')
    plt.close(fig)


def write_notebook():
    import nbformat as nbf
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nb.cells = [
        nbf.v4.new_markdown_cell('''# IST_robot：10–15 Hz 平均升力扫描

每隔 1 Hz 取点，固定基座、被动羽化与气动/电机参数保持一致。
通过调节速度指令补偿负载误差，实测频率误差不超过 0.01 Hz。
每组运行 2 s，排除前 0.6 s，仅在完整曲柄周期内积分并按时间加权平均。
Fz 为左右翼合计的世界坐标气动力，正值向上。15 Hz 点复用此前同参数实验的原始记录并重新计算。
图中误差棒表示各周期平均升力的最小值和最大值，**不是置信区间或模型误差**。'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import sys, json
import numpy as np
from IPython.display import display, Image, Markdown
here = Path.cwd().resolve()
ROOT = next(p for p in (here, *here.parents) if (p / 'flapsim' / 'simulator.py').is_file())
FOLDER = ROOT / 'experiments' / 'fixbase'
sys.path.insert(0, str(FOLDER))
from sweep_frequency import saved_run, plot_sweep, TOLERANCE_HZ
from run_15hz import cycle_statistics
OUTPUT = FOLDER / 'results_10_15hz'
results = json.loads((OUTPUT / 'sweep.json').read_text())
assert len(results) == 6
for r in results:
    data, summary = saved_run(FOLDER / r['source_folder'])
    calculated = cycle_statistics(data, target_hz=r['target_wing_frequency_hz'])
    assert abs(calculated['measured_wing_frequency_hz']-r['target_wing_frequency_hz']) <= TOLERANCE_HZ
    assert np.isclose(calculated['mean_lift_N'], r['mean_lift_N'])
plot_sweep(results)
display(Image(filename=str(OUTPUT / 'mean_Fz_vs_frequency.png')))
rows = ['| 目标 Hz | 实测 Hz | 平均 Fz / mN | 完整周期数 |', '|---:|---:|---:|---:|']
rows += [f"| {r['target_wing_frequency_hz']} | {r['measured_wing_frequency_hz']:.4f} | {r['mean_lift_mN']:.3f} | {r['complete_cycles']} |" for r in results]
display(Markdown(chr(10).join(rows)))'''),
        nbf.v4.new_markdown_cell('''汇总数据：`results_10_15hz/sweep.csv`、`sweep.json`。
每点 `source_folder` 指向原始 CSV/NPZ 和参数记录；各频率子目录保存调速过程。
PNG 与 SVG 图可直接导出。重新运行：`python experiments/fixbase/sweep_frequency.py`。
当前结果来自未实机标定的仿真模型。'''),
    ]
    path = FOLDER / 'IST_robot_10_15hz.ipynb'
    km = KernelManager(kernel_name='python3')
    km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    NotebookClient(nb, timeout=120, km=km, resources={'metadata': {'path': str(FOLDER)}}).execute()
    nbf.write(nb, path)
    print('Executed notebook:', path, flush=True)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(measure, target) for target in FREQUENCIES]
        for job in as_completed(jobs):
            results.append(job.result())
            results.sort(key=lambda r: r['target_wing_frequency_hz'])
            (OUTPUT / 'sweep.json').write_text(json.dumps(results, indent=2))
    keys = ['target_wing_frequency_hz', 'measured_wing_frequency_hz', 'mean_lift_N',
            'mean_lift_mN', 'complete_cycles', 'motor_command_rad_s', 'source_folder', 'reused_previous_run']
    with (OUTPUT / 'sweep.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    plot_sweep(results)
    write_notebook()


if __name__ == '__main__':
    if '--plot-only' in sys.argv:
        plot_sweep(json.loads((OUTPUT / 'sweep.json').read_text()))
        write_notebook()
    else:
        main()
