"""Measure lift at a measured 15 Hz wing-cycle frequency, with the same motor."""
import json
from pathlib import Path

import numpy as np

from experiment import ExperimentConfig, run_experiment, plot_results

OUTPUT = Path(__file__).resolve().parent / 'results_15hz'
GEAR_RATIO = 16
TARGET_HZ = 15.0


def cycle_statistics(data, after=0.6, *, target_hz=TARGET_HZ):
    """Integrate Fz between complete crank revolutions after the transient."""
    time = data['time_s']
    phase = data['motor_joint_angle_rad'] / (2 * np.pi * GEAR_RATIO)
    keep = time >= after
    if not np.all(np.diff(phase[keep]) > 0):
        raise ValueError('Motor must rotate monotonically in the measurement interval')
    integers = np.arange(np.ceil(phase[keep][0]), np.floor(phase[-1]) + 1)
    if len(integers) < 3:
        raise ValueError('Need at least two full steady measurement cycles')
    boundaries = np.interp(integers, phase[keep], time[keep])
    means = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        inside = (time > start) & (time < end)
        points = np.r_[start, time[inside], end]
        lift = np.interp(points, time, data['aero_Fz_N'])
        means.append(float(np.trapezoid(lift, points) / (end-start)))
    periods = np.diff(boundaries)
    return {
        'target_wing_frequency_hz': target_hz,
        'measured_wing_frequency_hz': float(len(periods) / periods.sum()),
        'mean_lift_N': float(np.average(means, weights=periods)),
        'mean_lift_mN': float(1000 * np.average(means, weights=periods)),
        'cycle_lift_min_max_N': [min(means), max(means)],
        'cycle_lift_N': means,
        'cycle_frequency_hz': (1 / periods).tolist(),
        'cycle_boundaries_s': boundaries.tolist(),
        'complete_cycles': len(periods),
        'averaging_interval_s': [float(boundaries[0]), float(boundaries[-1])],
        'definition': 'Supplied aerodynamic Fz, world +Z; time integral over full crank cycles',
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    command = TARGET_HZ * 2 * np.pi * GEAR_RATIO
    trials = []
    # The existing speed loop is proportional, so compensate its measured
    # load droop using the command only. Motor limits and all physics stay fixed.
    for trial in range(5):
        folder = OUTPUT / f'calibration_{trial+1}'
        data, _ = run_experiment(folder, ExperimentConfig(motor_speed=command, ramp_time=.15), video=False)
        stats = cycle_statistics(data)
        measured = stats['measured_wing_frequency_hz']
        trials.append({'command_rad_s': float(command), **stats})
        (OUTPUT / 'calibration.json').write_text(json.dumps(trials, indent=2))
        print(f'Calibration {trial+1}: command={command:.3f} rad/s, measured={measured:.5f} Hz', flush=True)
        if abs(measured - TARGET_HZ) <= .015:
            break
        if len(trials) == 1:
            slope = 1 / (2 * np.pi * GEAR_RATIO)
        else:
            previous = trials[-2]
            slope = (measured-previous['measured_wing_frequency_hz']) / (command-previous['command_rad_s'])
        if slope <= 0:
            raise RuntimeError('Frequency did not increase with the command; inspect motor saturation')
        next_command = command + (TARGET_HZ-measured) / slope
        if not 0 < next_command <= 2000:
            raise RuntimeError('Requested frequency exceeds the existing motor command range')
        command = next_command
    else:
        raise RuntimeError('Measured frequency did not converge to 15 Hz')

    data, _ = run_experiment(OUTPUT, ExperimentConfig(duration=2.0, motor_speed=command,
                                                      ramp_time=.15), video=True)
    stats = cycle_statistics(data, after=.6)
    stats['motor_command_rad_s'] = float(command)
    if abs(stats['measured_wing_frequency_hz'] - TARGET_HZ) > .05:
        raise RuntimeError('Final measured frequency differs from 15 Hz by more than .05 Hz')
    (OUTPUT / 'lift_summary.json').write_text(json.dumps(stats, indent=2))
    plot_results(data, OUTPUT)
    plot_lift(data, stats)
    print(json.dumps(stats, indent=2), flush=True)


def plot_lift(data, stats):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), layout='constrained')
    axes[0].plot(data['time_s'], data['aero_Fz_N']*1000, lw=.8, label='Instantaneous total lift')
    axes[0].axhline(stats['mean_lift_mN'], color='red', ls='--', label=f"Full-cycle mean: {stats['mean_lift_mN']:.2f} mN")
    axes[0].axvspan(*stats['averaging_interval_s'], alpha=.12, color='green', label='Averaging interval')
    axes[0].set(xlabel='Time [s]', ylabel='Aerodynamic Fz [mN]')
    axes[0].legend()
    boundaries = np.asarray(stats['cycle_boundaries_s'])
    centers = (boundaries[:-1]+boundaries[1:])/2
    axes[1].plot(centers, stats['cycle_frequency_hz'], 'o-')
    axes[1].axhline(TARGET_HZ, color='gray', ls='--')
    axes[1].set(xlabel='Cycle center time [s]', ylabel='Measured wing frequency [Hz]')
    axes[1].set_ylim(TARGET_HZ-.01, TARGET_HZ+.01)
    axes[1].ticklabel_format(axis='y', style='plain', useOffset=False)
    for ax in axes:
        ax.grid(alpha=.25)
    fig.savefig(OUTPUT / 'lift_15hz.png', dpi=180)
    plt.close('all')


def write_notebook():
    """Execute a compact report from the recorded run, including video playback."""
    import sys
    import nbformat as nbf
    from nbclient import NotebookClient
    from jupyter_client import KernelManager

    notebook = nbf.v4.new_notebook()
    notebook.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    notebook.cells = [
        nbf.v4.new_markdown_cell('''# IST_robot：末端 15 Hz 固定基座升力实验

沿用原实验的机器人、被动羽化、气动力与电机参数，仅调整输入轴转速指令。
16∶1 减速下的理想输入角速度为 `15 × 2π × 16 ≈ 1507.96 rad/s`。
比例调速有负载转速误差，因此通过实际完整周期测量修正指令，并记录在 `calibration.json`。
频率按完整曲柄周期计算；升力为左右翼气动力的世界坐标 Fz，正值向上。
排除前 0.6 s 后，使用完整周期边界内的时间积分求平均。
电机与气动参数未做实机标定，这里报告的是当前模型的仿真结果。'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import json
import sys
import numpy as np
from IPython.display import display, Image, Video, Markdown

here = Path.cwd().resolve()
ROOT = next(p for p in (here, *here.parents) if (p / 'flapsim' / 'simulator.py').is_file())
EXPERIMENT = ROOT / 'experiments' / 'fixbase'
sys.path.insert(0, str(EXPERIMENT))
from run_15hz import main, cycle_statistics, plot_lift
RUN_SIMULATION = False  # 改成 True 可重新调速并运行；默认读取已保存的真实数据。
if RUN_SIMULATION:
    main()
OUTPUT = EXPERIMENT / 'results_15hz'
with np.load(OUTPUT / 'measurements.npz') as saved:
    data = {k: saved[k] for k in saved.files}
summary = json.loads((OUTPUT / 'summary.json').read_text())
lift = json.loads((OUTPUT / 'lift_summary.json').read_text())
recomputed = cycle_statistics(data, after=.6)
assert np.isclose(recomputed['mean_lift_N'], lift['mean_lift_N'])
assert abs(lift['measured_wing_frequency_hz'] - 15) < .05
assert summary['max_base_pose_error'] == 0
assert not any(summary['numerical_warnings'])
plot_lift(data, lift)
print(f"实测末端频率：{lift['measured_wing_frequency_hz']:.5f} Hz")
print(f"平均总升力：{lift['mean_lift_N']:.6f} N = {lift['mean_lift_mN']:.3f} mN")
print('完整周期数：', lift['complete_cycles'])
print('统计时间区间 [s]：', lift['averaging_interval_s'])
print('各周期升力均值范围 [N]：', lift['cycle_lift_min_max_N'])
print('实际电机速度指令 [rad/s]：', lift['motor_command_rad_s'])'''),
        nbf.v4.new_code_cell("display(Image(filename=str(OUTPUT / 'lift_15hz.png')))"),
        nbf.v4.new_code_cell("display(Image(filename=str(OUTPUT / 'wrench_6d.png')))\ndisplay(Image(filename=str(OUTPUT / 'joint_motion.png')))"),
        nbf.v4.new_code_cell("display(Video(str(OUTPUT / 'fixbase.mp4'), embed=True, width=1100, html_attributes='controls loop muted'))"),
        nbf.v4.new_markdown_cell('''原始数据见 `results_15hz/measurements.csv` 与 `.npz`。
六维顺序为 Fx、Fy、Fz、Mx、My、Mz，力矩参考 base_link 原点。
`aero_*` 为气动力；`fixture_*` 为机器人作用于支座的总载荷，两者不混用。
完整周期统计见 `lift_summary.json`；视频以 250 仿真帧/s 捕获、30 fps 编码，约 8.33 倍慢放。
复现命令：`python experiments/fixbase/run_15hz.py`。'''),
    ]
    path = OUTPUT.parent / 'IST_robot_15hz.ipynb'
    km = KernelManager(kernel_name='python3')
    km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    client = NotebookClient(notebook, timeout=900, km=km, resources={'metadata': {'path': str(OUTPUT.parent)}})
    try:
        client.execute()
    finally:
        nbf.write(notebook, path)
    print('Executed notebook:', path)


if __name__ == '__main__':
    import sys
    if '--notebook-only' not in sys.argv:
        main()
    write_notebook()
