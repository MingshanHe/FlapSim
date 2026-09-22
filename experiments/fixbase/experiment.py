"""Reproducible fixed-base run, measurements and headless video.

Run this file directly, or import run_experiment/plot_results from the notebook.
No prescribed flapping or feathering angles are used.
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')  # set before importing MuJoCo

from dataclasses import asdict, dataclass
import json
import hashlib
from pathlib import Path
import sys
import time

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from flapsim import Simulator, SimulationConfig, SceneConfig, ControllerConfig
from flapsim.measurement import FixedBaseWrenchProbe

JOINTS = ('motor_joint', 'left_flapping_joint', 'right_flapping_joint',
          'left_feathering_joint', 'right_feathering_joint')
AXES = ('Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz')


@dataclass(frozen=True)
class ExperimentConfig:
    duration: float = 1.2
    motor_speed: float = 500.0  # input shaft rad/s, NOT wing frequency
    ramp_time: float = .1
    timestep: float = 5e-6
    sample_hz: int = 2000
    video_sample_hz: int = 250
    playback_fps: int = 30


class HeadlessVideo:
    """Two synchronized camera views and actual simulation-time labels."""
    def __init__(self, sim, path, config):
        import imageio.v2 as imageio
        from PIL import ImageFont
        from matplotlib.font_manager import findfont
        self.ImageFont = ImageFont
        self.font = ImageFont.truetype(findfont('DejaVu Sans'), 18)
        self.renderer = mujoco.Renderer(sim.model, height=480, width=640)
        self.cameras = []
        for lookat, distance, azimuth, elevation in (
            ((0, 0, .54), .245, 135, -20),
            ((0, 0, .557), .135, 90, -55),
        ):
            camera = mujoco.MjvCamera()
            camera.lookat[:] = lookat
            camera.distance, camera.azimuth, camera.elevation = distance, azimuth, elevation
            self.cameras.append(camera)
        self.writer = imageio.get_writer(str(path), fps=config.playback_fps,
                                         codec='libx264', quality=8, pixelformat='yuv420p')
        self.preview = Path(path).with_suffix('.png')
        self.slowdown = config.video_sample_hz/config.playback_fps
        self.count = 0

    def append(self, sim, command):
        from PIL import Image, ImageDraw
        panels = []
        for camera in self.cameras:
            self.renderer.update_scene(sim.data, camera=camera)
            panels.append(self.renderer.render().copy())
        canvas = Image.new('RGB', (1280, 560), (17, 24, 39))
        canvas.paste(Image.fromarray(np.concatenate(panels, axis=1)), (0, 80))
        draw = ImageDraw.Draw(canvas)
        def angle(name):
            if mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, name) < 0:
                return 'absent'
            return f'{np.degrees(sim.data.joint(name).qpos[0]):+6.1f}'
        speed = float(sim.data.joint(JOINTS[0]).qvel[0])
        draw.text((18, 7), f'IST_robot | fixed base | t = {sim.data.time:.3f} s '
                  f'| motor = {speed:.1f} / {command:.1f} rad/s | {self.slowdown:.2f}x slow motion',
                  font=self.font, fill='white')
        draw.text((18, 34), f'Flapping L/R: {angle(JOINTS[1])} / {angle(JOINTS[2])} deg'
                  f'     Feathering L/R: {angle(JOINTS[3])} / {angle(JOINTS[4])} deg',
                  font=self.font, fill=(160, 215, 255))
        draw.text((20, 92), 'Assembly', font=self.font, fill='white')
        draw.text((660, 92), 'Wing mechanism / passive pitch', font=self.font, fill='white')
        self.writer.append_data(np.asarray(canvas))
        if self.count == 0:
            canvas.save(self.preview)
        self.count += 1

    def close(self):
        self.writer.close()
        self.renderer.close()


def reversal_count(values, deadband=np.deg2rad(5)):
    signs = np.sign(values[np.abs(values) > deadband])
    return int(np.count_nonzero(signs[1:] != signs[:-1]))


def run_experiment(output=None, config=None, *, video=True, sim_config=None):
    config = config or ExperimentConfig()
    output = Path(output or Path(__file__).parent / 'results').resolve()
    output.mkdir(parents=True, exist_ok=True)
    if config.duration <= 0 or config.ramp_time < 0:
        raise ValueError('duration must be positive; ramp_time must be nonnegative')
    if min(config.timestep, config.sample_hz, config.video_sample_hz, config.playback_fps) <= 0:
        raise ValueError('time step and sampling rates must be positive')
    stride = round(1/(config.timestep*config.sample_hz))
    if stride < 1 or not np.isclose(stride*config.timestep*config.sample_hz, 1):
        raise ValueError('sample period must be an integer multiple of timestep')
    if config.sample_hz % config.video_sample_hz:
        raise ValueError('sample_hz must be divisible by video_sample_hz')
    n_samples = round(config.duration*config.sample_hz)
    if not np.isclose(n_samples/config.sample_hz, config.duration):
        raise ValueError('duration must contain an integer number of samples')

    sim_config = sim_config or SimulationConfig(
        scene=SceneConfig(timestep=config.timestep, fixed_base=True),
        controller=ControllerConfig(speed=(0.0,)))
    if not sim_config.scene.fixed_base or sim_config.scene.timestep != config.timestep:
        raise ValueError('fixed-base measurement requires matching scene timestep and a fixed base')
    sim = Simulator(sim_config)
    m = sim.model
    probe = FixedBaseWrenchProbe(m)
    root = m.body('base_link').id
    base_pose = np.r_[sim.data.xpos[root], sim.data.xquat[root]].copy()
    joints = tuple(j for j in JOINTS if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) >= 0)
    joint_ids = [m.joint(j).id for j in joints]
    wing_names = [w.body for w in sim_config.fluid.wings]
    wing_ids = [m.body(n).id for n in wing_names]
    wing_groups = [f'{n.removesuffix("_wing")}_aero' for n in wing_names]
    measurement = mujoco.MjData(m)
    # Colors distinguish the mirrored wings in the rendered playback only.
    for name, rgba in (('left_wing_visual', [.25, .65, .95, 1]),
                       ('right_wing_visual', [1, .60, .20, 1])):
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0:
            m.geom(name).rgba = rgba
    rows = []
    columns = ['time_s', 'command_rad_s']
    columns += [f'{j}_angle_rad' for j in joints]
    columns += [f'{j}_velocity_rad_s' for j in joints]
    columns += ['motor_torque_Nm'] + [f'motor_stator_M{a}_Nm' for a in 'xyz']
    wrench_groups = ['aero', 'support', 'fixture', 'gravity', 'fixture_dynamic'] + wing_groups
    wrench_groups += [f'{j}_parent_load' for j in joints]
    for group in wrench_groups:
        columns += [f'{group}_{a}_{"N" if a[0] == "F" else "Nm"}' for a in AXES]
    columns += ['base_pose_error', 'linear_momentum_x', 'linear_momentum_y', 'linear_momentum_z']
    recorder = HeadlessVideo(sim, output / 'fixbase.mp4', config) if video else None
    start = time.perf_counter()
    command = 0.0
    try:
        for sample in range(n_samples+1):
            if sample:
                phase = min((sample-1)/config.sample_hz/max(config.ramp_time, 1e-12), 1)
                command = config.motor_speed * (.5-.5*np.cos(np.pi*phase)) if config.ramp_time else config.motor_speed
                sim.set_speed(command)
                sim.step(stride)
            # Re-evaluate forces on a COPY at the logged time. This does not
            # alter the physical trajectory, solver warm start, or controller.
            mujoco.mj_copyData(measurement, m, sim.data)
            aero = sim.fluid.forces(measurement)
            measurement.xfrc_applied[:] = aero
            measurement.qfrc_applied[:] = sim.actuator.forces(measurement, np.array([command]))
            mujoco.mj_forward(m, measurement)
            wrenches = probe.read(measurement, aero)
            for name, body in zip(wing_groups, wing_ids):
                wrenches[name] = probe.sum_wrenches(measurement, aero, [body])
            for j, joint in zip(joints, joint_ids):
                wrenches[f'{j}_parent_load'] = probe.interaction_wrench(
                    measurement, int(m.jnt_bodyid[joint]), measurement.xanchor[joint])
            torque = float(measurement.qfrc_applied[sim.actuator.dof_ids[0]])
            motor_reaction = -torque * measurement.xaxis[joint_ids[0]]
            pose_error = np.max(np.abs(np.r_[measurement.xpos[root], measurement.xquat[root]] - base_pose))
            # COM linear momentum for an independent root-force balance check.
            angular = measurement.cvel[:, :3]
            velocity = measurement.cvel[:, 3:] + np.cross(
                angular, measurement.xipos - measurement.subtree_com[m.body_rootid])
            momentum = np.sum(m.body_mass[:, None]*velocity, axis=0)
            row = [float(sim.data.time), command]
            row += [float(measurement.joint(j).qpos[0]) for j in joints]
            row += [float(measurement.joint(j).qvel[0]) for j in joints]
            row += [torque, *motor_reaction]
            for group in wrench_groups:
                row.extend(wrenches[group])
            row.extend([pose_error, *momentum])
            rows.append(row)
            if recorder and sample % (config.sample_hz // config.video_sample_hz) == 0:
                recorder.append(sim, command)
            if sample and sample % config.sample_hz == 0:
                print(f't = {sim.data.time:.2f} s; {sample}/{n_samples} samples', flush=True)
    finally:
        if recorder:
            recorder.close()

    values = np.asarray(rows)
    if not np.isfinite(values).all() or np.any(sim.data.warning.number):
        raise RuntimeError('non-finite data or MuJoCo numerical warning; inspect before interpreting')
    data = {name: values[:, i] for i, name in enumerate(columns)}
    np.savetxt(output / 'measurements.csv', values, delimiter=',', header=','.join(columns), comments='')
    np.savez_compressed(output / 'measurements.npz', **data)
    steady = data['time_s'] >= max(.25, config.ramp_time)
    pitch_joints = [j for j in joints if 'feathering' in j]
    pitch = np.column_stack([data[f'{j}_angle_rad'] for j in pitch_joints])
    motor = data['motor_joint_angle_rad']
    summary = {
        'experiment': asdict(config), 'simulation': asdict(sim_config),
        'mujoco_version': mujoco.__version__, 'elapsed_wall_s': time.perf_counter()-start,
        'model_path': str(sim_config.path),
        'model_sha256': hashlib.sha256(sim_config.path.read_bytes()).hexdigest(),
        'initial_base_pose_xyz_wxyz': base_pose.tolist(),
        'recorded_joints': joints, 'wing_bodies': wing_names,
        'pitch_limits_deg': {j: np.degrees(m.joint(j).range).tolist() for j in pitch_joints},
        'samples': len(values), 'video_frames': recorder.count if recorder else 0,
        'playback_slowdown': config.video_sample_hz/config.playback_fps,
        'wing_cycles_measured': float((motor[-1]-motor[0])/(32*np.pi)),
        'mean_motor_speed_after_025s': float(data['motor_joint_velocity_rad_s'][steady].mean()),
        'pitch_min_deg': np.degrees(pitch[steady].min(axis=0)).tolist(),
        'pitch_max_deg': np.degrees(pitch[steady].max(axis=0)).tolist(),
        'pitch_reversals_after_025s': [reversal_count(pitch[steady, i]) for i in range(len(pitch_joints))],
        'max_base_pose_error': float(data['base_pose_error'].max()),
        'mean_aero_wrench_after_025s': [float(data[f'aero_{a}_{"N" if a[0] == "F" else "Nm"}'][steady].mean()) for a in AXES],
        'numerical_warnings': sim.data.warning.number.tolist(),
        'reference': 'world axes; moments about base_link origin; joint loads about their joint anchors',
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('samples', 'wing_cycles_measured',
          'pitch_reversals_after_025s', 'max_base_pose_error', 'mean_aero_wrench_after_025s')}, indent=2))
    return data, summary


def plot_results(data, output=None, *, include_weight=False, pitch_limits_deg=None):
    import matplotlib.pyplot as plt
    output = Path(output or Path(__file__).parent / 'results')
    colors = {'left_aero': '#2563eb', 'right_aero': '#d97706', 'aero': '#111827'}
    labels = {'left_aero': 'Left wing', 'right_aero': 'Right wing', 'aero': 'Total aerodynamic wrench'}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, layout='constrained')
    for ax, component in zip(axes.flat, AXES):
        unit = 'N' if component[0] == 'F' else 'Nm'
        for prefix in colors:
            if f'{prefix}_{component}_{unit}' not in data:
                continue
            ax.plot(data['time_s'], data[f'{prefix}_{component}_{unit}'],
                    color=colors[prefix], lw=1, alpha=.85, label=labels[prefix])
        ax.set_title(component)
        ax.set_ylabel('Force [N]' if unit == 'N' else 'Moment [N m]')
        ax.grid(alpha=.22)
    for ax in axes[1]:
        ax.set_xlabel('Simulation time [s]')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Aerodynamic six-axis wrench | world axes | moments about base_link origin')
    fig.savefig(output / 'wrench_6d.png', dpi=180)

    # Keep hard-stop impulses visible at their actual scale, without flattening
    # the aerodynamic curves in the primary figure or clipping measured peaks.
    fixture, fax = plt.subplots(2, 3, figsize=(14, 7), sharex=True, layout='constrained')
    traces = [('aero', '#2563eb', 'Aerodynamic wrench'),
              ('fixture_dynamic', '#dc2626', 'Fixture load minus gravity')]
    if include_weight:
        traces.append(('fixture', '#94a3b8', 'Robot on fixture (total)'))
    for ax, component in zip(fax.flat, AXES):
        unit = 'N' if component[0] == 'F' else 'Nm'
        for prefix, color, label in traces:
            ax.plot(data['time_s'], data[f'{prefix}_{component}_{unit}'], color=color, lw=1, label=label)
        ax.set_title(component)
        ax.set_ylabel('Force [N]' if unit == 'N' else 'Moment [N m]')
        ax.grid(alpha=.22)
    for ax in fax[1]:
        ax.set_xlabel('Simulation time [s]')
    fax[0, 0].legend(fontsize=8)
    fixture.suptitle('Fixture load includes inertia and hard-stop impulses | full sampled range')
    fixture.savefig(output / 'fixture_wrench_6d.png', dpi=180)
    plt.close(fixture)

    motion, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True, layout='constrained')
    t = data['time_s']
    axs[0].plot(t, data['command_rad_s'], '--', color='#64748b', label='Command')
    axs[0].plot(t, data['motor_joint_velocity_rad_s'], color='#7c3aed', label='Measured motor speed')
    axs[0].set_ylabel('Shaft speed [rad/s]')
    for i, (left, right) in enumerate(((JOINTS[1], JOINTS[2]), (JOINTS[3], JOINTS[4])), start=1):
        for name, color, label in ((left, '#2563eb', 'Left'), (right, '#d97706', 'Right')):
            if f'{name}_angle_rad' in data:
                axs[i].plot(t, np.degrees(data[f'{name}_angle_rad']), color=color, label=label)
        axs[i].set_ylabel(('Flapping' if i == 1 else 'Feathering') + ' [deg]')
    for ax in axs:
        ax.legend(loc='upper right')
        ax.grid(alpha=.22)
    if pitch_limits_deg is not None:
        for limit in pitch_limits_deg:
            axs[2].axhline(limit, color='gray', lw=.5, ls=':')
    axs[2].set_xlabel('Simulation time [s]')
    motion.suptitle('Motor-driven mechanism and passive feathering (no prescribed wing trajectory)')
    motion.savefig(output / 'joint_motion.png', dpi=180)
    return fig, motion


if __name__ == '__main__':
    measured, summary = run_experiment()
    plot_results(measured)
