"""Causal IMU dead-reckoning diagnostics; never feed position truth to a policy.

The historical training run is read-only.  A separate model copy adds a massless
accelerometer at the base_link origin. Existing policies remain frozen.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .compare_env import CompareEnv, make_policy
from .environment import rotation_to_rpy
from .train import write_json
from flapsim.kinematics import euler_matrix
from flapsim import load_config
from flapsim.rl.env import PRESET


def prepare(source, destination):
    source, destination = Path(source), Path(destination)
    (destination/'calibration').mkdir(parents=True, exist_ok=True)
    for name in ('phase_bank.npz', 'calibration/trim.json'):
        shutil.copyfile(source/name, destination/name)
    tree = ET.parse(source/'robot_upper7_lower8.xml')
    root = tree.getroot()
    original = load_config(PRESET).path
    compiler = ET.parse(original).getroot().find('compiler')
    root.find('compiler').set('meshdir',str((original.parent/compiler.get('meshdir','.')).resolve()))
    base = root.find(".//body[@name='base_link']")
    ET.SubElement(base, 'site', name='history_imu', pos='0 0 0', size='0.0001')
    sensors = root.find('sensor')
    if sensors is None:
        sensors = ET.SubElement(root, 'sensor')
    ET.SubElement(sensors, 'accelerometer', name='history_accel', site='history_imu')
    ET.SubElement(sensors, 'gyro', name='history_gyro', site='history_imu')
    tree.write(destination/'robot_upper7_lower8.xml', encoding='utf-8', xml_declaration=True)
    protocol = json.loads((source/'protocol.json').read_text())
    # New diagnostic cases, disjoint from the previous train/validation/test IDs.
    template = protocol['test_nominal'][0]
    cases = [dict(template, seed=2060810000+i) for i in range(8)]
    result = dict(source=str(source.resolve()), cases=cases, sampling_hz=2000,
                  policy='neat_seed836', sensor_site='base_link origin',
                  raw_channels=['roll','pitch','yaw','gyro_x','gyro_y','gyro_z','accel_x','accel_y','accel_z'],
                  position_use='Evaluation only, except clearly labeled oracle initial velocity upper bound',
                  scope='State estimation diagnostic, not new controller training or proof of hovering',
                  sensor_errors='Illustrative assumptions, not a measured hardware specification',
                  source_model_sha256=hashlib.sha256((source/'robot_upper7_lower8.xml').read_bytes()).hexdigest())
    write_json(destination/'protocol.json', result)
    shutil.copyfile(source/'neat_seed836/best_policy.json', destination/'policy.json')
    return result


def sample(env):
    data = env.sim.data
    site = env.sim.model.site('history_imu').id
    velocity = np.empty(6)
    mujoco.mj_objectVelocity(env.sim.model, data, mujoco.mjtObj.mjOBJ_SITE, site, velocity, 0)
    rotation = data.site_xmat[site].reshape(3,3).copy()
    return dict(time=float(data.time), position=data.site_xpos[site].copy(),
                velocity=velocity[3:].copy(), rotation=rotation,
                accel=data.sensor('history_accel').data.copy(),
                gyro=data.sensor('history_gyro').data.copy(), rpy=rotation_to_rpy(rotation))


def collect(root, case, sample_hz=2000, shift=None):
    root = Path(root)
    env = CompareEnv(root)
    env.configure_case(case)
    obs, _ = env.reset(seed=case['seed'])
    if shift is not None:
        env.sim.data.qpos[env.free_qpos:env.free_qpos+3] += shift
        mujoco.mj_forward(env.sim.model, env.sim.data)
        obs = env._observation()
    policy = make_policy(json.loads((root/'policy.json').read_text()))
    policy.reset(obs)
    rows = [sample(env)]
    observations, actions = [obs.copy()], []
    original = env.sim.step
    block = round(1./(sample_hz*env.sim.model.opt.timestep))
    assert block > 0 and env.substeps % block == 0

    def measured_step(steps=1):
        for start in range(0, steps, block):
            original(min(block, steps-start))
            rows.append(sample(env))
        return env.sim.data

    env.sim.step = measured_step
    try:
        while True:
            action = policy.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
            observations.append(obs.copy()); actions.append(action)
            if terminated or truncated:
                break
        arrays = {k:np.asarray([r[k] for r in rows]) for k in rows[0]}
        arrays['time'] -= arrays['time'][0]
        arrays['obs6'] = np.asarray(observations)
        arrays['actions'] = np.asarray(actions)
        return arrays, dict(seed=case['seed'], elapsed=info['elapsed'],
                           termination=info['termination_reason'],
                           tilt_rmse_deg=info['tilt_rmse_deg'],
                           tracking_rmse_m=info['tracking_rmse'])
    finally:
        env.close()


def integrate(time, rotation, specific_force, initial_velocity=None):
    """Trapezoidal strapdown integration with no position/velocity corrections."""
    gravity = np.array([0.,0.,-9.81])
    acceleration = np.einsum('nij,nj->ni', rotation, specific_force) + gravity
    velocity = np.zeros_like(acceleration)
    position = np.zeros_like(acceleration)
    if initial_velocity is not None:
        velocity[0] = initial_velocity
    dt = np.diff(time)
    velocity[1:] = velocity[0] + np.cumsum(.5*(acceleration[1:]+acceleration[:-1])*dt[:,None],axis=0)
    position[1:] = np.cumsum(.5*(velocity[1:]+velocity[:-1])*dt[:,None],axis=0)
    return position, velocity


def estimate_variants(arrays, sample_hz):
    rows, predictions = [], {}
    # These runs differ in acquisition/integration only, not controller actions.
    for rate, bias, pitch_error, initial in (
        (100,0.,0.,'oracle'), (500,0.,0.,'oracle'), (2000,0.,0.,'oracle'),
        (2000,0.,0.,'zero'), (2000,.02,0.,'zero'), (2000,0.,.5,'zero'),
    ):
        stride = sample_hz//rate
        assert sample_hz % rate == 0
        indices = np.arange(0,len(arrays['time']),stride)
        time = arrays['time'][indices]
        rotation = arrays['rotation'][indices] @ euler_matrix([0.,np.deg2rad(pitch_error),0.])
        accel = arrays['accel'][indices] + np.array([bias,0.,0.])
        initial_velocity = arrays['velocity'][0] if initial == 'oracle' else None
        position, velocity = integrate(time,rotation,accel,initial_velocity)
        truth = arrays['position'][indices]-arrays['position'][0]
        errors = np.linalg.norm(position-truth,axis=1)
        label = f'{rate}Hz_bias{bias:g}_pitch{pitch_error:g}_{initial}v0'
        rows.append(dict(label=label,rate_hz=rate,bias_m_s2=bias,pitch_error_deg=pitch_error,
                         initial_velocity=initial,elapsed_s=float(time[-1]),
                         displacement_rmse_m=float(np.sqrt(np.mean(errors**2))),
                         final_error_m=float(errors[-1]),
                         true_displacement_m=float(np.linalg.norm(truth[-1])),
                         final_velocity_error_m_s=float(np.linalg.norm(velocity[-1]-arrays['velocity'][indices[-1]]))))
        predictions[label] = np.column_stack([time,position,truth])
    return rows, predictions


def worker(task):
    root, case, hz = task
    arrays, episode = collect(root,case,hz)
    rows, predictions = estimate_variants(arrays,hz)
    directory = Path(root)/'traces'; directory.mkdir(exist_ok=True)
    np.savez_compressed(directory/f'{case["seed"]}.npz',**arrays)
    np.savez_compressed(directory/f'{case["seed"]}_estimates.npz',**predictions)
    return dict(episode=episode,estimates=rows,
                initial_speed_m_s=float(np.linalg.norm(arrays['velocity'][0])),
                accel_peak_m_s2=float(np.max(np.linalg.norm(arrays['accel'],axis=1))))


def run(source, root):
    protocol = prepare(source,root)
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(worker,[(str(root),c,protocol['sampling_hz']) for c in protocol['cases']]))
    summary = {}
    for i in range(len(results[0]['estimates'])):
        group = [r['estimates'][i] for r in results]
        summary[group[0]['label']] = dict(mean_rmse_m=float(np.mean([r['displacement_rmse_m'] for r in group])),
                                          mean_final_error_m=float(np.mean([r['final_error_m'] for r in group])),
                                          max_final_error_m=float(np.max([r['final_error_m'] for r in group])))
    # Horizontal translation changes target error but not inertial measurements.
    # This demonstrates absolute-position ambiguity, not impossibility of
    # relative dead reckoning from a known initial state.
    case = dict(protocol['cases'][0],seconds=.2)
    a, _ = collect(root,case,2000)
    shift = np.array([.1,-.08,0.])
    b, _ = collect(root,case,2000,shift)
    symmetry = dict(shift_m=shift.tolist(),duration_s=.2,
                    max_obs6_difference=float(np.max(np.abs(a['obs6']-b['obs6']))),
                    max_accel_difference_m_s2=float(np.max(np.abs(a['accel']-b['accel']))),
                    max_translation_residual_m=float(np.max(np.abs(b['position']-a['position']-shift))))
    result = dict(scope=protocol['scope'],episodes=results,summary=summary,translation_symmetry=symmetry)
    write_json(Path(root)/'results.json',result)
    plot(root,result)
    return result


def plot(root, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    seed = result['episodes'][0]['episode']['seed']
    traces = np.load(root/'traces'/f'{seed}_estimates.npz')
    raw = np.load(root/'traces'/f'{seed}.npz')
    fig, axes = plt.subplots(1,3,figsize=(15,4.3))
    selected = ['100Hz_bias0_pitch0_oraclev0','2000Hz_bias0_pitch0_oraclev0',
                '2000Hz_bias0_pitch0_zerov0','2000Hz_bias0.02_pitch0_zerov0','2000Hz_bias0_pitch0.5_zerov0']
    labels = ['100 Hz, ideal init','2 kHz, ideal init','2 kHz, zero init',
              '2 kHz, +0.02 m/s² bias','2 kHz, +0.5° pitch error']
    for name,label in zip(selected,labels):
        t = traces[name]
        axes[0].plot(t[:,0],100*np.linalg.norm(t[:,1:4]-t[:,4:7],axis=1),label=label)
    axes[0].set(xlabel='Flight time (s)',ylabel='Displacement error (cm)',title=f'First predeclared case {seed}')
    axes[0].legend(fontsize=7)
    axes[1].plot(raw['time'],raw['accel'][:,2],lw=.6)
    axes[1].set(xlabel='Flight time (s)',ylabel='Body-z specific force (m/s²)',title='Base-link accelerometer, 2 kHz')
    time = np.linspace(0,8,801)
    axes[2].plot(time,100*.5*.02*time**2,label='0.02 m/s² constant bias')
    axes[2].plot(time,100*.5*9.81*np.sin(np.deg2rad(.5))*time**2,label='0.5° gravity alignment error')
    axes[2].set(xlabel='Time (s)',ylabel='Position error (cm)',title='Analytical hover sensitivity (not flight test)')
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=.25)
    fig.suptitle('IMU history feasibility: frozen NEAT flight, no position feedback')
    fig.tight_layout()
    fig.savefig(root/'imu_history_probe.png',dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = run(args.source,args.output)
    print(json.dumps({k:v for k,v in result.items() if k!='episodes'},indent=2))
