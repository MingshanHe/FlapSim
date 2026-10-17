"""Freeze the 7+8 g comparison protocol and powered initial-state bank."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
from pathlib import Path
import zipfile
import numpy as np
from .mass_model import MassVariantEnv,describe
from .architecture import trim_network
from .activation import cases
from .train import write_json


def phase_state(task):
    root,index=task
    root=Path(root); trim=json.loads((root/'calibration/trim.json').read_text())
    env=MassVariantEnv(trim,model_path=root/'robot_upper7_lower8.xml',sensor_inputs=6,
                       initialization='trim',warmup_seconds=.8+index/8./trim['flapping_hz'])
    try: return env._initial_qpos.copy(),env._initial_qvel.copy()
    finally: env.close()


def prepare(root):
    root=Path(root).resolve()
    if (root/'protocol.json').exists(): raise RuntimeError('Protocol already frozen')
    trim=json.loads((root/'calibration/trim.json').read_text())
    write_json(root/'trim_seed.json',trim_network(trim))
    with ProcessPoolExecutor(max_workers=8) as pool:
        bank=list(pool.map(phase_state,[(str(root),i) for i in range(8)]))
    np.savez(root/'phase_bank.npz',qpos=np.array([p[0] for p in bank]),qvel=np.array([p[1] for p in bank]))
    protocol=dict(status='registered_before_training',date='2026-10-08',
        upper_mass_g=7.,lower_mass_g=8.,total_mass_g=15.,properties=describe(root/'robot_upper7_lower8.xml'),
        inputs=['base_link roll','base_link pitch','base_link yaw','base_link gyro_x','base_link gyro_y','base_link gyro_z'],
        outputs=['alpha','beta','motor_speed'],angle_range_rad=[-.3,.3],
        motor_center_rad_s=trim['command'][0],motor_fraction=[-.1,.1],
        action_smoothing_s=.02,physics_dt_s=.000005,control_dt_s=.01,
        initial_state='Eight fixed-base powered phases, 0.8 s plus k/8 periods; free-body momentum correction only at reset',
        actor_observation='Six current IMU values only. NEAT has evolved recurrent memory; PPO baseline is a feed-forward 64x64 MLP. Architecture is an explicit method difference, not an optimizer-only comparison.',
        seeds=[835,836],neat_population=128,neat_initial_hidden=8,neat_scenarios=2,
        target_training_steps_per_seed=524288,neat_generation_cap=100,
        budget_matching='NEAT completes a full generation crossing the target. PPO matches that seed total, rounded up to a 2048-step rollout; actual counts and validation costs reported.',
        reward='0.01*(1+2*exp(-(tilt/.35)^2)+2*exp(-(position_error/.2)^2)+exp(-(speed/.5)^2)-.1*min(sum((omega/10)^2),10)-.05*action_delta_squared-.005*action_squared), terminal -1, numerical -2',
        discount=.995,neat_fitness='Mean discounted shared reward across two shared cases; no teacher or old-model parent',
        curriculum='Training fraction <.3: 2 s, +/-2 deg; <.6: 4 s; then 8 s. Nominal/disturbed episodes alternate. Same schedule and disturbance generator for both.',
        neat_activations=['tanh','sigmoid','softsign','clamped','sin','gauss'],
        neat_output_activations=['tanh','clamped','bipolar_sigmoid','softsign'],
        activation_note='bipolar_sigmoid=2*sigmoid(5*x)-1 is mathematically equivalent to the library tanh(2.5*x); not an independent expressiveness gain',
        validation_cases=cases(range(905810000,905810006),8.)+cases(range(905820000,905820006),8.,True),
        test_nominal=cases(range(1053810000,1053810032),8.),test_stress=cases(range(1053820000,1053820032),8.,True),
        test_policy_selection='Use validation only; freeze all four selected policies before any final test.',
        success='Complete 8 s, position RMSE < .1 m, tilt RMS < 10 degrees',
        usable_simulation_target='At least 30/32 nominal and 28/32 disturbed successes; no hardware claim',
        limitations=['Six IMU values do not directly observe altitude or translational velocity',
                     'Rigid COM locations and inertia scaling remain CAD-based assumptions',
                     'Two training seeds provide descriptive evidence only, not a general NEAT-versus-PPO superiority claim'],
        versions={name:importlib.metadata.version(name) for name in ['neat-python','stable-baselines3','gymnasium','mujoco','torch']})
    write_json(root/'protocol.json',protocol)
    files=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).with_name('config.ini')]
    files+=list(Path('flapsim').rglob('*.py'))+list(Path('flapsim').rglob('*.cpp'))
    files += [root/'robot_upper7_lower8.xml',root/'calibration/trim.json',root/'phase_bank.npz']
    write_json(root/'source_manifest.json',{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
    with zipfile.ZipFile(root/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in files: archive.write(path,str(path))
    print(json.dumps(dict(stage='prepared',root=str(root),trim_converged=trim['converged'])),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    prepare(parser.parse_args().root)
