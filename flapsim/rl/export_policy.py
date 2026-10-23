"""Freeze a final/best PPO checkpoint with its exact normalizer and run config."""
import argparse
from dataclasses import replace
import hashlib
from io import BytesIO
from importlib.metadata import version
import json
from pathlib import Path
import zipfile
from flapsim import save_config
from .train import load_run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--best', action='store_true')
    a = p.parse_args()
    _, simulation, metadata = load_run(a.run)
    if a.output.exists() and any(a.output.iterdir()):
        p.error('output must be empty')
    source = a.run/'best' if a.best else a.run
    policy = (source/('best_model.zip' if a.best else 'policy.zip')).read_bytes()
    normalizer = (source/'vecnormalize.pkl').read_bytes()
    with zipfile.ZipFile(BytesIO(policy)) as archive:
        steps = json.loads(archive.read('data'))['num_timesteps']
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output/'policy.zip').write_bytes(policy)
    (a.output/'vecnormalize.pkl').write_bytes(normalizer)
    # Built-in model paths remain portable with this package; custom models keep
    # their explicit path rather than silently substituting a different robot.
    builtin = replace(simulation, model_path=None)
    if builtin.path.resolve() == simulation.path.resolve():
        simulation = builtin
    save_config(simulation, a.output/'simulation.json')
    metadata.update(status='exported', num_timesteps=steps,
                    exported_from=str(source.resolve()),
                    export_runtime_versions={name: version(name) for name in
                        ('mujoco', 'numpy', 'torch', 'gymnasium', 'stable_baselines3')},
                    exported_policy_sha256=hashlib.sha256(policy).hexdigest(),
                    exported_normalizer_sha256=hashlib.sha256(normalizer).hexdigest())
    (a.output/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Frozen policy at {steps} steps: {a.output.resolve()}')


if __name__ == '__main__':
    main()
