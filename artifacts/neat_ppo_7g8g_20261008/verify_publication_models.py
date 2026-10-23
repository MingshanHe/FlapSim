"""Check saved PPO models against their portable deterministic actors."""
import argparse
import ast
import inspect
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from experiments.algorithms.neat import ppo_refine
from experiments.algorithms.neat_imu.compare_env import NumpyPPO

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
root = parser.parse_args().root.resolve()
torch.set_num_threads(1)
probes = np.vstack((np.zeros((1, 6)), np.random.default_rng(732).uniform(-10, 10, (512, 6)))).astype(np.float32)
checks = []
for seed in (835, 836):
    run = root / f'ppo_seed{seed}'
    model = PPO.load(run / 'best_policy.zip', device='cpu')
    actor = NumpyPPO(json.loads((run / 'best_policy.json').read_text()))
    actual = model.predict(probes, deterministic=True)[0]
    expected = np.array([actor.act(observation) for observation in probes])
    error = float(np.max(np.abs(actual - expected)))
    assert error <= 2e-6, error
    optimizer_ids = {id(parameter) for group in model.policy.optimizer.param_groups for parameter in group['params']}
    assert all(id(parameter) in optimizer_ids for parameter in model.policy.action_net.parameters())
    checks.append(dict(seed=seed,selected_model_training_steps=model.num_timesteps,
                       probes=len(probes),maximum_absolute_action_difference=error,
                       tolerance=2e-6,output_layer_in_optimizer=True))
original = ast.parse((root / 'ppo_helper_original.py.txt').read_text())
original_class = next(node for node in original.body if isinstance(node, ast.ClassDef) and node.name == 'TanhMeanPolicy')
published_class = ast.parse(inspect.getsource(ppo_refine.TanhMeanPolicy)).body[0]
assert ast.dump(original_class) == ast.dump(published_class)
result = dict(status='passed',model_module=ppo_refine.__file__,
              published_class_matches_training_class=True,checks=checks)
(root / 'publication_model_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
