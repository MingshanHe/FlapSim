"""Test centroidal roll damping around a frozen PPO, without altering physics."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from flapsim.rl.train import load_run, make_env


def trial(job):
    run, kp, kd, seconds = job
    torch.set_num_threads(1)
    run = Path(run)
    tracking, simulation, metadata = load_run(run)
    monitored = make_env(replace(tracking, randomize=False, episode_seconds=seconds), simulation,
                         metadata['arguments'].get('split_step', False), metadata['arguments'].get('features') or 'full')
    env = monitored.unwrapped
    norm = VecNormalize.load(str(run/'best/vecnormalize.pkl'), DummyVecEnv([lambda: monitored]))
    norm.training = False
    policy = PPO.load(str(run/'best/best_model.zip'), device='cpu')
    obs, _ = env.reset(seed=1)
    am = np.zeros((3, env.sim.model.nv))
    errors = []
    while True:
        action = policy.predict(norm.normalize_obs(obs), deterministic=True)[0]
        m,d = env.sim.model, env.sim.data
        p,v,r = env._state()
        mujoco.mj_angmomMat(m,d,am,env.body)
        a = env.free_dof+3
        omega = np.linalg.solve(am[:,a:a+3],am@d.qvel)
        unit_z = -d.xmat[m.body('thrust_unit').id].reshape(3,3)[:,2]
        target, velocity, acceleration = env.reference.sample(env.steps*tracking.control_dt)
        desired = acceleration-1.5*(p-target)-2.*(v-velocity)
        desired[2] += 9.81
        desired /= np.linalg.norm(desired)
        error = np.cross(unit_z,desired) @ r
        roll = np.clip(-kp*error[0]+kd*omega[0],-.3,.3)
        action[2] = roll/.3
        obs,_,term,trunc,info = env.step(action)
        errors.append(info['position_error'])
        if term or trunc:
            break
    result = dict(kp=kp,kd=kd,seconds=info['elapsed'],rmse=info['tracking_rmse'],
                  reason=info['termination_reason'])
    norm.close()
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',default='runs/hover_refine')
    p.add_argument('--output',type=Path,default=Path('runs/hybrid_probe.json'))
    a=p.parse_args()
    jobs=[(a.run,kp,kd,4.) for kp in (.25,1.) for kd in (-.06,-.02,.02,.06)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=[]
        for row in pool.map(trial,jobs):
            print(json.dumps(row),flush=True)
            rows.append(row)
    a.output.write_text(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
