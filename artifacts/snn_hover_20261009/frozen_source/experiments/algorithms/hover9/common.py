"""Shared protocol, portable policies and held-out metrics."""
import json
from pathlib import Path
import numpy as np
from .environment import HistoryEnv
from ..neat_imu.compare_env import NumpyPPO,summarize,selection_key
from ..neat_imu.residual import ResidualPolicy


class ZeroPolicy:
    def reset(self,obs):pass
    def act(self,obs):return np.zeros(3)


def make_policy(payload):
    if payload.get('format')=='zero_residual':return ZeroPolicy()
    if payload.get('format')=='flapsim-ppo-history-v1':return NumpyPPO(payload)
    return ResidualPolicy(payload)


ENV=None
def init_worker(root):
    global ENV
    ENV=HistoryEnv(root)


def evaluate(task):
    payload,batch,trace=task
    policy=make_policy(payload);results=[]
    for case in batch:
        ENV.ablation=case.get('ablation')
        ENV.configure_case(case);obs,initial=ENV.reset(seed=case['seed']);policy.reset(obs)
        rows=[]
        while True:
            action=policy.act(obs)
            obs,reward,done,truncated,info=ENV.step(action)
            if trace:
                rows.append([info['elapsed'],*np.rad2deg(info['rpy']),info['alpha'],info['beta'],
                             *info['position'],info['motor_command'],*info['relative_position'],
                             *info['estimated_position'],*info['estimated_velocity'],info['observer_error_m']])
            if done or truncated:break
        results.append(dict(seed=case['seed'],case=case,steps=ENV.steps,phase=initial['phase_index'],
            elapsed=info['elapsed'],success=info['hover_success'],termination=info['termination_reason'],
            tracking_rmse_m=info['tracking_rmse'],tilt_rmse_deg=info['tilt_rmse_deg'],
            discounted_return=info['discounted_return'],return_sum=info['return_sum'],
            saturation_fraction=info['saturation_fraction'],observer_rmse_m=info['observer_rmse_m'],
            observer_final_error_m=info['observer_error_m'],trace=rows))
    return results


def summary(rows):
    return dict(summarize(rows),mean_observer_rmse_m=float(np.mean([r['observer_rmse_m'] for r in rows])),
                mean_observer_final_error_m=float(np.mean([r['observer_final_error_m'] for r in rows])))

