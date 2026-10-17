"""Replay a saved actor after relocating mesh paths in a temporary asset copy."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np

from ..pid_airborne.results import portable
from .environment import DEFAULT_SOURCE,HoverEnv,Zero,case,rollout
from .networks import NumpyPolicy
from .common import write


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--policy',type=Path,help='Saved best_policy.json; omit for the PID baseline')
    p.add_argument('--source',type=Path,default=Path(DEFAULT_SOURCE))
    p.add_argument('--output',type=Path,required=True,help='New directory for metrics and recorded physical states')
    p.add_argument('--seed',type=int,default=339900200)
    p.add_argument('--seconds',type=float,default=20.)
    p.add_argument('--condition',choices=['nominal','stress'],default='nominal')
    p.add_argument('--phase',type=int,choices=range(8),default=0)
    p.add_argument('--mode',choices=['residual','direct'],default='residual')
    a=p.parse_args()
    if a.seconds<=0:raise ValueError('seconds must be positive')
    spec=json.loads(a.policy.read_text()) if a.policy else None
    if spec and spec['inputs']!=11:raise ValueError('This replay uses the main experiment\'s 11-D height-aided interface')
    if a.policy and (a.policy.parent/'results.json').exists():
        method=json.loads((a.policy.parent/'results.json').read_text())['method']
        mode='direct' if method=='direct_ppo' else 'residual'
    else:mode=a.mode if a.policy else 'pid'
    a.output.mkdir(parents=True,exist_ok=False)
    with portable(a.source) as stage:
        shutil.copyfile(a.source/'selected_pid.json',stage/'selected_pid.json')
        env=HoverEnv(source=stage,mode=mode,sensor_mode='height')
        actor=NumpyPolicy(spec) if spec else Zero()
        result=rollout(env,actor,case(a.seed,a.seconds,a.condition,a.phase),trace=True)
        trajectory=np.asarray(result.pop('trace'))
        result['reference_position']=env.air.start.tolist()
        result['policy']=str(a.policy) if a.policy else 'PID'
        result['mode']=mode
        np.savez_compressed(a.output/'trajectory.npz',trace=trajectory)
        write(a.output/'result.json',result)
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
