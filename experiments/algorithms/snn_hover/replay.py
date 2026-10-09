"""Replay an exported temporal/ANN/PPO controller using relocated model assets."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from ..pid_airborne.results import portable
from ..topology_hover.environment import DEFAULT_SOURCE
from .environment import HoverEnv,Zero,case,rollout
from .networks import policy
from .common import record_criteria,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path);p.add_argument('--source',type=Path,default=Path(DEFAULT_SOURCE))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=430000000)
    p.add_argument('--seconds',type=float,default=20.);p.add_argument('--condition',choices=['nominal','stress'],default='nominal')
    p.add_argument('--phase',type=int,choices=range(8),default=0);p.add_argument('--tilt',type=float);p.add_argument('--direction',type=float,default=0.)
    a=p.parse_args();assert a.seconds>0
    spec=json.loads(a.policy.read_text()) if a.policy else None
    if spec:assert spec['inputs']==11,'This replay uses the frozen11-D interface'
    a.output.mkdir(parents=True,exist_ok=False);c=case(a.seed,a.seconds,a.condition,a.phase)
    if a.tilt is not None:c.update(tilt_deg=a.tilt,direction_deg=a.direction)
    with portable(a.source) as stage:
        shutil.copyfile(a.source/'selected_pid.json',stage/'selected_pid.json')
        env=HoverEnv(source=stage,mode='residual' if spec else 'pid');net=policy(spec) if spec else Zero()
        if hasattr(net,'capture'):net.capture=True
        r=record_criteria(rollout(env,net,c,True),env);trace=np.asarray(r.pop('trace'));r['reference_position']=env.air.start.tolist()
        np.savez_compressed(a.output/'trajectory.npz',trace=trace,internal=np.asarray(getattr(net,'internal_trace',[])))
        write(a.output/'result.json',dict(policy=str(a.policy) if spec else 'PID',scope='Additional replay, not part of the frozen benchmark',episode=r))
        print(json.dumps(r,ensure_ascii=False))
if __name__=='__main__':main()
