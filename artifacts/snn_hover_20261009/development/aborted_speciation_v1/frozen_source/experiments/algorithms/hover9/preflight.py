"""Sensor numerical checks and local open-loop actuator diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
from ..neat_imu.activation import cases
from ..neat_imu.environment import rotation_to_rpy
from ..neat_imu.train import write_json
from .plant import Plant,PID
from .imu import attach


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root
    env=Plant(root);sensor=attach(env.sim)
    results=[]
    for tau in (.0002,.002,.01):
        case=cases([301010000],.5)[0];case['degrees']=0.
        env.configure_case(case);obs,_=env.reset(seed=case['seed']);sensor.reset_imu(tau=tau)
        initial=env.sim.data.xpos[env.body].copy(); rows=[]
        while True:
            command=np.r_[env.reference_angles/.3,0.]
            obs,_,done,truncated,info=env.step(command)
            estimate=np.array(sensor.imu.position)
            truth=info['position']-initial
            rows.append(dict(time=info['elapsed'],estimate=estimate.tolist(),truth=truth.tolist(),
                error=float(np.linalg.norm(estimate-truth))))
            if done or truncated:break
        results.append(dict(tau=tau,elapsed=info['elapsed'],final_error=rows[-1]['error'],rows=rows))
    env.close()
    env=Plant(root);responses=[]
    for axis in (0,1):
        for delta in (-.03,0.,.03):
            case=cases([301010000],.3)[0];case['degrees']=0.
            env.configure_case(case);obs,_=env.reset(seed=case['seed']);rows=[]
            while True:
                command=np.r_[env.reference_angles/.3,0.];command[axis]+=delta/.3
                obs,_,done,truncated,info=env.step(command)
                rows.append(dict(time=info['elapsed'],base_rpy=info['rpy'].tolist(),
                    upper_rpy=rotation_to_rpy(env.sim.data.body('thrust_unit').xmat).tolist(),
                    actual_gimbal=[float(env.sim.data.joint(n).qpos[0]) for n in ('pitch_joint','roll_joint')]))
                if done or truncated:break
            responses.append(dict(axis=axis,delta=delta,rows=rows))
    env.close()
    write_json(root/'sensor_preflight.json',dict(sensor_runs=results,actuator_responses=responses))
    print(json.dumps(dict(sensor=[{k:v for k,v in r.items() if k!='rows'} for r in results],
                         response=[dict(axis=r['axis'],delta=r['delta'],last=r['rows'][-1]) for r in responses]),indent=2))

if __name__=='__main__':main()
