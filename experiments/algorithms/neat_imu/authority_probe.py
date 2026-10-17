"""Open-loop free-flight actuator authority diagnostic; no model changes."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
from .activation import SignedHoverEnv, cases
from .train import write_json


def measure(command):
    trim=json.loads(Path('artifacts/neat_imu_trim_20261007/calibration_fine/trim.json').read_text())
    env=SignedHoverEnv(trim,sensor_inputs=6)
    case=cases([894030000])[0]; case['degrees']=0.
    env.configure_case(case); obs,_=env.reset(seed=case['seed'])
    rows=[]
    while True:
        obs,_,terminated,truncated,info=env.step(command)
        rows.append(dict(t=info['elapsed'],rpy_deg=np.rad2deg(info['rpy']).tolist(),gyro_rad_s=(obs[3:]*10).tolist(),
                         actual_gimbal=[float(env.sim.data.joint(name).qpos[0]) for name in ('pitch_joint','roll_joint')],
                         xyz=info['position'].tolist(),alpha=info['alpha'],beta=info['beta']))
        if terminated or truncated: break
    env.close()
    return dict(command=command,elapsed=info['elapsed'],termination=info['termination_reason'],rows=rows)


if __name__=='__main__':
    commands=[[a,0.,0.] for a in (-1.,-.1,0.,.1,1.)]+[[0.,b,0.] for b in (-1.,-.1,.1,1.)]
    with ProcessPoolExecutor(max_workers=3) as pool: results=list(pool.map(measure,commands))
    write_json(Path('artifacts/neat_imu_activation_20261007/authority.json'),results)
    for row in results:
        print(json.dumps(dict(command=row['command'],elapsed=row['elapsed'],at_100ms=row['rows'][9],last=row['rows'][-1])),flush=True)
