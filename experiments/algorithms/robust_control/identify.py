"""Development-only physical direction/gain checks before command training."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from .environment import ResearchEnv, scenarios
from ..snn_pid_compare.common import write, sha


def one(task):
    source, c, output = task
    env = ResearchEnv(source, mode='pid', task='velocity')
    env.configure(c)
    env.reset()
    angles = []
    while True:
        _,_,done,truncated,info = env.step(np.zeros(3))
        angles.append(env.last_reading[:3].copy())
        if done or truncated:
            break
    rows = np.asarray(env.velocity_rows)
    angles = np.asarray(angles)
    settled = rows[:,0] >= 4.
    result = info['episode']
    result.update(mean_velocity=rows[settled,1:4].mean(0).tolist(),
                  mean_attitude_deg=np.rad2deg(angles[settled].mean(0)).tolist())
    target = Path(output)/f"{c['probe_name']}.npz"
    np.savez_compressed(target, rows=rows, measured_rpy=angles)
    return result


def run(source, output, workers, sweep):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tasks = []
    if sweep:
        for kp,ki in [(0.,0.),(.02,.01),(.05,.025),(.1,.05)]:
            for i,c in enumerate(scenarios('development',8,12.,task='velocity')):
                c.update(velocity_kp=kp,velocity_ki=ki,probe_name=f'kp{kp}_ki{ki}_case{i}')
                tasks.append((source,c,str(output)))
    else:
        for phase in (0,2,4,6):
            for tag,ref in [('zero',[0,0]),('roll_plus',[.08,0]),('roll_minus',[-.08,0]),
                            ('pitch_plus',[0,.08]),('pitch_minus',[0,-.08])]:
                c = scenarios('development',8,8.,task='velocity')[phase]
                c.update(velocity_command=[0,0,0],attitude_reference_override=ref,
                         probe_name=f'phase{phase}_{tag}')
                tasks.append((source,c,str(output)))
    from . import environment
    write(output/'plan.json',dict(scope='development; no neural training or final test',
                                 environment_sha256=sha(environment.__file__),
                                 cases=[t[1] for t in tasks]))
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(one,tasks))
    write(output/'results.json',rows)
    for r in rows:
        print(r['case']['probe_name'],r['mean_velocity'],r['mean_attitude_deg'],r['success'],flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--sweep',action='store_true')
    a=p.parse_args()
    run(a.source,a.output,a.workers,a.sweep)
