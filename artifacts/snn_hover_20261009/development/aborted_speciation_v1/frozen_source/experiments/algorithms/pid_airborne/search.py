"""Documented population search over PID gains, not a learned neural policy."""
import argparse,copy,itertools,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .core import init_worker,evaluate
from ..hover9.plant import prepare
from ..neat_imu.train import write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',choices=['screen','evolve'],default='screen')
    parser.add_argument('--generations',type=int,default=20)
    args=parser.parse_args();root=args.root
    if not root.exists():prepare(root,Path('artifacts/hover9_pid_20261008'))
    started=time.monotonic()
    seeds=[310100000,310100001,310100002,310100003]
    with ProcessPoolExecutor(max_workers=8,initializer=init_worker,initargs=(str(root),)) as pool:
        if args.stage=='screen':
            specs=[]
            for kp,kd,tau,sign in itertools.product([.1,.4,1.2],[.003,.02,.1],[.005,.04,.1],[-1,1]):
                specs.append(dict(kp=[sign*kp]*2,kd=[sign*kd]*2,sensor_tau=tau))
            for kp,kd in itertools.product([.1,.5,2.,5.],[.0002,.001,.005,.02]):
                specs.append(dict(kp=[0,0],kd=[0,0],servo_kp=kp,servo_kd=kd))
                specs.append(dict(kp=[.3,.3],kd=[.02,.02],servo_kp=kp,servo_kd=kd,sensor_tau=.04))
            rows=list(pool.map(evaluate,[(p,seeds,2.,1.) for p in specs]))
            rows.sort(key=lambda r:r['score'],reverse=True)
            write_json(root/'screen.json',rows)
            print(json.dumps(dict(stage='screen',wall=time.monotonic()-started,best=rows[:3])),flush=True)
            return
        records=json.loads((root/'screen.json').read_text())
        population=[r['parameters'] for r in records[:12]]
        rng=np.random.default_rng(1078);history=[]
        for gen in range(args.generations):
            candidates=copy.deepcopy(population)
            while len(candidates)<72:
                p=copy.deepcopy(population[rng.integers(len(population))])
                for name,base,scale,limit in [('kp',[.3,.3],.12,8.),('kd',[.02,.02],.006,1.),('ki',[0,0],.02,1.)]:
                    x=np.array(p.get(name,base))*np.exp(rng.normal(0,.25,2))+rng.normal(0,scale,2)
                    p[name]=np.clip(x,-limit,limit).tolist()
                p['trim']=(np.clip(np.array(p.get('trim',[.01366,.00088]))+rng.normal(0,.003,2),-.15,.15)).tolist()
                for key,base,lo,hi in [('sensor_tau',.02,.001,.2),('derivative_tau',.01,.001,.15),
                                       ('action_tau',.02,.001,.1),('servo_kp',.5,.05,5.),('servo_kd',.005,.0001,.02)]:
                    p[key]=float(np.clip(p.get(key,base)*np.exp(rng.normal(0,.25)),lo,hi))
                p['motor']=float(np.clip(p.get('motor',3461.36)+rng.normal(0,40),3115.23,3807.49))
                if gen>=3:
                    p['height_gains']=np.clip(np.array(p.get('height_gains',[200,100,0]))+
                        rng.normal(0,[70,30,15]),[0,0,0],[1500,800,500]).tolist()
                candidates.append(p)
            seconds=2. if gen<5 else 4. if gen<12 else 8.
            rows=list(pool.map(evaluate,[(p,seeds,seconds,2.) for p in candidates]))
            rows.sort(key=lambda r:r['score'],reverse=True)
            population=[r['parameters'] for r in rows[:12]]
            write_json(root/f'generation_{gen:03d}.json',rows)
            row=dict(generation=gen,wall=time.monotonic()-started,best=rows[0])
            history.append(row);write_json(root/'history.json',history);write_json(root/'best.json',rows[0])
            print(json.dumps(row),flush=True)

if __name__=='__main__':main()
