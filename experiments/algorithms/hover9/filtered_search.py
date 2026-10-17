"""PID tuning after anti-alias filtering at the physics sampling rate."""
import argparse,json,itertools,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from . import pid_search,compensated_search
from .filtered_plant import FilteredPlant
from ..neat_imu.activation import cases
from ..neat_imu.train import write_json

def init(root):
    pid_search.ENV=FilteredPlant(root,attitude_tau=.01)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    output=a.root/'filtered_search';output.mkdir(exist_ok=False)
    candidates=[]
    for kp,kd in itertools.product((.1,.3,1.,3.),(-.1,-.03,-.01,.003,.01,.03,.1)):
        candidates.append([kp,kp,kd,kd,0,0,.01366,.00088,.002])
    batch=cases([301050000,301050001],3.)
    for c in batch:c['degrees']=2.
    rng=np.random.default_rng(918);history=[];started=time.monotonic()
    with ProcessPoolExecutor(max_workers=6,initializer=init,initargs=(str(a.root),)) as pool:
        results=list(pool.map(pid_search.evaluate,[(s,batch) for s in candidates]))
        order=np.argsort([r['score'] for r in results])[::-1]
        population=np.array([candidates[i] for i in order[:8]])
        write_json(output/'grid.json',[dict(parameters=candidates[i],**results[i]) for i in order])
        print(json.dumps(dict(stage='filtered_grid',parameters=candidates[order[0]],best=results[order[0]])),flush=True)
        for gen in range(16):
            candidates=[]
            for i in range(48):
                x=population[rng.integers(8)].copy();x[:4]*=np.exp(rng.normal(0,.4,4))
                if i%8==0:x[rng.integers(4)]*=-1
                x[4:6]+=rng.normal(0,.03,2);x[6:8]+=rng.normal(0,.006,2)
                x[8]=np.clip(x[8]*np.exp(rng.normal(0,.4)),.001,.08)
                x[:2]=np.clip(x[:2],-10,10);x[2:4]=np.clip(x[2:4],-1,1)
                x[6:8]=np.clip(x[6:8],-.1,.1);candidates.append(x.tolist())
            candidates[:8]=population.tolist()
            results=list(pool.map(pid_search.evaluate,[(s,batch) for s in candidates]))
            order=np.argsort([r['score'] for r in results])[::-1]
            population=np.array([candidates[i] for i in order[:8]])
            row=dict(generation=gen,parameters=candidates[order[0]],best=results[order[0]],wall_seconds=time.monotonic()-started)
            history.append(row);write_json(output/'history.json',history);write_json(output/'best.json',row);print(json.dumps(row),flush=True)
        validation=cases(range(301060000,301060008),8.)
        val=list(pool.map(pid_search.evaluate,[(s,validation) for s in population.tolist()]))
        index=int(np.argmax([r['score'] for r in val]))
        write_json(a.root/'filtered_pid.json',dict(parameters=population[index].tolist(),validation=val[index],all_validation=val))

if __name__=='__main__':main()
