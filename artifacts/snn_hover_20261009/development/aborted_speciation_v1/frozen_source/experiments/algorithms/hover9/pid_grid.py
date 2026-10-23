"""Structured gain/phase diagnostic, independent of evolutionary search."""
import argparse,itertools,json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from .pid_search import init,evaluate
from ..neat_imu.activation import cases
from ..neat_imu.train import write_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    specs=[]
    for kp,kd,tau in itertools.product((.1,.3,1.,3.),(-.1,-.03,-.01,0.,.01,.03,.1),(.002,.01,.03)):
        specs.append([kp,kp,kd,kd,0.,0.,.01366,.00088,tau])
    batch=cases([301010002,301010003],2.)
    for c in batch:c['degrees']=1.
    with ProcessPoolExecutor(max_workers=6,initializer=init,initargs=(str(a.root),)) as pool:
        records=[dict(parameters=s,**r) for s,r in zip(specs,pool.map(evaluate,[(s,batch) for s in specs]))]
    records.sort(key=lambda r:r['score'],reverse=True)
    write_json(a.root/'pid_grid.json',records)
    print(json.dumps(records[:3],indent=2))
