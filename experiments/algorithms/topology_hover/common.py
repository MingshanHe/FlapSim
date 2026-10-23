import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from .environment import HoverEnv,rollout,case,Zero
from .networks import NumpyPolicy,complexity

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False));os.replace(tmp,path)

def compressed(path,value):
    with gzip.open(path,'wt',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False)

ENVS={}
def evaluate(task):
    spec,c,mode,sensor,trace=task
    key=(mode,sensor)
    if key not in ENVS:ENVS[key]=HoverEnv(mode=mode,sensor_mode=sensor)
    policy=Zero() if spec is None else NumpyPolicy(spec)
    return rollout(ENVS[key],policy,c,trace)

def summary(rows):
    return dict(episodes=len(rows),success_rate=float(np.mean([r['success'] for r in rows])),
        mean_return=float(np.mean([r['return_sum'] for r in rows])),mean_elapsed=float(np.mean([r['elapsed'] for r in rows])),
        tilt_rms_deg=float(np.mean([r['tilt_rms_deg'] for r in rows])),height_rmse_m=float(np.mean([r['height_rmse_m'] for r in rows])),
        steps=sum(r['steps'] for r in rows),minimum_clearance_m=min(r['minimum_clearance_m'] for r in rows))

def validate(pool,spec,mode,sensor,seed,seconds=8.):
    cases=[case(321000000+seed*10+i,seconds,'nominal' if i%2==0 else 'stress',i) for i in range(4)]
    rows=list(pool.map(evaluate,[(spec,c,mode,sensor,False) for c in cases]))
    return summary(rows),rows

def selection(s):return (s['success_rate'],s['mean_return'],-s['tilt_rms_deg'])

def benchmark(spec,repeats=2000):
    p=NumpyPolicy(spec);x=np.zeros(spec['inputs'],dtype=np.float32)
    for _ in range(100):p.act(x)
    times=[]
    for _ in range(5):
        start=time.perf_counter()
        for i in range(repeats):p.act(x)
        times.append((time.perf_counter()-start)/repeats*1e6)
    return dict(**complexity(spec),numpy_inference_us_median=float(np.median(times)),implementation='CPU NumPy dense evaluation of the exported actor; not an MCU measurement')

def freeze(root):
    paths=sorted(Path('experiments/algorithms/topology_hover').glob('*'))
    manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    write(Path(root)/'source_hashes.json',manifest)
    return manifest
