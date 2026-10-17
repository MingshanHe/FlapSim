import time
import numpy as np
from ..topology_hover.common import write,compressed,summary,selection
from ..topology_hover.evaluate import record_criteria
from .environment import HoverEnv,case,rollout,Zero
from .networks import policy,complexity

ENVS={}

def evaluate(task):
    spec,c,mode,sensor,trace=task;key=(mode,sensor)
    if key not in ENVS:ENVS[key]=HoverEnv(mode=mode,sensor_mode=sensor)
    env=ENVS[key];p=Zero() if spec is None else policy(spec)
    r=record_criteria(rollout(env,p,c,trace),env)
    if hasattr(p,'spike_count'):
        r.update(spikes=int(p.spike_count.sum()),hidden_spike_hz=(p.spike_count[p.hidden]/(p.ticks*.001)).tolist(),
            silent_hidden_fraction=float(np.mean(p.spike_count[p.hidden]==0)) if len(p.hidden) else 1.)
    return r

def validation_cases(seed,seconds=8.):
    return [case(410000000+seed*10+i,seconds,'nominal' if i%2==0 else 'stress',i) for i in range(8)]

def validate(pool,spec,mode,sensor,seed,seconds=8.):
    rows=list(pool.map(evaluate,[(spec,c,mode,sensor,False) for c in validation_cases(seed,seconds)]))
    return summary(rows),rows

def benchmark(spec,repeats=1000):
    p=policy(spec);x=np.zeros(spec['inputs'],dtype=np.float32)
    for _ in range(100):p.act(x)
    times=[]
    for _ in range(5):
        t=time.perf_counter()
        for _ in range(repeats):p.act(x)
        times.append((time.perf_counter()-t)*1e6/repeats)
    return dict(**complexity(spec),numpy_inference_us_median=float(np.median(times)),implementation='CPU NumPy; includes all neural substeps; no hardware energy inference')

def plateau(history):
    vals=[r['best_validation'] for r in history if 'best_validation' in r]
    if len(vals)<6:return False
    a,b=vals[-4],vals[-1]
    return a['success_rate']==b['success_rate'] and abs(b['mean_return']-a['mean_return'])<.002*max(1,abs(a['mean_return'])) and abs(b['tilt_rms_deg']-a['tilt_rms_deg'])<.01*max(1,a['tilt_rms_deg'])
