"""Develop a command-history compensated PID before training the outer loop."""
import argparse,itertools,json,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from .plant import Plant
from .compensated_pid import CompensatedPID
from ..neat_imu.activation import cases
from ..neat_imu.train import write_json

ENV=None
def init(root):
    global ENV
    ENV=Plant(root)

def evaluate(task):
    parameters,batch=task;rows=[]
    pid=CompensatedPID(parameters,ENV.reference_rpy[:2])
    for case in batch:
        ENV.configure_case(case);obs,_=ENV.reset(seed=case['seed']);pid.reset(obs);score=0.
        while True:
            obs,_,done,truncated,info=ENV.step(pid.act(obs))
            tilt=np.arccos(np.clip(ENV.sim.data.xmat[ENV.body,8],-1,1))
            score+=.01*(.2+np.exp(-(tilt/.3)**2))
            if done or truncated:break
        rows.append(dict(seed=case['seed'],score=score,elapsed=info['elapsed'],termination=info['termination_reason'],
                         tilt_rmse_deg=info['tilt_rmse_deg'],tracking_rmse_m=info['tracking_rmse']))
    return dict(score=float(np.mean([r['score'] for r in rows])),episodes=rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    output=a.root/'compensated_search';output.mkdir(exist_ok=False)
    specs=[]
    for kp,kd,kappa in itertools.product((.3,1.,3.),(.003,.01,.03,.1),(.4,.7,.9)):
        specs.append([kp,kp,kd,kd,0,0,.01366,.00088,.008,kappa,kappa])
    # A zero kappa recovers ordinary PID, useful for a paired baseline.
    specs.append([.3,.3,.01,.01,0,0,.01366,.00088,.008,0,0])
    batch=cases([301030000,301030001],2.)
    for c in batch:c['degrees']=1.
    rng=np.random.default_rng(9021);history=[];started=time.monotonic();steps=0
    with ProcessPoolExecutor(max_workers=6,initializer=init,initargs=(str(a.root),)) as pool:
        records=[dict(parameters=s,**r) for s,r in zip(specs,pool.map(evaluate,[(s,batch) for s in specs]))]
        records.sort(key=lambda r:r['score'],reverse=True)
        write_json(output/'grid.json',records)
        print(json.dumps(dict(stage='grid',best=records[0])),flush=True)
        population=np.array([r['parameters'] for r in records[:8]])
        for generation in range(20):
            # Preserve signed gains, and search actuator response compensation.
            candidates=[]
            for i in range(64):
                x=population[rng.integers(len(population))].copy()
                x[:4]*=np.exp(rng.normal(0,.4,4))
                if i%8==0:x[rng.integers(4)]*=-1
                x[4:6]+=rng.normal(0,.03,2);x[6:8]+=rng.normal(0,.006,2)
                x[8]=np.clip(x[8]*np.exp(rng.normal(0,.3)),.001,.08)
                x[9:11]=np.clip(x[9:11]+rng.normal(0,.07,2),0,1.2)
                x[:2]=np.clip(x[:2],-10,10);x[2:4]=np.clip(x[2:4],-1,1)
                x[6:8]=np.clip(x[6:8],-.1,.1);candidates.append(x.tolist())
            candidates[:len(population)]=population.tolist()
            results=list(pool.map(evaluate,[(s,batch) for s in candidates]))
            steps+=sum(round(e['elapsed']*100) for r in results for e in r['episodes'])
            order=np.argsort([r['score'] for r in results])[::-1]
            population=np.array([candidates[i] for i in order[:8]])
            row=dict(generation=generation,parameters=candidates[order[0]],best=results[order[0]],
                     steps=steps,wall_seconds=time.monotonic()-started)
            history.append(row);write_json(output/'history.json',history);write_json(output/'best.json',row)
            print(json.dumps(row),flush=True)
        validation=cases(range(301040000,301040008),8.)
        rows=list(pool.map(evaluate,[(s,validation) for s in population.tolist()]))
        index=int(np.argmax([r['score'] for r in rows]))
        write_json(a.root/'compensated_pid.json',dict(parameters=population[index].tolist(),validation=rows[index],all_validation=rows))

if __name__=='__main__':main()
