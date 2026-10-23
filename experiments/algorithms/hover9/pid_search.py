"""Development-only PID gain search. Final cases never select gains."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
from ..neat_imu.activation import cases
from ..neat_imu.train import write_json
from .plant import Plant,PID,prepare

ENV=None
def init(root):
    global ENV
    ENV=Plant(root)

def evaluate(task):
    parameters,batch=task; rows=[]
    pid=PID(parameters,ENV.reference_rpy[:2])
    for case in batch:
        ENV.configure_case(case);obs,_=ENV.reset(seed=case['seed']);pid.reset(obs)
        score=0.
        while True:
            obs,_,done,truncated,info=ENV.step(pid.act(obs))
            tilt=np.arccos(np.clip(ENV.sim.data.xmat[ENV.body,8],-1,1))
            score+=.01*(.2+np.exp(-(tilt/.3)**2))
            if done or truncated:break
        rows.append(dict(seed=case['seed'],score=score,elapsed=info['elapsed'],termination=info['termination_reason'],
                         tilt_rmse_deg=info['tilt_rmse_deg'],tracking_rmse_m=info['tracking_rmse']))
    return dict(score=float(np.mean([r['score'] for r in rows])),episodes=rows)

def decode(x):
    x=np.asarray(x)
    p=x.copy()
    p[:2]=np.sign(x[:2])*np.expm1(3*np.abs(x[:2]))*.2
    p[2:4]=np.sign(x[2:4])*np.expm1(3*np.abs(x[2:4]))*.01
    p[4:6]=x[4:6]*.3
    p[6]=.01366+x[6]*.06;p[7]=x[7]*.04
    p[8]=.004*np.exp(3*(x[8]+1)/2)
    return p.tolist()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--generations',type=int,default=24)
    args=parser.parse_args();prepare(args.root,args.source)
    output=args.root/'pid_search';output.mkdir(exist_ok=False)
    rng=np.random.default_rng(18041);population=rng.uniform(-1,1,(64,9))
    population[0]=0.;population[0,8]=-.4
    history=[];started=time.monotonic();steps=0
    write_json(output/'protocol.json',dict(population=64,generations=args.generations,seed=18041,
        development_seeds=[301010000,301010001],validation_seeds=list(range(301020000,301020008)),
        fitness='integrated attitude quality; fixed hover motor; 2 s then 4 s development',
        parameters=['kp_roll','kp_pitch','kd_roll','kd_pitch','ki_roll','ki_pitch','alpha_trim','beta_trim','filter_tau']))
    with ProcessPoolExecutor(max_workers=8,initializer=init,initargs=(str(args.root),)) as pool:
        for gen in range(args.generations):
            horizon=2. if gen<12 else 4.
            batch=cases([301010000,301010001],horizon)
            for c in batch:c['degrees']=2.
            results=list(pool.map(evaluate,[(decode(x),batch) for x in population]))
            steps+=sum(round(e['elapsed']*100) for r in results for e in r['episodes'])
            order=np.argsort([r['score'] for r in results])[::-1]
            row=dict(generation=gen,best=results[int(order[0])],parameters=decode(population[order[0]]),
                     control_steps=steps,wall_seconds=time.monotonic()-started)
            history.append(row);write_json(output/'history.json',history);write_json(output/'best.json',row)
            print(json.dumps(row),flush=True)
            elite=population[order[:10]].copy()
            spread=.25 if gen<12 else .12
            new=np.stack([elite[rng.integers(len(elite))]+rng.normal(0,spread,9) for _ in range(64)])
            new[:10]=elite;new[-8:]=rng.uniform(-1,1,(8,9));population=np.clip(new,-1,1)
        # Validate diverse finalists; selection uses validation only.
        batch=cases(range(301020000,301020008),8.)
        finalists=[]
        for x in elite:
            finalists.append((decode(x),batch))
        val=list(pool.map(evaluate,finalists));order=np.argsort([r['score'] for r in val])[::-1]
        winner=dict(parameters=finalists[int(order[0])][0],validation=val[int(order[0])],
                    all_validation=val,development_steps=steps,wall_seconds=time.monotonic()-started)
        write_json(args.root/'pid.json',winner)
        print(json.dumps(dict(stage='pid_complete',**winner)),flush=True)

if __name__=='__main__':main()
