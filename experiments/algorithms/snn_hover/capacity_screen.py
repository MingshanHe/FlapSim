"""Reproduction-only invariant checks, with no physical observations or rewards."""
import argparse,json,random,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import neat
from .evolution import METHODS,configuration,initialize
from .common import write

def screen(task):
    output,method,seed=task;output.mkdir(parents=True,exist_ok=True);random.seed(seed)
    cfg=configuration(output,method,64);pop=neat.Population(cfg);initialize(pop,cfg,seed);rng=random.Random(seed+827);history=[]
    for _ in range(80):
        old=set(pop.population)
        def fit(genomes,config):
            for gid,g in genomes:g.fitness=rng.random()
        pop.run(fit,1)
        row=dict(generation=pop.generation,species=len(pop.species.species),population=len(pop.population),newborn=len(set(pop.population)-old),merges=len(pop.species.merge_events))
        assert row['species']<=16 and row['population']==64 and row['newborn']>=32
        assert set(pop.species.genome_to_species)==set(pop.population)
        history.append(row)
    r=dict(method=method,seed=seed,status='passed',generations=80,history=history);write(output/'result.json',r);return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=a.root/'development/capacity_screen';out.mkdir(exist_ok=True)
    tasks=[(out/f'{m}_seed{s}',m,s) for m in METHODS for s in range(2181,2186)];start=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=16) as pool:
        for f in as_completed([pool.submit(screen,t) for t in tasks]):
            r=f.result();rows.append(r);print(json.dumps(dict(completed=len(rows),method=r['method'],seed=r['seed'])),flush=True)
    write(out/'summary.json',dict(status='passed',scope='25 independent reproduction-only runs; no physics or reward selection',generations_per_run=80,
        maximum_species=max(h['species'] for r in rows for h in r['history']),minimum_newborn=min(h['newborn'] for r in rows for h in r['history']),
        runs=len(rows),wall_seconds=time.monotonic()-start,physical_actor_steps=0))
if __name__=='__main__':main()
