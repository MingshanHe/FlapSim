"""Finite common-budget evolution; complete generations only at checkpoints."""
import argparse,gzip,json,pickle,random,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import neat,numpy as np
from .evolution import configuration,initialize,export,METHODS
from .environment import case,horizon
from .common import write,compressed,evaluate,validate,selection,benchmark,plateau
from .networks import complexity

def train(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False);random.seed(args.seed)
    cfg=configuration(out,args.method,args.population);pop=neat.Population(cfg);initialize(pop,cfg,args.seed)
    write(out/'arguments.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    state=dict(method=args.method,seed=args.seed,status='training',training_steps=0,allocated_steps=0,validation_steps=0,budget=args.steps)
    history=[];best=None;generation=0;start=time.monotonic();next_validation=51200
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        def validation(spec,gid,gen):
            nonlocal best
            val,rows=validate(pool,spec,'residual','height',args.seed,.1 if args.smoke else 8.)
            state['validation_steps']+=val['steps'];write(out/f'validation_{state["allocated_steps"]:07d}.json',rows)
            if best is None or selection(val)>selection(best):
                best=val;write(out/'best_policy.json',spec)
                write(out/'selection.json',dict(generation=gen,genome_id=gid,training_steps=state['training_steps'],allocated_steps=state['allocated_steps'],validation=val,complexity=complexity(spec)))
            return val
        zero=pop.population[min(pop.population)];initial=export(zero,args.method)
        validation(initial,zero.key,-1)
        history.append(dict(generation=-1,training_steps=0,allocated_steps=0,best_validation=best,validation=best))
        compressed(out/'initial_population.json.gz',[export(g,args.method) for g in pop.population.values()])
        def fitness(genomes,config):
            nonlocal generation,next_validation
            seconds=.05 if args.smoke else horizon(state['allocated_steps'])
            seed=360000000+args.seed*1000+generation
            c=case(seed,seconds,'nominal' if generation%3==0 else 'train',generation%8)
            specs=[export(g,args.method) for _,g in genomes]
            rows=list(pool.map(evaluate,[(s,c,'residual','height',False) for s in specs]))
            snaps=[]
            for (gid,g),s,r in zip(genomes,specs,rows):
                g.fitness=r['return_sum']/seconds-.00002*complexity(s)['connections']
                snaps.append(dict(id=gid,parents=getattr(g,'parents',[]),species=pop.species.get_species_id(gid),fitness=g.fitness,
                    mutation=getattr(g,'mutation_event',{}),policy=s,episode=r))
            champion=max(snaps,key=lambda x:x['fitness']);state['training_steps']+=sum(r['steps'] for r in rows)
            state['allocated_steps']+=round(seconds*100)*len(genomes)
            entry=dict(generation=generation,training_steps=state['training_steps'],allocated_steps=state['allocated_steps'],case=c,
                champion_id=champion['id'],fitness_max=champion['fitness'],fitness_mean=float(np.mean([s['fitness'] for s in snaps])),
                species=len(pop.species.species),**complexity(champion['policy']))
            compressed(out/f'generation_{generation:04d}.json.gz',dict(generation=generation,individuals=snaps,species_merges=pop.species.merge_events))
            if state['allocated_steps']>=next_validation or state['allocated_steps']>=args.steps:
                entry['validation']=validation(champion['policy'],champion['id'],generation);entry['best_validation']=best
                next_validation+=51200
            history.append(entry);write(out/'history.json',history)
            state.update(generation=generation,best_validation=best,wall_seconds=time.monotonic()-start)
            write(out/'status.json',state);print(json.dumps(state),flush=True);generation+=1
        while state['allocated_steps']+args.population*round((.05 if args.smoke else horizon(state['allocated_steps']))*100)<=args.steps:
            pop.run(fitness,1)
            with gzip.open(out/'checkpoint.pkl.gz','wb') as f:pickle.dump((pop,random.getstate(),state),f)
            if state['allocated_steps'] in (256000,512000,1024000) or args.smoke:
                checkpoint=out/f'budget_{state["allocated_steps"]}';checkpoint.mkdir(exist_ok=True)
                for filename in ['best_policy.json','selection.json']:(checkpoint/filename).write_bytes((out/filename).read_bytes())
                write(checkpoint/'accounting.json',state)
    state.update(status='completed',wall_seconds=time.monotonic()-start,validation_plateau=plateau(history),
        stop_reason='predeclared_complete_generation_budget',inference=benchmark(json.loads((out/'best_policy.json').read_text())))
    write(out/'results.json',state);write(out/'status.json',state)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int,default=1024000);p.add_argument('--population',type=int,default=64)
    p.add_argument('--workers',type=int,default=16);p.add_argument('--smoke',action='store_true');train(p.parse_args())

if __name__=='__main__':main()
