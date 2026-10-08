"""NEAT with every evaluated genome, its ancestry, and mutation events archived."""
import argparse,copy,gzip,json,math,pickle,random,time,itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import neat
import numpy as np
from .networks import export_genome,complexity
from .environment import case
from .common import write,compressed,evaluate,validate,selection,benchmark

def plain_tanh(x):return math.tanh(x)

class AuditGenome(neat.DefaultGenome):
    def configure_crossover(self,g1,g2,config):
        super().configure_crossover(g1,g2,config)
        self.parents=[g1.key,g2.key]
    def mutate(self,config):
        before={k:(g.weight,g.enabled) for k,g in self.connections.items()};nodes=set(self.nodes)
        super().mutate(config)
        after=set(self.connections)
        self.mutation_event=dict(added_nodes=sorted(set(self.nodes)-nodes),removed_nodes=sorted(nodes-set(self.nodes)),
            added_edges=[list(k) for k in sorted(after-set(before))],removed_edges=[list(k) for k in sorted(set(before)-after)],
            changed_weights=[dict(edge=list(k),before=before[k][0],after=self.connections[k].weight) for k in sorted(after&set(before)) if before[k][0]!=self.connections[k].weight],
            toggled_edges=[dict(edge=list(k),enabled=self.connections[k].enabled) for k in sorted(after&set(before)) if before[k][1]!=self.connections[k].enabled])


def config_file(output,pop,ninputs,fixed=False):
    base=(Path(__file__).resolve().parents[1]/'neat_imu/config.ini').read_text()
    changes={'pop_size':pop,'num_inputs':ninputs,'feed_forward':'False','activation_default':'plain_tanh',
        'activation_options':'plain_tanh','weight_init_stdev':.015,'bias_init_stdev':.005,
        'weight_mutate_power':.04,'weight_mutate_rate':.5,'bias_mutate_power':.015,
        'conn_add_prob':0. if fixed else .30,'conn_delete_prob':0. if fixed else .06,
        'node_add_prob':0. if fixed else .20,'node_delete_prob':0. if fixed else .03,
        'enabled_mutate_rate':0. if fixed else .015,'compatibility_threshold':1.5,'max_stagnation':12}
    lines=[]
    for line in base.splitlines():
        key=line.split('=')[0].strip()
        lines.append(f'{key} = {changes[key]}' if key in changes else line)
    path=output/'neat.ini';path.write_text('\n'.join(lines).replace('[DefaultGenome]','[AuditGenome]'))
    cfg=neat.Config(AuditGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation,str(path))
    cfg.genome_config.activation_defs.add('plain_tanh',plain_tanh)
    return cfg


def initialize(population,cfg,seed,fixed=False):
    rng=np.random.default_rng(seed);gc=cfg.genome_config
    hidden=list(range(3,11 if fixed else 5));gc.node_indexer=itertools.count(max(hidden)+1)
    for index,g in enumerate(population.population.values()):
        for key in hidden:
            node=g.create_node(gc,key);node.bias=0.;g.nodes[key]=node
        for target in hidden:
            sources=list(gc.input_keys)+hidden if fixed else list(rng.choice(gc.input_keys,3,replace=False))+[target]
            for source in sources:
                source=int(source);innovation=gc.innovation_tracker.get_innovation_number(source,target,'shared_initial')
                c=g.create_connection(gc,source,target,innovation);c.weight=float(rng.normal(0,.04));g.connections[(source,target)]=c
        for target in gc.output_keys:
            for source in hidden:
                innovation=gc.innovation_tracker.get_innovation_number(source,target,'shared_initial')
                c=g.create_connection(gc,source,target,innovation);c.weight=float(rng.normal(0,.015));g.connections[(source,target)]=c
        if index==0:
            for key in gc.output_keys:
                g.nodes[key].bias=0.
                for (s,t),edge in g.connections.items():
                    if t==key:edge.weight=0.
        g.parents=[];g.mutation_event={}


def train(args):
    output=args.output;output.mkdir(parents=True,exist_ok=False);random.seed(args.seed)
    ninputs=9 if args.sensor=='imu' else 11
    cfg=config_file(output,args.population,ninputs,args.method=='fixed_evo')
    pop=neat.Population(cfg);initialize(pop,cfg,args.seed,args.method=='fixed_evo')
    start=time.monotonic();state=dict(method=args.method,seed=args.seed,sensor=args.sensor,status='training',training_steps=0,validation_steps=0,budget=args.steps)
    best=None;history=[];generation=0
    write(output/'arguments.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        def fitness(genomes,config):
            nonlocal generation,best
            # Training scenarios rotate, but all individuals in a generation share the scenario.
            horizon=.05 if args.smoke else (2. if state['training_steps']<args.steps*.5 else 4.)
            horizon=min(horizon,max(.01,((args.steps-state['training_steps'])//len(genomes))*.01))
            c=case(330000000+args.seed*1000+generation,horizon,'nominal' if generation%3==0 else 'train',generation%8)
            specs=[export_genome(g,ninputs) for _,g in genomes]
            rows=list(pool.map(evaluate,[(s,c,'residual',args.sensor,False) for s in specs]))
            snapshots=[]
            for (gid,g),s,row in zip(genomes,specs,rows):
                # This weak size penalty affects selection only; report raw task outcomes separately.
                g.fitness=row['return_sum']/horizon-.00002*complexity(s)['connections']
                snapshots.append(dict(id=gid,parents=getattr(g,'parents',[]),fitness=g.fitness,species=pop.species.get_species_id(gid),
                    mutation=getattr(g,'mutation_event',{}),policy=s,episode=row))
            champion=max(snapshots,key=lambda x:x['fitness'])
            state['training_steps']+=sum(r['steps'] for r in rows)
            row=dict(generation=generation,training_steps=state['training_steps'],fitness_max=champion['fitness'],
                fitness_mean=float(np.mean([g.fitness for _,g in genomes])),fitness_min=float(min(g.fitness for _,g in genomes)),
                champion_id=champion['id'],species=len(pop.species.species),case=c,**complexity(champion['policy']))
            compressed(output/f'generation_{generation:04d}.json.gz',dict(generation=generation,individuals=snapshots,
                species={str(sid):list(s.members) for sid,s in pop.species.species.items()}))
            if generation%4==0 or state['training_steps']>=args.steps:
                vs,vr=validate(pool,champion['policy'],'residual',args.sensor,args.seed,.1 if args.smoke else 8.)
                state['validation_steps']+=vs['steps'];row['validation']=vs
                write(output/f'validation_{generation:04d}.json',vr)
                if best is None or selection(vs)>selection(best):
                    best=vs;write(output/'best_policy.json',champion['policy'])
                    write(output/'selection.json',dict(generation=generation,genome_id=champion['id'],validation=vs,complexity=complexity(champion['policy'])))
            history.append(row);write(output/'history.json',history)
            state.update(generation=generation,best_validation=best,wall_seconds=time.monotonic()-start)
            write(output/'status.json',state);print(json.dumps(state),flush=True)
            generation+=1
        while state['training_steps']<=args.steps-args.population:
            pop.run(fitness,1)
            with gzip.open(output/'checkpoint.pkl.gz','wb') as f:pickle.dump((pop,random.getstate(),state),f)
    policy=json.loads((output/'best_policy.json').read_text())
    state.update(status='completed',wall_seconds=time.monotonic()-start,inference=benchmark(policy))
    write(output/'results.json',state);write(output/'status.json',state)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--method',choices=['neat','fixed_evo'],default='neat');p.add_argument('--seed',type=int,required=True)
    p.add_argument('--steps',type=int,default=262144);p.add_argument('--workers',type=int,default=16)
    p.add_argument('--population',type=int,default=128);p.add_argument('--sensor',choices=['height','imu'],default='height')
    p.add_argument('--smoke',action='store_true')
    train(p.parse_args())

if __name__=='__main__':main()
