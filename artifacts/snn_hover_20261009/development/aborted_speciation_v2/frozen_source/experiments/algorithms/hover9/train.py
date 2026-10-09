"""Matched-budget residual NEAT/PPO training with the same causal IMU history."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import itertools
import math
from pathlib import Path
import random
import time
import neat
import numpy as np
from . import common
from .environment import HistoryEnv,training_case,NAMES
from ..neat_imu.robust import BoundedSpeciesSet
from ..neat_imu.residual import encode
from ..neat_imu.signed_activations import register
from ..neat_imu.train import write_json


class HistoryGenome(neat.DefaultGenome):
    def mutate(self,config):
        super().mutate(config)
        for k in config.output_keys:
            if self.nodes[k].activation not in ('tanh','bipolar_sigmoid','softsign','clamped'):
                self.nodes[k].activation='tanh'


def mapping():
    return dict(center=[0.,0.,3461.360088271976],span=[.3,.3,346.1360088271976],
                observation_offset_rad=[0.,0.,0.],meaning='residual correction; see protocol for prior and actuator composition')


def export_neat(g,config):
    p=encode(g,config,mapping());p['input_names']=NAMES;p['priming_steps']=20
    p['output_names']=['alpha_correction','beta_correction','motor_residual'];return p


def config_for(path,population,protocol):
    source=Path(__file__).parents[1]/'neat_imu/config.ini'
    path.write_text(source.read_text().replace('[DefaultGenome]','[HistoryGenome]').replace('[DefaultSpeciesSet]','[BoundedSpeciesSet]'))
    c=neat.Config(HistoryGenome,neat.DefaultReproduction,BoundedSpeciesSet,neat.DefaultStagnation,str(path))
    c.pop_size=population;c.reset_on_extinction=True;g=c.genome_config
    g.num_inputs=15;g.input_keys=list(range(-1,-16,-1));g.num_hidden=12;g.feed_forward=False
    register(g.activation_defs);g.activation_options=protocol['neat_activations'];g.activation_mutate_rate=.08
    g.weight_mutate_power=.09;g.weight_mutate_rate=.5;g.weight_replace_rate=.005
    g.bias_mutate_power=.025;g.bias_mutate_rate=.3;g.bias_replace_rate=.005
    g.conn_add_prob=.15;g.conn_delete_prob=.01;g.node_add_prob=.06;g.node_delete_prob=.01
    g.enabled_mutate_rate=.005;c.species_set_config.compatibility_threshold=.8
    c.reproduction_config.elitism=2;c.stagnation_config.max_stagnation=12
    c.save(str(path));return c


def validate(pool,payload,protocol):
    rows=[r for group in pool.map(common.evaluate,[(payload,[c],False) for c in protocol['validation_cases']]) for r in group]
    return common.summary(rows),rows


def train_neat(a,protocol,out):
    random.seed(a.seed);rng=np.random.default_rng(a.seed)
    config=config_for(out/'config.ini',8 if a.smoke else protocol['population'],protocol)
    pop=neat.Population(config);gcfg=config.genome_config
    for i,g in enumerate(pop.population.values()):
        # Homologous initial hidden nodes must share keys and innovations.
        remap={key:3+j for j,key in enumerate(sorted(k for k in g.nodes if k not in gcfg.output_keys))}
        for key,node in list(g.nodes.items()):node.key=remap.get(key,key)
        g.nodes={node.key:node for node in g.nodes.values()}
        connections={}
        for (source,target),gene in g.connections.items():
            edge=(remap.get(source,source),remap.get(target,target));gene.key=edge
            gene.innovation=gcfg.innovation_tracker.get_innovation_number(*edge,'initial_history')
            connections[edge]=gene
        g.connections=connections
        for k in gcfg.output_keys:g.nodes[k].bias=0.;g.nodes[k].activation='tanh'
        for edge,gene in g.connections.items():
            if edge[1] in gcfg.output_keys:gene.weight=0. if i==0 else float(rng.normal(0,.018))
        for k,node in g.nodes.items():
            if k not in gcfg.output_keys:node.activation=str(rng.choice(protocol['neat_activations']))
    gcfg.node_indexer=itertools.count(15)
    pop.species.speciate(config,pop.population,0)
    state=dict(method='neat',seed=a.seed,status='training',training_steps=0,validation_steps=0)
    best=None;started=time.monotonic();history=[];validations=[];next_validation=0
    with ProcessPoolExecutor(max_workers=a.workers,initializer=common.init_worker,initargs=(str(a.root),)) as pool:
        def check(payload,label):
            nonlocal best
            summary,rows=validate(pool,payload,protocol);state['validation_steps']+=summary['control_steps']
            validations.append(dict(training_steps=state['training_steps'],label=label,summary=summary,episodes=rows))
            if best is None or common.selection_key(summary)>common.selection_key(best):
                best=summary;write_json(out/'best_policy.json',payload)
            write_json(out/'validation.json',validations)
        check(dict(format='zero_residual'),'shared_zero_residual')
        def evaluate(genomes,config):
            nonlocal next_validation
            fraction=min(1.,state['training_steps']/a.steps)
            batch=[training_case(int(rng.integers(10000,800000000)),fraction,False)]
            if fraction>=.4:batch.append(training_case(int(rng.integers(10000,800000000)),fraction,True))
            if a.smoke:
                for c in batch:c['seconds']=.03
            payloads=[export_neat(g,config) for _,g in genomes]
            results=list(pool.map(common.evaluate,[(p,batch,False) for p in payloads]))
            for (_,genome),rows in zip(genomes,results):
                returns=[r['discounted_return'] for r in rows]
                genome.fitness=float(.7*np.mean(returns)+.3*np.min(returns))
                state['training_steps']+=sum(r['steps'] for r in rows)
            order=np.argsort([g.fitness for _,g in genomes])[::-1]
            if state['training_steps']>=next_validation or state['training_steps']>=a.steps:
                check(payloads[int(order[0])],'generation_best');next_validation=state['training_steps']+32768
            row=dict(generation=pop.generation,training_steps=state['training_steps'],
                     best_fitness=genomes[int(order[0])][1].fitness,species=len(pop.species.species),
                     best_validation=best,wall_seconds=time.monotonic()-started,cases=batch)
            history.append(row);state.update(generation=pop.generation,best_validation=best,wall_seconds=row['wall_seconds'])
            write_json(out/'history.json',history);write_json(out/'status.json',state)
            print(json.dumps({k:v for k,v in row.items() if k!='cases'}),flush=True)
        while state['training_steps']<a.steps:pop.run(evaluate,1)
    state.update(status='completed',generations=pop.generation,wall_seconds=time.monotonic()-started)
    write_json(out/'results.json',state);write_json(out/'status.json',state)


def export_ppo(model):
    import torch
    layers=[l for l in model.policy.mlp_extractor.policy_net if isinstance(l,torch.nn.Linear)]
    layers +=[l for l in model.policy.action_net.modules() if isinstance(l,torch.nn.Linear)]
    return dict(format='flapsim-ppo-history-v1',input_names=NAMES,
                layers=[dict(weight=l.weight.detach().cpu().numpy().tolist(),bias=l.bias.detach().cpu().numpy().tolist()) for l in layers])


def train_ppo(a,protocol,out):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from experiments.algorithms.neat.ppo_refine import TanhMeanPolicy
    torch.set_num_threads(1)
    counterpart=a.root/f'neat_seed{a.seed}/results.json'
    target=json.loads(counterpart.read_text())['training_steps'] if not a.smoke else a.steps
    n_steps=32 if a.smoke else 128
    budget=math.ceil(target/(n_steps*a.workers))*n_steps*a.workers
    env=SubprocVecEnv([partial(HistoryEnv,str(a.root),training=True,seed=a.seed,rank=i,workers=a.workers,budget=a.steps)
                      for i in range(a.workers)],start_method='spawn')
    model=PPO(TanhMeanPolicy,env,seed=a.seed,device='cpu',learning_rate=3e-4,n_steps=n_steps,
              batch_size=min(256,n_steps*a.workers),n_epochs=10,gamma=.995,gae_lambda=.95,
              clip_range=.2,ent_coef=.003,target_kl=.03,
              policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64]),activation_fn=torch.nn.Tanh,log_std_init=-2.5))
    with torch.no_grad():
        model.policy.action_net[0].weight.zero_();model.policy.action_net[0].bias.zero_()
    state=dict(method='ppo',seed=a.seed,status='training',training_steps=0,validation_steps=0,
               matched_neat_steps=target,target_steps=budget)
    started=time.monotonic();best=None;history=[]
    pool=ProcessPoolExecutor(max_workers=4,initializer=common.init_worker,initargs=(str(a.root),))
    class Callback(BaseCallback):
        def __init__(self):super().__init__();self.next_validation=32768;self.next_progress=8192
        def validate(self):
            nonlocal best
            payload=export_ppo(model)
            probe=np.random.default_rng(819).normal(0,.3,(16,15)).astype(np.float32)
            np.testing.assert_allclose(model.predict(probe,deterministic=True)[0],
                np.array([common.make_policy(payload).act(x) for x in probe]),atol=2e-6)
            summary,rows=validate(pool,payload,protocol);state['validation_steps']+=summary['control_steps']
            if best is None or common.selection_key(summary)>common.selection_key(best):
                best=summary;write_json(out/'best_policy.json',payload);model.save(out/'best_policy.zip')
            history.append(dict(training_steps=model.num_timesteps,summary=summary,episodes=rows));write_json(out/'validation.json',history)
            print(json.dumps(dict(stage='validation',training_steps=model.num_timesteps,best=best)),flush=True)
        def _on_step(self):
            if self.num_timesteps>=self.next_validation:self.validate();self.next_validation=self.num_timesteps+32768
            if self.num_timesteps>=self.next_progress:
                state.update(training_steps=self.num_timesteps,best_validation=best,wall_seconds=time.monotonic()-started)
                write_json(out/'status.json',state);print(json.dumps(state),flush=True);self.next_progress=self.num_timesteps+8192
            return True
    cb=Callback()
    try:
        cb.validate();model.learn(total_timesteps=budget,callback=cb);cb.validate();model.save(out/'latest.zip')
        state.update(status='completed',training_steps=model.num_timesteps,best_validation=best,wall_seconds=time.monotonic()-started)
        write_json(out/'results.json',state);write_json(out/'status.json',state)
    finally:env.close();pool.shutdown(wait=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--method',choices=['neat','ppo'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int,default=262144);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--smoke',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args();a.root=a.root.resolve()
    out=a.output or a.root/f'{a.method}_seed{a.seed}';out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((a.root/'protocol.json').read_text())
    if a.smoke:
        protocol['validation_cases']=protocol['validation_cases'][:2]
        for c in protocol['validation_cases']:c['seconds']=.03
    write_json(out/'arguments.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
    try:(train_neat if a.method=='neat' else train_ppo)(a,protocol,out)
    except BaseException as e:write_json(out/'failure.json',dict(type=type(e).__name__,message=str(e)));raise


if __name__=='__main__':main()
