"""Budgeted NEAT or PPO training using exactly the same six-IMU environment."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from functools import partial
import itertools
import json
import math
from pathlib import Path
import random
import time
import numpy as np
from . import compare_env as common
from .activation import signed_map,export
from .architecture import initialize
from .robust_train import make_config
from .signed_activations import register
from .train import write_json


def checkpoint_validation(pool,payload,protocol):
    rows=[r for group in pool.map(common.evaluate,[(payload,[case],False) for case in protocol['validation_cases']]) for r in group]
    return common.summarize(rows),rows


def train_neat(args,protocol,output):
    import neat
    random.seed(args.seed); rng=np.random.default_rng(args.seed)
    trim=json.loads((args.root/'calibration/trim.json').read_text())
    mapping=signed_map(trim)
    config=make_config(output/'neat_config.ini',8 if args.smoke else 128,'exploratory','gyro6')
    cfg=config.genome_config
    register(cfg.activation_defs)
    cfg.activation_options=protocol['neat_activations']; cfg.activation_mutate_rate=.2
    cfg.signed_output_activations=protocol['neat_output_activations']; cfg.output_activation_mutate_rate=.15
    cfg.hidden_seed_count=8
    population=neat.Population(config)
    cfg.node_indexer=itertools.count(12)
    parent=json.loads((args.root/'trim_seed.json').read_text())
    for i,genome in enumerate(population.population.values()):
        initialize(genome,config,mapping,rng,parent if i<16 else None)
        for k in cfg.initial_hidden_keys: genome.nodes[k].activation=str(rng.choice(cfg.activation_options))
        if 0<i<16: genome.mutate(cfg)
    config.save(str(output/'neat_config.ini'))
    population.add_reporter(neat.Checkpointer(generation_interval=4,filename_prefix=str(output/'checkpoint-')))
    state=dict(method='neat',seed=args.seed,status='training',training_steps=0,validation_steps=0,generation=0)
    history=[]; validations=[]; best=None; best_payload=parent; started=time.monotonic(); next_validation=0
    with ProcessPoolExecutor(max_workers=args.workers,initializer=common.init_worker,initargs=(str(args.root),)) as pool:
        def validate(payload,label):
            nonlocal best,best_payload
            summary,rows=checkpoint_validation(pool,payload,protocol)
            state['validation_steps']+=summary['control_steps']
            validations.append(dict(training_steps=state['training_steps'],label=label,summary=summary,episodes=rows))
            if best is None or common.selection_key(summary)>common.selection_key(best):
                best=summary; best_payload=payload
                write_json(output/'best_policy.json',payload)
            write_json(output/'validation.json',validations)
        validate(parent,'constant_trim')
        def evaluate(genomes,config):
            nonlocal next_validation
            fraction=min(1.,state['training_steps']/args.steps)
            seeds=rng.integers(10000,800000000,2)
            batch=[common.training_case(int(seed),fraction,i==1) for i,seed in enumerate(seeds)]
            if args.smoke:
                for case in batch: case['seconds']=.05
            anneal=.5 if fraction>.7 else 1.
            cfg.weight_mutate_power=.18*anneal; cfg.bias_mutate_power=.05*anneal
            payloads=[export(genome,config,mapping) for _,genome in genomes]
            results=list(pool.map(common.evaluate,[(p,batch,False) for p in payloads]))
            for (_,genome),rows in zip(genomes,results):
                genome.fitness=float(np.mean([r['discounted_return'] for r in rows]))
                state['training_steps']+=sum(r['steps'] for r in rows)
            order=np.argsort([g.fitness for _,g in genomes])[::-1]
            if state['training_steps']>=next_validation or state['training_steps']>=args.steps:
                validate(payloads[int(order[0])],'generation_best')
                next_validation=state['training_steps']+65536
            row=dict(generation=population.generation,training_steps=state['training_steps'],
                     best_fitness=genomes[int(order[0])][1].fitness,
                     species=len(population.species.species),best_validation=best,
                     nodes=len(genomes[int(order[0])][1].nodes),
                     elapsed_seconds=time.monotonic()-started,cases=batch)
            state.update(generation=population.generation,best_validation=best,wall_seconds=row['elapsed_seconds'])
            history.append(row); write_json(output/'history.json',history); write_json(output/'status.json',state)
            print(json.dumps({k:v for k,v in row.items() if k!='cases'}),flush=True)
        while state['training_steps']<args.steps and population.generation<100:
            population.run(evaluate,1)
        write_json(output/'selected.json',dict(summary=best,policy='best_policy.json'))
    state.update(status='completed',generations=population.generation,wall_seconds=time.monotonic()-started)
    write_json(output/'results.json',state); write_json(output/'status.json',state)


def export_ppo(model,mapping):
    import torch
    layers=[layer for layer in model.policy.mlp_extractor.policy_net if isinstance(layer,torch.nn.Linear)]
    layers+=[layer for layer in model.policy.action_net.modules() if isinstance(layer,torch.nn.Linear)]
    return dict(format='flapsim-ppo-imu6-v1',action_map=mapping,
                layers=[dict(weight=l.weight.detach().cpu().numpy().tolist(),bias=l.bias.detach().cpu().numpy().tolist()) for l in layers])


def train_ppo(args,protocol,output):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from experiments.algorithms.neat.ppo_refine import TanhMeanPolicy
    torch.set_num_threads(1)
    trim=json.loads((args.root/'calibration/trim.json').read_text()); mapping=signed_map(trim)
    # NEAT overshoots only to complete a generation; PPO matches the observed count.
    counterpart=args.root/f'neat_seed{args.seed}/results.json'
    if not counterpart.exists() and not args.smoke: raise RuntimeError('NEAT budget must be known before matched PPO')
    target=json.loads(counterpart.read_text())['training_steps'] if counterpart.exists() else args.steps
    rollout_steps=32 if args.smoke else 256
    steps=math.ceil(target/(args.workers*rollout_steps))*args.workers*rollout_steps
    env=SubprocVecEnv([partial(common.CompareEnv,str(args.root),training=True,seed=args.seed,rank=i,workers=args.workers,budget=args.steps)
                      for i in range(args.workers)],start_method='spawn')
    model=PPO(TanhMeanPolicy,env,seed=args.seed,device='cpu',learning_rate=3e-4,n_steps=rollout_steps,batch_size=min(256,rollout_steps*args.workers),
              n_epochs=10,gamma=common.GAMMA,gae_lambda=.95,clip_range=.2,ent_coef=.003,
              target_kl=.03,policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64]),activation_fn=torch.nn.Tanh,log_std_init=-1.8))
    with torch.no_grad():
        model.policy.action_net[0].bias.copy_(torch.tensor(np.arctanh(np.clip(np.r_[np.asarray(trim['command'][1:])/.3,0.],-.99,.99)),dtype=torch.float32))
    state=dict(method='ppo',seed=args.seed,status='training',training_steps=0,validation_steps=0,target_steps=steps,matched_neat_steps=target)
    started=time.monotonic(); history=[]; best=None
    pool=ProcessPoolExecutor(max_workers=4,initializer=common.init_worker,initargs=(str(args.root),))
    class Callback(BaseCallback):
        def __init__(self):
            super().__init__(); self.next_validation=65536; self.next_progress=8192
        def validate(self):
            nonlocal best
            payload=export_ppo(model,mapping)
            probe=np.random.default_rng(631).normal(0,.3,(16,6)).astype(np.float32)
            predicted=model.predict(probe,deterministic=True)[0]
            np.testing.assert_allclose(predicted,np.array([common.NumpyPPO(payload).act(x) for x in probe]),atol=2e-6)
            summary,rows=checkpoint_validation(pool,payload,protocol)
            state['validation_steps']+=summary['control_steps']
            if best is None or common.selection_key(summary)>common.selection_key(best):
                best=summary; write_json(output/'best_policy.json',payload); model.save(output/'best_policy.zip')
            history.append(dict(training_steps=model.num_timesteps,summary=summary,episodes=rows))
            write_json(output/'validation.json',history)
            model.save(output/'latest.zip')
            print(json.dumps(dict(stage='validation',training_steps=model.num_timesteps,best=best)),flush=True)
        def _on_step(self):
            if self.num_timesteps>=self.next_validation:
                self.validate(); self.next_validation=self.num_timesteps+65536
            if self.num_timesteps>=self.next_progress:
                state.update(training_steps=self.num_timesteps,best_validation=best,wall_seconds=time.monotonic()-started)
                write_json(output/'status.json',state)
                print(json.dumps(state),flush=True); self.next_progress=self.num_timesteps+8192
            return True
    callback=Callback()
    try:
        # Before callback.init_callback, validation uses the enclosing model only.
        callback.validate()
        model.learn(total_timesteps=steps,callback=callback)
        callback.validate()
        state.update(status='completed',training_steps=model.num_timesteps,best_validation=best,wall_seconds=time.monotonic()-started)
        write_json(output/'results.json',state); write_json(output/'status.json',state)
    finally:
        env.close(); pool.shutdown(wait=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--method',choices=['neat','ppo'],required=True); parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--workers',type=int,default=8); parser.add_argument('--steps',type=int,default=524288)
    parser.add_argument('--smoke',action='store_true'); parser.add_argument('--output',type=Path)
    args=parser.parse_args(); args.root=args.root.resolve()
    output=args.output or args.root/f'{args.method}_seed{args.seed}'
    output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((args.root/'protocol.json').read_text())
    if args.smoke:
        protocol=copy.deepcopy(protocol)
        protocol['validation_cases']=protocol['validation_cases'][:2]
        for case in protocol['validation_cases']: case['seconds']=.05
    write_json(output/'arguments.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    try:
        (train_neat if args.method=='neat' else train_ppo)(args,protocol,output)
    except BaseException as error:
        write_json(output/'failure.json',dict(type=type(error).__name__,message=str(error)))
        raise


if __name__=='__main__': main()
