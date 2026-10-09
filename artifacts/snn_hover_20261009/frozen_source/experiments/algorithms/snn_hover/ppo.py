"""Clipped PPO with causal graph BPTT and an identical feedforward critic.

The actor's pre-tanh Gaussian is used for likelihood ratios (the action Jacobian
cancels). Time-limit transitions bootstrap from terminal observations; actual
failures do not. Stored rollout hidden states initialize truncated BPTT chunks.
"""
import argparse,copy,json,math,time
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.vec_env import SubprocVecEnv
from .environment import HoverEnv,case
from ..topology_hover.networks import torch_actor,random_graph,complexity,NumpyPolicy
from .common import plateau
from .common import write,evaluate,validate,selection,benchmark


def gae(rewards,values,dones,last_value,gamma=.995,lam=.95):
    adv=np.zeros_like(rewards);last=np.zeros_like(last_value)
    for t in reversed(range(len(rewards))):
        next_value=last_value if t==len(rewards)-1 else values[t+1]
        live=1-dones[t]
        delta=rewards[t]+gamma*next_value*live-values[t]
        last=delta+gamma*lam*live*last;adv[t]=last
    return adv,adv+values


def demonstrate(task):
    seed,sensor=task;e=HoverEnv(mode='pid',sensor_mode=sensor);e.configure(case(seed,4.,'nominal' if seed%3==0 else 'train'))
    obs,_=e.reset(seed=seed);xs=[];ys=[]
    while True:
        old=obs.copy();obs,_,done,truncated,info=e.step(np.zeros(3));xs.append(old);ys.append(info['teacher_action'])
        if done or truncated:break
    return np.array(xs),np.array(ys)


def clone_pid(actor,episodes,seed):
    rng=np.random.default_rng(seed);opt=torch.optim.Adam(actor.parameters(),lr=.002);losses=[]
    chunks=[]
    for x,y in episodes:
        for start in range(0,len(x)-64+1,32):chunks.append((x[start:start+64],y[start:start+64]))
    if not chunks:return losses
    for epoch in range(120):
        idx=rng.choice(len(chunks),min(16,len(chunks)),replace=False)
        x=torch.tensor(np.stack([chunks[i][0] for i in idx],1));y=torch.tensor(np.stack([chunks[i][1] for i in idx],1))
        h=torch.zeros(len(idx),actor.state_size);loss=0.
        for t in range(64):
            mean,h=actor(x[t],h);loss=loss+((torch.tanh(mean)-y[t])**2).mean()
        loss=loss/64;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),.5);opt.step();losses.append(float(loss.detach()))
    return losses


def train(args):
    torch.set_num_threads(1);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    out=args.output;out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    ninputs=9 if args.sensor=='imu' else 11
    direct=args.method=='direct_ppo';mode='direct' if direct else 'residual'
    spec=json.loads(args.topology.read_text()) if args.topology else None
    if spec is not None:
        # Reinitialize all weights; graph transfer never carries evolved weights.
        if args.method=='random_graph_ppo':spec=random_graph(ninputs,seed=args.seed,reference=spec)
        else:
            spec=copy.deepcopy(spec)
            for n in spec['nodes']:n['bias']=0.
            for e in spec['edges']:e['weight']=float(rng.normal(0,.03))
    actor=torch_actor('mlp' if args.method=='ff_ppo' else 'graph',ninputs,args.seed,spec=spec,hidden=args.hidden)
    critic=nn.Sequential(nn.Linear(ninputs,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
    log_std=nn.Parameter(torch.full((3,),-2.5));parameters=list(actor.parameters())+list(critic.parameters())+[log_std]
    optimizer=torch.optim.Adam(parameters,lr=3e-4)
    state=dict(method=args.method,seed=args.seed,sensor=args.sensor,status='training',training_steps=0,validation_steps=0,demonstration_steps=0,budget=args.steps)
    write(out/'arguments.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    history=[];best=None;N=args.workers;T=16 if args.smoke else 128;L=8 if args.smoke else 32
    pool=ProcessPoolExecutor(max_workers=min(args.workers,8))
    if direct and not args.smoke:
        examples=list(pool.map(demonstrate,[(322000000+i,args.sensor) for i in range(16)]))
        state['demonstration_steps']=sum(len(x) for x,y in examples)
        write(out/'bc_losses.json',clone_pid(actor,examples,args.seed))
        np.savez_compressed(out/'demonstrations.npz',**{f'{tag}{i}':v for i,(x,y) in enumerate(examples) for tag,v in [('obs',x),('action',y)]})
    env=SubprocVecEnv([partial(HoverEnv,mode=mode,sensor_mode=args.sensor,training=True,seed=args.seed,rank=i,horizon=.1 if args.smoke else 4.) for i in range(N)],start_method='spawn')
    obs=env.reset();hidden=torch.zeros(N,actor.state_size);next_validation=0;update=0
    def validation():
        nonlocal best
        policy=actor.export();numpy=NumpyPolicy(policy);h=torch.zeros(1,actor.state_size)
        for x in rng.normal(0,.3,(32,ninputs)).astype(np.float32):
            with torch.no_grad():m,h=actor(torch.tensor(x[None]),h)
            np.testing.assert_allclose(numpy.act(x),torch.tanh(m)[0].numpy(),atol=2e-6)
        s,rows=validate(pool,policy,mode,args.sensor,args.seed,.1 if args.smoke else 8.)
        state['validation_steps']+=s['steps']
        entry=dict(training_steps=state['training_steps'],validation=s,update=update)
        history.append(entry);write(out/'history.json',history);write(out/f'validation_{update:04d}.json',rows)
        if best is None or selection(s)>selection(best):
            best=s;write(out/'best_policy.json',policy);write(out/'selection.json',entry)
        entry['best_validation']=best
        write(out/'history.json',history)
        return s
    try:
        validation();next_validation=51200
        while state['training_steps']<args.steps:
            env.env_method('set_training_progress',state['training_steps'])
            observations=[];states=[];actions=[];old_logps=[];values=[];rewards=[];dones=[];episode_rows=[]
            for t in range(T):
                x=torch.tensor(obs);states.append(hidden.detach().numpy().copy());observations.append(obs.copy())
                with torch.no_grad():
                    mean,newhidden=actor(x,hidden);dist=torch.distributions.Normal(mean,log_std.exp())
                    u=dist.sample();lp=dist.log_prob(u).sum(-1);value=critic(x).squeeze(-1)
                newobs,r,done,infos=env.step(torch.tanh(u).numpy())
                adjusted=r.copy()
                for i,info in enumerate(infos):
                    if done[i] and info.get('TimeLimit.truncated',False):
                        with torch.no_grad():adjusted[i]+=.995*float(critic(torch.tensor(info['terminal_observation'])[None]).item())
                    if done[i] and 'episode' in info:episode_rows.append(info['episode'])
                actions.append(u.numpy());old_logps.append(lp.numpy());values.append(value.numpy());rewards.append(adjusted);dones.append(done.copy())
                hidden=newhidden.detach();hidden[done]=0.;obs=newobs
            with torch.no_grad():last=critic(torch.tensor(obs)).squeeze(-1).numpy()
            adv,returns=gae(np.array(rewards),np.array(values),np.array(dones),last)
            adv=(adv-adv.mean())/(adv.std()+1e-8)
            X,U,LP,A,R=[torch.tensor(np.array(z),dtype=torch.float32) for z in (observations,actions,old_logps,adv,returns)]
            H=torch.tensor(np.array(states));D=torch.tensor(np.array(dones));chunks=[(e,t) for e in range(N) for t in range(0,T,L)]
            losses=[];kls=[]
            for epoch in range(6):
                rng.shuffle(chunks)
                for offset in range(0,len(chunks),8):
                    selected=chunks[offset:offset+8];es=torch.tensor([e for e,t in selected]);ts=torch.tensor([t for e,t in selected]);h=H[ts,es].detach()
                    batch_loss=0.;kl=0.
                    for k in range(L):
                        ix=ts+k;mean,h=actor(X[ix,es],h)
                        dist=torch.distributions.Normal(mean,log_std.exp());lp=dist.log_prob(U[ix,es]).sum(-1)
                        delta=lp-LP[ix,es];ratio=torch.exp(delta);advantages=A[ix,es]
                        pg=-torch.minimum(ratio*advantages,torch.clamp(ratio,.8,1.2)*advantages).mean()
                        vl=.5*((critic(X[ix,es]).squeeze(-1)-R[ix,es])**2).mean()
                        batch_loss=batch_loss+pg+.5*vl-.001*dist.entropy().sum(-1).mean()
                        kl=kl+((ratio-1)-delta).mean().detach();h=h*(~D[ix,es]).float()[:,None]
                    loss=batch_loss/L;optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(parameters,.5);optimizer.step()
                    with torch.no_grad():log_std.clamp_(-5,-1)
                    losses.append(float(loss.detach()));kls.append(float(kl/L))
                if np.mean(kls[-len(chunks):])>.03:break
            update+=1;state['training_steps']+=N*T
            state.update(update=update,loss=float(np.mean(losses)),approx_kl=float(np.mean(kls)),wall_seconds=time.monotonic()-start)
            if state['training_steps']>=next_validation or state['training_steps']>=args.steps:
                validation();next_validation=state['training_steps']+51200
                torch.save(dict(actor=actor.state_dict(),critic=critic.state_dict(),optimizer=optimizer.state_dict(),log_std=log_std.detach(),state=state),out/'checkpoint.pt')
            if state['training_steps'] in (256000,512000,1024000):
                budget=out/f'budget_{state["training_steps"]}';budget.mkdir(exist_ok=True)
                for name in ['best_policy.json','selection.json']:(budget/name).write_bytes((out/name).read_bytes())
                write(budget/'accounting.json',state)
            state['best_validation']=best;write(out/'status.json',state)
            with (out/'training.jsonl').open('a') as f:f.write(json.dumps(dict(**state,episodes=episode_rows))+'\n')
            print(json.dumps(state),flush=True)
        policy=json.loads((out/'best_policy.json').read_text());state.update(status='completed',wall_seconds=time.monotonic()-start,validation_plateau=plateau(history),inference=benchmark(policy))
        write(out/'results.json',state);write(out/'status.json',state)
    finally:env.close();pool.shutdown(wait=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--method',choices=['ff_ppo','rnn_ppo'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int,default=1024000)
    p.add_argument('--workers',type=int,default=16);p.add_argument('--hidden',type=int,default=8)
    p.add_argument('--sensor',choices=['height','imu'],default='height');p.add_argument('--topology',type=Path)
    p.add_argument('--smoke',action='store_true');train(p.parse_args())

if __name__=='__main__':main()
