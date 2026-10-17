"""Recurrent PPO for the same exported ANN/SNN actor, at 500 Hz."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.vec_env import SubprocVecEnv
from ..snn_pid_compare.common import read, write, append
from .environment import ResearchEnv, scenarios
from .networks import Actor
from .evaluate import worker_init, evaluate, selection, validation_cases
from .train_neat import curriculum_durations

GAMMA = .995**.2
LAMBDA = .95**.2


def advantages(rewards, values, next_values, terminated, boundaries):
    """Truncations bootstrap terminal observations but stop the GAE trace."""
    advantage = np.zeros_like(rewards)
    carry = np.zeros(rewards.shape[1])
    for t in reversed(range(len(rewards))):
        delta = rewards[t]+GAMMA*next_values[t]*(1-terminated[t])-values[t]
        carry = delta+GAMMA*LAMBDA*(1-boundaries[t])*carry
        advantage[t] = carry
    return advantage, advantage+values


def train(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    resume = (out/'checkpoint.pt').exists() and not (out/'results.json').exists()
    if resume and not args.resume_reset_episodes:
        raise RuntimeError('Resume requires --resume-reset-episodes: simulator state is not checkpointed.')
    actor = Actor(read(args.initial))
    critic = nn.Sequential(nn.Linear(actor.inputs, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)).double()
    log_std = nn.Parameter(torch.full((3,), -3., dtype=torch.float64))
    parameters = list(actor.parameters())+list(critic.parameters())+[log_std]
    optimizer = torch.optim.Adam(parameters, lr=3e-4)
    N, T, L = args.workers, args.rollout, args.window
    durations = curriculum_durations(args.task)
    fractions = np.cumsum(durations)/sum(durations)
    def horizon_at(steps):
        return durations[min(2,int(np.searchsorted(fractions,steps/args.steps,side='right')))]
    if T % L or args.steps % (N*T):
        raise ValueError('Budget/rollout/window must align exactly')
    env = SubprocVecEnv([partial(ResearchEnv, args.source, mode=args.mode, task=args.task, training=True,
                                seed=args.seed, rank=i, horizon=durations[0]) for i in range(N)], start_method='spawn')
    state = dict(status='training', allocated_steps=0, training_steps=0, validation_steps=0,
                 mode=args.mode, kind=actor.kind, seed=args.seed, budget=args.steps, update=0)
    best = None
    if resume:
        saved = torch.load(out/'checkpoint.pt', weights_only=False)
        actor.load_state_dict(saved['actor'])
        critic.load_state_dict(saved['critic'])
        optimizer.load_state_dict(saved['optimizer'])
        with torch.no_grad():
            log_std.copy_(saved['log_std'])
        torch.set_rng_state(saved['torch_rng'])
        rng.bit_generator.state = saved['rng']
        state = saved['state']
        state['resume_count'] = state.get('resume_count', 0)+1
        best = state.get('best_validation')
        for i, index in enumerate(saved['env_episode_index']):
            env.set_attr('episode_index', index, indices=i)
        append(out/'resume_events.jsonl', dict(unix=time.time(), update=state['update'],
            policy='restore optimizer and RNG; reset unfinished episodes and neural state',
            episode_indices=saved['env_episode_index']))
    env.set_attr('horizon', horizon_at(state['training_steps']))
    obs = env.reset()
    hidden = actor.initial_state(N)
    previous_wall = state.get('wall_seconds', 0.)
    start = time.monotonic()
    write(out/'implementation.json',dict(module=__file__,task=args.task))
    write(out/'status.json',state)
    cases = (scenarios('validation', 2, .02, task=args.task) if args.smoke else
             validation_cases(args.task))
    pool = ProcessPoolExecutor(args.workers, initializer=worker_init, initargs=(args.source, args.task))
    def validation():
        nonlocal best
        result = evaluate(pool, actor.export(), args.mode, cases)
        state['validation_steps'] += result['summary']['steps']
        write(out/f'validation_{state["update"]:05d}.json', result)
        if best is None or selection(result['summary']) > selection(best):
            best = result['summary']
            write(out/'best_policy.json', actor.export())
            write(out/'selection.json', dict(update=state['update'], summary=best))
    try:
        if best is None:
            validation()
        # Three post-initial selections, matching NEAT's generation checkpoints.
        validation_budgets = [b for b in args.steps*fractions
                              if b > state['training_steps']]
        while state['training_steps'] < args.steps:
            env.set_attr('horizon', horizon_at(state['training_steps']))
            X, H, U, LP, V, NV, rewards, terminated, boundaries = ([] for _ in range(9))
            for _ in range(T):
                x = torch.tensor(obs, dtype=torch.float64)
                X.append(obs.copy())
                H.append(hidden.detach().numpy().copy())
                with torch.no_grad():
                    mean, new_hidden = actor(x, hidden)
                    distribution = torch.distributions.Normal(mean, log_std.exp())
                    raw = distribution.sample()
                    lp = distribution.log_prob(raw).sum(-1)
                    value = critic(x).squeeze(-1)
                new_obs, reward, done, infos = env.step(raw.tanh().numpy())
                terminal_obs = np.array([info.get('terminal_observation', new_obs[i]) for i, info in enumerate(infos)])
                with torch.no_grad():
                    next_value = critic(torch.tensor(terminal_obs, dtype=torch.float64)).squeeze(-1).numpy()
                terminal = np.array([done[i] and not info.get('TimeLimit.truncated', False) for i, info in enumerate(infos)])
                U.append(raw.numpy()); LP.append(lp.numpy()); V.append(value.numpy()); NV.append(next_value)
                rewards.append(reward); terminated.append(terminal); boundaries.append(done.copy())
                hidden = new_hidden.detach()
                hidden[done] = 0.
                obs = new_obs
            adv, returns = advantages(np.array(rewards), np.array(V), np.array(NV), np.array(terminated), np.array(boundaries))
            adv = (adv-adv.mean())/(adv.std()+1e-8)
            X, H, U, LP, A, R = [torch.tensor(np.array(z), dtype=torch.float64) for z in (X, H, U, LP, adv, returns)]
            D = torch.tensor(np.array(boundaries))
            chunks = [(environment, t) for environment in range(N) for t in range(0, T, L)]
            losses, divergences = [], []
            for epoch in range(args.epochs):
                rng.shuffle(chunks)
                epoch_divergences = []
                for offset in range(0, len(chunks), 8):
                    selected = chunks[offset:offset+8]
                    es = torch.tensor([e for e, _ in selected])
                    ts = torch.tensor([t for _, t in selected])
                    h = H[ts, es].detach()
                    batch_loss = 0.
                    kl = 0.
                    for k in range(L):
                        ix = ts+k
                        mean, h = actor(X[ix, es], h)
                        distribution = torch.distributions.Normal(mean, log_std.exp())
                        lp = distribution.log_prob(U[ix, es]).sum(-1)
                        delta = lp-LP[ix, es]
                        ratio = delta.exp()
                        pg = -torch.minimum(ratio*A[ix, es], ratio.clamp(.8, 1.2)*A[ix, es]).mean()
                        vl = .5*(critic(X[ix, es]).squeeze(-1)-R[ix, es]).square().mean()
                        batch_loss = batch_loss+pg+.5*vl-.001*distribution.entropy().sum(-1).mean()
                        kl = kl+((ratio-1)-delta).mean().detach()
                        h = h*(~D[ix, es])[:, None]
                    loss = batch_loss/L
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(parameters, .5)
                    optimizer.step()
                    with torch.no_grad():
                        log_std.clamp_(-5., -1.)
                    losses.append(float(loss.detach()))
                    epoch_divergences.append(float(kl/L))
                divergences.extend(epoch_divergences)
                if np.mean(epoch_divergences) > .03:
                    break
            state['training_steps'] += N*T
            state['allocated_steps'] += N*T
            state['update'] += 1
            if validation_budgets and state['training_steps'] >= validation_budgets[0]:
                validation()
                validation_budgets.pop(0)
            state.update(loss=float(np.mean(losses)), approx_kl=float(np.mean(divergences)),
                         wall_seconds=previous_wall+time.monotonic()-start, best_validation=best)
            temporary = out/'checkpoint.tmp.pt'
            torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(),
                            optimizer=optimizer.state_dict(), log_std=log_std.detach(),
                            state=state, torch_rng=torch.get_rng_state(), rng=rng.bit_generator.state,
                            env_episode_index=env.get_attr('episode_index')), temporary)
            temporary.replace(out/'checkpoint.pt')
            write(out/'last_policy.json', actor.export())
            write(out/'status.json', state)
            append(out/'history.jsonl', state)
            print(state, flush=True)
        state['status'] = 'completed'
        write(out/'results.json', state)
    finally:
        env.close()
        pool.shutdown(wait=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--initial', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['direct', 'residual'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--rollout', type=int, default=500)
    parser.add_argument('--window', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--task', choices=['hover', 'velocity'], default='hover')
    parser.add_argument('--resume-reset-episodes', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    train(parser.parse_args())


if __name__ == '__main__':
    main()
