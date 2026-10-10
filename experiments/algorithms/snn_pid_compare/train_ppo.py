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
from .common import read, write, append
from .environment import ControlEnv, scenarios
from .networks import Actor
from .evaluate import worker_init, evaluate, selection

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
    if (out/'checkpoint.pt').exists() and not (out/'results.json').exists():
        raise RuntimeError('Incomplete PPO run retained. Environment-exact mid-run resume is not implemented; do not silently reset it.')
    actor = Actor(read(args.initial))
    critic = nn.Sequential(nn.Linear(11, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)).double()
    log_std = nn.Parameter(torch.full((3,), -3., dtype=torch.float64))
    parameters = list(actor.parameters())+list(critic.parameters())+[log_std]
    optimizer = torch.optim.Adam(parameters, lr=3e-4)
    N, T, L = args.workers, args.rollout, args.window
    if T % L or args.steps % (N*T):
        raise ValueError('Budget/rollout/window must align exactly')
    env = SubprocVecEnv([partial(ControlEnv, args.source, mode=args.mode, training=True,
                                seed=args.seed, rank=i, horizon=2.) for i in range(N)], start_method='spawn')
    obs = env.reset()
    hidden = actor.initial_state(N)
    state = dict(status='training', allocated_steps=0, training_steps=0, validation_steps=0,
                 mode=args.mode, kind=actor.kind, seed=args.seed, budget=args.steps, update=0)
    best = None
    start = time.monotonic()
    cases = (scenarios('validation', 2, .02) if args.smoke else
             scenarios('validation', 16, 20.)+scenarios('stress_validation', 16, 20., 'stress'))
    pool = ProcessPoolExecutor(args.workers, initializer=worker_init, initargs=(args.source,))
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
        validation()
        # Three post-initial selections, matching NEAT's generation checkpoints.
        validation_budgets = [args.steps/7, 3*args.steps/7, args.steps]
        while state['training_steps'] < args.steps:
            fraction = state['training_steps']/args.steps
            env.set_attr('horizon', 2. if fraction < 1/7 else 4. if fraction < 3/7 else 8.)
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
                         wall_seconds=time.monotonic()-start, best_validation=best)
            temporary = out/'checkpoint.tmp.pt'
            torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(),
                            optimizer=optimizer.state_dict(), log_std=log_std.detach(),
                            state=state, torch_rng=torch.get_rng_state(), rng=rng.bit_generator.state), temporary)
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
    parser.add_argument('--smoke', action='store_true')
    train(parser.parse_args())


if __name__ == '__main__':
    main()
