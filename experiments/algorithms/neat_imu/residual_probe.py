"""Short development rollouts for physics-informed recurrent initialization."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path
import time
import numpy as np
from .residual import ResidualHoverEnv, ResidualPolicy, action_map, gain_network

ENV = None


def init_worker(settings):
    global ENV
    ENV = ResidualHoverEnv(**settings)


def run(task):
    payload, seeds, seconds, initial_degrees, trace = task
    ENV.set_seconds(seconds)
    ENV.initial_degrees = initial_degrees
    policy = ResidualPolicy(payload)
    results = []
    for seed in seeds:
        obs, _ = ENV.reset(seed=seed)
        policy.reset(obs)
        total, rows = 0., []
        while True:
            obs, reward, terminated, truncated, info = ENV.step(policy.act(obs))
            total += reward
            if trace:
                rows.append([info['elapsed'], *np.rad2deg(info['rpy']), info['alpha'], info['beta'],
                             *info['position'], info['motor_command']])
            if terminated or truncated:
                break
        results.append(dict(seed=seed, fitness=total/ENV.horizon, steps=ENV.steps,
                            elapsed=info['elapsed'], success=info['is_success'],
                            tracking_rmse_m=info['tracking_rmse'],
                            attitude_rmse_deg=info['attitude_rmse_deg'].tolist(),
                            termination=info['termination_reason'], trace=rows))
    return results


def summary(rows):
    return dict(successes=sum(r['success'] for r in rows), episodes=len(rows),
                mean_elapsed=float(np.mean([r['elapsed'] for r in rows])),
                mean_tracking_rmse_m=float(np.mean([r['tracking_rmse_m'] for r in rows])),
                mean_fitness=float(np.mean([r['fitness'] for r in rows])),
                control_steps=sum(r['steps'] for r in rows),
                termination_counts={k: sum(r['termination'] == k for r in rows)
                                    for k in sorted({r['termination'] for r in rows})})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trim', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--signed', action='store_true', help='also probe opposite feedback signs')
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        p.error('output must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    trim = json.loads(args.trim.read_text())
    settings = dict(trim=trim)
    mapping = action_map(trim)
    gains = list(itertools.product((-.3, -1., -3., -.1, .1, .3, 1., 3.) if args.signed else (.1, .3, 1., 3.),
                                  (-.1, -.03, 0., .03, .1) if args.signed else (0., .01, .03, .1), (.02, .05)))
    gains.append((0., 0., .05))
    payloads = [gain_network(kp, kd, tau, mapping) for kp, kd, tau in gains]
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(settings,)) as pool:
        rows = list(pool.map(run, [(payload, [851007000, 851007001], 2., 1., False) for payload in payloads]))
    records = [dict(kp=g[0], kd=g[1], tau=g[2], **summary(r), results=r, payload=payload)
               for g, r, payload in zip(gains, rows, payloads)]
    records.sort(key=lambda r: (r['successes'], r['mean_fitness']), reverse=True)
    (args.output/'candidates.json').write_text(json.dumps(records, indent=2)+'\n')
    (args.output/'best_seed.json').write_text(json.dumps(records[0]['payload'], indent=2)+'\n')
    report = dict(wall_seconds=time.monotonic()-start, control_steps=sum(r['control_steps'] for r in records),
                  candidates=len(records), best={k: v for k, v in records[0].items() if k not in ('payload', 'results')})
    (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
