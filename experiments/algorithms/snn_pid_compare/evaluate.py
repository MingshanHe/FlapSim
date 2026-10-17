"""Shared closed-loop evaluation; validation is separate from final tests."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import time
import numpy as np
from .common import read, write
from .environment import ControlEnv, scenarios
from .networks import policy

_ENV = None


def worker_init(source):
    global _ENV
    _ENV = ControlEnv(source)


def evaluate_task(task):
    spec, mode, case = task
    env = _ENV
    env.mode = mode
    env.configure(case)
    obs, _ = env.reset(seed=case['seed'])
    actor = policy(spec) if spec is not None else None
    start = time.monotonic()
    inference = 0.
    adapter_inference = 0.
    while True:
        tick = time.perf_counter()
        action = actor.act(obs) if actor is not None else np.zeros(3)
        inference += time.perf_counter()-tick
        obs, _, done, truncated, info = env.step(action)
        adapter_inference += info['controller_seconds']
        if done or truncated:
            result = info['episode']
            result.update(wall_seconds=time.monotonic()-start,
                          actor_inference_seconds=inference,
                          complete_controller_seconds=inference+adapter_inference,
                          inference_scope='actor plus PID/action adapter; common sensing and physics excluded')
            if actor is not None and spec['kind'] == 'snn':
                result['spikes'] = int(actor.spike_count.sum())
            return result


def summarize(rows):
    return dict(episodes=len(rows), successes=sum(r['success'] for r in rows),
                completed=sum(r['termination'] == 'time_limit' for r in rows),
                steps=sum(r['steps'] for r in rows),
                tilt_rms_deg=float(np.mean([r['tilt_rms_deg'] for r in rows])),
                height_rmse_m=float(np.mean([r['height_rmse_m'] for r in rows])),
                mean_return=float(np.mean([r['return_sum'] for r in rows])))


def selection(summary):
    return (summary['successes'], summary['completed'], summary['mean_return'],
            -summary['height_rmse_m'], -summary['tilt_rms_deg'])


def evaluate(pool, spec, mode, cases):
    rows = list(pool.map(evaluate_task, [(spec, mode, c) for c in cases]))
    return dict(summary=summarize(rows), episodes=rows)


def gate(nominal, long, teacher_nominal, offline_rmse):
    student = nominal['summary']
    teacher = teacher_nominal['summary']
    checks = dict(nominal_success=student['successes'] >= 15,
                  tilt=student['tilt_rms_deg'] <= teacher['tilt_rms_deg']+1.,
                  height=student['height_rmse_m'] <= teacher['height_rmse_m']+.02,
                  long_success=long['summary']['successes'] == 4)
    return dict(passed=all(checks.values()), checks=checks,
                offline_diagnostic_pass=bool(max(offline_rmse) <= .05),
                offline_rmse=offline_rmse, nominal=student, long=long['summary'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--policy')
    parser.add_argument('--mode', choices=['pid', 'direct', 'residual'], default='direct')
    parser.add_argument('--split', choices=['validation', 'long_validation', 'stress_validation'], default='validation')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cases = scenarios(args.split, 4 if args.split == 'long_validation' else 16,
                      60. if args.split == 'long_validation' else 20.,
                      'stress' if args.split == 'stress_validation' else 'nominal')
    with ProcessPoolExecutor(args.workers, initializer=worker_init, initargs=(args.source,)) as pool:
        write(args.output, evaluate(pool, read(args.policy) if args.policy else None, args.mode, cases))


if __name__ == '__main__':
    main()
