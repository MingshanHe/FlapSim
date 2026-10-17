"""Shared original-computation evaluation for the v3 tasks."""
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .environment import ResearchEnv, scenarios
from .networks import policy
from ..snn_pid_compare.evaluate import summarize

_ENV = None


def worker_init(source, task='hover'):
    global _ENV
    _ENV = ResearchEnv(source, task=task)


def evaluate_task(task):
    spec, mode, c = task
    env = _ENV
    env.mode = mode
    env.configure(c)
    obs, _ = env.reset(seed=c['seed'])
    legacy = spec is not None and spec.get('legacy_100hz', False)
    if legacy:
        from ..snn_hover.networks import policy as historical_policy
        actor = historical_policy(spec['policy'])
    else:
        actor = policy(spec) if spec is not None else None
    start = time.monotonic()
    inference = 0.
    adapter_inference = 0.
    actor_calls = 0
    while True:
        tic = time.perf_counter()
        if not legacy or env.steps % 5 == 0:
            action = actor.act(obs) if actor is not None else np.zeros(3)
            actor_calls += int(actor is not None)
        inference += time.perf_counter()-tic
        obs, _, done, truncated, info = env.step(action)
        adapter_inference += info['controller_seconds']
        if done or truncated:
            result = info['episode']
            samples=np.asarray(env.samples)
            late=samples[samples[:,0] >= max(0.,samples[-1,0]-5.)]
            result['height_peak_to_peak_m']=float(np.ptp(late[:,2]))
            checks=dict(full_duration=result['termination']=='time_limit',
                        clearance=result['minimum_clearance_m']>.03)
            if env.task == 'hover':
                checks.update(tilt=result['tilt_rms_deg']<10.,height=result['height_rmse_m']<.15,
                              height_peak_to_peak=result['height_peak_to_peak_m']<.2,
                              height_drift=abs(result['height_slope_m_s'])<.03)
            else:
                checks.update(tilt=result['tilt_rms_deg']<12.,
                              speed=result['velocity_vector_rmse_m_s']<.1,
                              normalized_tracking=result['tracking_nrmse']<.6,
                              height=result['reference_height_rmse_m']<.15)
            result['success_checks']=checks
            result.update(wall_seconds=time.monotonic()-start, actor_inference_seconds=inference)
            result.update(complete_controller_seconds=inference+adapter_inference,actor_calls=actor_calls,
                          inference_scope='actor and PID/velocity/action adapter; common sensing and physics excluded')
            result['policy_hz'] = 100 if legacy else 500
            if actor is not None and (spec['policy'] if legacy else spec)['kind'] == 'snn':
                result['spikes'] = int(actor.spike_count.sum())
            return result


def evaluate(pool, spec, mode, cases):
    rows = list(pool.map(evaluate_task, [(spec, mode, c) for c in cases]))
    summary = summarize(rows)
    if rows and 'velocity_vector_rmse_m_s' in rows[0]:
        summary['velocity_vector_rmse_m_s'] = float(np.mean([r['velocity_vector_rmse_m_s'] for r in rows]))
    summary['minimum_clearance_m'] = min(r['minimum_clearance_m'] for r in rows)
    return dict(summary=summary, episodes=rows)


def selection(summary):
    return (summary['successes'], summary['completed'],
            -summary.get('velocity_vector_rmse_m_s', 0.),
            summary['mean_return'], -summary['height_rmse_m'], -summary['tilt_rms_deg'])


def validation_cases(task):
    seconds = 20. if task == 'velocity' else 12.
    return (scenarios('validation', 8, seconds, task=task)
            + scenarios('validation', 8, seconds, 'stress', task))


def development(source, output, workers=16):
    from pathlib import Path
    from ..snn_pid_compare.common import write, read
    output = Path(output)
    result = {}
    for task in ('hover', 'velocity'):
        target = output/f'{task}_baseline.json'
        if target.exists():
            result[task] = read(target)
            continue
        cases = scenarios('development', 8, 8., task=task)
        if task == 'hover':
            cases += scenarios('development', 8, 8., 'stress', task)
        with ProcessPoolExecutor(workers, initializer=worker_init, initargs=(source, task)) as pool:
            result[task] = evaluate(pool, None, 'pid', cases)
        write(target, result[task])
        print(task, result[task]['summary'], flush=True)
    return result


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=16)
    a = p.parse_args()
    development(a.source, a.output, a.workers)
