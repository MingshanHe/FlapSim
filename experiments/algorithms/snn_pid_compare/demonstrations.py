"""Timestamp-aligned teacher sequences, split by whole episodes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from .common import write, sha
from .environment import ControlEnv, CENTER, HALF, scenarios

_ENV = None


def initialize(source):
    global _ENV
    _ENV = ControlEnv(source, mode='pid')


def collect_case(task):
    case, output, index = task
    env = _ENV
    env.configure(case)
    obs, _ = env.reset(seed=case['seed'])
    observations, targets, raw, applied = [], [], [], []
    while True:
        command = env.teacher_command()
        observations.append(obs.copy())
        targets.append((command-CENTER)/HALF)
        raw.append(env.teacher.raw.copy())
        obs, _, done, truncated, info = env.step(np.zeros(3))
        applied.append(info['applied_command'])
        if done or truncated:
            break
    path = Path(output)/f'episode_{index:03d}.npz'
    np.savez_compressed(path, obs=np.asarray(observations), action=np.asarray(targets),
                        teacher_raw=np.asarray(raw), applied=np.asarray(applied),
                        time=np.arange(len(targets))*.002)
    return dict(file=path.name, sha256=sha(path), samples=len(targets), case=case,
                accepted=info['episode']['success'], metrics=info['episode'])


def collect(source, output, workers=8, count=32, seconds=20., split='demo'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cases = scenarios(split, count, seconds)
    write(output/'cases.json', cases)
    rows = []
    with ProcessPoolExecutor(workers, initializer=initialize, initargs=(str(source),)) as pool:
        for row in pool.map(collect_case, [(c, str(output), i) for i, c in enumerate(cases)]):
            rows.append(row)
            write(output/'progress.json', dict(completed=len(rows), total=len(cases), samples=sum(r['samples'] for r in rows)))
            print('demonstration', split, len(rows), row['accepted'], row['samples'], flush=True)
    manifest = dict(split=split, control_hz=500, input_dimensions=11,
                    acceptance='strict_success_on_entire_episode', episodes=rows,
                    samples=sum(r['samples'] for r in rows),
                    accepted_samples=sum(r['samples'] for r in rows if r['accepted']))
    write(output/'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--split', choices=['demo', 'offline'], default='demo')
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--seconds', type=float, default=20.)
    args = parser.parse_args()
    collect(args.source, args.output, args.workers, args.count, args.seconds, args.split)


if __name__ == '__main__':
    main()
