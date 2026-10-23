"""Measure zero-command releases before choosing a new training initialization."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
from .environment import ImuAttitudeEnv


def probe(setting):
    warmup, speed = setting
    env = ImuAttitudeEnv(seconds=8., motor_speed=speed, initial_degrees=0., warmup_seconds=warmup)
    rows = []
    for seed in (850007000, 850007001):
        env.reset(seed=seed)
        while True:
            _, _, terminated, truncated, info = env.step(np.zeros(3))
            if terminated or truncated:
                break
        rows.append(dict(seed=seed, seconds=info['elapsed'], reason=info['termination_reason'],
                         rmse_m=info['tracking_rmse'], attitude_rmse_deg=info['attitude_rmse_deg'].tolist()))
    env.close()
    return dict(warmup=warmup, speed=speed, results=rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    with ProcessPoolExecutor(max_workers=5) as pool:
        rows = list(pool.map(probe, [(t, 3376.8672644667213) for t in (.1, .2, .4, .8, 1.)]))
    args.output.write_text(json.dumps(rows, indent=2)+'\n')
    print(json.dumps(rows, indent=2))
