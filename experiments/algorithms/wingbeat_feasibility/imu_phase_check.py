"""How much of each v3 IMU channel is locked to the wingbeat, and how well an
offline (non-causal) band-pass + Hilbert estimate recovers the crank phase.

PID-only flights, no modulation. The causal estimator actually used in closed
loop is WingbeatPhase in preliminary.py (stage e4).

    python -m experiments.algorithms.wingbeat_feasibility.imu_phase_check \
      --scratch <dir> --output artifacts/wingbeat_feasibility_20261012/p0_imu_phase_check.json
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from . import preliminary as P

NAMES = ['gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z']


def flight(seed, phase, seconds=4.):
    rows = []
    spec = dict(seed=seed, phase=phase, seconds=seconds, law=None, pulses=[])
    env = P._ENV
    c = P.make_case(seed, seconds, 'nominal', phase)
    c['task'] = 'hover'
    env.mode = 'pid'
    env.configure(c)
    env.drive = dict(window=None, hold=None, law=P.make_law(spec), disturbance=P.make_disturbance([]),
                     sensor=P.VelocitySensor(seed), estimator=None, phase_errors=[],
                     electric={k: dict(n=0, power=0., peak_current=0., saturated=0) for k in ('all', 'window')})
    env.reset(seed=seed)
    while True:
        _, _, terminated, truncated, _ = env.step(np.zeros(3))
        rows.append([env.air.steps*P.DT, env.sim.data.joint('carrier_2_joint').qpos[0], *env.last_reading[3:9]])
        if terminated or truncated:
            return np.asarray(rows)


def hilbert_phase(y):
    n = len(y)
    h = np.zeros(n)
    h[0] = 1
    h[1:(n+1)//2] = 2
    if n % 2 == 0:
        h[n//2] = 1
    return np.angle(np.fft.ifft(np.fft.fft(y)*h))


def analyse(rows, settle=1.5):
    rows = rows[rows[:, 0] >= settle]
    phi = rows[:, 1]
    hz = (phi[-1]-phi[0])/(2*np.pi*(rows[-1, 0]-rows[0, 0]))
    X = np.c_[np.cos(phi), np.sin(phi), np.cos(2*phi), np.sin(2*phi), np.ones(len(phi))]
    locked = {}
    for i, name in enumerate(NAMES):
        y = rows[:, 2+i]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        locked[name] = float(1-np.var(y-X@coef)/np.var(y))
    best = max(locked, key=locked.get)
    y = rows[:, 2+NAMES.index(best)]
    spectrum = np.fft.rfft(y-y.mean())
    f = np.fft.rfftfreq(len(y), P.DT)
    spectrum[(f < .7*hz) | (f > 1.3*hz)] = 0
    error = np.angle(np.exp(1j*(hilbert_phase(np.fft.irfft(spectrum, len(y)))-phi)))
    offset = math.atan2(np.mean(np.sin(error)), np.mean(np.cos(error)))
    residual = np.degrees(np.angle(np.exp(1j*(error-offset))))[50:-50]  # drop FFT edges
    return dict(wingbeat_hz=float(hz), phase_locked_variance=locked, best_channel=best,
                offset_deg=math.degrees(offset), residual_rms_deg=float(np.sqrt(np.mean(residual**2))),
                residual_p95_abs_deg=float(np.quantile(np.abs(residual), .95)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scratch', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    source = a.scratch/'source'
    if not (source/'robot_upper7_lower8.xml').exists():
        P.prepare_source(source)
    P.worker_init(str(source))
    result = {str(P.SEED0+3000+k): analyse(flight(P.SEED0+3000+k, k)) for k in range(3)}
    result['scope'] = 'PID-only nominal flights; offline non-causal estimate = upper bound on accuracy'
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
