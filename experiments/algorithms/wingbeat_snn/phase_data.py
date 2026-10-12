"""Wingbeat-phase estimation from raw IMU: recorded flights, offline metrics, closed-loop adapter.

Flights are flown by the P1-retuned hand law with the classical PLL, so the
recorded IMU contains the drive modulation the estimator will see in use.

    python -m experiments.algorithms.wingbeat_snn.phase_data record --scratch <dir> --output runs/wingbeat_snn/phase_data
"""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import numpy as np
from ..snn_pid_compare.common import prepare_source
from .plant import Plant, HandLaw, cases, gusts, DT
from ..wingbeat_feasibility.preliminary import wrap

RETUNED = Path(__file__).resolve().parents[3]/'artifacts/wingbeat_snn_p1/retune.json'
PEAKS = (0., .5, 1., 1.25)
SPLIT_PLAN = dict(training=(48, 700_000), validation=(16, 300_000), test=(16, 300_000))
COLUMNS = ['time', 'roll', 'pitch', 'yaw', 'gx', 'gy', 'gz', 'ax', 'ay', 'az', 'crank', 'pll_crank', 'pll_omega',
           'wind_x', 'wind_y']
_PLANT = None


def raw_inputs(reading):
    """The 8 raw IMU inputs (same scaling as the controller observations)."""
    r = reading
    return np.array([r[0]/.35, r[1]/.35, r[3]/10., r[4]/10., r[5]/10., r[6]/9.81, r[7]/9.81, r[8]/9.81])


def worker_init(source):
    global _PLANT
    _PLANT = Plant(source)


def record_task(task):
    c, peak, params = task
    out = _PLANT.run(c, HandLaw(**params), gusts(peak) if peak else (), record=True)
    return dict(seed=c['seed'], peak=peak, termination=out['termination'], trace=out['trace'])


def record(source, output, workers=18, scale=1.):
    source, output = Path(source), Path(output)
    if not (source/'robot_upper7_lower8.xml').exists():
        prepare_source(source)
    params = json.loads(RETUNED.read_text())['best']['params']
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    with ProcessPoolExecutor(workers, initializer=worker_init, initargs=(str(source),)) as pool:
        for split, (count, offset) in SPLIT_PLAN.items():
            case_list = cases(split, max(1, round(count*scale)), 8., offset=offset)
            tasks = [(c, PEAKS[i % len(PEAKS)], params) for i, c in enumerate(case_list)]
            rows = list(pool.map(record_task, tasks))
            np.savez_compressed(output/f'{split}.npz', columns=np.array(COLUMNS),
                                seeds=np.array([r['seed'] for r in rows]), peaks=np.array([r['peak'] for r in rows]),
                                terminations=np.array([r['termination'] for r in rows]),
                                **{f'episode_{i:03d}': r['trace'] for i, r in enumerate(rows)})
            summary[split] = dict(episodes=len(rows), completed=sum(r['termination'] == 'time_limit' for r in rows),
                                  ticks=int(sum(len(r['trace']) for r in rows)))
    (output/'summary.json').write_text(json.dumps(summary, indent=1))
    return summary


def load(path):
    data = np.load(path)
    episodes = [data[k] for k in sorted(k for k in data.files if k.startswith('episode_'))]
    return dict(episodes=episodes, peaks=data['peaks'], seeds=data['seeds'], terminations=data['terminations'])


def inputs_targets(trace):
    x = np.stack([raw_inputs(row[1:10]) for row in trace]) if len(trace) else np.zeros((0, 8))
    crank = trace[:, 10].astype(np.float64)
    return x, np.stack([np.cos(crank), np.sin(crank)], axis=1)


def phase_metrics(trace, estimate, settle=1.5, lock_deg=15., hold=.5):
    """Phase error of an estimate (unwrapped radians, one per tick) against the recorded true crank."""
    t = trace[:, 0].astype(np.float64)
    err = np.degrees([wrap(e-c) for e, c in zip(estimate, trace[:, 10].astype(np.float64))])
    wind = np.hypot(trace[:, 13], trace[:, 14]) > .05
    after = t >= settle
    ok = np.abs(err) < lock_deg
    lock = None
    need = int(round(hold/DT))
    run = 0
    for i, flag in enumerate(ok):
        run = run+1 if flag else 0
        if run >= need:
            lock = float(t[i-need+1])
            break
    def stats(mask):
        e = err[mask]
        return dict(rms=float(np.sqrt(np.mean(e**2))), p95=float(np.quantile(np.abs(e), .95)), n=int(mask.sum())) \
            if mask.any() else None
    return dict(all=stats(after), calm=stats(after & ~wind), gust=stats(after & wind), lock_time_s=lock)


class _Notched:
    """Shared by phase sources other than the PLL: ``slow`` is the pitch gyro minus a sliding
    least-squares wingbeat fit (fundamental + 2nd harmonic) on this source's own phase."""
    def _notch(self, y):
        self.buffer.append((self.theta, y))
        self.slow = y
        if len(self.buffer) == self.window:
            theta = np.array([b[0] for b in self.buffer])
            X = np.c_[np.cos(theta), np.sin(theta), np.cos(2*theta), np.sin(2*theta), np.ones_like(theta)]
            coef, *_ = np.linalg.lstsq(X, np.array([b[1] for b in self.buffer]), rcond=None)
            self.slow = y-float(X[-1, :4]@coef[:4])

    def crank(self, t):
        return self.theta+self.omega*(t-self.t)


class NetPhase(_Notched):
    """Closed-loop adapter: a model mapping raw IMU to (cos, sin) of the crank replaces the PLL.

    Frequency is a 50 ms low-pass of successive phase increments."""
    def __init__(self, model, window=25, tau=.05):
        self.model, self.window, self.tau = model, window, tau
        self.model.reset()
        self.theta, self.t, self.omega, self.slow = None, None, 2*math.pi*25., 0.
        self.buffer = deque(maxlen=window)

    def update_reading(self, t, reading):
        c, s = self.model.step(raw_inputs(reading))
        phi = math.atan2(s, c)
        if self.theta is None:
            self.theta = phi
        else:
            step = wrap(phi-wrap(self.theta))
            dt = max(t-self.t, 1e-9)
            self.omega += (1-math.exp(-dt/self.tau))*(step/dt-self.omega)
            self.theta += step
        self.t = t
        self._notch(float(reading[4]))


class TruthPhase(_Notched):
    """Perfect phase: the true crank angle and rate (an upper bound, not a sensor)."""
    def __init__(self, window=25):
        self.window, self.plant = window, None
        self.theta, self.t, self.omega, self.slow = None, None, 2*math.pi*25., 0.
        self.buffer = deque(maxlen=window)

    def attach(self, plant):
        self.plant = plant

    def update_reading(self, t, reading):
        d = self.plant.env.sim.data
        self.theta, self.omega, self.t = float(d.qpos[self.plant.crank]), float(d.qvel[self.plant.crank_dof]), t
        self._notch(float(reading[4]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['record'])
    p.add_argument('--scratch', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=18)
    p.add_argument('--scale', type=float, default=1., help='fraction of the 48/16/16 flights (smoke tests)')
    a = p.parse_args()
    print(json.dumps(record(a.scratch/'source', a.output, a.workers, a.scale), indent=1))


if __name__ == '__main__':
    main()
