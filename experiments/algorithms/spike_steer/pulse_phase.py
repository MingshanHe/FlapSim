"""Where in the wingbeat does the learned layer fire? True crank phase (carrier_2_joint, the convention of
neuro_gates.timing: 135 deg raises pitch, 315 deg lowers it) at every motor-pulse onset, for the v11 SNN arms and the
classical opponent, on the plan-v15 nominal cases and on the plan-v12 primary kick cases.

Onsets are labelled by source: 'net' (the SNN's own output spike while it is in charge: command on or within the tail;
Learned8.onsets increments) and 'law' (the shared phase pulse law outside the gate, and every onset of the classical arm).
Per onset: phase, IMU pitch / roll and their rates (instantaneous, and averaged over the last wingbeat: the raw gyro
rocks with the wingbeat, so its sign is tied to the crank phase), the layer's pitch / roll reference, time from t0, task, direction.

    python -m experiments.algorithms.spike_steer.pulse_phase run --per-task 16 --workers 19
"""
import argparse
import collections
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from . import bench as B
from . import bench8 as E
from . import bench9 as T
from . import bench10 as T10
from . import descend8 as D8
from . import descend9 as D9
from .blind12 import load_arms, cases12, TESTS as T12
from .blind15 import cases15

ART = Path('artifacts/spike_steer/pulse_phase')
_R = {}
BEAT_TICKS = 19                  # 38 ms, one wingbeat: wingbeat-averaged rates remove the rocking


def _wrap(cls, net):
    orig = cls.act

    def act(self, env, now, t, x, cmd, rel):
        n0 = getattr(self, 'onsets', 0)
        r = env.last_reading
        _R['buf'].append((float(r[4]), float(r[3]), float(r[1])))
        out = orig(self, env, now, t, x, cmd, rel)
        fire = bool(out[0])
        if fire and not _R['prev']:
            d, m = env.sim.data, env.sim.model
            q = float(d.qpos[m.joint('carrier_2_joint').qposadr[0]])
            b = np.array(_R['buf'])
            ref = np.atleast_1d(np.asarray(out[1], float))
            src = 'net' if (net and getattr(self, 'onsets', 0) > n0) else 'law'
            _R['rows'].append(dict(src=src, phase_deg=math.degrees(q) % 360., t=float(t), rel=float(rel), roll=float(r[0]), pitch=float(r[1]),
                                   roll_rate=float(r[3]), pitch_rate=float(r[4]), ref=ref.tolist(),
                                   pitch_rate_beat=float(b[:, 0].mean()), roll_rate_beat=float(b[:, 1].mean()),
                                   pitch_trend=float(b[-1, 2]-b[0, 2])/max(len(b)-1, 1)/.002))
        _R['prev'] = fire
        return out
    cls.act = act


def _init():
    E._init(str(E.SOURCE), .15, dict(E.SIGNS))
    _wrap(D8.Learned8, True)
    _wrap(T.Classical9, False)


def job(args):
    name, kind, task, c, payload, setname = args
    _R.update(rows=[], prev=False, buf=collections.deque(maxlen=BEAT_TICKS))
    r = T10.job((name, task, c, kind, payload, False))
    for o in _R['rows']:
        o.update(arm=name, set=setname, task=task, seed=c['seed'], dir_deg=c.get('dir_deg'), flown=r['flown'])
    return _R['rows']


def stage_run(a):
    arms, law = load_arms()
    keep = ['classical']+[f'snn11_s{s}' for s in range(1, 6)]
    K = T12['kick']
    sets = dict(nominal={t: cases15(t, 'nominal', a.per_task) for t in D9.TASKS},
                kick={t: cases12(t, K['base'], a.per_task, K['condition'], K['kick'], K.get('mag'), K.get('kick_t')) for t in D9.TASKS})
    jobs = []
    for name, kind, payload in arms:
        if name not in keep:
            continue
        for sname, cs in sets.items():
            for t in D9.TASKS:
                for c in cs[t]:
                    arg = dict(law=law, schedule=payload[t][T.klass(c)]) if kind == 'classical' else payload
                    jobs.append((name, kind, t, c, arg, sname))
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
        rows = [o for rr in ex.map(job, jobs, chunksize=1) for o in rr]
    ART.mkdir(parents=True, exist_ok=True)
    (ART/'onsets.json').write_text(json.dumps(rows))
    for name in keep:
        for src in ('net', 'law'):
            ph = np.radians([o['phase_deg'] for o in rows if o['arm'] == name and o['src'] == src])
            if len(ph) == 0:
                continue
            z = np.exp(1j*ph).mean()
            print('%-10s %-3s onsets %5d  circular mean %6.1f deg  resultant length %.3f' % (name, src, len(ph), math.degrees(np.angle(z)) % 360, abs(z)), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['run'])
    p.add_argument('--per-task', type=int, default=16)
    p.add_argument('--workers', type=int, default=19)
    a = p.parse_args()
    E.load_signs()
    B.DASH_M['value'] = .15
    stage_run(a)


if __name__ == '__main__':
    main()
