"""Plan v10 controller inputs: bench10 tasks and scoring unchanged, the learned layer sees 12 more input channels.

v9 diagnosis (plan-v8 blind rows, v7-full controllers): SNN + ES reaches dash targets ~30 % sooner than the classical
opponent but brakes worse (roll-class dash overshoot 0.10-0.17 m vs 0.04 m, longer settling; larger redirect overrun);
the classical schedule brakes on a timer (go_beats / brake_beats) while the network has no notion of time since the
command (its LIF / delay memory spans tens of ms) and no height signal (v8: upward height escapes). v10 adds, inside the
policy (so tasks, scoring, opponents and the prefix cache stay bench10's):

  * 8 phasic command channels: the 8 rectified command channels times exp(-(t - t_on)/TAU_PH), t_on = the last command
    onset or direction change (an adapting copy of the tonic command, as in insect descending pathways);
  * 4 height channels: rectified z / Z_SCALE and vz / VZ_SCALE from the original interface's height reading
    (reading[9], reading[10]: range-finder height relative to the start, vertical speed), clipped to +-CLIP.

Input order: features8 (25) | phasic (8) | height (z+, vz+, z-, vz-) -> N_IN = 37.

    python -m experiments.algorithms.spike_steer.bench11 selftest
"""
import math
import numpy as np
from . import bench as B
from . import bench8 as E
from . import bench9 as T
from . import bench10 as T10
from . import descend8 as D8

TAU_PH, Z_SCALE, VZ_SCALE, CLIP = .3, .1, .5, 5.
N_IN = 25+8+4
CMD = slice(17, 25)                      # features8 command channels


class Augment:
    """Stateful 25 -> 37 input map (state: last command vector and its onset)."""

    def reset_aug(self):
        self.cmd_prev, self.t_on = None, None

    def augment(self, env, t, x):
        c8 = np.asarray(x[CMD], float)
        if c8.any():
            if self.cmd_prev is None or not np.array_equal(c8, self.cmd_prev):
                self.t_on = t
            ph = c8*math.exp(-(t-self.t_on)/TAU_PH)
        else:
            ph = np.zeros(8)
        self.cmd_prev = c8 if c8.any() else None
        r = env.last_reading
        h = np.clip(np.array([float(r[9])/Z_SCALE, float(r[10])/VZ_SCALE]), -CLIP, CLIP)
        return np.r_[x, ph, np.maximum(h, 0.), np.maximum(-h, 0.)]


class Learned11(D8.Learned8, Augment):
    def reset(self, c):
        super().reset(c)
        self.reset_aug()
        self.xs11 = []

    def act(self, env, now, t, x, cmd, rel):
        x11 = self.augment(env, t, x)
        if rel < 0:
            self.xs11.append((t, x11))
        return super().act(env, now, t, x11, cmd, rel)


class Traced11(T.Classical9, Augment):
    """Classical opponent logging the 37-D input (imitation data; descend9.Traced9 with augmented x)."""

    def __init__(self, law, schedule):
        super().__init__(law, schedule)

    def reset(self, c):
        super().reset(c)
        self.reset_aug()
        self.log = []

    def act(self, env, now, t, x, cmd, rel):
        x11 = self.augment(env, t, x)
        pulse, ref = super().act(env, now, t, x, cmd, rel)
        if t >= 2.:
            u = -(self.law['k_theta']*env.last_reading[1]+self.law['k_q']*self.steer.estimator.slow)
            self.log.append(np.r_[x11, float(pulse), float(np.sign(u)), np.asarray(ref, float)])
        return pulse, ref


def job(args):
    name, task, c, kind, arg = args[:5]
    trace = args[5] if len(args) > 5 else False
    pol = T.Classical9(arg.get('law'), arg.get('schedule')) if kind == 'classical' else Learned11(arg)
    return dict(T10.run_episode10(B._ENV, task, c, pol, trace), controller=name)


def run_jobs(ex, jobs):
    return list(ex.map(job, jobs, chunksize=1))


def run_cases(env, task, c, specs, trace=False):
    """bench10.run_cases with Learned11: the prefix is flown once; each candidate's network replays the recorded 37-D
    pre-command inputs (they do not depend on the network: the gate is closed before t0)."""
    from . import cache8 as C8
    ep = T10.Episode(env, task, c, Learned11(specs[0][1]))
    while not ep.s['done'] and ep.t() < c['t0']-1e-9:
        ep.tick()
    base = ep.s
    out = []
    for name, spec in specs:
        st = C8.snapshot(base)
        if not base['done']:
            pol = st['policy']
            pol.spec, pol.net, pol.twitch = spec, D8.Net8(spec), 0
            for (t, x) in pol.xs11:
                pol.replay(t, x)
        e = T10.Episode.__new__(T10.Episode)
        e.s = st
        while not e.s['done']:
            e.tick()
        out.append(dict(e.finish(trace), controller=name))
    return out


def job_group(args):
    task, c, specs = args[:3]
    return run_cases(B._ENV, task, c, specs, args[3] if len(args) > 3 else False)


def run_grouped(ex, items, chunk=8):
    groups = {}
    for k, (name, task, c, spec) in enumerate(items):
        groups.setdefault((task, c['seed']), (task, c, []))[2].append((k, name, spec))
    jobs, index = [], []
    for task, c, lst in groups.values():
        for i in range(0, len(lst), chunk):
            part = lst[i:i+chunk]
            jobs.append((task, c, [(n, s) for _, n, s in part]))
            index.append([k for k, _, _ in part])
    out = [None]*len(items)
    for idx, rows in zip(index, ex.map(job_group, jobs, chunksize=1)):
        for k, r in zip(idx, rows):
            out[k] = r
    return out


def widen(spec, seed=0, scale=0.):
    """A 25-input spec padded to 37 inputs (new weights = scale * N(0,1); scale 0 reproduces the 25-input net)."""
    s = dict(spec)
    W = np.asarray(spec['W_in'], float)
    H, n, K = W.shape
    rng = np.random.default_rng(seed)
    s['W_in'] = np.concatenate([W, scale*rng.standard_normal((H, N_IN-n, K))], 1).tolist()
    s['inputs'] = N_IN
    return s


def selftest():
    import json
    from pathlib import Path
    from . import cache8 as _C8  # noqa: F401
    E.load_signs()
    B.DASH_M['value'] = .15
    B.worker_init(str(E.SOURCE), .15)
    spec25 = json.loads(Path('artifacts/spike_steer/plan_v7/full/snn/s5/selected.json').read_text())
    # 1) zero-padded 37-input net == the 25-input net (the augmentation is inert with zero weights)
    for task in ('dash', 'redirect'):
        c = T.cases9(task, 2098200000, 4, offset=777)[1]
        r25 = T10.run_episode10(B._ENV, task, c, D8.Learned8(spec25))
        r37 = T10.run_episode10(B._ENV, task, c, Learned11(widen(spec25)))
        assert r25['score'] == r37['score'] and r25['pulses'] == r37['pulses'], (task, r25['score'], r37['score'])
        print('zero-pad', task, 'ok %.4f' % r25['score'])
    # 2) prefix cache == plain episode, with live new weights
    specs = [('a', widen(spec25, 1, .3)), ('b', widen(spec25, 2, .3)), ('z', widen(spec25))]
    for task in ('evade', 'redirect', 'dash'):
        c = T.cases9(task, 2098200000, 4, offset=888)[2]
        cached = run_cases(B._ENV, task, c, specs)
        for (name, sp), rc in zip(specs, cached):
            rp = T10.run_episode10(B._ENV, task, c, Learned11(sp))
            for k, v in rp.items():
                w = rc[k]
                same = (v == w) or (isinstance(v, float) and isinstance(w, float) and math.isnan(v) and math.isnan(w))
                assert same, (task, name, k, v, w)
            print('cache', task, name, 'ok %.4f' % rp['score'], rp['termination'])
    # 3) phasic channels: onset resets on direction change, decays with TAU_PH
    a = Augment()
    a.reset_aug()

    class Env:
        last_reading = np.zeros(11)
    x = np.zeros(25)
    x[17] = 1.
    p0 = a.augment(Env, 1., x)[25]
    p1 = a.augment(Env, 1.3, x)[25]
    assert p0 == 1. and abs(p1-math.exp(-1.)) < 1e-12
    x2 = np.zeros(25)
    x2[18] = 1.
    assert a.augment(Env, 1.4, x2)[26] == 1.
    print('bench11 selftest ok')


if __name__ == '__main__':
    selftest()
