"""Shared-prefix evaluation for plan v6: every learned candidate flies the same physical trajectory until the command
onset T0 (the gate is closed: the reflex PID and the classical pulse law act, the network only ticks), so the prefix is
simulated once per case, snapshotted, and each candidate continues from a copy. The candidate's network state at T0 is
rebuilt by replaying the recorded inputs through it (bookkeeping identical to descend8.Learned8.act).

Exactness: `selftest` checks that every result field equals bench8.run_episode8 with the same policy (bit for bit).

Snapshot: copy.deepcopy of the episode (env, oracle, policy, logs) with the static native pieces shared (model, model
joint views, native libraries, wing geometry); the ctypes Motor / SensorState are copied by value; the upset wrapper
(a closure over the env, upset.install) is re-installed on the copy.
"""
import copy
import copyreg
import math
import numpy as np
from flapsim.rl import native as _N
from ..topology_hover import sensors as _S
from ..snn_pid_compare.common import DT
from ..snn_pid_compare.environment import LOW, HIGH, CENTER, HALF
from ..robust_control.environment import ReferencePID
from ..neuro_gates import gates
from . import bench as B
from . import bench8 as E
from . import upset as U
from . import descend8 as D8

for _T in (_N.Motor, _S.SensorState):
    copyreg.pickle(_T, lambda s, T=_T: (T.from_buffer_copy, (bytes(s),)))


def _memo(env):
    be = env.air.env.sim.backend
    shared = [be.lib, be.sensor_lib, be.wings, env.sim.model, *getattr(be, 'arrays', []), *env.air.env.internal]
    return {id(x): x for x in shared}


def snapshot(state):
    """state: dict with 'env' and anything referencing it."""
    out = copy.deepcopy(state, _memo(state['env']))
    env = out['env']
    del env.air.__dict__['advance']
    U.install(env)
    env.upset_axes = dict(state['env'].upset_axes)
    return out


class Episode:
    """bench8.run_episode8 split into setup / tick / finish (same code path, same order of operations)."""

    def __init__(self, env, task, c, policy):
        env.configure(c)
        env.reset(seed=c['seed'])
        env.upset_axes = {}
        p, (sp, sd) = env.p, B.PID_GAINS
        env.teacher = ReferencePID(dict(p, kp=[sp*k for k in p['kp']], kd=[sd*k for k in p['kd']], ki=[sp*k for k in p['ki']]), env.air.last_rpy)
        env._teacher_step = -1
        policy.reset(c)
        self.s = dict(env=env, oracle=gates.Oracle(env, 'derocked'), policy=policy, rows=[], pos0=None, fwd=None, left=None,
                      goal_done=False, goal_t=None, pulses=0, big=0, out=None, done=False, xs=[], task=task, c=c)

    def t(self):
        return self.s['env'].air.steps*DT

    def tick(self):
        s = self.s
        env, task, c = s['env'], s['task'], s['c']
        d = env.sim.data
        T0 = B.T0
        phi = math.radians(c['dir_deg'])
        u_att = E.att_dir(c['dir_deg'])
        hold = int(round(B.HOLD_S/DT))
        t, now = env.air.steps*DT, float(d.time)
        rel = t-T0
        if s['pos0'] is None and t >= T0:
            pos, _, _ = env.air.env._state()
            s['pos0'] = pos.copy()
            yaw = float(env.last_reading[2])
            s['fwd'] = np.array([math.cos(yaw), math.sin(yaw)])
            s['left'] = np.array([-s['fwd'][1], s['fwd'][0]])
        cmd = np.zeros(4)
        if rel >= 0 and not s['goal_done']:
            k = 0 if task == 'nod' else 2
            cmd[k], cmd[k+1] = math.cos(phi), math.sin(phi)
        x = E.features8(env.last_reading, cmd)
        if rel < 0:
            s['xs'].append((t, x))
        fire, ref = s['policy'].act(env, now, t, x, cmd, rel)
        ref = np.asarray(ref, float)
        if ref.ndim == 0:
            ref = np.array([0., float(ref)])
        ref = np.clip(ref, -B.REF_MAX, B.REF_MAX)
        env.teacher.rpy_reference = ref
        command = env.teacher_command()
        if fire and t >= gates.HANDOVER_T:
            command[2] = HIGH[2]
            s['pulses'] += 1
        _, _, term, trunc, out = env.step((np.clip(command, LOW, HIGH)-CENTER)/HALF)
        s['oracle'].update()
        pos, vel, R = env.air.env._state()
        roll_true, pitch_true = (float(v) for v in s['oracle'].queue[-1][:2])
        tnow = env.air.steps*DT
        along = float(np.degrees(np.array([roll_true, pitch_true])@u_att))
        s['rows'].append([tnow, pitch_true, roll_true, *pos, *vel, math.acos(min(1., max(-1., float(R[2, 2])))), *ref, float(fire), along])
        if task == 'nod' and not s['goal_done'] and rel >= 0:
            s['big'] = s['big']+1 if along >= E.NOD_DEG else 0
            if s['big'] >= hold:
                s['goal_done'], s['goal_t'] = True, tnow-T0
        if task == 'dash' and not s['goal_done'] and s['pos0'] is not None:
            dvec = math.cos(phi)*s['fwd']+math.sin(phi)*s['left']
            if float((pos[:2]-s['pos0'][:2])@dvec) >= B.DASH_M['value']:
                s['goal_done'], s['goal_t'] = True, tnow-T0
        s['out'] = out
        s['done'] = bool(term or trunc)

    def finish(self, trace=False):
        s = self.s
        return E.score8(s['task'], s['c'], s['out']['episode'], np.array(s['rows']), s['pos0'], s['fwd'], s['left'],
                        s['pulses'], s['goal_t'], trace)


def run_cases(env, task, c, specs, trace=False):
    """Fly every spec in `specs` ([(name, spec)]) on case c; the pre-T0 prefix is simulated once."""
    ep = Episode(env, task, c, D8.Learned8(specs[0][1]))
    while not ep.s['done'] and ep.t() < B.T0-1e-9:
        ep.tick()
    base = ep.s
    out = []
    for name, spec in specs:
        if base['done']:                                       # crashed before T0: identical for every candidate
            st = snapshot(base)
        else:
            st = snapshot(base)
            pol = st['policy']
            pol.spec = spec
            pol.net = D8.Net8(spec)
            pol.twitch = 0
            for (t, x) in st['xs']:
                pol.replay(t, x)
        e = Episode.__new__(Episode)
        e.s = st
        while not e.s['done']:
            e.tick()
        out.append(dict(e.finish(trace), controller=name))
    return out


def job(args):
    """(task, case, [(name, spec)], trace) -> rows, run in a bench8 pool worker."""
    task, c, specs = args[:3]
    trace = args[3] if len(args) > 3 else False
    return run_cases(B._ENV, task, c, specs, trace)


def run_grouped(ex, items, chunk=8):
    """items: [(name, task, case, spec)]; groups by (task, case seed), chunks of `chunk` candidates per job; returns rows
    in the input order."""
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
    for idx, rows in zip(index, ex.map(job, jobs, chunksize=1)):
        for k, r in zip(idx, rows):
            out[k] = r
    return out


def selftest(n_specs=3):
    """Exactness against bench8.run_episode8 for nod and dash, several random specs, one case each."""
    import json
    E.load_signs()
    B.DASH_M['value'] = .15
    B.worker_init(str(E.SOURCE), .15)
    env = B._ENV
    law = B.law_tuned()
    up = json.loads(open('artifacts/spike_steer/plan_v4/r5/train/imitate_spec_s1.json').read())['upset_law']
    specs = []
    for k in range(n_specs):
        sp = D8.SNN8(seed=10+k).export(gate='command', tail_s=1., base_law=law, upset_law=up, upset_tilt_deg=25., refractory_s=.037)
        th = D8.flatten(sp)
        th = th+np.random.default_rng(k).standard_normal(th.size)*.3      # make the outputs active
        specs.append((f's{k}', D8.unflatten(sp, th)))
    bad = 0
    for task in ('nod', 'dash'):
        c = E.cases8(task, E.BASES['tune'], 3)[2]
        fast = run_cases(env, task, c, specs, trace=True)
        for (name, spec), f in zip(specs, fast):
            ref = E.run_episode8(env, task, c, D8.Learned8(spec), trace=True)
            same = all(np.array_equal(np.asarray(ref[k], dtype=object), np.asarray(f[k], dtype=object)) for k in ref)
            bad += not same
            print(task, name, 'identical' if same else 'DIFFERENT', 'score', ref['score'], f['score'], 'pulses', ref['pulses'], f['pulses'])
    assert bad == 0, f'{bad} mismatches'
    print('cache8 selftest ok')


if __name__ == '__main__':
    selftest()
