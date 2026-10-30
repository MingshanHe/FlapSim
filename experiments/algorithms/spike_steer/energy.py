"""Electrical energy of the motor per episode for every arm (classical, classical + ES, SNN11, MLP + PPO) on the plan-v15
nominal cases. The native step computes the motor voltage and current per physics substep from the shaft speed and the
speed command (flapsim/rl/native_step.cpp: V = R (g (target - v) / kt + i0 tanh v) + ke v, clamped; I = (V - ke v) / R,
clamped); here backend.advance is split into CHUNK-substep calls (bit-identical, checked by selftest) and V I is sampled at
the start of every chunk. Electrical power P = V I; the shaft power kt (I - loss) v is kept too.

    python -m experiments.algorithms.spike_steer.energy selftest
    python -m experiments.algorithms.spike_steer.energy run --per-task 16 --workers 19
"""
import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from . import bench as B
from . import bench8 as E
from . import bench9 as T
from . import bench10 as T10
from . import descend9 as D9
from .blind12 import load_arms
from .blind15 import cases15

ART = Path('artifacts/spike_steer/energy')
CHUNK = 20                       # substeps per sample: 0.1 ms
WINDOW = 2.5                     # manoeuvre window after t0 (s)
_A = {}


def _motor_vi(be, command):
    mo, v = be.motor, float(be.data.qvel[be.motor.dof])
    loss = mo.i0*math.tanh(v)
    target = min(max(command, -mo.velocity), mo.velocity)
    V = mo.resistance*(mo.gain*(target-v)/mo.kt+loss)+mo.ke*v
    V = min(max(V, -mo.voltage), mo.voltage)
    I = min(max((V-mo.ke*v)/mo.resistance, -mo.current), mo.current)
    return V, I, v, loss


def install(env, chunk=CHUNK):
    be = env.sim.backend
    orig = be.advance
    dt = env.sim.model.opt.timestep

    def advance(steps, command, joint_forces, body_forces, controls):
        done = 0
        while done < steps:
            n = min(chunk, steps-done)
            V, I, v, loss = _motor_vi(be, float(command[0]))
            t = float(be.data.time)
            _A['e'] += V*I*n*dt
            _A['shaft'] += be.motor.kt*(I-loss)*v*n*dt
            if t >= _A['t0']:
                _A['e_win'] += V*I*n*dt*(t < _A['t0']+WINDOW)
            _A['vsat'] += n*(abs(V) >= be.motor.voltage-1e-12)
            _A['n'] += n
            orig(n, command, joint_forces, body_forces, controls)
            done += n
    be.advance = advance


def _reset(c):
    _A.update(e=0., shaft=0., e_win=0., vsat=0, n=0, t0=float(c['t0']))


def _init(kind):
    if kind == 'ppo':
        from . import ppo_blind14 as Q
        Q._init(str(E.SOURCE))
        _A['Q'] = Q
    else:
        E._init(str(E.SOURCE), .15, dict(E.SIGNS))
    install(B._ENV)


def job(args):
    name, kind, task, c, payload = args
    _reset(c)
    if kind == 'ppo':
        r = _A['Q'].job((int(name[-1]), payload, task, c))
    else:
        r = T10.job((name, task, c, kind, payload, False))
    dur = _A['n']*B._ENV.sim.model.opt.timestep
    return dict(name=name, task=task, seed=c['seed'], cls=r.get('cls'), flown=r['flown'], score=r['score'], seconds=dur,
                energy_J=_A['e'], shaft_J=_A['shaft'], window_J=_A['e_win'], mean_W=_A['e']/max(dur, 1e-9), vsat_frac=_A['vsat']/max(_A['n'], 1))


def ppo_ckpts():
    P = Path('artifacts/spike_steer/ppo_direct/v7full')
    out = {}
    for s in range(1, 6):
        g = json.loads((P/f'reselect_s{s}.json').read_text())['selected']['generation']
        out[f'mlp_ppo_s{s}'] = str(P/f's{s}/centres/ckpt_g{g}.zip')
    return out


def stage_run(a):
    arms, law = load_arms()
    keep = ['classical', 'cls_es_s1', 'cls_es_s2']+[f'snn11_s{s}' for s in range(1, 6)]
    cs = {t: cases15(t, 'nominal', a.per_task) for t in D9.TASKS}
    jobs = []
    for name, kind, payload in arms:
        if name not in keep:
            continue
        for t in D9.TASKS:
            for c in cs[t]:
                arg = dict(law=law, schedule=payload[t][T.klass(c)]) if kind == 'classical' else payload
                jobs.append((name, kind, t, c, arg))
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=('bench',)) as ex:
        rows = list(ex.map(job, jobs, chunksize=1))
    pj = [(n, 'ppo', t, c, ck) for n, ck in ppo_ckpts().items() for t in D9.TASKS for c in cs[t]]
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=('ppo',)) as ex:
        rows += list(ex.map(job, pj, chunksize=1))
    ART.mkdir(parents=True, exist_ok=True)
    (ART/'rows.json').write_text(json.dumps(rows))
    ref = {(r['task'], r['seed']): r for r in rows if r['name'] == 'classical'}
    out = {}
    for n in dict.fromkeys(r['name'] for r in rows):
        R = [r for r in rows if r['name'] == n]
        both = [r for r in R if r['flown'] and ref[(r['task'], r['seed'])]['flown']]
        out[n] = dict(episodes=len(R), crashes=sum(not r['flown'] for r in R), mean_W=float(np.mean([r['mean_W'] for r in R if r['flown']])),
                      window_J=float(np.mean([r['window_J'] for r in R if r['flown']])),
                      window_J_minus_classical=float(np.mean([r['window_J']-ref[(r['task'], r['seed'])]['window_J'] for r in both])),
                      energy_J_per_task={t: float(np.mean([r['energy_J'] for r in R if r['task'] == t and r['flown']])) for t in D9.TASKS},
                      vsat_frac=float(np.mean([r['vsat_frac'] for r in R])))
        print('%-12s crashes %d  mean %.3f W  window %.3f J  (vs classical %+.4f J, paired on %d)  V-saturated %.2f %%' % (
            n, out[n]['crashes'], out[n]['mean_W'], out[n]['window_J'], out[n]['window_J_minus_classical'], len(both), 100*out[n]['vsat_frac']), flush=True)
    (ART/'energy.json').write_text(json.dumps(dict(chunk_substeps=CHUNK, window_s=WINDOW, cases_per_task=a.per_task, arms=out), indent=1))


def selftest():
    """CHUNK splitting is bit-identical to the unsplit step, and the sampled energy converges (CHUNK 20 vs 4)."""
    arms, law = load_arms()
    spec = dict((n, p) for n, k, p in arms)['snn11_s4']
    c = dict(cases15('dash', 'nominal', 1)[0], seconds=3.2)
    res = {}
    for ch in (None, 20, 4):
        E._init(str(E.SOURCE), .15, dict(E.SIGNS))
        if ch:
            install(B._ENV, ch)
        _reset(c)
        r = T10.job(('snn', 'dash', c, 'learned', spec, True))
        res[ch] = (np.array(r['trace']), _A['e'])
        print('chunk', ch, 'energy %.6f J' % _A['e'] if ch else '')
    assert np.array_equal(res[None][0], res[20][0], equal_nan=True) and np.array_equal(res[None][0], res[4][0], equal_nan=True), 'chunked stepping changed the trajectory'
    print('trajectories bit-identical; energy chunk 20 vs 4: %.4f %%' % (100*(res[20][1]/res[4][1]-1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['selftest', 'run'])
    p.add_argument('--per-task', type=int, default=16)
    p.add_argument('--workers', type=int, default=19)
    a = p.parse_args()
    E.load_signs()
    B.DASH_M['value'] = .15
    selftest() if a.stage == 'selftest' else stage_run(a)


if __name__ == '__main__':
    main()
