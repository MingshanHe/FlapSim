"""Plan v4 round 4: command-gated descending layer with a wingbeat refractory period (at most one learned pulse onset per
37 ms wingbeat, like an insect steering muscle), searched WITH the constraint in the loop, a tilt penalty only near the
termination line, and a fresh nod/dash blind set.

Why (artifacts/spike_steer/plan_v4/r3/diag/height_escape.md, plan_v4/r4/precheck/):
  * round-3 'height_escape' crashes: the learned layer fires 10-24 pulses/s in the 1 s hand-back tail, each pulse drives
    the motor to 3807 rad/s while the height PID is clamped at the 3115 rad/s floor, so the robot climbs out of +1 m;
  * capping onsets to one per wingbeat removes every height escape (4 -> 0 over 896 training episodes of 7 selected
    specs) but, bolted on without re-tuning, raises tilt crashes 5 -> 19: the pulse timing doubles as pitch braking,
    so the cap must be present during the search;
  * the round-3 tilt penalty from 45 deg made the nod conservative (round-2 specs on the round-3 blind set: 0.943 /
    0.952; round-3 specs: 0.97-1.00), so round 4 penalises only near the 80 deg termination line (from 60 deg, x0.3).

Everything else (network, inputs, gate, tail, righting reflex, base/upset laws, ES settings, start specs = round-2 gated
imitation specs, 16-case zero-crash selection) is the round-3 code. This module only rebinds round-3 hooks at run time;
no existing module is edited.

    python -W ignore -m experiments.algorithms.spike_steer.descend_r4 selftest
    python -W ignore -m experiments.algorithms.spike_steer.descend_r4 prepare  --output runs/spike_steer/bench_r4
    python -W ignore -m experiments.algorithms.spike_steer.descend_r4 finetune --output runs/spike_steer/bench_r4 --seed 1 --generations 60 --hours 14 --workers 10
    python -W ignore -m experiments.algorithms.spike_steer.descend_r4 reselect --output runs/spike_steer/bench_r4 --seed 1 --workers 30
    python -W ignore -m experiments.algorithms.spike_steer.descend_r4 evaluate --output runs/spike_steer/bench_r4 --seed 1 --spec <selected> --tag r4_s1 --variants intact
"""
import json
import sys
import time
from pathlib import Path
from . import bench as B
from . import descend_gated as G
from . import descend_r3 as R
from . import maneuver_v2 as M

REFRACTORY_S = .037            # one wingbeat (26.7 Hz)
SOFT_TILT_DEG = 60.            # ES / validation penalty starts here (true tilt peak); round 3: 45
PENALTY_PER_DEG = .3           # ratio units per degree beyond SOFT_TILT_DEG; round 3: 0.1
HARD_TILT_DEG = 70.            # selection: centres reaching this on any of the 16 validation cases are not eligible; round 3: 60
TEST_BASE_R4 = 2_096_500_000   # fresh nod/dash blind set (r1-2: 2_096_100_000, r3: 2_096_400_000; train cases < 2_096_310_000)
SEEDS = (1, 2, 3, 4, 5)


class RefractoryLearned(G.GatedLearned):
    """GatedLearned (round 2/3, no envelope) + refractory period: while the learned layer is in charge, a learned pulse
    onset within refractory_s of the previous accepted onset is suppressed. Reflex/classical pulses are untouched, so
    hover and upset (command never on) are identical to rounds 2-3 by construction."""

    def reset(self):
        super().reset()
        self.refr = float(self.spec['refractory_s'])
        self.last_onset = None
        self.cancelled = 0
        self.onsets = 0

    def act(self, env, now, t, x, cmd, rel):
        fire, ref = G._Learned.act(self, env, now, t, x, cmd, rel)       # net ticks every step (keeps its state)
        law_pulse = False
        if self.steer:
            law_pulse, _ = self.steer.step(env, now, t)                   # pulse law ticks every step (keeps the PLL)
        on = bool(cmd.any())
        if self.was_on and not on:
            self.off_t = t
        self.was_on = on
        in_charge = on or (self.off_t is not None and t-self.off_t < self.tail)
        if self.steer_up:
            up_pulse, _ = self.steer_up.step(env, now, t)
            if (not self.upset and not in_charge and env is not None
                    and R.math.hypot(float(env.last_reading[0]), float(env.last_reading[1])) > self.tilt_thr):
                self.upset = True                                        # righting reflex (round-2 amendment 1)
            if self.upset:
                law_pulse = up_pulse
        if in_charge:
            if fire and self.twitch == B.TWITCH-1:                        # a fresh learned onset this tick
                if self.last_onset is not None and t-self.last_onset < self.refr:
                    fire, self.twitch = False, 0
                    self.cancelled += 1
                else:
                    self.last_onset = t
                    self.onsets += 1
            return fire, ref
        return bool(law_pulse), 0.


def dispatch_learned(spec):
    if spec.get('gate') == G.GATE and spec.get('refractory_s') is not None:
        return RefractoryLearned(spec)
    return _dispatch_r3(spec)


def cases_r4(task, split, levels=None, n=None, offset=0):
    if task in ('nod', 'dash') and split == 'test':
        return M.cases(TEST_BASE_R4, n or 16, task)
    return R._cases0(task, split, levels, n, offset)


_install_r3 = R.install
_dispatch_r3 = R.dispatch_learned


def install(fresh_test=True):
    """Round-3 hooks, then the round-4 dispatch, blind set and penalty slope. R.install is rebound to this function so
    every round-3 entry point (finetune, evaluate, reselect, pool workers) runs the round-4 configuration."""
    _install_r3(fresh_test=fresh_test)
    B.Learned = dispatch_learned
    B.cases = cases_r4 if fresh_test else R._cases0
    R.PENALTY_PER_DEG = PENALTY_PER_DEG


R.install = install


# ----------------------------------------------------------------------------------------- prepare (pre-registration)
def prepare(a):
    install()
    out, src = Path(a.output), Path(a.source)
    tr = json.loads((out/'train'/'baseline.json').read_text())          # written by _local/r4_baseline.py (claude-ea)
    ev = json.loads((out/'baseline.json').read_text())
    assert ev['normaliser_notes']['test_base_nod_dash'] == TEST_BASE_R4
    tr['prereg'] = dict(tr['prereg'], round=4, envelope_deg=None, soft_tilt_deg=SOFT_TILT_DEG, hard_tilt_deg=HARD_TILT_DEG,
                        penalty_per_deg=PENALTY_PER_DEG, refractory_s=REFRACTORY_S)
    (out/'train'/'baseline.json').write_text(json.dumps(tr))
    seeds = []
    for s in SEEDS:
        f = src/'train'/f'imitate_spec_s{s}.json'
        spec = json.loads(f.read_text())
        spec.pop('cmd_envelope_deg', None)
        spec['refractory_s'] = REFRACTORY_S
        (out/'train'/f'imitate_spec_s{s}.json').write_text(json.dumps(spec, indent=1))
        seeds.append(s)
    print(json.dumps(dict(prepared=str(out), seeds=seeds, normalisers=tr['normalisers'], prereg=dict(
        soft=SOFT_TILT_DEG, slope=PENALTY_PER_DEG, hard=HARD_TILT_DEG, refractory=REFRACTORY_S))), flush=True)


def reselect(a):
    from . import reselect_r3 as S
    sys.argv = [sys.argv[0], '--seed', str(a.seed), '--output', a.output, '--workers', str(a.workers),
                '--tag', a.tag or f'r4_s{a.seed}'] + (['--centres', a.spec] if a.spec else [])
    S.main()


# ----------------------------------------------------------------------------------------- selftest
def selftest():
    install()
    spec = dict(R.Dn.DescendingSNN(hidden=2, seed=0).export(), gate=G.GATE, base_law=None, tail_s=.5, refractory_s=REFRACTORY_S)
    pol = B.Learned(spec)
    assert isinstance(pol, RefractoryLearned)
    assert type(B.Learned({k: v for k, v in spec.items() if k != 'refractory_s'})) is G.GatedLearned
    pol.reset()

    class FakeEnv:
        last_reading = R.np.zeros(11)
    env = FakeEnv()
    x = R.np.zeros(B.N_IN)
    calls = iter([(True, .1), (False, .1), (False, .1), (False, .1), (True, .1), (False, .1), (False, .1), (False, .1)]
                 + [(False, .1)]*8 + [(True, .1)])
    orig = G._Learned.act

    def fake(self, env_, now, t, x_, cmd, rel):                         # learned layer requests onsets at t = 0, 16, 64 ms
        f, r = next(calls)
        if f:
            self.twitch = B.TWITCH-1
        elif self.twitch > 0:
            self.twitch -= 1
            f = True
        return f, r
    G._Learned.act = fake
    try:
        fired = [pol.act(env, 0., k*.004, x, R.np.array([1., 0.]), 0.)[0] for k in range(17)]
    finally:
        G._Learned.act = orig
    assert pol.onsets == 2 and pol.cancelled == 1, (pol.onsets, pol.cancelled, fired)
    assert B.cases('nod', 'test')[0]['seed'] == TEST_BASE_R4 and len(B.cases('dash', 'test')) == 16
    assert B.cases('hover', 'test')[0]['seed'] == R._cases0('hover', 'test')[0]['seed']
    R.SOFT['value'] = SOFT_TILT_DEG
    rows = [dict(task='nod', score=1., tilt_peak_deg=50.), dict(task='nod', score=1., tilt_peak_deg=70.)]
    comp, _ = R.composite_r3(rows, ['nod'], dict(nod=1.), dict(nod=1.))
    assert abs(comp-(1.+(0.+3.)/2)) < 1e-9, comp
    print('selftest ok (refractory dispatch and onset cap, fresh blind set, penalty 60 deg x0.3)')


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else ''
    if stage == 'selftest':
        return selftest()
    if stage in ('prepare', 'reselect'):
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument('stage')
        p.add_argument('--output', default='runs/spike_steer/bench_r4')
        p.add_argument('--source', default='runs/spike_steer/bench_r3')
        p.add_argument('--seed', type=int, default=1)
        p.add_argument('--workers', type=int, default=30)
        p.add_argument('--tag', default='')
        p.add_argument('--spec', default=None, help='reselect: centres directory')
        a = p.parse_args()
        return (prepare if stage == 'prepare' else reselect)(a)
    R.main()                      # finetune / evaluate: round-3 entry points, round-4 hooks (R.install is rebound above)


if __name__ == '__main__':
    main()
