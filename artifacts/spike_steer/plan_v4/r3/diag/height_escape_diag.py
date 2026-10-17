"""Round-3 diagnostic (HomePC, 2026-10-16): mechanism of the 'height_escape' crashes of the gated descending layer.

Stage trace: tick-by-tick replay of
  * r3 seed 1 gen 59 (artifacts/spike_steer/plan_v4/r3/finetune_s1/spec_val_g59.json) on the blind dash case 2096400008
    (termination height_escape in claude_ea's blind test) and the classical opponent (schedule) on the same case;
  * r2 seed 3 CloudPC gen 59 (plan_v4/r2/finetune_s3_cloudpc/spec_val_g59.json) on its two height_escape blind cases
    (nod 2096100001, dash 2096100000; round-2 blind set).
  Logged per 2 ms tick: t, true z (relative to the start height, the quantity the simulator tests against +-1 m), vz,
  applied motor command omega (rad/s), the height PID's unclipped motor demand, learned pulse, learned layer in charge
  (command on or hand-back tail), command on, pitch reference, true tilt, AHRS tilt, measured z.
Stage counterfactual (training-split cases only, never a blind set): r3 s1 g59 on 64 nod + 64 dash cases at a fresh
  offset (OFFSET below), intact and three variants of the SAME spec (no re-tuning):
  (a) no learned pulses once the goal is reached (command dropped; the hand-back tail keeps the pitch reference);
  (b) at most one learned pulse onset per 37 ms wingbeat;
  (c) |z_meas - z_ref| > 0.3 m while the learned layer is in charge -> hand back to the classical reflex layer (latched).
No experiment module is modified; the variants subclass descend_gated.GatedLearned and re-implement its act() with hooks.

    cd <repo root>; W=18 python -W ignore artifacts/spike_steer/plan_v4/r3/diag/height_escape_diag.py all   # stages: trace | counterfactual | all
"""
import json
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, os.getcwd())      # run from the repo root as a plain script
from experiments.algorithms.spike_steer import bench as B, descend_gated as G, descend_r3 as R
from experiments.algorithms.snn_pid_compare.environment import LOW, HIGH, CENTER, HALF
from experiments.algorithms.snn_pid_compare.common import DT

OUT = Path(os.environ.get('OUT', 'runs/spike_steer/r3_diag_height'))
ART = Path('artifacts/spike_steer/plan_v4/r3')
SRC = Path('runs/neuro_gates/scratch/source')
OFFSET = 80_000            # fresh training-split offset: ES used 100_000+gen*100, validation 60_000, envelope diag 120_000
N_CASES = 64
BEAT_S = .037              # one wingbeat (variant b)
Z_HANDBACK_M = .3          # variant c
SPEC_R3_S1 = ART/'finetune_s1'/'spec_val_g59.json'
SPEC_R2_S3C = Path('artifacts/spike_steer/plan_v4/r2/finetune_s3_cloudpc/spec_val_g59.json')
BASE_R3 = ART/'claude_ea'/'baseline.json'
BASE_R3_TRAIN = ART/'claude_ea'/'train'/'baseline.json'
BASE_R2 = Path('artifacts/spike_steer/plan_v4/baseline.json')
TRACE_CASES = [('r3_s1_g59', 'learned', SPEC_R3_S1, 'dash', 2096400008, 'r3'),
               ('opponent_schedule', 'classical', None, 'dash', 2096400008, 'r3'),
               ('r2_s3_cloudpc_g59', 'learned', SPEC_R2_S3C, 'nod', 2096100001, 'r2'),
               ('r2_s3_cloudpc_g59', 'learned', SPEC_R2_S3C, 'dash', 2096100000, 'r2'),
               ('opponent_schedule', 'classical', None, 'nod', 2096100001, 'r2'),
               ('opponent_schedule', 'classical', None, 'dash', 2096100000, 'r2')]


# ----------------------------------------------------------------------------------------- policy variants
class Variant(G.GatedLearned):
    """descend_gated.GatedLearned.act re-implemented line by line (variant 'intact' must reproduce it exactly; checked in
    stage trace) plus the three counterfactual hooks. Exposes in_charge for the trace."""

    def reset(self):
        super().reset()
        self.v = self.spec.get('diag_variant', 'intact')
        self.seen_on = False
        self.last_onset = None
        self.handback = False
        self.handback_t = None
        self.cancelled = 0
        self.in_charge = False

    def act(self, env, now, t, x, cmd, rel):
        fire, ref = G._Learned.act(self, env, now, t, x, cmd, rel)       # net ticks every step (keeps its state)
        law_pulse = False
        if self.steer:
            law_pulse, _ = self.steer.step(env, now, t)                   # pulse law ticks every step (keeps the PLL)
        on = bool(cmd.any())
        if self.was_on and not on:
            self.off_t = t
        self.was_on = on
        if on:
            self.seen_on = True
        in_charge = on or (self.off_t is not None and t-self.off_t < self.tail)
        if self.steer_up:
            up_pulse, _ = self.steer_up.step(env, now, t)
            if (not self.upset and not in_charge
                    and math.hypot(float(env.last_reading[0]), float(env.last_reading[1])) > self.tilt_thr):
                self.upset = True
            if self.upset:
                law_pulse = up_pulse
        if self.v == 'c' and in_charge and not self.handback and abs(float(env.last_reading[9])-env.teacher.z_reference) > Z_HANDBACK_M:
            self.handback, self.handback_t = True, t
        self.in_charge = bool(in_charge and not self.handback)
        if self.in_charge:
            if self.v == 'a' and self.seen_on and not on and fire:
                fire, self.twitch = False, 0
                self.cancelled += 1
            if self.v == 'b' and fire and self.twitch == B.TWITCH-1:        # a fresh onset this tick
                if self.last_onset is not None and t-self.last_onset < BEAT_S:
                    fire, self.twitch = False, 0
                    self.cancelled += 1
                else:
                    self.last_onset = t
            return fire, ref
        return bool(law_pulse), 0.


def make_policy(kind, arg):
    if kind == 'classical':
        return B.Classical(arg.get('law'), arg.get('schedule'))
    return Variant(arg)


# ----------------------------------------------------------------------------------------- traced episode
def traced_episode(env, task, c, pol):
    log, state = [], {}
    orig_act = pol.act

    def act(env_, now, t, x, cmd, rel):
        fire, ref = orig_act(env_, now, t, x, cmd, rel)
        r = env_.last_reading
        state.update(fire=bool(fire), ref=float(ref), on=bool(cmd.any()), in_charge=getattr(pol, 'in_charge', None),
                     tilt_ahrs=math.degrees(math.hypot(float(r[0]), float(r[1]))), z_meas=float(r[9]))
        return fire, ref
    pol.act = act
    orig_step = env.step

    def step(action):
        cmdv = CENTER+HALF*np.asarray(action, float)
        pos, vel, Rm = env.air.env._state()
        log.append([env.air.steps*DT, float(pos[2]-env.air.start[2]), float(vel[2]), float(cmdv[2]), float(env.teacher.raw[2]),
                    float(state['fire']), (-1. if state['in_charge'] is None else float(state['in_charge'])), float(state['on']),
                    state['ref'], math.degrees(math.acos(min(1., max(-1., float(Rm[2, 2]))))), state['tilt_ahrs'], state['z_meas']])
        return orig_step(action)
    env.step = step
    try:
        res = B.run_episode(env, task, c, pol)
    finally:
        del env.step
    return res, np.array(log)


COLS = ['t', 'z', 'vz', 'omega', 'pid_motor_raw', 'fire', 'in_charge', 'cmd_on', 'ref', 'tilt_true_deg', 'tilt_ahrs_deg', 'z_meas']


def window_stats(a, t_lo, t_hi):
    w = a[(a[:, 0] >= t_lo) & (a[:, 0] < t_hi)]
    if len(w) < 2:
        return None
    fire = w[:, 5] > .5
    onsets = int(np.sum(fire[1:] & ~fire[:-1])+(1 if fire[0] else 0))
    dur = float(w[-1, 0]-w[0, 0]+DT)
    return dict(t=[round(t_lo, 3), round(t_hi, 3)], onsets=onsets, onsets_per_s=round(onsets/dur, 2), pulse_duty=round(float(fire.mean()), 4),
                learned_in_charge_frac=round(float((w[:, 6] > .5).mean()), 3), omega_mean=round(float(w[:, 3].mean()), 1),
                pid_motor_raw_mean=round(float(w[:, 4].mean()), 1), pid_saturated_low_frac=round(float((w[:, 4] <= LOW[2]+1e-6).mean()), 3),
                z_start=round(float(w[0, 1]), 3), z_end=round(float(w[-1, 1]), 3), vz_mean=round(float(w[:, 2].mean()), 3),
                tilt_true_max=round(float(w[:, 9].max()), 1), ref_abs_mean=round(float(np.abs(w[:, 8]).mean()), 3))


def analyse(res, a, task):
    t_end = float(a[-1, 0])
    t0 = B.T0
    goal_t = res.get('goal_t')
    t_off = t0+goal_t if goal_t is not None else None
    tail = G.TAIL_S
    out = dict(termination=res['termination'], elapsed=res['elapsed'], score=res['score'], flown=res['flown'], pulses=res['pulses'],
               goal_t=goal_t, t_cmd_off=t_off, t_tail_end=(t_off+tail) if t_off is not None else None,
               z_end=round(float(a[-1, 1]), 3), vz_end=round(float(a[-1, 2]), 3),
               direction=('up' if a[-1, 1] > 0 else 'down') if res['termination'] == 'height_escape' else None,
               z_max=round(float(a[:, 1].max()), 3), z_min=round(float(a[:, 1].min()), 3), tilt_true_max=round(float(a[:, 9].max()), 1),
               tilt_true_at_end=round(float(a[-1, 9]), 1), tilt_ahrs_at_end=round(float(a[-1, 10]), 1))
    phases = dict(before_cmd=(2., t0))
    if t_off is not None:
        phases.update(cmd_on=(t0, t_off), tail=(t_off, min(t_off+tail, t_end)), after_tail=(t_off+tail, t_end))
    else:
        phases.update(cmd_on=(t0, t_end))
    out['phases'] = {k: window_stats(a, lo, hi) for k, (lo, hi) in phases.items()}
    out['last_2s'] = window_stats(a, t_end-2., t_end-1.)
    out['last_1s'] = window_stats(a, t_end-1., t_end+DT)
    out['half_s_windows'] = [window_stats(a, s, s+.5) for s in np.arange(2., t_end, .5)]
    # vertical speed reached while the learned layer was in charge, and the climb during the tail
    ic = a[:, 6] > .5
    if ic.any():
        out['while_in_charge'] = dict(vz_max=round(float(a[ic, 2].max()), 3), vz_min=round(float(a[ic, 2].min()), 3), z_at_handback=round(float(a[ic, 1][-1]), 3),
                                      vz_at_handback=round(float(a[ic, 2][-1]), 3), t_handback=round(float(a[ic, 0][-1]), 3),
                                      onsets=window_stats(a, float(a[ic, 0][0]), float(a[ic, 0][-1])+DT)['onsets'])
    return out


def stage_trace(w):
    OUT.mkdir(parents=True, exist_ok=True)
    base_r3, base_r2 = json.loads(BASE_R3.read_text()), json.loads(BASE_R2.read_text())
    assert (base_r3['prereg']['dash_m'] or .25) == (base_r2['prereg']['dash_m'] or .25), 'dash_m differs between rounds'
    if not (SRC/'robot_upper7_lower8.xml').exists():
        B.prepare_source(SRC)
    B.worker_init(str(SRC), base_r3['prereg']['dash_m'] or .25)
    env = B._ENV
    report, traces = {}, {}
    for label, kind, spec_path, task, seed, rnd in TRACE_CASES:
        R.install(fresh_test=(rnd == 'r3'))
        base = base_r3 if rnd == 'r3' else base_r2
        c = [c for c in B.cases(task, 'test') if c['seed'] == seed][0]
        arg = json.loads(spec_path.read_text()) if kind == 'learned' else base['arms_full'][task][base['opponent'][task]]
        key = f'{label}|{task}|{seed}'
        if kind == 'learned':   # the Variant('intact') must reproduce the stock gated policy exactly
            ref_res = B.run_episode(env, task, c, B.Learned(arg))
            res, a = traced_episode(env, task, c, Variant(dict(arg, diag_variant='intact')))
            same = (ref_res['termination'] == res['termination'] and abs(ref_res['score']-res['score']) < 1e-9 and ref_res['pulses'] == res['pulses'])
            print(f'{key}: stock policy term {ref_res["termination"]} score {ref_res["score"]:.3f} pulses {ref_res["pulses"]} | Variant(intact) identical: {same}', flush=True)
            assert same, (ref_res, res)
        else:
            res, a = traced_episode(env, task, c, make_policy(kind, arg))
        an = analyse(res, a, task)
        an['opponent_row'] = next(({k: r.get(k) for k in ('score', 'termination', 'height_loss_cm', 'tilt_peak_deg', 'pulses', 'goal_t')}
                                   for r in base['test_rows'][task] if r['seed'] == seed and r['controller'] == base['opponent'][task]), None)
        report[key] = an
        traces[key] = dict(columns=COLS, rows=np.round(a, 4).tolist())
        print(json.dumps({key: {k: an[k] for k in ('termination', 'elapsed', 'score', 'direction', 'z_end', 'vz_end', 'goal_t', 't_cmd_off', 't_tail_end', 'z_max', 'z_min')}}), flush=True)
        print('   phases:', json.dumps(an['phases']), flush=True)
        print('   last_2s:', json.dumps(an['last_2s']), '\n   last_1s:', json.dumps(an['last_1s']), flush=True)
        if 'while_in_charge' in an:
            print('   while_in_charge:', json.dumps(an['while_in_charge']), flush=True)
    (OUT/'height_escape_traces.json').write_text(json.dumps(dict(columns=COLS, summary=report, traces=traces, LOW=LOW.tolist(), HIGH=HIGH.tolist(),
                                                                 hover_motor=env.p['motor'], height_gains=env.p['height_gains'])))
    (OUT/'height_escape_trace_summary.json').write_text(json.dumps(dict(summary=report, LOW=LOW.tolist(), HIGH=HIGH.tolist(), hover_motor=env.p['motor'],
                                                                        height_gains=env.p['height_gains']), indent=1))
    print('wrote', OUT/'height_escape_traces.json', flush=True)


# ----------------------------------------------------------------------------------------- counterfactuals (training split)
def _init(source, dash_m):
    R.install(fresh_test=False)
    B.worker_init(source, dash_m)


def cf_task(args):
    variant, task, c, spec = args
    env = B._ENV
    pol = Variant(dict(spec, diag_variant=variant))
    res = B.run_episode(env, task, c, pol, trace=True)
    a = np.array(res.pop('trace'))
    z = a[:, 4]-env.air.start[2]
    res.update(variant=variant, z_max=float(z.max()), z_min=float(z.min()), vz_max=float(a[:, 7].max()), vz_min=float(a[:, 7].min()),
               cancelled=pol.cancelled, handback_t=pol.handback_t, onsets=pol.onsets)
    return res


def stage_counterfactual(w):
    OUT.mkdir(parents=True, exist_ok=True)
    R.install(fresh_test=False)
    base_tr = json.loads(BASE_R3_TRAIN.read_text())
    norm = base_tr['normalisers']
    spec = json.loads(SPEC_R3_S1.read_text())
    tasks = ('nod', 'dash')
    cases = {t: B.cases(t, 'train', None, n=N_CASES, offset=OFFSET) for t in tasks}
    variants = ('intact', 'a', 'b', 'c')
    jobs = [(v, t, c, spec) for v in variants for t in tasks for c in cases[t]]
    if not (SRC/'robot_upper7_lower8.xml').exists():
        B.prepare_source(SRC)
    t_start = time.monotonic()
    with ProcessPoolExecutor(w, mp_context=mp.get_context('fork'), initializer=_init, initargs=(str(SRC), base_tr['prereg']['dash_m'] or .25)) as ex:
        rows = list(ex.map(cf_task, jobs, chunksize=1))
    table = {}
    for v in variants:
        table[v] = {}
        for t in tasks:
            rs = [r for r in rows if r['variant'] == v and r['task'] == t]
            terms = {}
            for r in rs:
                terms[r['termination']] = terms.get(r['termination'], 0)+1
            fl = [r for r in rs if r['flown']]
            table[v][t] = dict(episodes=len(rs), crashes=sum(not r['flown'] for r in rs), terminations=terms,
                               height_escape=sum(r['termination'] == 'height_escape' for r in rs), height_escape_up=sum(r['termination'] == 'height_escape' and r['z_max'] > .99 for r in rs),
                               score_mean=float(np.mean([r['score'] for r in rs])), ratio=float(np.mean([r['score'] for r in rs]))/norm[t],
                               ratio_flown=(float(np.mean([r['score'] for r in fl]))/norm[t]) if fl else None,
                               hits=sum(r.get('t30' if t == 'nod' else 't_reach') is not None for r in rs),
                               pulses_mean=float(np.mean([r['pulses'] for r in rs])), cancelled_mean=float(np.mean([r['cancelled'] for r in rs])),
                               handbacks=sum(r['handback_t'] is not None for r in rs),
                               z_max_mean=float(np.mean([r['z_max'] for r in rs])), z_max_max=float(max(r['z_max'] for r in rs)),
                               z_over_0p5=sum(r['z_max'] > .5 for r in rs), z_min_min=float(min(r['z_min'] for r in rs)),
                               tilt_peak_max=float(max(r['tilt_peak_deg'] or 0. for r in rs)), tilt_over_45=sum((r['tilt_peak_deg'] or 0.) > 45. for r in rs))
            print(f"{v:7s} {t:5s} crashes {table[v][t]['crashes']}/{len(rs)} {terms} ratio {table[v][t]['ratio']:.3f} (flown {table[v][t]['ratio_flown']}) "
                  f"pulses {table[v][t]['pulses_mean']:.1f} z_max mean {table[v][t]['z_max_mean']:.3f} max {table[v][t]['z_max_max']:.3f} >0.5m {table[v][t]['z_over_0p5']} "
                  f"tilt>45 {table[v][t]['tilt_over_45']}", flush=True)
    # paired comparison of each variant with intact on the same cases
    paired = {}
    for v in variants[1:]:
        paired[v] = {}
        for t in tasks:
            A = {r['seed']: r for r in rows if r['variant'] == 'intact' and r['task'] == t}
            Bv = {r['seed']: r for r in rows if r['variant'] == v and r['task'] == t}
            d = np.array([Bv[s]['score']-A[s]['score'] for s in A])
            paired[v][t] = dict(mean_delta=float(d.mean()), wins=int((d < -1e-9).sum()), losses=int((d > 1e-9).sum()), ties=int((np.abs(d) <= 1e-9).sum()),
                                rescued=sum((not A[s]['flown']) and Bv[s]['flown'] for s in A), newly_crashed=sum(A[s]['flown'] and not Bv[s]['flown'] for s in A))
    out = dict(spec=str(SPEC_R3_S1), offset=OFFSET, n_cases=N_CASES, seeds={t: [c['seed'] for c in cases[t]] for t in tasks}, normalisers=norm,
               variants=dict(intact='round-3 gated policy unchanged', a='no learned pulses after the goal (command dropped); classical law after the tail unchanged',
                             b=f'at most one learned pulse onset per {BEAT_S*1000:.0f} ms', c=f'|z_meas - z_ref| > {Z_HANDBACK_M} m while in charge -> classical reflex layer (latched)'),
               table=table, paired_vs_intact=paired, wall_s=round(time.monotonic()-t_start),
               rows=[{k: r[k] for k in r if k not in ('trace',)} for r in rows])
    (OUT/'height_escape_counterfactual.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(dict(paired_vs_intact=paired)), flush=True)
    print('wrote', OUT/'height_escape_counterfactual.json', 'wall_s', out['wall_s'], flush=True)


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    W = int(os.environ.get('W', max(1, os.cpu_count()-2)))
    if stage in ('trace', 'all'):
        stage_trace(W)
    if stage in ('counterfactual', 'all'):
        stage_counterfactual(W)
