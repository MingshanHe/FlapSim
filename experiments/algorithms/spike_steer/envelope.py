"""Maneuver envelope inside the original interface: what can open-loop pulse trains and gimbal steps do?

Prerequisite for the combined hover + maneuver benchmark (HANDOFF 4d). Nominal case, PID flying; at t0 = 2.4 s an
open-loop maneuver replaces part of the command for a while, then the PID gets the aircraft back. Truth is logged for
analysis only.

Maneuvers:
  pulse_<dir>_<N>      motor HIGH in a 45 deg crank window at 135 deg (up) or 315 deg (down), every wingbeat, N beats
  gimbal_<axis>_<s>_<dur>  alpha (pitch) or beta (roll) gimbal held at +-0.3 rad for dur seconds
  flip_<N>             N beats up-pulses then N beats down-pulses (brake)
Metrics: peak de-rocked base pitch/roll and rates, thrust-unit peak tilt, time to 30 deg, recovered (flown to the
end), height loss, forward displacement.

    python -m experiments.algorithms.spike_steer.envelope --output artifacts/spike_steer/envelope.json --workers 2
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import numpy as np
from ..snn_pid_compare.common import DT, prepare_source
from ..snn_pid_compare.environment import LOW, HIGH, CENTER, HALF
from ..robust_control.environment import ResearchEnv
from ..topology_hover.environment import case as make_case
from ..neuro_gates import gates

T0, SECONDS = 2.4, 6.
WINDOW = math.radians(45.)
UP, DOWN = math.radians(135.), math.radians(315.)
BEAT = .04  # s, 25 Hz


def plan(name):
    """List of (start, end, kind, arg) segments relative to t0."""
    p = name.split('_')
    if p[0] == 'pulse':
        n = int(p[2])
        return [(0., n*BEAT, 'pulse', UP if p[1] == 'up' else DOWN)]
    if p[0] == 'gimbal':
        axis, sign, dur = (0 if p[1] == 'alpha' else 1), (1. if p[2] == 'p' else -1.), float(p[3])
        return [(0., dur, 'gimbal', (axis, sign))]
    if p[0] == 'flip':
        n = int(p[1])
        return [(0., n*BEAT, 'pulse', UP), (n*BEAT, 2*n*BEAT, 'pulse', DOWN)]
    raise ValueError(name)


def run(env, c, name):
    env.configure(dict(c, seconds=SECONDS))
    env.reset(seed=c['seed'])
    p = env.p
    env.teacher.p = dict(p, kp=[1.5*k for k in p['kp']], kd=[.85*k for k in p['kd']], ki=[1.5*k for k in p['ki']])
    m, d = env.sim.model, env.sim.data
    crank = int(m.joint('carrier_2_joint').qposadr[0])
    thrust = m.body('thrust_unit').id
    oracle = gates.Oracle(env, 'derocked')
    segs = plan(name)
    rows, pulses = [], 0
    pos0 = None
    while True:
        t = env.air.steps*DT
        cmd = env.teacher_command()
        rel = t-T0
        for s0, s1, kind, arg in segs:
            if s0 <= rel < s1:
                if kind == 'pulse':
                    if abs(math.remainder(float(d.qpos[crank])-arg, 2*math.pi)) < WINDOW/2:
                        cmd[2] = HIGH[2]
                        pulses += 1
                else:
                    axis, sign = arg
                    cmd[axis] = .3*sign
        _, _, term, trunc, out = env.step((np.clip(cmd, LOW, HIGH)-CENTER)/HALF)
        oracle.update()
        pos, vel, R = env.air.env._state()
        if pos0 is None and t >= T0:
            pos0 = pos.copy()
        up = math.copysign(1., float(d.xmat[thrust][8]))
        rows.append([env.air.steps*DT, oracle.queue[-1][0], oracle.queue[-1][1], oracle.rate[0], oracle.rate[1],
                     math.acos(min(1., max(-1., up*float(d.xmat[thrust][8])))), pos[0], pos[1], pos[2], vel[0], vel[1], vel[2],
                     math.acos(min(1., max(-1., float(R[2, 2]))))])
        if term or trunc:
            break
    a = np.array(rows)
    ep = out['episode']
    after = a[:, 0] >= T0
    w = a[after]
    pitch, roll = np.degrees(w[:, 2]), np.degrees(w[:, 1])
    t_rel = w[:, 0]-T0
    def first(cond):
        i = np.flatnonzero(cond)
        return float(t_rel[i[0]]) if len(i) else None
    heading = np.array([math.cos(float(env.last_reading[2])), math.sin(float(env.last_reading[2]))])
    return dict(name=name, seed=c['seed'], termination=ep['termination'], flown=ep['termination'] == 'time_limit',
                elapsed=ep['elapsed'], pulses=pulses, duration_s=max(s1 for _, s1, _, _ in segs),
                peak_pitch_deg=float(pitch[np.argmax(np.abs(pitch))]), peak_roll_deg=float(roll[np.argmax(np.abs(roll))]),
                peak_pitch_rate=float(np.degrees(w[:, 4][np.argmax(np.abs(w[:, 4]))])),
                peak_roll_rate=float(np.degrees(w[:, 3][np.argmax(np.abs(w[:, 3]))])),
                thrust_peak_tilt_deg=float(np.degrees(w[:, 5].max())), base_peak_tilt_deg=float(np.degrees(w[:, 12].max())),
                t_to_30deg=first(np.abs(pitch) >= 30.), t_to_45deg=first(np.abs(pitch) >= 45.),
                t_back_below_10deg=first((t_rel > max(s1 for _, s1, _, _ in segs)) & (np.abs(pitch) < 10.)),
                height_loss_cm=float(100*max(0., (pos0[2]-w[:, 8].min()) if pos0 is not None else 0.)),
                forward_m=float((w[-1, 6:8]-pos0[:2])@heading) if pos0 is not None else None,
                peak_forward_speed=float(np.abs(w[:, 9:11]@heading).max()))


_ENV = None


def worker_init(source):
    global _ENV
    _ENV = ResearchEnv(str(source), mode='direct', task='hover')


def task(args):
    return run(_ENV, *args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    source = Path('runs/neuro_gates/scratch/source')
    if not (source/'robot_upper7_lower8.xml').exists():
        prepare_source(source)
    names = [f'pulse_{dirn}_{n}' for dirn in ('up', 'down') for n in (3, 6, 12, 25)]
    names += [f'gimbal_{ax}_{sg}_{dur}' for ax in ('alpha', 'beta') for sg in ('p', 'n') for dur in (.1, .3)]
    names += [f'flip_{n}' for n in (3, 6, 12)]
    cases = [make_case(2_090_000_000+i, SECONDS, 'nominal', i % 8) for i in range(2)]
    with ProcessPoolExecutor(a.workers, initializer=worker_init, initargs=(str(source),)) as ex:
        rows = list(ex.map(task, [(c, n) for n in names for c in cases], chunksize=1))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(t0=T0, names=names, rows=rows), indent=1)+'\n')
    f = lambda v: 'None' if v is None else f'{v:.2f}'
    for n in names:
        items = [r for r in rows if r['name'] == n]
        g = lambda k: np.mean([r[k] for r in items if r[k] is not None]) if any(r[k] is not None for r in items) else None
        print(f"{n:20s} flown {sum(r['flown'] for r in items)}/{len(items)} dur {items[0]['duration_s']:.2f}s pulses {g('pulses'):.0f} "
              f"peak pitch {g('peak_pitch_deg'):+6.1f} deg (rate {g('peak_pitch_rate'):+6.0f} deg/s) roll {g('peak_roll_deg'):+5.1f} "
              f"thrust tilt {g('thrust_peak_tilt_deg'):5.1f} t30 {f(g('t_to_30deg'))} t45 {f(g('t_to_45deg'))} back<10 {f(g('t_back_below_10deg'))} "
              f"dz {g('height_loss_cm'):5.1f} cm fwd {g('forward_m'):+.2f} m vmax {g('peak_forward_speed'):.2f}", flush=True)


if __name__ == '__main__':
    main()
