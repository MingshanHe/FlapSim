"""Wingbeat-timed actuation INSIDE the original v3 interface.

Phase-locked patterns are added to the PID command every 2 ms. ResearchEnv then clips the command to
LOW/HIGH and passes it through the same 8.2 ms smoothing, servo and motor speed loop as in v3. Nothing in
the plant or the actuator path changes. Timing uses the true crank angle, so this measures an upper bound on
the authority a phase-locked controller could have.

Each pattern runs as a 0.1 s burst at t = 3 s and is compared with the same case without a burst. The
effect of timing alone is half the difference between opposite phases, (d(psi) - d(psi+180)) / 2,
which cancels the change in mean lift that one-sided pulses cause.

    python -m experiments.algorithms.neuro_gates.timing --output artifacts/neuro_gates/timing_authority.json
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
from . import gates

BURST = (3., 3.1)
PHASES = [45.*k for k in range(8)]


def pattern_command(kind, pid, phi, psi, amount):
    """Command for one 2 ms tick; phi is the true crank angle and psi the phase offset (rad)."""
    cmd = pid.copy()
    if kind == 'gimbal_step':
        cmd[0] += amount
    elif kind == 'gimbal_sine':  # pitch gimbal oscillates at the wingbeat
        cmd[0] += amount*math.sin(phi+psi)
    elif kind == 'motor_sine':  # symmetric request; the bounds clip the downward half
        cmd[2] += amount*math.sin(phi+psi)
    elif kind in ('motor_pulse', 'motor_dip'):  # full-range pulse during a window of `amount` rad of crank phase
        if abs(math.remainder(phi-psi, 2*math.pi)) < amount/2:
            cmd[2] = HIGH[2] if kind == 'motor_pulse' else LOW[2]
    elif kind != 'none':
        raise ValueError(kind)
    return cmd


def run(env, c, kind, psi_deg, amount):
    env.configure(dict(c, seconds=BURST[1]+.2))
    env.reset(seed=c['seed'])
    m, d = env.sim.model, env.sim.data
    crank = int(m.joint('carrier_2_joint').qposadr[0])
    oracle = gates.Oracle(env, 'derocked')
    psi = math.radians(psi_deg)
    rows = []
    while True:
        t = env.air.steps*DT
        pid = env.teacher_command()  # the original PID, stepped once per tick
        cmd = pattern_command(kind, pid, float(d.qpos[crank]), psi, amount) if BURST[0] <= t < BURST[1] else pid
        _, _, term, trunc, _ = env.step((np.clip(cmd, LOW, HIGH)-CENTER)/HALF)
        oracle.update()
        pos, vel, rotation = env.air.env._state()
        rows.append([env.air.steps*DT, *oracle.queue[-1], *vel, *pos])
        if term or trunc:
            break
    return np.array(rows)


def measure(ref, hit, heading):
    """Paired differences at the burst end: de-rocked pitch and roll rates, forward and vertical velocity."""
    n = min(len(ref), len(hit))
    t = ref[:n, 0]
    i = int(np.argmin(abs(t-BURST[1])))
    j = int(np.argmin(abs(t-(BURST[1]-.02))))
    rate = lambda a, k: (a[i, k]-a[j, k])/(t[i]-t[j])
    forward = np.array([math.cos(heading), math.sin(heading)])
    return dict(pitch_rate=rate(hit, 2)-rate(ref, 2), roll_rate=rate(hit, 1)-rate(ref, 1),
                forward_velocity=float((hit[i, 3:5]-ref[i, 3:5])@forward), vertical_velocity=float(hit[i, 5]-ref[i, 5]),
                pitch_deg=math.degrees(hit[i, 2]-ref[i, 2]))


_ENV = None


def worker_init(source):
    global _ENV
    _ENV = ResearchEnv(str(source), mode='direct', task='hover')


def task(args):
    c, kind, psi, amount = args
    ref = run(_ENV, c, 'none', 0., 0.)
    hit = run(_ENV, c, kind, psi, amount)
    # heading: yaw of the base at the burst start (reading index 2 is the lagged yaw; truth is close enough)
    out = measure(ref, hit, float(_ENV.last_reading[2]))
    return dict(seed=c['seed'], kind=kind, psi_deg=psi, amount=amount, **out)


PULSE_PHASE = math.radians(135.)  # pulses at this crank phase raise pitch rate; +180 deg lowers it (authority scan)


def closed_loop(env, c, gains, law=None):
    """Retuned PID, optionally with an event-triggered, phase-locked motor-pulse law.

    The pulse law reads the oracle (true, de-rocked) pitch state, so this is an upper bound. It computes
    u = -(k_theta*pitch + k_q*pitch_rate). When |u| exceeds the threshold, the motor command goes to HIGH
    during a window of `width` rad of crank phase, centred on the phase that pushes pitch toward zero."""
    env.configure(c)
    env.reset(seed=c['seed'])
    p, (sp, sd) = env.p, gains
    env.teacher.p = dict(p, kp=[sp*k for k in p['kp']], kd=[sd*k for k in p['kd']], ki=[sp*k for k in p['ki']])
    d = env.sim.data
    crank = int(env.sim.model.joint('carrier_2_joint').qposadr[0])
    oracle = gates.Oracle(env, 'derocked')
    imu = law is not None and law.get('imu')
    if imu:  # IMU only: PLL phase and notched pitch gyro (P0-E4), lagged AHRS pitch; offset calibrated in P0
        from ..wingbeat_feasibility.preliminary import WingbeatPhase
        from ..wingbeat_snn.plant import PHASE_OFFSET
        estimator = WingbeatPhase(PHASE_OFFSET)
    pitch, pulses = [], 0
    while True:
        t = env.air.steps*DT
        cmd = env.teacher_command()
        if imu:
            estimator.update(float(d.time), float(env.last_reading[4]))
        if law and t >= gates.HANDOVER_T:
            if imu:
                u = -(law['k_theta']*env.last_reading[1]+law['k_q']*estimator.slow)
                phase = estimator.crank(float(d.time))
            else:
                u = -(law['k_theta']*oracle.queue[-1][1]+law['k_q']*oracle.rate[1])
                phase = float(d.qpos[crank])
            if abs(u) > law['threshold']:
                psi = PULSE_PHASE if u > 0 else PULSE_PHASE+math.pi
                offset = abs(math.remainder(phase-psi, 2*math.pi))
                if offset < law['width']/2:
                    cmd[2] = HIGH[2]
                    pulses += 1
                elif law.get('dip') and offset > math.pi-law['width']/2:  # half a cycle later: shed lift
                    cmd[2] = LOW[2]
        _, _, term, trunc, out = env.step((np.clip(cmd, LOW, HIGH)-CENTER)/HALF)
        oracle.update()
        pitch.append((env.air.steps*DT, math.degrees(oracle.queue[-1][1])))
        if term or trunc:
            break
    m = out['episode']
    pt = np.array(pitch)
    late = pt[:, 0] >= 2.
    return dict(seed=c['seed'], gains=list(gains), law=law, termination=m['termination'], success=m['success'],
                tilt_rms_deg=m['tilt_rms_deg'], height_rmse_m=m['height_rmse_m'],
                pitch_peak_deg=float(np.abs(pt[late, 1]).max()) if late.any() else float('nan'),
                pitch_rms_deg=float(np.sqrt(np.mean(pt[late, 1]**2))) if late.any() else float('nan'),
                pulse_duty=pulses/max(1, env.air.steps))


def closed_task(args):
    return closed_loop(_ENV, *args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', nargs='?', default='authority', choices=['authority', 'closed_loop', 'law_grid', 'imu_law'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=18)
    p.add_argument('--count', type=int, default=4)
    a = p.parse_args()
    source = Path('runs/neuro_gates/scratch/source')
    if not (source/'robot_upper7_lower8.xml').exists():
        prepare_source(source)
    if a.stage in ('closed_loop', 'law_grid', 'imu_law'):
        cs = gates.cases('tune', 'stress', a.count, 8.)
        if a.stage == 'imu_law':  # the same pulse law fed only by the IMU, versus its oracle version
            specs = [((1.5, 1.), None)]
            for kt, kq, u0, w in ((5., .4, .5, 60.), (5., .4, .8, 90.), (3., .3, .5, 90.), (5., .4, 1.2, 90.)):
                for imu in (False, True):
                    specs.append(((1.5, 1.), dict(k_theta=kt, k_q=kq, threshold=u0, width=math.radians(w), imu=imu)))
        elif a.stage == 'closed_loop':
            laws = [None]+[dict(k_theta=kt, k_q=kq, threshold=u0, width=math.radians(w)) for kt, kq, u0, w in
                           ((5., .4, .2, 90.), (5., .4, .5, 90.), (5., .4, .2, 180.), (10., .8, .5, 90.))]
            specs = [((1., 1.), None)]+[((1.5, 1.), law) for law in laws]
        else:  # how far a hand-tuned pulse law can go: gains, threshold, window, lift-shedding dip
            specs = [((1.5, 1.), None)]+[((1.5, 1.), dict(k_theta=kt, k_q=kq, threshold=u0, width=math.radians(w),
                                                          dip=dip))
                                         for kt, kq in ((5., .4), (3., .3)) for u0 in (.5, .8, 1.2)
                                         for w in (60., 90.) for dip in (False, True)]
        with ProcessPoolExecutor(a.workers, initializer=worker_init, initargs=(str(source),)) as ex:
            rows = list(ex.map(closed_task, [(c, g, law) for g, law in specs for c in cs], chunksize=1))
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(dict(stage=a.stage, rows=rows), indent=1)+'\n')
        for g, law in specs:
            items = [r for r in rows if r['gains'] == list(g) and r['law'] == law]
            flown = [r for r in items if r['termination'] == 'time_limit']
            f = lambda k: np.mean([r[k] for r in flown]) if flown else float('nan')
            name = 'none' if law is None else f"k=({law['k_theta']:g},{law['k_q']:g}) u0={law['threshold']:g} " \
                                              f"w={math.degrees(law['width']):g}{' dip' if law.get('dip') else ''}" \
                                              f"{' IMU' if law.get('imu') else ''}"
            print(f'PID x{g} law {name:28s} flown {len(flown)}/{len(items)} tilt_rms {f("tilt_rms_deg"):5.2f} '
                  f'pitch_rms {f("pitch_rms_deg"):5.2f} pitch_peak {f("pitch_peak_deg"):5.2f} '
                  f'height_rmse {100*f("height_rmse_m"):5.2f} cm duty {f("pulse_duty"):.3f}')
        return
    cs = gates.cases('tune', 'nominal', a.count, 4.)
    specs = [('gimbal_step', 0., .035)]
    specs += [(k, psi, amt) for psi in PHASES for k, amt in (('motor_pulse', math.pi/2), ('motor_pulse', math.pi),
                                                            ('motor_dip', math.pi), ('motor_sine', 300.),
                                                            ('gimbal_sine', .1))]
    tasks = [(c, k, psi, amt) for k, psi, amt in specs for c in cs]
    with ProcessPoolExecutor(a.workers, initializer=worker_init, initargs=(str(source),)) as ex:
        rows = list(ex.map(task, tasks, chunksize=1))
    keys = ('pitch_rate', 'roll_rate', 'forward_velocity', 'vertical_velocity', 'pitch_deg')
    mean = {}
    for r in rows:
        mean.setdefault((r['kind'], r['amount'], r['psi_deg']), []).append([r[k] for k in keys])
    summary = {f'{k}|{amt:g}|{psi:g}': dict(zip(keys, np.mean(v, axis=0).tolist())) for (k, amt, psi), v in mean.items()}
    timing = {}
    for (k, amt, psi), v in mean.items():  # timing-only effect: half the difference between opposite phases
        if k != 'gimbal_step' and psi < 180:
            opp = np.mean(mean[(k, amt, psi+180.)], axis=0)
            timing[f'{k}|{amt:g}|{psi:g}'] = dict(zip(keys, ((np.mean(v, axis=0)-opp)/2).tolist()))
    result = dict(burst_s=list(BURST), cases=[c['seed'] for c in cs], summary=summary, timing_effect=timing, rows=rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=1)+'\n')
    base = summary['gimbal_step|0.035|0']
    print('gimbal step +0.035 rad:', {k: round(v, 4) for k, v in base.items()})
    for k, v in sorted(timing.items(), key=lambda kv: -abs(kv[1]['pitch_rate'])):
        print(f'{k:24s} pitch_rate {v["pitch_rate"]:+.3f} ({abs(v["pitch_rate"]/base["pitch_rate"]):4.1f}x)'
              f'  forward_v {v["forward_velocity"]:+.4f} ({abs(v["forward_velocity"]/base["forward_velocity"]):4.1f}x)'
              f'  vertical_v {v["vertical_velocity"]:+.4f}')


if __name__ == '__main__':
    main()
