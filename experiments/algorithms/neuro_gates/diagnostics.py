"""Diagnostics of the original v3 hover plant (no interface change): why the PID needs its 0.13 s attitude.

    python -m experiments.algorithms.neuro_gates.diagnostics rocking     # wingbeat rocking in base pitch
    python -m experiments.algorithms.neuro_gates.diagnostics lagscan     # PID survival vs attitude lag and gains
    python -m experiments.algorithms.neuro_gates.diagnostics gimbal_step # base vs thrust-unit response to a pulse
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import numpy as np
from ..snn_pid_compare.common import DT
from ..robust_control.environment import ResearchEnv
from . import gates

SCRATCH = Path('runs/neuro_gates/scratch')


def env_for(mode='pid'):
    source = SCRATCH/'source'
    if not (source/'robot_upper7_lower8.xml').exists():
        from ..snn_pid_compare.common import prepare_source
        prepare_source(source)
    return ResearchEnv(str(source), mode=mode, task='hover')


def rocking():
    """Amplitude of base pitch at 1x/2x/3x the wingbeat, for the truth, each oracle and the sensor."""
    env = env_for()
    c = gates.cases('tune', 'nominal', 1, 4.)[0]
    env.configure(c)
    env.reset(seed=c['seed'])
    kinds = {k: gates.Oracle(env, k) for k in ('raw', 'lp30', 'derocked')}
    series = {k: [] for k in list(kinds)+['sensor']}
    crank_v, speeds = int(env.sim.model.joint('carrier_2_joint').dofadr[0]), []
    while True:
        _, _, term, trunc, _ = env.step(np.zeros(3))
        for k, o in kinds.items():
            o.update()
            series[k].append(o.queue[0][1])
        series['sensor'].append(env.last_reading[1])
        speeds.append(abs(float(env.sim.data.qvel[crank_v])))
        if term or trunc:
            break
    f_wb = float(np.mean(speeds[500:])/(2*math.pi))
    out = dict(wingbeat_hz=f_wb, amplitude_deg={})
    for k, s in series.items():
        x = np.degrees(np.array(s[1000:]))
        x = x-x.mean()
        w = np.hanning(len(x))
        spec = np.abs(np.fft.rfft(x*w))*2/np.sum(w)
        freq = np.fft.rfftfreq(len(x), DT)
        out['amplitude_deg'][k] = [float(spec[max(0, np.argmin(abs(freq-h*f_wb))-1):np.argmin(abs(freq-h*f_wb))+2].max())
                                   for h in (1, 2, 3)]
    return out


def lagscan(workers=18, count=3):
    """De-rocked true attitude with a chosen lag replaces the sensor after a 1.5 s takeoff handover."""
    tune = gates.cases('tune', 'stress', count, 6.)
    infos = [f'truth_derocked@{t}' for t in (.13, .06, .03, .01)]+['truth_derocked']
    scan = [(sp, sd) for sp in (.25, .5, 1., 2.) for sd in (.5, 1., 2., 4., 8.)]
    tasks = [(c, (1., 1.), 'env', 'none') for c in tune]+[(c, g, i, 'none') for i in infos for g in scan for c in tune]
    with gates.pool(SCRATCH, workers) as ex:
        rows = list(ex.map(gates.task, tasks, chunksize=1))
    by = defaultdict(list)
    for r in rows:
        by[f"{r['info']}|{r['gains'][0]:g},{r['gains'][1]:g}"].append(r)
    table = {k: dict(flown=sum(r['termination'] == 'time_limit' for r in v), episodes=len(v),
                     tilt_rms_deg=float(np.mean([r['tilt_rms_deg'] for r in v if r['termination'] == 'time_limit']))
                     if any(r['termination'] == 'time_limit' for r in v) else None,
                     crash_time_s=float(np.mean([r['elapsed'] for r in v if r['termination'] != 'time_limit']))
                     if any(r['termination'] != 'time_limit' for r in v) else None) for k, v in by.items()}
    return dict(handover_s=gates.HANDOVER_T, table=table, episodes=rows)


def gimbal_step(count=2):
    """Paired runs: a 50 ms +0.035 rad pitch-gimbal residual at 3 s minus the same case without it."""
    env = env_for('residual')
    m = env.sim.model
    site, thrust = m.site('history_imu').id, m.body('thrust_unit').id
    joint = int(m.joint('pitch_joint').qposadr[0])
    pitch = lambda R: math.degrees(math.asin(max(-1., min(1., -R[6]))))

    def run(c, pulse):
        env.configure(c)
        env.reset(seed=c['seed'])
        rows = []
        while True:
            t = env.air.steps*DT
            _, _, term, trunc, _ = env.step(np.array([pulse if 3. <= t < 3.05 else 0., 0., 0.]))
            d = env.sim.data
            rows.append([env.air.steps*DT, pitch(d.site_xmat[site]), pitch(d.xmat[thrust]),
                         math.degrees(float(d.qpos[joint])), math.degrees(env.last_reading[1])])
            if term or trunc:
                break
        return np.array(rows)
    out = []
    for c in gates.cases('tune', 'nominal', count, 3.6):
        ref, hit = run(c, 0.), run(c, 1.)
        n = min(len(ref), len(hit))
        diff = hit[:n]-ref[:n]
        keep = (ref[:n, 0] >= 2.99) & (ref[:n, 0] <= 3.6)
        out.append(dict(seed=c['seed'], columns=['t', 'base_pitch_deg', 'thrust_unit_pitch_deg', 'pitch_joint_deg',
                                                 'sensor_pitch_deg'],
                        difference=np.column_stack([ref[:n, 0][keep], diff[keep, 1:]]).round(4).tolist()))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['rocking', 'lagscan', 'gimbal_step'])
    p.add_argument('--output', type=Path)
    p.add_argument('--workers', type=int, default=18)
    a = p.parse_args()
    result = dict(rocking=rocking, gimbal_step=gimbal_step)[a.stage]() if a.stage != 'lagscan' else lagscan(a.workers)
    text = json.dumps(result, indent=1)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(text+'\n')
    print(text[:4000])


if __name__ == '__main__':
    main()
