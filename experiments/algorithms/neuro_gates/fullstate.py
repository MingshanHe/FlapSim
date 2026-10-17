"""Gate for gates_report section 5.6: how much headroom does multivariable dynamic control have on hover?

Upper bound: an ideal full-state linear feedback per axis, fed TRUTH (never the observation):
    gimbal_i = trim_i - a*theta_k - b*theta_dot_k - c*int(theta_k) + d*(phi_i - phi_ref) + e*phi_dot_i
with theta the de-rocked base angle and rate (gates.Oracle 'derocked', about zero lag), phi the gimbal joint
angle and rate. The motor command stays the original PID height loop. Ten gains, CEM-tuned on 6 stress
cases (common random numbers), compared on 16 held-out stress cases with the original PID, the pilot's kp x1.5
PID and the best PID of the 25-point grid tuned on the same 6 cases.

Reading the result: if the full-state bound beats the best PID by only a little, no controller (SNN or
classical) has room on hover inside the original interface; if it beats it a lot, an IMU-only dynamic
compensator becomes a research question.

    python -m experiments.algorithms.neuro_gates.fullstate --output runs/neuro_gates/fullstate
"""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from ..snn_pid_compare.common import DT
from ..snn_pid_compare.environment import LOW, HIGH, CENTER, HALF
from ..robust_control.environment import ResearchEnv
from . import gates

HEIGHT_WEIGHT = .3  # J = tilt RMS (deg, t >= 2 s) + 0.3 x height RMSE (cm); crash = 40 - elapsed
# The controller is the (grid-best) PID command plus truth-state terms per axis: gains on the filtered de-rocked
# angle and rate, on the gimbal angle and rate, and the log of the first-order filter time constant applied to
# the de-rocked angle and rate. x = 0 is exactly the PID, so the bound can only improve on it, and the result
# measures the increment that fast multivariable feedback adds. (A lagged de-rocked truth is not equivalent
# to the AHRS reading: the same PID gains on it fly 13.6 deg against 8.6 deg, so the PID is not re-expressed.)
NAMES = ('da_theta', 'db_rate', 'd_gimbal', 'e_gimbal_rate', 'log_tau_f')
TAU_F = (.002, .3)
CEM = dict(pop=16, iters=10, elite=4)


def gains_from_pid(p=None, tau_f=.03):
    """CEM start: zero increments (= the PID), filter at 30 ms. Order: roll axis, pitch axis."""
    return np.array([[0., 0., 0., 0., math.log(tau_f)] for k in range(2)]).ravel()


SD0 = np.array([.05, .03, .15, .005, .7]*2)


def run(env, c, mode, x=None, gains=(1., 1.)):
    """mode 'pid': PID with multiplied gains. mode 'fullstate': x = 10 gains, PID until the handover."""
    env.configure(c)
    env.reset(seed=c['seed'])
    p, (sp, sd) = env.p, gains
    env.teacher.p = dict(p, kp=[sp*k for k in p['kp']], kd=[sd*k for k in p['kd']], ki=[sp*k for k in p['ki']])
    m, d = env.sim.model, env.sim.data
    joints = [m.joint('pitch_joint'), m.joint('roll_joint')]  # command index 0 (alpha) <-> pitch axis, 1 (beta) <-> roll
    qadr = [int(j.qposadr[0]) for j in joints]
    vadr = [int(j.dofadr[0]) for j in joints]
    ref = np.asarray(env.air.env.reference_angles, dtype=float)
    oracle = gates.Oracle(env, 'derocked')
    K = None if x is None else np.asarray(x, dtype=float).reshape(2, 5)
    alpha = None if K is None else 1.-np.exp(-DT/np.clip(np.exp(K[:, 4]), *TAU_F))  # per-axis filter coefficient
    filtered, handed = np.zeros((2, 2)), False
    saturated = 0
    while True:
        t = env.air.steps*DT
        cmd = env.teacher_command()
        if mode == 'fullstate' and t >= gates.HANDOVER_T:
            state = np.column_stack([oracle.queue[-1], oracle.rate])  # rows: axis (roll, pitch); cols: angle, rate
            if not handed:
                filtered, handed = state.copy(), True
            filtered += alpha[:, None]*(state-filtered)
            for k in range(2):  # axis k: 0 roll, 1 pitch; gimbal index i = 1-k
                i = 1-k
                phi, phid = float(d.qpos[qadr[i]])-ref[i], float(d.qvel[vadr[i]])
                da, db, dd, e, _ = K[k]
                cmd[i] += -da*filtered[k, 0]-db*filtered[k, 1]+dd*phi+e*phid
            saturated += int((np.abs(cmd[:2]) >= .3).any())
        _, _, term, trunc, out = env.step((np.clip(cmd, LOW, HIGH)-CENTER)/HALF)
        oracle.update()
        if term or trunc:
            break
    ep = out['episode']
    flown = ep['termination'] == 'time_limit'
    s = np.asarray(env.samples)
    late = s[:, 0] >= 2.
    tilt = np.degrees(s[late, 1])
    J = ep['tilt_rms_deg']+HEIGHT_WEIGHT*100*ep['height_rmse_m'] if flown else 40.-ep['elapsed']
    return dict(seed=c['seed'], mode=mode, gains=list(gains), termination=ep['termination'], elapsed=ep['elapsed'],
                objective=float(J), tilt_rms_deg=ep['tilt_rms_deg'], tilt_peak_deg=float(tilt.max()) if late.any() else None,
                height_rmse_m=ep['height_rmse_m'], saturated_fraction=saturated/max(1, env.air.steps))


_ENV = None


def worker_init(source):
    global _ENV
    _ENV = ResearchEnv(str(source), mode='direct', task='hover')


def task(args):
    name, c, mode, x, gains = args
    return dict(run(_ENV, c, mode, x, gains), controller=name)


def stats(rows):
    flown = [r for r in rows if r['termination'] == 'time_limit']
    f = lambda k: float(np.mean([r[k] for r in flown])) if flown else None
    return dict(episodes=len(rows), flown=len(flown), J=float(np.mean([r['objective'] for r in rows])),
                tilt_rms_deg=f('tilt_rms_deg'), tilt_peak_deg=f('tilt_peak_deg'),
                height_rmse_cm=None if not flown else 100*f('height_rmse_m'), saturated=f('saturated_fraction'))


def paired(rows_a, rows_b, key='objective'):
    a = {r['seed']: r for r in rows_a}
    b = {r['seed']: r for r in rows_b}
    common = [s for s in a if s in b and a[s]['termination'] == b[s]['termination'] == 'time_limit']
    d = np.array([a[s][key]-b[s][key] for s in common])
    rng = np.random.default_rng(0)
    boot = [rng.choice(d, len(d)).mean() for _ in range(2000)] if len(d) else []
    return dict(n=len(d), mean=float(d.mean()) if len(d) else None, wins=int((d < 0).sum()),
                ci95=[float(np.percentile(boot, q)) for q in (2.5, 97.5)] if len(d) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=18)
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    source = Path('runs/neuro_gates/scratch/source')
    train = gates.cases('tune', 'stress', 6, 6.)
    test = gates.cases('evaluate', 'stress', 16, 8.)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(a.workers, initializer=worker_init, initargs=(str(source),)) as ex:
        p = json.loads((source/'selected_pid.json').read_text())['parameters']
        t0 = time.monotonic()
        # 1. PID grid on the training cases (same-budget classical baseline)
        rows = list(ex.map(task, [(f'{g[0]:g},{g[1]:g}', c, 'pid', None, g) for g in gates.GRID for c in train], chunksize=1))
        table = {}
        for g in gates.GRID:
            name = f'{g[0]:g},{g[1]:g}'
            table[name] = stats([r for r in rows if r['controller'] == name])
        best_name = min(table, key=lambda k: (-table[k]['flown'], table[k]['J']))
        grid_best = tuple(float(v) for v in best_name.split(','))
        print(json.dumps(dict(stage='pid_grid', best=grid_best, J=table[best_name]['J'], wall_s=round(time.monotonic()-t0)),
                         indent=None), flush=True)
        # 2. CEM on the full-state gains, same training cases
        rng = np.random.default_rng(a.seed)
        mu, sd = gains_from_pid(p), SD0.copy()
        history, evaluated = [], []
        for it in range(CEM['iters']):
            xs = mu+sd*rng.standard_normal((CEM['pop'], len(mu)))
            if it == 0:
                xs[0] = mu
            rows = list(ex.map(task, [(str(i), c, 'fullstate', x.tolist(), grid_best) for i, x in enumerate(xs) for c in train],
                               chunksize=1))
            Js = np.array([stats([r for r in rows if r['controller'] == str(i)])['J'] for i in range(len(xs))])
            flown = np.array([stats([r for r in rows if r['controller'] == str(i)])['flown'] for i in range(len(xs))])
            for i in range(len(xs)):
                evaluated.append(dict(iter=it, x=xs[i].tolist(), J=float(Js[i]), flown=int(flown[i])))
            order = np.lexsort((Js, -flown))
            elite = xs[order[:CEM['elite']]]
            mu, sd = elite.mean(0), np.maximum(elite.std(0), SD0*.1)
            history.append(dict(iter=it, best_J=float(Js[order[0]]), best_flown=int(flown[order[0]]), mean_J=float(Js.mean()),
                                mu=mu.tolist(), wall_s=round(time.monotonic()-t0)))
            print(json.dumps(history[-1]), flush=True)
        best = min(evaluated, key=lambda e: (-e['flown'], e['J']))
        # 3. held-out test
        arms = [('pid_original', 'pid', None, (1., 1.)), ('pid_retuned', 'pid', None, (1.5, 1.)),
                ('pid_grid_best', 'pid', None, grid_best), ('fullstate_bound', 'fullstate', best['x'], grid_best),
                ('fullstate_cem_mean', 'fullstate', mu.tolist(), grid_best)]
        rows = list(ex.map(task, [(n, c, m, x, g) for n, m, x, g in arms for c in test], chunksize=1))
        by = {n: [r for r in rows if r['controller'] == n] for n, *_ in arms}
        summary = {n: stats(v) for n, v in by.items()}
        pairs = {f'{n}-vs-pid_grid_best': {k: paired(by[n], by['pid_grid_best'], k) for k in ('objective', 'tilt_rms_deg', 'height_rmse_m')}
                 for n in by if n != 'pid_grid_best'}
        result = dict(stage='fullstate', names=NAMES, pid=p, grid=table, grid_best=grid_best, cem=history, best=best,
                      summary=summary, paired=pairs, rows=rows)
        (a.output/'fullstate.json').write_text(json.dumps(result, indent=1)+'\n')
        for n, s in summary.items():
            print(f"{n:20s} flown {s['flown']:2d}/16 J {s['J']:.2f} tilt {s['tilt_rms_deg'] or float('nan'):.2f} "
                  f"peak {s['tilt_peak_deg'] or float('nan'):.1f} height {s['height_rmse_cm'] or float('nan'):.1f} cm")
        print('best gains (roll a,b,c,d,e | pitch a,b,c,d,e):', [round(v, 4) for v in best['x']])
        print(json.dumps(pairs['fullstate_bound-vs-pid_grid_best'], indent=1))


if __name__ == '__main__':
    main()
