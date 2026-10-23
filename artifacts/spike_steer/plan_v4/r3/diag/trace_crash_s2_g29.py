import json, math, numpy as np
from pathlib import Path
from experiments.algorithms.spike_steer import bench as B, descend_r3 as R
R.install(fresh_test=False)
src = Path('runs/neuro_gates/scratch/source')
base = json.loads(Path('runs/spike_steer/bench_r2/baseline.json').read_text())
B.worker_init(str(src), base['prereg']['dash_m'] or .25)
env = B._ENV
spec = json.loads(Path('artifacts/spike_steer/plan_v4/r2/finetune_s2/spec_val_g29.json').read_text())
c = [c for c in B.cases('nod', 'test') if c['seed'] == 2096100008][0]
for E in (None, 50.):
    s = dict(spec)
    if E is None: s.pop('cmd_envelope_deg', None)
    else: s['cmd_envelope_deg'] = E
    pol = B.Learned(s)
    print('policy class', type(pol).__name__, 'E', E)
    log = []
    orig = pol.act
    def act(env_, now, t, x, cmd, rel):
        out = orig(env_, now, t, x, cmd, rel)
        r = env_.last_reading
        pos, vel, Rm = env_.air.env._state()
        true_tilt = math.degrees(math.acos(min(1., max(-1., float(Rm[2, 2])))))
        log.append((t, math.degrees(math.hypot(float(r[0]), float(r[1]))), math.degrees(float(r[1])), true_tilt, bool(cmd.any()), getattr(pol, 'envelope', None), out[1]))
        return out
    pol.act = act
    res = B.run_episode(env, 'nod', c, pol)
    a = np.array([(l[0], l[1], l[2], l[3]) for l in log])
    i = int(np.argmax(a[:, 3]))
    print(f"term {res['termination']} elapsed {res['elapsed']:.2f} score {res['score']:.2f} peak_pitch {res.get('peak_pitch_deg')}")
    print(f"max true tilt {a[:,3].max():.1f} at t={a[i,0]:.2f}; AHRS tilt there {a[i,1]:.1f}, AHRS pitch {a[i,2]:.1f}; max AHRS tilt {a[:,1].max():.1f} at t={a[int(np.argmax(a[:,1])),0]:.2f}")
    print('envelope_t', getattr(pol, 'envelope_t', None), 'reading len', len(env.last_reading), 'reading sample', np.round(np.asarray(env.last_reading, float)[:6], 3))
    sel = a[(a[:, 0] > 2.3) & (a[:, 0] < a[i, 0]+.05)][::50]
    print(' t    ahrs_tilt ahrs_pitch true_tilt'); [print(f" {r[0]:.2f} {r[1]:7.1f} {r[2]:8.1f} {r[3]:8.1f}") for r in sel]
