"""Raw pitch gyro vs true crank: wingbeat component amplitude, slow component, and the gyro->crank phase offset per cycle,
against the commanded motor speed. Single process, PID only, 4 stress cases of 8 s."""
import json, math, sys, numpy as np
from experiments.algorithms.spike_steer import pilot as P
from experiments.algorithms.snn_pid_compare.common import DT
from experiments.algorithms.snn_pid_compare.environment import LOW, HIGH, CENTER, HALF
from experiments.algorithms.wingbeat_snn.plant import PHASE_OFFSET
P.worker_init('runs/neuro_gates/scratch/source'); env = P._ENV
crank = int(env.sim.model.joint('carrier_2_joint').qposadr[0])
out = []
for c in P.cases('test', 16, 8.)[:4]:
    env.configure(c); env.reset(seed=c['seed'])
    p = env.p; env.teacher.p = dict(p, kp=[1.5*k for k in p['kp']], kd=p['kd'], ki=[1.5*k for k in p['ki']])
    log = []
    while True:
        cmd = env.teacher_command()
        _, _, term, trunc, _ = env.step((np.clip(cmd, LOW, HIGH)-CENTER)/HALF)
        r = env.last_reading; d = env.sim.data
        log.append((env.air.steps*DT, float(d.qpos[crank]), float(r[3]), float(r[4]), float(r[5]), float(r[6]), float(r[7]), float(r[8]), float(env.command[2]), float(d.qvel[int(env.sim.model.joint('carrier_2_joint').dofadr[0])])))
        if term or trunc: break
    a = np.array(log); a = a[a[:, 0] >= 2.]
    t, phi, gx, gy, gz, ax, ay, az, motor, crank_rate = a.T
    # per-cycle fit of gy = A cos(phi - delta) + slow (slow from the fit's constant)
    cyc = np.floor(phi/(2*math.pi)); rows = []
    for k in np.unique(cyc):
        m = cyc == k
        if m.sum() < 15: continue
        X = np.c_[np.cos(phi[m]), np.sin(phi[m]), np.ones(m.sum())]
        for name, y in (('gy', gy), ('gx', gx), ('ax', ax), ('az', az)):
            co, *_ = np.linalg.lstsq(X, y[m], rcond=None)
            rows.append((k, name, math.hypot(co[0], co[1]), math.degrees(math.atan2(co[1], co[0])) % 360, co[2], motor[m].mean(), crank_rate[m].mean()/(2*math.pi)))
    out.append(dict(seed=c['seed'], rows=rows))
    def circ(deg):
        z = np.mean(np.exp(1j*np.radians(deg)))
        return math.degrees(np.angle(z)) % 360, math.degrees(math.sqrt(-2*math.log(max(1e-9, abs(z)))))
    r = np.array([x[2:] for x in rows if x[1] == 'gy'])
    om, osd = circ(r[:, 1])
    print(f"seed {c['seed']}: pitch gyro wingbeat amp {r[:,0].mean():.2f} rad/s (sd {r[:,0].std():.2f}); slow |gy| rms {np.sqrt(np.mean(r[:,2]**2)):.2f}; "
          f"gyro->crank offset mean {om:.1f} deg, circ sd {osd:.1f}; wingbeat {r[:,4].mean():.1f} Hz (range {r[:,4].min():.1f}-{r[:,4].max():.1f}); "
          f"motor cmd {r[:,3].mean():.0f} ({r[:,3].min():.0f}-{r[:,3].max():.0f})")
    slope = np.polyfit(r[:, 3], r[:, 1], 1)[0]*100
    print(f"   corr(offset, motor cmd) {np.corrcoef(r[:,1], r[:,3])[0,1]:+.2f}; corr(offset, wingbeat Hz) {np.corrcoef(r[:,1], r[:,4])[0,1]:+.2f}; slope {slope:+.2f} deg per 100 rad/s")
    for name in ('gx', 'ax', 'az'):
        q = np.array([x[2:] for x in rows if x[1] == name])
        qm, qsd = circ(q[:, 1])
        print(f"   {name}: wingbeat amp {q[:,0].mean():.3f}, offset {qm:.0f} deg, circ sd {qsd:.0f}")
print('PHASE_OFFSET (PLL calibration) =', round(math.degrees(PHASE_OFFSET), 1))
json.dump(out, open(sys.argv[1], 'w'))
