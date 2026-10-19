"""Plan v6: the descending SNN for 8-direction manoeuvres (bench8). Same algorithm as plan v4 round 5 (descend.py,
descend_gated.py, descend_r4.py, descend_r5.py); what changes is the interface of the learned layer:

  inputs   25 = 16 rectified IMU halves + bias + 8 rectified direction-command channels
           (nod*cos phi, nod*sin phi, dash*cos phi, dash*sin phi, each split into +/- halves; bench8.features8)
  outputs  up / down spiking units (motor pulse, timing = pitch torque direction; unchanged)
           + 2 graded units -> (roll, pitch) set-point = 0.7*tanh(g) each (round 5: pitch only)

Gate (command on or < 1 s after), righting reflex (25 deg), refractory 37 ms, imitation (BCE + 5 * ref MSE, 40
epochs) and OpenAI-ES (pop 32, sigma 0.03, Adam 0.02, rank utility) are those of round 5. Hover and upset never
switch the learned layer on, so they are identical to round 5 / the classical opponent by construction and are not
re-run here.

    python -m experiments.algorithms.spike_steer.descend8 selftest
    python -m experiments.algorithms.spike_steer.descend8 baseline --output runs/spike_steer/v6 --workers 30
    python -m experiments.algorithms.spike_steer.descend8 collect  --output runs/spike_steer/v6 --workers 30
    python -m experiments.algorithms.spike_steer.descend8 train    --output runs/spike_steer/v6 --seed 1
    python -m experiments.algorithms.spike_steer.descend8 finetune --output runs/spike_steer/v6 --seed 1 --workers 30
    python -m experiments.algorithms.spike_steer.descend8 reselect --output runs/spike_steer/v6 --seed 1 --workers 30
    python -m experiments.algorithms.spike_steer.descend8 evaluate --output runs/spike_steer/v6 --seed 1 --workers 30
"""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from . import bench as B
from . import bench8 as E
from . import pilot_v3 as V

N_IN = 25
N_G = 2                                   # graded units: 0 = roll set-point, 1 = pitch set-point
K = len(B.IN_DELAYS_TICKS)
D = len(B.HID_DELAYS_MS)
TASKS = ('nod', 'dash')
CLASSES = dict(nod=('pitch', 'roll'), dash=('pitch', 'roll', 'diag'))
CLASS = {0: 'pitch', 180: 'pitch', 90: 'roll', 270: 'roll', 45: 'diag', 135: 'diag', 225: 'diag', 315: 'diag'}
TAIL_S, UPSET_TILT_DEG, REFRACTORY_S = 1.0, 25., .037
SOFT_TILT_DEG, PENALTY_PER_DEG, HARD_TILT_DEG = 60., .3, 70.
WEIGHTS = dict(nod=2/3, dash=1/3)
N_COLLECT = dict(nod=32, dash=32)          # nod 8 / dash 4 per direction
N_VAL_IMIT = dict(nod=8, dash=8)
ES = dict(pop=32, sigma=.03, lr=.02, cases=dict(nod=8, dash=8), validate_every=5)
N_VAL_SELECT = 64                          # per task (nod 16 / dash 8 per direction)
N_TEST = 32                                # per task (nod 8 / dash 4 per direction)
OFF = dict(collect=0, collect_val=50_000, select=60_000, es=100_000)
TEST_BASE = 2_097_100_000                  # blind set (nod / dash); never used before the evaluation stage


# ----------------------------------------------------------------------------------------- network (numpy)
class Net8:
    """bench.DescendingNet with N_G graded units (W_g [G, H, D], b_g [G])."""

    def __init__(self, spec):
        s = spec
        self.W_in, self.b_h = np.asarray(s['W_in'], float), np.asarray(s['b_h'], float)
        self.W_out, self.b_o = np.asarray(s['W_out'], float), np.asarray(s['b_o'], float)
        self.W_g, self.b_g = np.asarray(s['W_g'], float), np.asarray(s['b_g'], float)
        self.H, self.n_in, _ = self.W_in.shape
        self.S = float(s['spike_scale'])
        self.a_m, self.a_s, self.a_g = math.exp(-.001/s['tau_m']), math.exp(-.001/s['tau_s']), math.exp(-.001/s['tau_g'])
        self.din, self.dhid = np.asarray(B.IN_DELAYS_TICKS), np.asarray(B.HID_DELAYS_MS)
        self.L = int(self.dhid.max())+2
        self.reset()

    def reset(self):
        self.xbuf = np.zeros((self.n_in, int(self.din.max())+1))
        self.cur_h, self.v_h = np.zeros(self.H), np.zeros(self.H)
        self.cur_o, self.v_o = np.zeros(2), np.zeros(2)
        self.cur_g, self.g = np.zeros(N_G), np.zeros(N_G)
        self.hist = np.zeros((self.L, self.H))
        self.p = 0
        self.spikes = 0

    def tick(self, x):
        self.xbuf = np.roll(self.xbuf, 1, axis=1)
        self.xbuf[:, 0] = x
        drive_h = np.einsum('hik,ik->h', self.W_in, self.xbuf[:, self.din])+self.b_h
        fired = np.zeros(2, bool)
        for _ in range(2):
            self.cur_h = self.a_s*self.cur_h+(1-self.a_s)*drive_h
            self.v_h = self.a_m*self.v_h+(1-self.a_m)*self.cur_h
            spk = self.v_h >= 1.
            self.v_h = self.v_h-spk
            self.hist[self.p] = spk
            self.spikes += int(spk.sum())
            rec = self.hist[(self.p-1-self.dhid) % self.L].T                       # H, D
            drive_o = self.S*np.einsum('ohd,hd->o', self.W_out, rec)+self.b_o
            self.cur_o = self.a_s*self.cur_o+(1-self.a_s)*drive_o
            self.v_o = self.a_m*self.v_o+(1-self.a_m)*self.cur_o
            spk_o = self.v_o >= 1.
            self.v_o = self.v_o-spk_o
            fired |= spk_o
            drive_g = self.S*np.einsum('ghd,hd->g', self.W_g, rec)+self.b_g
            self.cur_g = self.a_s*self.cur_g+(1-self.a_s)*drive_g
            self.g = self.a_g*self.g+(1-self.a_g)*self.cur_g
            self.p = (self.p+1) % self.L
        return fired, B.REF_MAX*np.tanh(self.g)


class Learned8:
    """Round-5 controller generalised: gate + 1 s tail, righting reflex, refractory period; the net always ticks."""

    def __init__(self, spec):
        self.spec = spec

    def reset(self, c):
        s = self.spec
        self.net = Net8(s)
        self.twitch = 0
        self.steer = V.SteerV3(None, s['base_law'], False)
        self.steer.reset()
        self.steer_up = V.SteerV3(None, s['upset_law'], False)
        self.steer_up.reset()
        self.tilt_thr = math.radians(float(s.get('upset_tilt_deg', UPSET_TILT_DEG)))
        self.tail, self.refr = float(s.get('tail_s', TAIL_S)), float(s.get('refractory_s', REFRACTORY_S))
        self.upset, self.off_t, self.was_on, self.last_onset = False, None, False, None
        self.onsets = self.cancelled = 0

    def net_step(self, t, x):
        """Network tick + twitch bookkeeping (the part of act that depends on the weights)."""
        fired, ref = self.net.tick(x)
        fire = False
        if t >= B.gates.HANDOVER_T:
            if self.twitch > 0:
                self.twitch -= 1
                fire = True
            elif fired.any():
                self.twitch = B.TWITCH-1
                fire = True
        else:
            ref = np.zeros(N_G)
        return fire, ref

    def replay(self, t, x):
        """Rebuild the network state on a recorded pre-command input (gate closed: output unused; cache8)."""
        self.net_step(t, x)

    def act(self, env, now, t, x, cmd, rel):
        fire, ref = self.net_step(t, x)
        law_pulse, _ = self.steer.step(env, now, t)
        up_pulse, _ = self.steer_up.step(env, now, t)
        on = bool(np.any(cmd))
        if self.was_on and not on:
            self.off_t = t
        self.was_on = on
        in_charge = on or (self.off_t is not None and t-self.off_t < self.tail)
        if not self.upset and not in_charge and math.hypot(float(env.last_reading[0]), float(env.last_reading[1])) > self.tilt_thr:
            self.upset = True
        if self.upset:
            law_pulse = up_pulse
        if in_charge:
            if fire and self.twitch == B.TWITCH-1:
                if self.last_onset is not None and t-self.last_onset < self.refr:
                    fire, self.twitch = False, 0
                    self.cancelled += 1
                else:
                    self.last_onset = t
                    self.onsets += 1
            return fire, ref
        return bool(law_pulse), np.zeros(N_G)


def learned_factory(arg):
    return Learned8(arg['spec'])


def learned_job(name, task, c, spec, trace=False):
    return (name, task, c, 'learned', dict(factory=learned_factory, spec=spec), trace)


# ----------------------------------------------------------------------------------------- network (torch)
def spike_fn(v, beta=.25):
    import torch
    s = (v >= 1.).to(v.dtype)
    sg = torch.sigmoid((v-1.)/beta)
    return s.detach()+sg-sg.detach()


class SNN8:
    def __init__(self, hidden=8, seed=0):
        import torch
        dt = torch.float64
        g = torch.Generator().manual_seed(seed)
        r = lambda *shape: torch.randn(*shape, generator=g, dtype=dt)
        self.H = hidden
        self.W_in = (r(hidden, N_IN, K)*.3/math.sqrt(K)).requires_grad_()
        self.b_h = (r(hidden)*.3-.5).requires_grad_()
        self.W_out = (r(2, hidden, D)*.3/math.sqrt(D)).requires_grad_()
        self.b_o = torch.full((2,), -2., dtype=dt).requires_grad_()
        self.W_g = (r(N_G, hidden, D)*.3/math.sqrt(D)).requires_grad_()
        self.b_g = torch.zeros(N_G, dtype=dt).requires_grad_()
        self.log_tau_m = torch.tensor(math.log(1e-3), dtype=dt).requires_grad_()
        self.log_tau_s = torch.tensor(math.log(.5e-3), dtype=dt).requires_grad_()

    def parameters(self):
        return [self.W_in, self.b_h, self.W_out, self.b_o, self.W_g, self.b_g, self.log_tau_m, self.log_tau_s]

    def taus(self):
        import torch
        return torch.clamp(self.log_tau_m.exp(), 3e-4, 1e-2), torch.clamp(self.log_tau_s.exp(), 3e-4, 1e-2)

    def forward(self, x):
        """x [T, B, N_IN] -> proxy [T,B,2], spikes [T,B,2], ref [T,B,G], hidden spikes [2T,B,H]."""
        import torch
        T, Bn, _ = x.shape
        tm, ts = self.taus()
        a_m, a_s = torch.exp(-.001/tm), torch.exp(-.001/ts)
        a_g = math.exp(-.001/B.TAU_G)
        kmax = max(B.IN_DELAYS_TICKS)
        xp = torch.cat([torch.zeros(kmax, Bn, N_IN, dtype=x.dtype), x], 0)
        x_del = torch.stack([xp[kmax-k:kmax-k+T] for k in B.IN_DELAYS_TICKS], -1)
        drive_h = torch.einsum('tbik,hik->tbh', x_del, self.W_in)+self.b_h
        off = max(B.HID_DELAYS_MS)+1
        z = lambda *s: torch.zeros(*s, dtype=x.dtype)
        cur_h, v_h, cur_o, v_o, cur_g, g = z(Bn, self.H), z(Bn, self.H), z(Bn, 2), z(Bn, 2), z(Bn, N_G), z(Bn, N_G)
        hist = [z(Bn, self.H) for _ in range(off)]
        proxy, out_spk, refs = [], [], []
        for t in range(T):
            p_t, s_t = z(Bn, 2), z(Bn, 2)
            for sub in range(2):
                s = t*2+sub
                cur_h = a_s*cur_h+(1-a_s)*drive_h[t]
                v_h = a_m*v_h+(1-a_m)*cur_h
                spk_h = spike_fn(v_h)
                v_h = v_h-spk_h
                hist.append(spk_h)
                rec = torch.stack([hist[off+s-1-d] for d in B.HID_DELAYS_MS], -1)
                drive_o = B.SPIKE_SCALE*torch.einsum('bhd,ohd->bo', rec, self.W_out)+self.b_o
                cur_o = a_s*cur_o+(1-a_s)*drive_o
                v_o = a_m*v_o+(1-a_m)*cur_o
                spk_o = spike_fn(v_o)
                p_t = torch.maximum(p_t, torch.sigmoid((v_o-1.)/.25))
                s_t = torch.maximum(s_t, spk_o.detach())
                v_o = v_o-spk_o
                drive_g = B.SPIKE_SCALE*torch.einsum('bhd,ghd->bg', rec, self.W_g)+self.b_g
                cur_g = a_s*cur_g+(1-a_s)*drive_g
                g = a_g*g+(1-a_g)*cur_g
            proxy.append(p_t)
            out_spk.append(s_t)
            refs.append(B.REF_MAX*torch.tanh(g))
        return torch.stack(proxy), torch.stack(out_spk), torch.stack(refs), torch.stack(hist[off:off+2*T])

    def export(self, **extra):
        tm, ts = (float(v) for v in self.taus())
        f = lambda t: t.detach().numpy().tolist()
        return dict(kind='descending_snn8', inputs=N_IN, hidden=self.H, graded=N_G, W_in=f(self.W_in), b_h=f(self.b_h),
                    W_out=f(self.W_out), b_o=f(self.b_o), W_g=f(self.W_g), b_g=f(self.b_g), tau_m=tm, tau_s=ts, tau_g=B.TAU_G,
                    spike_scale=B.SPIKE_SCALE, **extra)


KEYS = ('W_in', 'b_h', 'W_out', 'b_o', 'W_g', 'b_g')


def flatten(spec):
    return np.concatenate([np.asarray(spec[k], float).ravel() for k in KEYS])


def unflatten(spec, theta):
    s, i = dict(spec), 0
    for k in KEYS:
        arr = np.asarray(spec[k], float)
        s[k] = theta[i:i+arr.size].reshape(arr.shape).tolist()
        i += arr.size
    return s


# ----------------------------------------------------------------------------------------- objective
def ratios(rows, norm):
    """Per task: mean over cases of score / normaliser of the case's direction class."""
    return {t: float(np.mean([r['score']/norm[t][CLASS[int(r['dir_deg'])]] for r in rows if r['task'] == t])) for t in TASKS
            if any(r['task'] == t for r in rows)}


def penalties(rows):
    return {t: float(np.mean([max(0., (r['tilt_peak_deg'] or 0.)-SOFT_TILT_DEG)*PENALTY_PER_DEG for r in rows if r['task'] == t]))
            for t in TASKS if any(r['task'] == t for r in rows)}


def objective(rows, norm, weights=WEIGHTS):
    rt, pn = ratios(rows, norm), penalties(rows)
    tot = sum(weights[t] for t in rt)
    return float(sum(weights[t]*(rt[t]+pn[t]) for t in rt)/tot), rt


# ----------------------------------------------------------------------------------------- stages
def load(out):
    return json.loads((Path(out)/'baseline.json').read_text())


def stage_baseline(a):
    """Classical opponents: the 2-D schedule tuned per task and direction class on the tune set (the pilot grid, same
    budget for every class); normalisers = opponent mean score per (task, class) on the ES training distribution."""
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    pilot = json.loads(Path(a.pilot).read_text())
    law = B.law_tuned()
    opp = {t: {cl: pilot[t]['best_per_class'][cl]['config'] for cl in CLASSES[t]} for t in TASKS}
    norm = {}
    with E.pool(a.workers) as ex:
        for t in TASKS:
            cs = E.cases8(t, E.BASES['train'], 64, offset=OFF['es']-1000)
            rows = E.run_jobs(ex, [('opp', t, c, 'classical', dict(law=law, schedule=opp[t][CLASS[int(c['dir_deg'])]])) for c in cs])
            norm[t] = {cl: float(np.mean([r['score'] for r in rows if CLASS[int(r['dir_deg'])] == cl])) for cl in CLASSES[t]}
            print(t, json.dumps(norm[t]), 'crashes', sum(not r['flown'] for r in rows), flush=True)
    (out/'baseline.json').write_text(json.dumps(dict(opponent=opp, law=law, normalisers=norm, signs=E.SIGNS, dash_m=B.DASH_M['value'],
                                                     upset_law=json.loads(Path(a.upset_law).read_text())['upset_law']), indent=1))


class Traced2(E.Classical2):
    def reset(self, c):
        super().reset(c)
        self.log = []

    def act(self, env, now, t, x, cmd, rel):
        pulse, ref = super().act(env, now, t, x, cmd, rel)
        if t >= 2.:
            u = -(self.law['k_theta']*env.last_reading[1]+self.law['k_q']*self.steer.estimator.slow)
            self.log.append(np.r_[x, float(pulse), float(np.sign(u)), np.asarray(ref, float)])
        return pulse, ref


def collect_job(args):
    name, t, c, law, sched = args
    pol = Traced2(law, sched)
    res = E.run_episode8(B._ENV, t, c, pol)
    return dict(seed=c['seed'], task=t, score=res['score'], flown=res['flown'], log=np.array(pol.log, np.float32))


def stage_collect(a):
    out = Path(a.output)
    base = load(out)
    with E.pool(a.workers) as ex:
        for split, off, n in (('train', OFF['collect'], N_COLLECT), ('val', OFF['collect_val'], N_VAL_IMIT)):
            jobs = [('opp', t, c, base['law'], base['opponent'][t][CLASS[int(c['dir_deg'])]])
                    for t in TASKS for c in E.cases8(t, E.BASES['train'], n[t], offset=off)]
            rows = list(ex.map(collect_job, jobs, chunksize=1))
            L = np.array([len(r['log']) for r in rows])
            X = np.zeros((len(rows), int(L.max()), N_IN+4), np.float32)   # x(25), pulse, sign(u), ref(roll, pitch)
            for i, r in enumerate(rows):
                X[i, :L[i]] = r['log']
            np.savez_compressed(out/f'collect_{split}.npz', X=X, lengths=L, tasks=np.array([r['task'] for r in rows]))
            v = np.concatenate([X[i, :n_] for i, n_ in enumerate(L)])
            print(json.dumps(dict(split=split, episodes=len(rows), pulse_rate=float(v[:, N_IN].mean()),
                                  ref_abs=np.abs(v[:, N_IN+2:]).mean(0).round(4).tolist(), flown=sum(r['flown'] for r in rows))), flush=True)


def targets(X):
    pulse, sign = X[:, :, N_IN], X[:, :, N_IN+1]
    y = np.stack([pulse*(sign > 0), pulse*(sign < 0)], -1).astype(np.float32)
    return X[:, :, :N_IN], y, (X[:, :, N_IN+2:N_IN+4]/B.REF_MAX).astype(np.float32)


def stage_train(a):
    import torch
    torch.set_num_threads(a.threads)
    out = Path(a.output)
    base = load(out)
    tr, va = np.load(out/'collect_train.npz'), np.load(out/'collect_val.npz')
    Xtr, ytr, rtr = targets(tr['X'])
    Xva, yva, rva = targets(va['X'])
    model = SNN8(seed=a.seed)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    bce = torch.nn.BCELoss(reduction='none')
    W = 200
    starts = [(e, s) for e in range(Xtr.shape[0]) for s in range(0, int(tr['lengths'][e])-W+1, W//2)]
    vst = [(e, s) for e in range(Xva.shape[0]) for s in range(0, int(va['lengths'][e])-W+1, W)]
    T = lambda A, idx: torch.tensor(np.stack([A[e, s:s+W] for e, s in idx], 1), dtype=torch.float64)
    xv, yv, rv = T(Xva, vst), T(yva, vst), T(rva, vst)
    rng = np.random.default_rng(a.seed)
    best, hist, t0 = None, [], time.monotonic()
    extra = dict(gate='command', tail_s=TAIL_S, base_law=base['law'], upset_law=base['upset_law'], upset_tilt_deg=UPSET_TILT_DEG,
                 refractory_s=REFRACTORY_S, seed=a.seed)
    for epoch in range(a.epochs):
        rng.shuffle(starts)
        losses = []
        for b in range(0, len(starts), 32):
            idx = starts[b:b+32]
            xb, yb, rb = T(Xtr, idx), T(ytr, idx), T(rtr, idx)
            proxy, _, ref, _ = model.forward(xb)
            loss = (bce(proxy.clamp(1e-5, 1-1e-5), yb)*(1.+2.*yb)).mean()+5.*((ref/B.REF_MAX-rb)**2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            losses.append(float(loss))
        with torch.no_grad():
            proxy, spk, ref, hid = model.forward(xv)
            tp, fp, fn = float((spk*yv).sum()), float((spk*(1-yv)).sum()), float(((1-spk)*yv).sum())
            vloss = float((bce(proxy.clamp(1e-5, 1-1e-5), yv)*(1.+2.*yv)).mean()+5.*((ref/B.REF_MAX-rv)**2).mean())
        row = dict(epoch=epoch, loss=float(np.mean(losses)), val_loss=vloss, precision=tp/max(1., tp+fp), recall=tp/max(1., tp+fn),
                   ref_rmse=np.sqrt(((ref/B.REF_MAX-rv)**2).mean((0, 1))).tolist(), wall_s=round(time.monotonic()-t0))
        hist.append(row)
        print(json.dumps(row), flush=True)
        if best is None or vloss < best:
            best = vloss
            (out/f'imitate_spec_s{a.seed}.json').write_text(json.dumps(model.export(trained_by='descend8 train', epoch=epoch, **extra), indent=1))
        (out/f'imitate_history_s{a.seed}.json').write_text(json.dumps(hist, indent=1))


def stage_finetune(a):
    out = Path(a.output)
    base = load(out)
    norm = base['normalisers']
    spec0 = json.loads((out/f'imitate_spec_s{a.seed}.json').read_text())
    run = out/f'finetune_s{a.seed}'
    run.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1000+a.seed)
    theta = flatten(spec0)
    m_ad, v_ad = np.zeros_like(theta), np.zeros_like(theta)
    pop, sigma, lr = ES['pop'], ES['sigma'], ES['lr']
    gen0 = 0
    if (run/'checkpoint.npz').exists():
        ck = np.load(run/'checkpoint.npz')
        theta, m_ad, v_ad, gen0 = ck['theta'], ck['m'], ck['v'], int(ck['gen'])
        rng = np.random.default_rng([1000+a.seed, gen0])
        print(f'resumed at generation {gen0}', flush=True)
    t_start = time.monotonic()
    with E.pool(a.workers) as ex:
        if gen0 == 0:
            (run/'spec_val_g-1.json').write_text(json.dumps(dict(spec0, generation=-1), indent=1))
        for gen in range(gen0, a.generations):
            if (time.monotonic()-t_start)/3600. > a.hours:
                break
            t_gen = time.monotonic()
            eps = rng.standard_normal((pop//2, theta.size))
            eps = np.concatenate([eps, -eps], 0)
            batch = [(t, c) for t in TASKS for c in E.cases8(t, E.BASES['train'], ES['cases'][t], offset=OFF['es']+gen*100)]
            cands = [theta+sigma*e for e in eps]+[theta]
            from . import cache8                                    # shared pre-command prefix (exact; cache8.selftest)
            rows = cache8.run_grouped(ex, [(str(i), t, c, unflatten(spec0, th)) for i, th in enumerate(cands) for t, c in batch], chunk=7)
            objs = [objective([r for r in rows if r['controller'] == str(i)], norm)[0] for i in range(len(cands))]
            f = -np.array(objs[:pop])
            ranks = np.empty(pop)
            ranks[np.argsort(-f)] = np.arange(pop)
            util = np.maximum(0., math.log(pop/2+1)-np.log(ranks+1))
            util = util/util.sum()-1./pop
            grad = (util[:, None]*eps).sum(0)/sigma
            m_ad = .9*m_ad+.1*grad
            v_ad = .999*v_ad+.001*grad*grad
            step = lr*(m_ad/(1-.9**(gen+1)))/(np.sqrt(v_ad/(1-.999**(gen+1)))+1e-8)
            theta = theta+step
            centre_rows = [r for r in rows if r['controller'] == str(pop)]
            row = dict(generation=gen, center=objs[-1], center_ratios=objective(centre_rows, norm)[1], best_candidate=float(min(objs[:pop])),
                       mean=float(np.mean(objs[:pop])), flown_center=sum(r['flown'] for r in centre_rows), episodes_per_candidate=len(batch),
                       step_norm=float(np.linalg.norm(step)), wall_s=round(time.monotonic()-t_gen))
            if gen % ES['validate_every'] == ES['validate_every']-1:
                (run/f'spec_val_g{gen}.json').write_text(json.dumps(dict(unflatten(spec0, theta), generation=gen), indent=1))
            with (run/'history.jsonl').open('a') as h:
                h.write(json.dumps(row)+'\n')
            np.savez(run/'checkpoint.npz', theta=theta, m=m_ad, v=v_ad, gen=gen+1)
            print(json.dumps(row), flush=True)


def stage_reselect(a):
    """Round-5 selection: every saved centre on 64 + 64 validation cases (8 per direction); eligible = no crash and
    true tilt peak < HARD_TILT_DEG on every case; pick the lowest penalised objective."""
    out = Path(a.output)
    base = load(out)
    norm = base['normalisers']
    run = out/f'finetune_s{a.seed}'
    cs = {t: E.cases8(t, E.BASES['train'], N_VAL_SELECT, offset=OFF['select']) for t in TASKS}
    specs = sorted(run.glob('spec_val_g*.json'), key=lambda f: int(f.stem.split('g')[-1]))
    res = []
    from . import cache8
    loaded = [json.loads(f.read_text()) for f in specs]
    with E.pool(a.workers) as ex:                                   # all centres on a case share the pre-command prefix
        all_rows = cache8.run_grouped(ex, [(str(k), t, c, s) for k, s in enumerate(loaded) for t in TASKS for c in cs[t]], chunk=len(loaded))
        for k, (f, s) in enumerate(zip(specs, loaded)):
            rows = [r for r in all_rows if r['controller'] == str(k)]
            obj, rt = objective(rows, norm)
            e = dict(generation=s['generation'], objective=obj, ratios=rt, crashes=sum(not r['flown'] for r in rows),
                     over_hard=sum((r['tilt_peak_deg'] or 0) >= HARD_TILT_DEG for r in rows),
                     tilt_peak_max=max(r['tilt_peak_deg'] or 0 for r in rows), spec=str(f))
            res.append(e)
            print(json.dumps(e), flush=True)
    ok = [e for e in res if e['crashes'] == 0 and e['over_hard'] == 0]
    sel = min(ok, key=lambda e: e['objective']) if ok else None
    (out/f'reselect_s{a.seed}.json').write_text(json.dumps(dict(centres=res, selected=sel), indent=1))
    print(json.dumps(dict(selected=sel)), flush=True)


def stage_evaluate(a):
    """Blind set (TEST_BASE): selected controller vs the classical opponent on 32 nod + 32 dash cases, paired.
    Task ratio = mean over cases of (learned score / opponent median score of the case's direction class); per-class
    ratios = learned class mean / opponent class median (preregistration_v6.json)."""
    out = Path(a.output)
    base = load(out)
    sel = json.loads((out/f'reselect_s{a.seed}.json').read_text())['selected']
    spec = json.loads(Path(sel['spec']).read_text())
    cs = {t: E.cases8(t, TEST_BASE, N_TEST) for t in TASKS}
    with E.pool(a.workers) as ex:
        opp_file = out/'test_opponent.json'
        if opp_file.exists():
            opp_rows = json.loads(opp_file.read_text())
        else:
            opp_rows = E.run_jobs(ex, [('opp', t, c, 'classical', dict(law=base['law'], schedule=base['opponent'][t][CLASS[int(c['dir_deg'])]]))
                                       for t in TASKS for c in cs[t]])
            opp_file.write_text(json.dumps(opp_rows))
        rows = E.run_jobs(ex, [learned_job('learned', t, c, spec) for t in TASKS for c in cs[t]])
    rep = {}
    for t in TASKS:
        lr_ = [r for r in rows if r['task'] == t]
        op = [r for r in opp_rows if r['task'] == t]
        med = {cl: float(np.median([r['score'] for r in op if CLASS[int(r['dir_deg'])] == cl])) for cl in CLASSES[t]}
        norm_case = [r['score']/med[CLASS[int(r['dir_deg'])]] for r in lr_]      # per case: / opponent median of its class
        by = {}
        for cl in CLASSES[t]:
            l_ = [r for r in lr_ if CLASS[int(r['dir_deg'])] == cl]
            o_ = [r for r in op if CLASS[int(r['dir_deg'])] == cl]
            by[cl] = dict(ratio=float(np.mean([r['score'] for r in l_])/med[cl]), opponent_median=med[cl],
                          wins=sum(x['score'] < y['score'] for x, y in zip(l_, o_)), n=len(l_), crashes=sum(not r['flown'] for r in l_),
                          opponent_crashes=sum(not r['flown'] for r in o_))
        rep[t] = dict(ratio=float(np.mean(norm_case)), wins=sum(x['score'] < y['score'] for x, y in zip(lr_, op)),
                      crashes=sum(not r['flown'] for r in lr_), opponent_crashes=sum(not r['flown'] for r in op), by_class=by)
    rep['manoeuvre_mean'] = (rep['nod']['ratio']+rep['dash']['ratio'])/2
    (out/f'evaluate_s{a.seed}.json').write_text(json.dumps(dict(selected=sel, report=rep, rows=rows), indent=1))
    print(json.dumps(rep, indent=1), flush=True)


def selftest():
    """Shapes, numpy/torch agreement on random input, flatten round trip, gate/refractory without simulation."""
    import torch
    m = SNN8(seed=3)
    spec = m.export(gate='command', base_law=dict(k_theta=8., k_q=.4, threshold=.5, width=45.), upset_law=None)
    x = np.abs(np.random.default_rng(0).standard_normal((60, N_IN)))*.8
    with torch.no_grad():
        _, spk, ref, _ = m.forward(torch.tensor(x[:, None, :]))
    n = Net8(spec)
    outs = [n.tick(xi) for xi in x]
    assert np.allclose(np.array([o[1] for o in outs]), ref[:, 0].numpy(), atol=1e-9), 'ref mismatch'
    assert np.array_equal(np.array([o[0] for o in outs]), spk[:, 0].numpy().astype(bool)), 'spike mismatch'
    th = flatten(spec)
    assert th.size == 8*N_IN*K+8+2*8*D+2+N_G*8*D+N_G
    assert np.allclose(flatten(unflatten(spec, th)), th)
    print(f'selftest ok: {th.size} parameters, numpy == torch over {len(x)} ticks, spikes {int(spk.sum())}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['selftest', 'baseline', 'collect', 'train', 'finetune', 'reselect', 'evaluate'])
    p.add_argument('--output', default='runs/spike_steer/v6')
    p.add_argument('--pilot', default='runs/spike_steer/bench8_pilot/pilot_summary.json')
    p.add_argument('--upset-law', default='artifacts/spike_steer/plan_v4/r5/train/imitate_spec_s1.json', help='spec whose upset_law (pulse_tuned) is reused')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--workers', type=int, default=30)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--generations', type=int, default=60)
    p.add_argument('--hours', type=float, default=30.)
    a = p.parse_args()
    E.load_signs()
    B.DASH_M['value'] = .15
    if a.stage == 'selftest':
        return selftest()
    {'baseline': stage_baseline, 'collect': stage_collect, 'train': stage_train, 'finetune': stage_finetune,
     'reselect': stage_reselect, 'evaluate': stage_evaluate}[a.stage](a)


if __name__ == '__main__':
    main()
