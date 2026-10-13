"""Learned descending-command layer (research plan v4, stages S2 and S3) for the unified benchmark (bench.py).

Model (torch twin of bench.DescendingNet, selftest-equal tick by tick): 20 inputs (18 rectified raw-IMU features +
nod/dash command) -> H hidden LIF with a delay bank per input (0..20 ms) -> 2 spiking motor outputs (delay bank per
hidden unit, 0..20 ms) and 1 graded unit (20 ms leaky integrator) whose tanh is the pitch reference. Current-based
LIF, 1 ms substeps, two per 2 ms tick, threshold 1, subtractive reset; shared learnable tau_m, tau_s.

S2 imitation (surrogate gradient, BPTT over 200-tick windows): targets are the classical opponents of bench's
baseline on fresh training cases of every task: pulse this tick (by direction) -> weighted BCE on the fire proxy;
pitch reference -> MSE. S3 fine-tune: antithetic policy search (OpenAI-ES, rank-shaped, Adam) on the composite score
of fresh training cases (common random numbers per generation), selection on a fixed validation set.

    python -m experiments.algorithms.spike_steer.descend selftest
    python -m experiments.algorithms.spike_steer.descend collect  --output runs/spike_steer/bench --workers 30
    python -m experiments.algorithms.spike_steer.descend train    --output runs/spike_steer/bench --seed 1 --hidden 8 --epochs 40
    python -m experiments.algorithms.spike_steer.descend finetune --output runs/spike_steer/bench --seed 1 --generations 40 --hours 4
"""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from . import bench as B
from . import pilot_v3 as V

K, D = len(B.IN_DELAYS_TICKS), len(B.HID_DELAYS_MS)
N_COLLECT = dict(hover=24, nod=16, dash=16, upset=16)
N_VAL = dict(hover=4, nod=4, dash=4, upset=4)
COLLECT_OFFSET = dict(train=0, val=50_000)
ES = dict(pop=32, sigma=.03, lr=.02, cases_per_task=2, validate_every=5)


# ----------------------------------------------------------------------------------------- torch model
def spike_fn(v, beta=.25):
    import torch
    s = (v >= 1.).to(v.dtype)
    sg = torch.sigmoid((v-1.)/beta)
    return s.detach()+sg-sg.detach()


class DescendingSNN:
    def __init__(self, n_in=B.N_IN, hidden=8, seed=0, spec=None):
        import torch
        dt = torch.float64
        self.n_in, self.H = n_in, hidden
        if spec is not None:
            t = lambda a: torch.tensor(np.asarray(a, float), dtype=dt).requires_grad_()
            self.W_in, self.b_h, self.W_out, self.b_o = t(spec['W_in']), t(spec['b_h']), t(spec['W_out']), t(spec['b_o'])
            self.W_g, self.b_g = t(spec['W_g']), t(spec['b_g'])
            self.log_tau_m, self.log_tau_s = t(math.log(spec['tau_m'])), t(math.log(spec['tau_s']))
            self.H = self.W_in.shape[0]
            return
        g = torch.Generator().manual_seed(seed)
        r = lambda *shape: torch.randn(*shape, generator=g, dtype=dt)
        self.W_in = (r(hidden, n_in, K)*.3/math.sqrt(K)).requires_grad_()
        self.b_h = (r(hidden)*.3-.5).requires_grad_()
        self.W_out = (r(2, hidden, D)*.3/math.sqrt(D)).requires_grad_()
        self.b_o = torch.full((2,), -2., dtype=dt).requires_grad_()
        self.W_g = (r(hidden, D)*.3/math.sqrt(D)).requires_grad_()
        self.b_g = torch.zeros((), dtype=dt).requires_grad_()
        self.log_tau_m = torch.tensor(math.log(1e-3), dtype=dt).requires_grad_()
        self.log_tau_s = torch.tensor(math.log(.5e-3), dtype=dt).requires_grad_()

    def parameters(self):
        return [self.W_in, self.b_h, self.W_out, self.b_o, self.W_g, self.b_g, self.log_tau_m, self.log_tau_s]

    def taus(self):
        import torch
        return torch.clamp(self.log_tau_m.exp(), 3e-4, 1e-2), torch.clamp(self.log_tau_s.exp(), 3e-4, 1e-2)

    def forward(self, x):
        """x [T, B, n_in] -> proxy [T,B,2], out spikes [T,B,2], ref [T,B] (REF_MAX tanh g at tick end), hidden [2T,B,H]."""
        import torch
        T, Bn, _ = x.shape
        tm, ts = self.taus()
        a_m, a_s = torch.exp(-.001/tm), torch.exp(-.001/ts)
        a_g = math.exp(-.001/B.TAU_G)
        kmax = max(B.IN_DELAYS_TICKS)
        xp = torch.cat([torch.zeros(kmax, Bn, self.n_in, dtype=x.dtype), x], 0)
        x_del = torch.stack([xp[kmax-k:kmax-k+T] for k in B.IN_DELAYS_TICKS], -1)        # T,B,n_in,K
        drive_h = torch.einsum('tbik,hik->tbh', x_del, self.W_in)+self.b_h
        S = 2*T
        off = max(B.HID_DELAYS_MS)+1
        cur_h = torch.zeros(Bn, self.H, dtype=x.dtype)
        v_h = torch.zeros(Bn, self.H, dtype=x.dtype)
        cur_o = torch.zeros(Bn, 2, dtype=x.dtype)
        v_o = torch.zeros(Bn, 2, dtype=x.dtype)
        cur_g = torch.zeros(Bn, dtype=x.dtype)
        g = torch.zeros(Bn, dtype=x.dtype)
        hist = [torch.zeros(Bn, self.H, dtype=x.dtype) for _ in range(off)]
        proxy, out_spk, refs = [], [], []
        for t in range(T):
            p_t, s_t = torch.zeros(Bn, 2, dtype=x.dtype), torch.zeros(Bn, 2, dtype=x.dtype)
            for sub in range(2):
                s = t*2+sub
                cur_h = a_s*cur_h+(1-a_s)*drive_h[t]
                v_h = a_m*v_h+(1-a_m)*cur_h
                spk_h = spike_fn(v_h)
                v_h = v_h-spk_h
                hist.append(spk_h)
                rec = torch.stack([hist[off+s-1-d] for d in B.HID_DELAYS_MS], -1)          # B,H,D
                drive_o = B.SPIKE_SCALE*torch.einsum('bhd,ohd->bo', rec, self.W_out)+self.b_o
                cur_o = a_s*cur_o+(1-a_s)*drive_o
                v_o = a_m*v_o+(1-a_m)*cur_o
                spk_o = spike_fn(v_o)
                p_t = torch.maximum(p_t, torch.sigmoid((v_o-1.)/.25))
                s_t = torch.maximum(s_t, spk_o.detach())
                v_o = v_o-spk_o
                drive_g = B.SPIKE_SCALE*torch.einsum('bhd,hd->b', rec, self.W_g)+self.b_g
                cur_g = a_s*cur_g+(1-a_s)*drive_g
                g = a_g*g+(1-a_g)*cur_g
            proxy.append(p_t)
            out_spk.append(s_t)
            refs.append(B.REF_MAX*torch.tanh(g))
        return torch.stack(proxy), torch.stack(out_spk), torch.stack(refs), torch.stack(hist[off:off+S])

    def export(self, **extra):
        tm, ts = (float(v) for v in self.taus())
        f = lambda t: t.detach().numpy().tolist()
        return dict(kind='descending_snn', inputs=self.n_in, hidden=self.H, W_in=f(self.W_in), b_h=f(self.b_h), W_out=f(self.W_out),
                    b_o=f(self.b_o), W_g=f(self.W_g), b_g=float(self.b_g), tau_m=tm, tau_s=ts, tau_g=B.TAU_G, spike_scale=B.SPIKE_SCALE, **extra)


# ----------------------------------------------------------------------------------------- data
class TracedClassical(B.Classical):
    """The opponent as teacher: logs per tick [features(20), pulse, sign(u), ref]."""

    def reset(self):
        super().reset()
        self.log = []

    def act(self, env, now, t, x, cmd, rel):
        pulse, ref = super().act(env, now, t, x, cmd, rel)
        if t >= 2.:
            u = 0.
            if self.steer is not None and self.law is not None:
                u = -(self.law['k_theta']*env.last_reading[1]+self.law['k_q']*self.steer.estimator.slow)
            self.log.append(np.r_[x, float(pulse), float(np.sign(u)), float(ref)])
        return pulse, ref


def collect_task(args):
    name, tname, c, arg = args
    pol = TracedClassical(arg.get('law'), arg.get('schedule'))
    res = B.run_episode(B._ENV, tname, c, pol)
    return dict(seed=c['seed'], task=tname, score=res['score'], flown=res['flown'], log=np.array(pol.log, dtype=np.float32))


def collect(a):
    out = Path(a.output)
    base = B.load_baseline(out)
    levels = B.upset_levels(out)
    with B.pool(a.workers, base['prereg']['dash_m'] or .25) as ex:
        for split, off in COLLECT_OFFSET.items():
            jobs = []
            for tname, opp in base['opponent'].items():
                n = (N_COLLECT if split == 'train' else N_VAL)[tname]
                if B.QUICK:
                    n = 2
                for c in B.cases(tname, 'train', levels, n=n, offset=off):
                    jobs.append((opp, tname, c, base['arms_full'][tname][opp]))
            rows = list(ex.map(collect_task, jobs, chunksize=1))
            lengths = np.array([len(r['log']) for r in rows])
            X = np.zeros((len(rows), int(lengths.max()), B.N_IN+3), np.float32)   # episodes x ticks x 23, zero-padded
            for i, r in enumerate(rows):
                X[i, :len(r['log'])] = r['log']
            np.savez_compressed(out/f'collect_{split}.npz', X=X, lengths=lengths, seeds=np.array([r['seed'] for r in rows]),
                                tasks=np.array([r['task'] for r in rows]), scores=np.array([r['score'] for r in rows]))
            valid = np.concatenate([X[i, :n] for i, n in enumerate(lengths)])
            print(json.dumps(dict(split=split, episodes=X.shape[0], ticks=[int(lengths.min()), int(lengths.max())], pulse_rate=float(valid[:, B.N_IN].mean()),
                                  ref_abs=float(np.abs(valid[:, B.N_IN+2]).mean()),
                                  flown={t: sum(r['flown'] for r in rows if r['task'] == t) for t in base['opponent']},
                                  score={t: float(np.mean([r['score'] for r in rows if r['task'] == t])) for t in base['opponent']})), flush=True)


def targets(X):
    x = X[:, :, :B.N_IN]
    pulse, sign, ref = X[:, :, B.N_IN], X[:, :, B.N_IN+1], X[:, :, B.N_IN+2]
    y = np.stack([pulse*(sign > 0), pulse*(sign < 0)], -1).astype(np.float32)
    return x, y, (ref/B.REF_MAX).astype(np.float32)


# ----------------------------------------------------------------------------------------- S2 imitation
def train(a):
    import torch
    torch.set_num_threads(1)
    out = Path(a.output)
    tr, va = np.load(out/'collect_train.npz'), np.load(out/'collect_val.npz')
    Xtr, ytr, rtr = targets(tr['X'])
    Xva, yva, rva = targets(va['X'])
    Ltr, Lva = tr['lengths'], va['lengths']
    model = DescendingSNN(hidden=a.hidden, seed=a.seed)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    bce = torch.nn.BCELoss(reduction='none')
    E, T, _ = Xtr.shape
    W = a.window
    starts = [(e, s) for e in range(E) for s in range(0, int(Ltr[e])-W+1, W//2)]
    vstarts = [(e, s) for e in range(Xva.shape[0]) for s in range(0, int(Lva[e])-W+1, W)]
    xv = torch.tensor(np.stack([Xva[e, s:s+W] for e, s in vstarts], 1), dtype=torch.float64)
    yv = torch.tensor(np.stack([yva[e, s:s+W] for e, s in vstarts], 1), dtype=torch.float64)
    rv = torch.tensor(np.stack([rva[e, s:s+W] for e, s in vstarts], 1), dtype=torch.float64)
    rng = np.random.default_rng(a.seed)
    tag = f'_s{a.seed}{a.tag}'
    history, t0, best = [], time.monotonic(), None
    print(json.dumps(dict(seed=a.seed, hidden=a.hidden, epochs=a.epochs, pos_weight=a.pos_weight, lam_ref=a.lam_ref, windows=len(starts), val_windows=len(vstarts))), flush=True)
    for epoch in range(a.epochs):
        rng.shuffle(starts)
        losses = []
        for b in range(0, len(starts), a.batch):
            idx = starts[b:b+a.batch]
            xb = torch.tensor(np.stack([Xtr[e, s:s+W] for e, s in idx], 1), dtype=torch.float64)
            yb = torch.tensor(np.stack([ytr[e, s:s+W] for e, s in idx], 1), dtype=torch.float64)
            rb = torch.tensor(np.stack([rtr[e, s:s+W] for e, s in idx], 1), dtype=torch.float64)
            proxy, _, ref, _ = model.forward(xb)
            w = 1.+(a.pos_weight-1.)*yb
            loss = (bce(proxy.clamp(1e-5, 1-1e-5), yb)*w).mean()+a.lam_ref*((ref/B.REF_MAX-rb)**2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            losses.append(float(loss))
        with torch.no_grad():
            proxy, spk, ref, hid = model.forward(xv)
            tp = float((spk*yv).sum()); fp = float((spk*(1-yv)).sum()); fn = float(((1-spk)*yv).sum())
            vloss = float((bce(proxy.clamp(1e-5, 1-1e-5), yv)*(1.+(a.pos_weight-1.)*yv)).mean()+a.lam_ref*((ref/B.REF_MAX-rv)**2).mean())
            ref_rmse = float(torch.sqrt(((ref/B.REF_MAX-rv)**2).mean()))
            ref_rmse_active = float(torch.sqrt(((ref/B.REF_MAX-rv)**2)[rv.abs() > .05].mean())) if bool((rv.abs() > .05).any()) else None
            tm, ts = (1e3*float(v) for v in model.taus())
        row = dict(epoch=epoch, loss=float(np.mean(losses)), val_loss=vloss, val_precision=tp/max(1., tp+fp), val_recall=tp/max(1., tp+fn),
                   ref_rmse=ref_rmse, ref_rmse_active=ref_rmse_active, tau_ms=[tm, ts], hidden_spike_rate=float(hid.mean()), wall_s=round(time.monotonic()-t0))
        history.append(row)
        print(json.dumps(row), flush=True)
        if best is None or vloss < best:
            best = vloss
            (out/f'imitate_spec{tag}.json').write_text(json.dumps(model.export(trained_by='descend.py train', seed=a.seed, epoch=epoch), indent=1))
        (out/f'imitate_history{tag}.json').write_text(json.dumps(history, indent=1))
    (out/f'imitate_spec{tag}_last.json').write_text(json.dumps(model.export(trained_by='descend.py train', seed=a.seed, epoch=a.epochs-1), indent=1))
    print(f'best val loss {best:.4f}; wrote imitate_spec{tag}.json', flush=True)


# ----------------------------------------------------------------------------------------- S3 policy search
def flatten(spec):
    parts = [np.asarray(spec[k], float).ravel() for k in ('W_in', 'b_h', 'W_out', 'b_o', 'W_g')]+[np.array([float(spec['b_g'])])]
    return np.concatenate(parts)


def unflatten(spec, theta):
    s = dict(spec)
    i = 0
    for k in ('W_in', 'b_h', 'W_out', 'b_o', 'W_g'):
        arr = np.asarray(spec[k], float)
        s[k] = theta[i:i+arr.size].reshape(arr.shape).tolist()
        i += arr.size
    s['b_g'] = float(theta[i])
    return s


def composite_of(rows, tasks, normalisers, weights):
    scores = {t: float(np.mean([r['score'] for r in rows if r['task'] == t])) for t in tasks}
    return B.composite(scores, normalisers, weights), scores


def finetune(a):
    out = Path(a.output)
    base = B.load_baseline(out)
    levels = B.upset_levels(out)
    tasks = list(base['opponent'])
    norm, weights = base['normalisers'], base['prereg']['weights']
    spec0 = json.loads(Path(a.spec or out/f'imitate_spec_s{a.seed}.json').read_text())
    run = out/f'finetune_s{a.seed}{a.tag}'
    run.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1000+a.seed)
    theta = flatten(spec0)
    m_adam, v_adam = np.zeros_like(theta), np.zeros_like(theta)
    pop, sigma, lr = (ES['pop'], ES['sigma'], ES['lr']) if not B.QUICK else (4, ES['sigma'], ES['lr'])
    cpt = ES['cases_per_task'] if not B.QUICK else 1
    gen0, best, history = 0, None, []
    if (run/'checkpoint.npz').exists():
        ck = np.load(run/'checkpoint.npz', allow_pickle=True)
        theta, m_adam, v_adam, gen0 = ck['theta'], ck['m'], ck['v'], int(ck['gen'])
        rng = np.random.default_rng([1000+a.seed, gen0])
        best = json.loads((run/'best.json').read_text()) if (run/'best.json').exists() else None
        history = [json.loads(l) for l in (run/'history.jsonl').read_text().splitlines()]
        print(f'resumed at generation {gen0}', flush=True)
    val_cases = {t: B.cases(t, 'train', levels, n=N_VAL[t] if not B.QUICK else 1, offset=COLLECT_OFFSET['val']+10_000) for t in tasks}
    t_start = time.monotonic()
    with B.pool(a.workers, base['prereg']['dash_m'] or .25) as ex:
        def validate(th, gen):
            nonlocal best
            s = unflatten(spec0, th)
            rows = B.run_jobs(ex, [('val', t, c, 'learned', s) for t in tasks for c in val_cases[t]])
            comp, scores = composite_of(rows, tasks, norm, weights)
            flown = sum(r['flown'] for r in rows)
            entry = dict(generation=gen, composite=comp, scores=scores, flown=flown, episodes=len(rows))
            if best is None or (flown, -comp) > (best['flown'], -best['composite']):
                best = entry
                (run/'spec_best.json').write_text(json.dumps(dict(s, trained_by='descend.py finetune', seed=a.seed, generation=gen), indent=1))
                (run/'best.json').write_text(json.dumps(best, indent=1))
            return entry
        if gen0 == 0:
            v = validate(theta, -1)
            print(json.dumps(dict(validation=v)), flush=True)
        for gen in range(gen0, a.generations):
            if (time.monotonic()-t_start)/3600. > a.hours:
                print(f'time budget reached before generation {gen}', flush=True)
                break
            t_gen = time.monotonic()
            eps = rng.standard_normal((pop//2, theta.size))
            eps = np.concatenate([eps, -eps], 0)
            batch = [(t, c) for t in tasks for c in B.cases(t, 'train', levels, n=cpt, offset=100_000+gen*100)]
            cands = [theta+sigma*e for e in eps]+[theta]
            jobs = [(str(i), t, c, 'learned', unflatten(spec0, th)) for i, th in enumerate(cands) for t, c in batch]
            rows = B.run_jobs(ex, jobs)
            comps = []
            for i in range(len(cands)):
                comp, _ = composite_of([r for r in rows if r['controller'] == str(i)], tasks, norm, weights)
                comps.append(comp)
            f = -np.array(comps[:pop])
            ranks = np.empty(pop)
            ranks[np.argsort(-f)] = np.arange(pop)
            util = np.maximum(0., math.log(pop/2+1)-np.log(ranks+1))
            util = util/util.sum()-1./pop
            grad = (util[:, None]*eps).sum(0)/sigma
            m_adam = .9*m_adam+.1*grad
            v_adam = .999*v_adam+.001*grad*grad
            step = lr*(m_adam/(1-.9**(gen+1)))/(np.sqrt(v_adam/(1-.999**(gen+1)))+1e-8)
            theta = theta+step
            row = dict(generation=gen, center=comps[-1], best_candidate=float(min(comps[:pop])), mean=float(np.mean(comps[:pop])),
                       flown_center=sum(r['flown'] for r in rows if r['controller'] == str(pop)), episodes_per_candidate=len(batch),
                       step_norm=float(np.linalg.norm(step)), wall_s=round(time.monotonic()-t_gen))
            if gen % ES['validate_every'] == ES['validate_every']-1:
                row['validation'] = validate(theta, gen)
            history.append(row)
            with (run/'history.jsonl').open('a') as h:
                h.write(json.dumps(row)+'\n')
            np.savez(run/'checkpoint.npz', theta=theta, m=m_adam, v=v_adam, gen=gen+1)
            (run/'spec_last.json').write_text(json.dumps(dict(unflatten(spec0, theta), trained_by='descend.py finetune', seed=a.seed, generation=gen), indent=1))
            print(json.dumps(row), flush=True)
        if history and 'validation' not in history[-1]:
            print(json.dumps(dict(validation=validate(theta, history[-1]['generation']))), flush=True)
    print(json.dumps(dict(best=best)), flush=True)


# ----------------------------------------------------------------------------------------- selftest
def selftest():
    import torch
    model = DescendingSNN(hidden=3, seed=1)
    with torch.no_grad():
        model.W_in.mul_(6.)
        model.b_h.fill_(.2)
        model.W_out.mul_(3.)
        model.b_o.fill_(-.5)
        model.W_g.mul_(3.)
    T = 80
    rng = np.random.default_rng(0)
    xs = rng.random((T, 1, B.N_IN))*(rng.random((T, 1, B.N_IN)) < .4)
    proxy, spk, ref, hid = model.forward(torch.tensor(xs, dtype=torch.float64))
    assert float(hid.sum()) > 0 and float(spk.sum()) > 0, (float(hid.sum()), float(spk.sum()))
    (proxy.mean()+ref.mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    net = B.DescendingNet(model.export())
    mism, dref = 0, 0.
    for t in range(T):
        f, r = net.tick(xs[t, 0])
        mism += int(np.abs(f.astype(float)-spk[t, 0].numpy()).sum())
        dref = max(dref, abs(r-float(ref[t, 0])))
    assert mism == 0 and dref < 1e-9, (mism, dref)
    spec = model.export()
    th = flatten(spec)
    s2 = unflatten(spec, th)
    assert np.allclose(np.asarray(s2['W_in']), np.asarray(spec['W_in'])) and s2['b_g'] == spec['b_g'] and th.size == 3*B.N_IN*K+3+2*3*D+2+3*D+1
    s3 = unflatten(spec, th+1.)
    assert abs(np.asarray(s3['W_out'])[0, 0, 0]-np.asarray(spec['W_out'])[0, 0, 0]-1.) < 1e-12
    X = np.zeros((1, 5, B.N_IN+3), np.float32)
    X[0, :, B.N_IN] = [1, 1, 0, 1, 0]
    X[0, :, B.N_IN+1] = [1, -1, 1, 1, -1]
    X[0, :, B.N_IN+2] = [0., .35, 0., 0., -.7]
    _, y, r = targets(X)
    assert y[0, :, 0].tolist() == [1, 0, 0, 1, 0] and y[0, :, 1].tolist() == [0, 1, 0, 0, 0] and abs(r[0, 1]-.5) < 1e-6 and abs(r[0, 4]+1.) < 1e-6
    print(f'selftest ok (torch twin == DescendingNet on {T} ticks, {int(spk.sum())} output ticks, {int(float(hid.sum()))} hidden spikes)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['selftest', 'collect', 'train', 'finetune'])
    p.add_argument('--output', default='runs/spike_steer/bench')
    p.add_argument('--workers', type=int, default=30)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--hidden', type=int, default=8)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--window', type=int, default=200)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--pos-weight', type=float, default=3.)
    p.add_argument('--lam-ref', type=float, default=5.)
    p.add_argument('--tag', default='')
    p.add_argument('--spec', default=None)
    p.add_argument('--generations', type=int, default=40)
    p.add_argument('--hours', type=float, default=4.)
    p.add_argument('--quick', action='store_true')
    a = p.parse_args()
    B.QUICK = a.quick
    dict(selftest=lambda a: selftest(), collect=collect, train=train, finetune=finetune)[a.stage](a)


if __name__ == '__main__':
    main()
