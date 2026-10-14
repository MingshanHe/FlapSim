"""H4 control for research plan v4: a rate-coded (non-spiking) descending-command network of the SAME size and
wiring as bench.DescendingNet, trained and fine-tuned by the SAME pipeline (descend.py imitation targets, descend.py
OpenAI-ES, bench.py blind evaluation and ablation variants).

Model: 20 inputs with the same 11-tap delay bank (0..20 ms) -> H tanh units -> 2 sigmoid motor outputs (pulse when
> 0.5; same 11-tap hidden delay bank, 0..20 ms in 2 ms ticks) and 1 graded unit (20 ms leaky integrator of a linear
read-out, REF_MAX tanh). Parameter tensors have identical names and shapes to the SNN spec (W_in, b_h, W_out, b_o,
W_g, b_g), so descend.flatten/unflatten and the bench ablation variants (clamp_fast, nodelay_all, noref, nopulse)
apply unchanged; the spec carries kind='descending_ann' and is dispatched by install() without editing bench.py.

Pre-registered expectation (preregistration.json H4): the ANN matches the SNN on the composite; no SNN performance
claim is made. Evaluated by the same blind test sets.

    python -m experiments.algorithms.spike_steer.descend_ann selftest
    python -m experiments.algorithms.spike_steer.descend_ann train    --output runs/spike_steer/bench --seed 1 --epochs 40
    python -m experiments.algorithms.spike_steer.descend_ann finetune --output runs/spike_steer/bench --seed 1 --generations 30 --hours 4.5
    python -m experiments.algorithms.spike_steer.descend_ann evaluate --output runs/spike_steer/bench --spec ... --tag ann_ft_s1 --variants intact,...
"""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from . import bench as B
from . import descend as Dn

KIND = 'descending_ann'
TICK = .002
HID_DELAYS_TICKS = tuple(d//2 for d in B.HID_DELAYS_MS)


# ----------------------------------------------------------------------------------------- numpy runtime
class DescendingANN:
    """Tick-level rate network; same interface as bench.DescendingNet (tick(x) -> fired[2] bool, ref)."""

    def __init__(self, spec):
        s = spec
        self.W_in, self.b_h = np.asarray(s['W_in'], float), np.asarray(s['b_h'], float)
        self.W_out, self.b_o = np.asarray(s['W_out'], float), np.asarray(s['b_o'], float)
        self.W_g, self.b_g = np.asarray(s['W_g'], float), float(s['b_g'])
        self.H, self.n_in, self.K = self.W_in.shape
        self.a_g = math.exp(-TICK/s.get('tau_g', B.TAU_G))
        self.mask = np.ones(self.n_in)
        if s.get('mask') == 'ahrs':
            for i in [2, 3, 4, 5, 6, 7]:
                self.mask[i] = self.mask[i+len(B.Y.CHANNELS)] = 0.
        self.delays_in = np.asarray(s.get('delays_in', B.IN_DELAYS_TICKS), int)
        self.delays_hid = np.asarray(s.get('delays_hid_ticks', HID_DELAYS_TICKS), int)
        self.L = int(self.delays_hid.max())+1
        self.reset()

    def reset(self):
        self.xbuf = np.zeros((self.n_in, int(self.delays_in.max())+1))
        self.hist = np.zeros((self.L, self.H))
        self.p = 0
        self.g = 0.
        self.spikes = 0

    def tick(self, x):
        self.xbuf = np.roll(self.xbuf, 1, axis=1)
        self.xbuf[:, 0] = np.asarray(x, float)*self.mask
        x_del = self.xbuf[:, self.delays_in]
        h = np.tanh(np.einsum('hik,ik->h', self.W_in, x_del)+self.b_h)
        self.hist[self.p] = h
        rec = self.hist[(self.p-self.delays_hid) % self.L].T                 # H, D (delay 0 = this tick)
        o = 1./(1.+np.exp(-(np.einsum('ohd,hd->o', self.W_out, rec)+self.b_o)))
        fired = o > .5
        self.spikes += int(fired.sum())
        drive_g = float(np.sum(self.W_g*rec))+self.b_g
        self.g = self.a_g*self.g+(1-self.a_g)*drive_g
        self.p = (self.p+1) % self.L
        return fired, B.REF_MAX*math.tanh(self.g)


_SNN = B.DescendingNet


def dispatch(spec):
    return DescendingANN(spec) if spec.get('kind') == KIND else _SNN(spec)


def _worker_init(source, dash_m):
    """Pool initializer: install the dispatch in the worker itself (needed with the spawn start method, e.g. macOS)."""
    install()
    B.worker_init(source, dash_m)


def _pool(workers, dash_m=None):
    from concurrent.futures import ProcessPoolExecutor
    source = Path('runs/neuro_gates/scratch/source')
    if not (source/'robot_upper7_lower8.xml').exists():
        B.prepare_source(source)
    return ProcessPoolExecutor(workers, initializer=_worker_init, initargs=(str(source), dash_m if dash_m is not None else B.DASH_M['value']))


def install():
    """Route bench.Learned to the ANN for kind='descending_ann' specs, in this process and in every pool worker
    (bench.pool is replaced by a twin whose initializer installs the dispatch, so fork and spawn both work)."""
    B.DescendingNet = dispatch
    B.pool = _pool


# ----------------------------------------------------------------------------------------- torch twin
class DescendingRateNet:
    def __init__(self, n_in=B.N_IN, hidden=8, seed=0, spec=None):
        import torch
        dt = torch.float64
        self.n_in, self.H = n_in, hidden
        K, D = len(B.IN_DELAYS_TICKS), len(HID_DELAYS_TICKS)
        if spec is not None:
            t = lambda a: torch.tensor(np.asarray(a, float), dtype=dt).requires_grad_()
            self.W_in, self.b_h, self.W_out, self.b_o = t(spec['W_in']), t(spec['b_h']), t(spec['W_out']), t(spec['b_o'])
            self.W_g, self.b_g = t(spec['W_g']), t(spec['b_g'])
            self.H = self.W_in.shape[0]
            return
        g = torch.Generator().manual_seed(seed)
        r = lambda *shape: torch.randn(*shape, generator=g, dtype=dt)
        self.W_in = (r(hidden, n_in, K)*.3/math.sqrt(K)).requires_grad_()
        self.b_h = torch.zeros(hidden, dtype=dt).requires_grad_()
        self.W_out = (r(2, hidden, D)*.3/math.sqrt(D)).requires_grad_()
        self.b_o = torch.full((2,), -2., dtype=dt).requires_grad_()
        self.W_g = (r(hidden, D)*.3/math.sqrt(D)).requires_grad_()
        self.b_g = torch.zeros((), dtype=dt).requires_grad_()

    def parameters(self):
        return [self.W_in, self.b_h, self.W_out, self.b_o, self.W_g, self.b_g]

    def forward(self, x):
        """x [T,B,n_in] -> proxy [T,B,2] (sigmoid), pulses [T,B,2] (proxy > .5), ref [T,B], hidden [T,B,H]."""
        import torch
        T, Bn, _ = x.shape
        kmax, dmax = max(B.IN_DELAYS_TICKS), max(HID_DELAYS_TICKS)
        xp = torch.cat([torch.zeros(kmax, Bn, self.n_in, dtype=x.dtype), x], 0)
        x_del = torch.stack([xp[kmax-k:kmax-k+T] for k in B.IN_DELAYS_TICKS], -1)          # T,B,n_in,K
        h = torch.tanh(torch.einsum('tbik,hik->tbh', x_del, self.W_in)+self.b_h)             # T,B,H
        hp = torch.cat([torch.zeros(dmax, Bn, self.H, dtype=x.dtype), h], 0)
        h_del = torch.stack([hp[dmax-d:dmax-d+T] for d in HID_DELAYS_TICKS], -1)              # T,B,H,D
        proxy = torch.sigmoid(torch.einsum('tbhd,ohd->tbo', h_del, self.W_out)+self.b_o)
        drive_g = torch.einsum('tbhd,hd->tb', h_del, self.W_g)+self.b_g
        a_g = math.exp(-TICK/B.TAU_G)
        g, gs = torch.zeros(Bn, dtype=x.dtype), []
        for t in range(T):
            g = a_g*g+(1-a_g)*drive_g[t]
            gs.append(g)
        ref = B.REF_MAX*torch.tanh(torch.stack(gs))
        return proxy, (proxy > .5).to(x.dtype), ref, h

    def export(self, **extra):
        f = lambda t: t.detach().numpy().tolist()
        return dict(kind=KIND, inputs=self.n_in, hidden=self.H, W_in=f(self.W_in), b_h=f(self.b_h), W_out=f(self.W_out),
                    b_o=f(self.b_o), W_g=f(self.W_g), b_g=float(self.b_g), tau_g=B.TAU_G, **extra)


# ----------------------------------------------------------------------------------------- imitation (mirror of descend.train)
def train(a):
    import torch
    torch.set_num_threads(1)
    out = Path(a.output)
    tr, va = np.load(out/'collect_train.npz'), np.load(out/'collect_val.npz')
    Xtr, ytr, rtr = Dn.targets(tr['X'])
    Xva, yva, rva = Dn.targets(va['X'])
    Ltr, Lva = tr['lengths'], va['lengths']
    model = DescendingRateNet(hidden=a.hidden, seed=a.seed)
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
    n_par = sum(p.numel() for p in model.parameters())
    print(json.dumps(dict(kind=KIND, seed=a.seed, hidden=a.hidden, params=n_par, epochs=a.epochs, pos_weight=a.pos_weight, lam_ref=a.lam_ref,
                          windows=len(starts), val_windows=len(vstarts))), flush=True)
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
            act = rv.abs() > .05
            ref_rmse_active = float(torch.sqrt(((ref/B.REF_MAX-rv)**2)[act].mean())) if bool(act.any()) else None
        row = dict(epoch=epoch, loss=float(np.mean(losses)), val_loss=vloss, val_precision=tp/max(1., tp+fp), val_recall=tp/max(1., tp+fn),
                   ref_rmse=ref_rmse, ref_rmse_active=ref_rmse_active, hidden_abs_mean=float(hid.abs().mean()), wall_s=round(time.monotonic()-t0))
        history.append(row)
        print(json.dumps(row), flush=True)
        if best is None or vloss < best:
            best = vloss
            (out/f'imitate_spec{tag}.json').write_text(json.dumps(model.export(trained_by='descend_ann.py train', seed=a.seed, epoch=epoch), indent=1))
        (out/f'imitate_history{tag}.json').write_text(json.dumps(history, indent=1))
    (out/f'imitate_spec{tag}_last.json').write_text(json.dumps(model.export(trained_by='descend_ann.py train', seed=a.seed, epoch=a.epochs-1), indent=1))
    print(f'best val loss {best:.4f}; wrote imitate_spec{tag}.json', flush=True)


# ----------------------------------------------------------------------------------------- selftest
def selftest():
    import torch
    install()
    model = DescendingRateNet(hidden=3, seed=1)
    with torch.no_grad():
        model.W_in.mul_(6.)
        model.W_out.mul_(6.)
        model.b_o.fill_(0.)
        model.W_g.mul_(3.)
    T = 80
    rng = np.random.default_rng(0)
    xs = rng.random((T, 1, B.N_IN))*(rng.random((T, 1, B.N_IN)) < .4)
    proxy, spk, ref, hid = model.forward(torch.tensor(xs, dtype=torch.float64))
    assert 0 < float(spk.sum()) < 2*T, float(spk.sum())
    (proxy.mean()+ref.mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    spec = model.export()
    net = B.DescendingNet(spec)                       # dispatched
    assert isinstance(net, DescendingANN) and isinstance(B.DescendingNet(Dn.DescendingSNN(hidden=2, seed=0).export()), _SNN)
    mism, dref = 0, 0.
    for t in range(T):
        f, r = net.tick(xs[t, 0])
        mism += int(np.abs(f.astype(float)-spk[t, 0].numpy()).sum())
        dref = max(dref, abs(r-float(ref[t, 0])))
    assert mism == 0 and dref < 1e-9, (mism, dref)
    th = Dn.flatten(spec)
    s2 = Dn.unflatten(spec, th)
    assert s2['kind'] == KIND and np.allclose(np.asarray(s2['W_out']), np.asarray(spec['W_out']))
    n_snn = Dn.flatten(Dn.DescendingSNN(hidden=3, seed=0).export()).size
    assert th.size == n_snn, (th.size, n_snn)
    pol = B.Learned(dict(spec, b_o=[-50., -50.]))
    pol.reset()
    assert isinstance(pol.net, DescendingANN)
    print(f'selftest ok (torch twin == DescendingANN on {T} ticks, {int(spk.sum())} output pulses; {th.size} ES parameters, same as the SNN)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['selftest', 'train', 'finetune', 'evaluate'])
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
    p.add_argument('--tag', default='_ann')
    p.add_argument('--spec', default=None)
    p.add_argument('--variants', default='')
    p.add_argument('--generations', type=int, default=40)
    p.add_argument('--hours', type=float, default=4.)
    p.add_argument('--quick', action='store_true')
    a = p.parse_args()
    B.QUICK = a.quick
    install()
    if a.stage == 'selftest':
        selftest()
    elif a.stage == 'train':
        train(a)
    elif a.stage == 'finetune':
        a.spec = a.spec or str(Path(a.output)/f'imitate_spec_s{a.seed}{a.tag}.json')
        Dn.finetune(a)
    else:
        B.stage_evaluate(a)


if __name__ == '__main__':
    main()
