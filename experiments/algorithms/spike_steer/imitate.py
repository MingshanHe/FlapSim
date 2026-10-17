"""Gradient-trained spiking baseline for plan B': imitate the classical pulse law from RAW IMU inputs.

Question answered: does a small delay-LIF network that sees only the 18 raw-IMU features of pilot_v6 (no PLL,
no phase cells) contain a solution that reproduces the engineered law's phase-locked pulses? Supervised
surrogate-gradient training (BPTT) is the strongest search tool we have, so a failure here means the target
of plan B' does not exist in this input space, and a success gives NEAT a concrete target circuit.

Pipeline:
  collect   run law_tuned (PID + PLL law) on stratified stress cases; log per tick the 18 features, the law's
            pulse decision, its direction, and the true crank phase (analysis only)
  train     torch model with EXACTLY the SpikingGraph dynamics (current-based LIF, 1 ms substeps, two per 2 ms
            tick, threshold 1, subtractive reset, integer delays, spike_scale 5); inputs -> hidden -> 2 outputs,
            a delay bank per connection (0..10 ticks for inputs, 0..20 ms for hidden->output); shared learnable
            tau_m, tau_s. Loss: BCE on "output fires this tick" vs the law's pulse, positives up-weighted.
  export    write a pilot_v6-compatible spec (multiple edges per node pair are allowed by the Graph arrays)
  evaluate  closed loop on the old/fresh 16-case tests through pilot_v6.task, next to law_tuned
  (mechanism probes: python -m experiments.algorithms.spike_steer.pilot_v6 analyze --spec <exported spec>)

    python -m experiments.algorithms.spike_steer.imitate selftest
    python -m experiments.algorithms.spike_steer.imitate collect --output runs/spike_steer/imitate --workers 2
    python -m experiments.algorithms.spike_steer.imitate train --output runs/spike_steer/imitate --hidden 8 --epochs 30
    python -m experiments.algorithms.spike_steer.imitate evaluate --output runs/spike_steer/imitate --workers 18
"""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from . import pilot as P
from . import pilot_v3 as V
from . import pilot_v6 as Y

NEURAL_DT = .001
TICK_SUBSTEPS = 2
IN_DELAYS_TICKS = tuple(range(0, 11))     # 0..10 ticks = 0..20 ms
HID_DELAYS_MS = tuple(range(0, 21, 2))    # 0..20 ms hidden->output
SPIKE_SCALE = float(P.SPEC['spike_scale'])
CALIB = dict(g_theta=1., g_q=1., d_out=0., twitch=3)
BASES = dict(train=2_080_100_000, val=2_080_200_000)


# ----------------------------------------------------------------------------------------- data
class TracedLaw(Y.SteerV6):
    """The classical law (PLL + thresholds) as the teacher; logs features and decisions per tick."""

    def reset(self):
        super().reset()
        self.log = []

    def step(self, env, now, t):
        pulse, onset = super().step(env, now, t)
        if t >= 2.:
            x = self.features(env.last_reading, 0.)
            u = -(self.law['k_theta']*env.last_reading[1]+self.law['k_q']*self.estimator.slow)
            crank = int(env.sim.model.joint('carrier_2_joint').qposadr[0])
            self.log.append(np.r_[x, float(pulse), float(np.sign(u)), math.degrees(float(env.sim.data.qpos[crank])) % 360.])
        return pulse, onset


def collect_task(args):
    c, law = args
    steer = TracedLaw(None, law, False)
    steer.calib = dict(CALIB)
    res = P.run(P._ENV, c, steer)
    return dict(seed=c['seed'], J=res['objective'], log=np.array(steer.log, dtype=np.float32))


def collect(a):
    law = json.loads(V.V2_LAW.read_text())['best']
    sets = dict(train=V.stratified(BASES['train'], a.cases, 6.), val=V.stratified(BASES['val'], max(4, a.cases//4), 6.))
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    with P.pool(a.workers) as ex:
        for name, cases in sets.items():
            rows = list(ex.map(collect_task, [(c, law) for c in cases], chunksize=1))
            X = np.stack([r['log'] for r in rows])  # episodes x ticks x (18 + 3)
            np.savez_compressed(out/f'{name}.npz', X=X, seeds=np.array([r['seed'] for r in rows]))
            pulses = X[:, :, Y.INPUTS].mean()
            print(f'{name}: {X.shape[0]} episodes x {X.shape[1]} ticks, pulse rate {pulses:.4f}, '
                  f'law J {np.mean([r["J"] for r in rows]):.2f}', flush=True)


# ----------------------------------------------------------------------------------------- model
def spike_fn(v, beta=.25):
    """Heaviside forward, fast-sigmoid surrogate backward."""
    import torch
    s = (v >= 1.).float()
    sg = torch.sigmoid((v-1.)/beta)
    return s.detach()+sg-sg.detach()


class DelaySNN:
    """Torch twin of pilot.SpikingGraph: inputs -> H hidden LIF (delay bank per input) -> 2 output LIF (delay bank
    per hidden unit). Shared tau_m, tau_s (log-parametrised). Outputs spike; a tick 'fires' if either substep spikes."""

    def __init__(self, n_in=Y.INPUTS, hidden=8, seed=0):
        import torch
        g = torch.Generator().manual_seed(seed)
        self.n_in, self.H, self.K, self.D = n_in, hidden, len(IN_DELAYS_TICKS), len(HID_DELAYS_MS)
        self.W_in = (torch.randn(hidden, n_in, self.K, generator=g)*.3/math.sqrt(self.K)).requires_grad_()
        self.b_h = (torch.randn(hidden, generator=g)*.3-.5).requires_grad_()
        self.W_out = (torch.randn(2, hidden, self.D, generator=g)*.3/math.sqrt(self.D)).requires_grad_()
        self.b_o = (torch.full((2,), -2.)).requires_grad_()
        self.log_tau_m = torch.tensor(math.log(1e-3)).requires_grad_()
        self.log_tau_s = torch.tensor(math.log(.5e-3)).requires_grad_()

    def parameters(self):
        return [self.W_in, self.b_h, self.W_out, self.b_o, self.log_tau_m, self.log_tau_s]

    def taus(self):
        import torch
        tm = torch.clamp(self.log_tau_m.exp(), 3e-4, 1e-2)
        ts = torch.clamp(self.log_tau_s.exp(), 3e-4, 1e-2)
        return tm, ts

    def forward(self, x):
        """x: [T ticks, B, n_in] -> fire probability proxy [T, B, 2], hidden spikes [T*2, B, H], out spikes [T, B, 2]."""
        import torch
        T, B, _ = x.shape
        tm, ts = self.taus()
        a_m, a_s = torch.exp(-NEURAL_DT/tm), torch.exp(-NEURAL_DT/ts)
        # delayed inputs: x_del[k][t] = x[t-k] (zero before the window)
        pad = torch.zeros(max(IN_DELAYS_TICKS), B, self.n_in)
        xp = torch.cat([pad, x], 0)
        x_del = torch.stack([xp[max(IN_DELAYS_TICKS)-k:max(IN_DELAYS_TICKS)-k+T] for k in IN_DELAYS_TICKS], -1)  # T,B,n_in,K
        drive_h = torch.einsum('tbik,hik->tbh', x_del, self.W_in)+self.b_h  # constant within a tick
        S = T*TICK_SUBSTEPS
        cur_h, v_h = torch.zeros(B, self.H), torch.zeros(B, self.H)
        cur_o, v_o = torch.zeros(B, 2), torch.zeros(B, 2)
        hist = torch.zeros(S+max(HID_DELAYS_MS)+1, B, self.H)  # hidden spikes by substep, offset by max delay + 1
        off = max(HID_DELAYS_MS)+1
        proxy, out_spk = [], []
        for t in range(T):
            p_t, s_t = torch.zeros(B, 2), torch.zeros(B, 2)
            for sub in range(TICK_SUBSTEPS):
                s = t*TICK_SUBSTEPS+sub
                cur_h = a_s*cur_h+(1-a_s)*drive_h[t]
                v_h = a_m*v_h+(1-a_m)*cur_h
                spk_h = spike_fn(v_h)
                v_h = v_h-spk_h
                hist = hist.clone()
                hist[off+s] = spk_h
                # output drive from hidden spikes emitted 1+d substeps earlier (pilot.SpikingGraph: spike_history[p-1-d])
                rec = torch.stack([hist[off+s-1-d] for d in HID_DELAYS_MS], -1)  # B,H,D
                drive_o = SPIKE_SCALE*torch.einsum('bhd,ohd->bo', rec, self.W_out)+self.b_o
                cur_o = a_s*cur_o+(1-a_s)*drive_o
                v_o = a_m*v_o+(1-a_m)*cur_o
                spk_o = spike_fn(v_o)
                p_t = torch.maximum(p_t, torch.sigmoid((v_o-1.)/.25))
                s_t = torch.maximum(s_t, spk_o.detach())
                v_o = v_o-spk_o
            proxy.append(p_t)
            out_spk.append(s_t)
        return torch.stack(proxy), hist[off:off+S], torch.stack(out_spk)

    def export(self, eps=1e-3):
        """pilot_v6-compatible spec; one edge per (source, target, delay) with |w| > eps."""
        tm, ts = (float(v) for v in self.taus())
        nodes = [dict(id=0, bias=float(self.b_o[0])), dict(id=1, bias=float(self.b_o[1]))]
        nodes += [dict(id=2+h, bias=float(self.b_h[h])) for h in range(self.H)]
        edges = []
        W_in, W_out = self.W_in.detach().numpy(), self.W_out.detach().numpy()
        for h in range(self.H):
            for i in range(self.n_in):
                for k, kt in enumerate(IN_DELAYS_TICKS):
                    if abs(W_in[h, i, k]) > eps:
                        edges.append(dict(source=-(i+1), target=2+h, weight=float(W_in[h, i, k]), delay=float(kt*TICK_SUBSTEPS), enabled=True))
            for o in range(2):
                for d, dm in enumerate(HID_DELAYS_MS):
                    if abs(W_out[o, h, d]) > eps:
                        edges.append(dict(source=2+h, target=o, weight=float(W_out[o, h, d]), delay=float(dm), enabled=True))
        return dict(kind='snn', inputs=self.n_in, outputs=[0, 1], tau_m=tm, tau_s=ts, spike_scale=SPIKE_SCALE,
                    arm=dict(Y.ARMS['main']), tied=False, nodes=nodes, edges=edges, trained_by='imitate.py',
                    calib=dict(CALIB))


def targets(X):
    """From a collected block [E, T, 21]: features [E, T, 18]; per-tick target [E, T, 2] = law pulses this tick in
    direction (up: sign u > 0 -> output 0, down -> output 1)."""
    x = X[:, :, :Y.INPUTS]
    pulse, sign = X[:, :, Y.INPUTS], X[:, :, Y.INPUTS+1]
    y = np.stack([pulse*(sign > 0), pulse*(sign < 0)], -1).astype(np.float32)
    return x, y


def train(a):
    import torch
    out = Path(a.output)
    Xtr, ytr = targets(np.load(out/'train.npz')['X'])
    Xva, yva = targets(np.load(out/'val.npz')['X'])
    model = DelaySNN(hidden=a.hidden, seed=a.seed)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    pos = a.pos_weight if a.pos_weight else max(1., float((1-ytr.mean())/max(ytr.mean(), 1e-4)))
    print(json.dumps(dict(pos_weight=pos, hidden=a.hidden, epochs=a.epochs, tag=a.tag)), flush=True)
    bce = torch.nn.BCELoss(reduction='none')
    E, T, _ = Xtr.shape
    W = a.window
    starts = [(e, s) for e in range(E) for s in range(0, T-W+1, W//2)]
    rng = np.random.default_rng(a.seed)
    history, t0 = [], time.monotonic()
    for epoch in range(a.epochs):
        rng.shuffle(starts)
        losses = []
        for b in range(0, len(starts), a.batch):
            idx = starts[b:b+a.batch]
            xb = torch.tensor(np.stack([Xtr[e, s:s+W] for e, s in idx], 1))   # W,B,18
            yb = torch.tensor(np.stack([ytr[e, s:s+W] for e, s in idx], 1))   # W,B,2
            proxy, _, _ = model.forward(xb)
            w = 1.+(pos-1.)*yb
            loss = (bce(proxy.clamp(1e-5, 1-1e-5), yb)*w).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            losses.append(float(loss))
        with torch.no_grad():
            xv = torch.tensor(Xva.transpose(1, 0, 2))
            _, hid, spk = model.forward(xv)
            yv = torch.tensor(yva.transpose(1, 0, 2))
            tp = float((spk*yv).sum()); fp = float((spk*(1-yv)).sum()); fn = float(((1-spk)*yv).sum())
            prec, rec = tp/max(1., tp+fp), tp/max(1., tp+fn)
            tm, ts = (1e3*float(v) for v in model.taus())
            hid_rate = float(hid.mean())
        row = dict(epoch=epoch, loss=float(np.mean(losses)), val_precision=prec, val_recall=rec, tau_ms=[tm, ts],
                   hidden_spike_rate=hid_rate, wall_s=round(time.monotonic()-t0))
        history.append(row)
        print(json.dumps(row), flush=True)
    spec = model.export()
    (out/f'imitate_spec{a.tag}.json').write_text(json.dumps(spec, indent=1))
    (out/f'train_history{a.tag}.json').write_text(json.dumps(history, indent=1))
    print(f'exported {len(spec["edges"])} edges, {len(spec["nodes"])} nodes, tau {spec["tau_m"]*1e3:.2f}/{spec["tau_s"]*1e3:.2f} ms')


def evaluate(a):
    out = Path(a.output)
    spec = json.loads((out/f'imitate_spec{a.tag}.json').read_text())
    law = json.loads(V.V2_LAW.read_text())['best']
    V.task = Y.task
    sets = dict(old16=P.cases('test', 16, 8.), fresh16=V.consecutive(V.BASES['test3'], 16, 8.))
    report = {}
    with P.pool(a.workers) as ex:
        for label, cases in sets.items():
            rows = V.run_all(ex, [(n, c, s, l, False) for n, s, l in (('imitate', spec, None), ('law_tuned', None, law)) for c in cases])
            by = {n: [r for r in rows if r['controller'] == n] for n in ('imitate', 'law_tuned')}
            table = {n: V.stats(v) for n, v in by.items()}
            d = dict(V.directions(by['imitate']), R_up=Y.concentration(by['imitate'], 1), R_down=Y.concentration(by['imitate'], -1))
            report[label] = dict(table=table, directions=d, paired=V.paired(by['imitate'], by['law_tuned']),
                                 J_ratio=table['imitate']['J']/table['law_tuned']['J'], rows=rows)
            print(label, json.dumps(dict(J_ratio=round(report[label]['J_ratio'], 3), imitate=dict(flown=table['imitate']['flown'], J=round(table['imitate']['J'], 2),
                                                                                                   pitch=round(table['imitate']['pitch_rms_deg'] or 0, 2), height=round(table['imitate']['height_rmse_cm'] or 0, 1)),
                                         law=round(table['law_tuned']['J'], 2), directions=d)), flush=True)
    (out/f'evaluate{a.tag}.json').write_text(json.dumps(report, indent=1))


# ----------------------------------------------------------------------------------------- selftest
def selftest():
    """Torch twin == pilot.SpikingGraph on the same input sequence (export round trip), gradients flow, targets split."""
    import torch
    torch.manual_seed(0)
    model = DelaySNN(hidden=3, seed=1)
    with torch.no_grad():  # make the random net active enough to spike
        model.W_in.mul_(4.)
        model.b_h.fill_(.2)
        model.W_out.mul_(3.)
        model.b_o.fill_(-.5)
    T, B = 60, 1
    x = torch.rand(T, B, Y.INPUTS)*(torch.rand(T, B, Y.INPUTS) < .4)
    proxy, hid, spk = model.forward(x)
    assert proxy.shape == (T, B, 2) and hid.shape == (T*2, B, 3) and float(hid.sum()) > 0, 'hidden never spiked'
    loss = proxy.mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    # round trip: the exported spec, run through pilot.SpikingGraph with the same inputs, fires identically per tick
    spec = model.export(eps=0.)
    net = P.SpikingGraph(spec)
    fired = []
    for t in range(T):
        xi = x[t, 0].numpy()
        f = net.snn_substep(xi) | net.snn_substep(xi)
        fired.append(f.astype(float))
    fired = np.array(fired)
    mism = int(np.abs(fired-spk[:, 0].numpy()).sum())
    assert mism == 0, f'export mismatch: {mism} tick-output pairs differ (of {T*2})'
    hidden_spikes = net.spikes  # includes outputs; just ensure the graph was active
    assert hidden_spikes > 0
    # targets split by direction
    X = np.zeros((1, 5, Y.INPUTS+3), np.float32)
    X[0, :, Y.INPUTS] = [1, 1, 0, 1, 0]
    X[0, :, Y.INPUTS+1] = [1, -1, 1, 1, -1]
    _, y = targets(X)
    assert y[0, :, 0].tolist() == [1, 0, 0, 1, 0] and y[0, :, 1].tolist() == [0, 1, 0, 0, 0]
    print(f'selftest ok (torch twin == SpikingGraph on {T} ticks, {int(spk.sum())} output ticks fired, {int(float(hid.sum()))} hidden spikes)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['selftest', 'collect', 'train', 'evaluate'])
    p.add_argument('--output', default='runs/spike_steer/imitate')
    p.add_argument('--workers', type=int, default=18)
    p.add_argument('--cases', type=int, default=32)
    p.add_argument('--hidden', type=int, default=8)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--window', type=int, default=200)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--pos-weight', type=float, default=None, help='BCE weight of pulse ticks (default: class ratio)')
    p.add_argument('--tag', default='', help='suffix for spec/history/evaluate files')
    a = p.parse_args()
    dict(selftest=lambda a: selftest(), collect=collect, train=train, evaluate=evaluate)[a.stage](a)


if __name__ == '__main__':
    main()
