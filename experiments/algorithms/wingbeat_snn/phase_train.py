"""Learned wingbeat-phase estimators from raw IMU, compared with the classical PLL.

Every model reads the 8 raw IMU features at 500 Hz and outputs (cos, sin) of the
crank angle at the reading time. Training uses recorded flights only (no physics
in the loop); models are selected on the validation split; the test split is
used once, offline and then in closed loop (see ``closed_loop``).

  gru        ANN baseline, gradient (truncated BPTT)
  snn_grad   LIF SNN with the exact semantics of networks.Graph, surrogate gradients
  snn_neat   LIF SNN evolved by NEAT with evolvable integer synaptic delays

    python -m experiments.algorithms.wingbeat_snn.phase_train gru --data runs/wingbeat_snn/phase_data --output <dir>
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from .networks import Graph, NEURAL_DT
from .phase_data import load, inputs_targets, phase_metrics

WINDOW, BURN_IN = 300, 100  # ticks (0.6 s, of which the first 0.2 s are not scored)


def dataset(path):
    data = load(path)
    out = []
    for trace in data['episodes']:
        x, y = inputs_targets(trace)
        out.append((x.astype(np.float32), y.astype(np.float32), trace))
    return out


def windows(episodes, count, rng):
    picks = []
    for _ in range(count):
        e = int(rng.integers(len(episodes)))
        start = int(rng.integers(0, len(episodes[e][0])-WINDOW))
        picks.append((e, start))
    return picks


def batch(episodes, picks):
    x = np.stack([episodes[e][0][s:s+WINDOW] for e, s in picks])
    y = np.stack([episodes[e][1][s:s+WINDOW] for e, s in picks])
    return torch.from_numpy(x), torch.from_numpy(y)


# ---------------------------------------------------------------- models with a common step interface
class GRUPhase(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.gru = nn.GRU(8, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out)


class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v)
        return (v >= 0).to(v.dtype)

    @staticmethod
    def backward(ctx, g):
        v, = ctx.saved_tensors
        return g/(1.+5.*v.abs()).square()


class TorchSNN(nn.Module):
    """Dense LIF graph identical to networks.Graph (delays 0, two 1 ms substeps per 2 ms tick)."""
    def __init__(self, hidden=24, tau_m=.01, tau_s=.005, spike_scale=5., seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.hidden, self.spike_scale = hidden, spike_scale
        self.w_in = nn.Parameter(torch.randn(hidden, 8, generator=g)*.5)
        self.w_rec = nn.Parameter(torch.randn(hidden, hidden, generator=g)*.05)
        self.b_h = nn.Parameter(torch.full((hidden,), .8))
        self.w_out = nn.Parameter(torch.randn(2, hidden, generator=g)*.05)
        self.b_out = nn.Parameter(torch.zeros(2))
        self.a_m, self.a_s = math.exp(-NEURAL_DT/tau_m), math.exp(-NEURAL_DT/tau_s)
        self.tau_m, self.tau_s = tau_m, tau_s

    def forward(self, x):
        B, L, _ = x.shape
        z = x.new_zeros
        i_h, v_h, i_o, v_o, s = z(B, self.hidden), z(B, self.hidden), z(B, 2), z(B, 2), z(B, self.hidden)
        out = []
        for k in range(L):
            for _ in range(2):
                signal = self.spike_scale*s
                d_h = self.b_h+x[:, k]@self.w_in.T+signal@self.w_rec.T
                d_o = self.b_out+signal@self.w_out.T
                i_h = self.a_s*i_h+(1-self.a_s)*d_h
                v_h = self.a_m*v_h+(1-self.a_m)*i_h
                i_o = self.a_s*i_o+(1-self.a_s)*d_o
                v_o = self.a_m*v_o+(1-self.a_m)*i_o
                s = Spike.apply(v_h-1.)
                v_h = v_h-s
            out.append(torch.tanh(v_o))
        return torch.stack(out, 1)

    def export(self):
        h = self.hidden
        w_in, w_rec = self.w_in.detach().double().numpy(), self.w_rec.detach().double().numpy()
        w_out, b_h, b_out = self.w_out.detach().double().numpy(), self.b_h.detach().double().numpy(), \
            self.b_out.detach().double().numpy()
        nodes = [dict(id=k, bias=float(b_out[k])) for k in range(2)]+[dict(id=2+j, bias=float(b_h[j])) for j in range(h)]
        edges = [dict(source=-(i+1), target=2+j, weight=float(w_in[j, i]), delay=0., enabled=True)
                 for j in range(h) for i in range(8)]
        edges += [dict(source=2+a, target=2+b, weight=float(w_rec[b, a]), delay=0., enabled=True)
                  for a in range(h) for b in range(h)]
        edges += [dict(source=2+j, target=o, weight=float(w_out[o, j]), delay=0., enabled=True)
                  for j in range(h) for o in range(2)]
        return dict(kind='snn', method='snn_grad', mode='phasor', inputs=8, outputs=[0, 1], tau_m=self.tau_m,
                    tau_s=self.tau_s, spike_scale=self.spike_scale, nodes=nodes, edges=edges)


class GraphStep:
    """NumPy deployment of an exported spec (SNN or ANN)."""
    def __init__(self, spec):
        self.g = Graph(spec)

    def reset(self):
        self.g.reset()

    def step(self, x):
        if self.g.kind == 'snn':
            self.g.snn_substep(x)
            y = self.g.snn_substep(x)
        else:
            y = self.g.ann_step(x)
        return float(y[0]), float(y[1])


class TorchStep:
    def __init__(self, module):
        self.m = module.eval()
        self.h = None

    def reset(self):
        self.h = None

    def step(self, x):
        with torch.no_grad():
            out, self.h = self.m.gru(torch.as_tensor(x, dtype=torch.float32).view(1, 1, -1), self.h)
            y = self.m.head(out[0, 0])
        return float(y[0]), float(y[1])


def run_estimator(stepper, x):
    stepper.reset()
    return np.array([math.atan2(*reversed(stepper.step(row))) for row in x])


def evaluate_offline(make_stepper, episodes):
    rows = [phase_metrics(trace, run_estimator(make_stepper(), x)) for x, _, trace in episodes]
    return aggregate(rows)


def aggregate(rows):
    def mean(key, field):
        vals = [r[key][field] for r in rows if r[key]]
        return float(np.mean(vals)) if vals else None
    locks = [r['lock_time_s'] for r in rows]
    return dict(rms_all=mean('all', 'rms'), rms_calm=mean('calm', 'rms'), rms_gust=mean('gust', 'rms'),
                p95_gust=mean('gust', 'p95'), lock_time_s_mean=float(np.mean([l for l in locks if l is not None]))
                if any(l is not None for l in locks) else None, never_locked=sum(l is None for l in locks),
                episodes=len(rows))


def pll_offline(episodes):
    return aggregate([phase_metrics(trace, trace[:, 11].astype(np.float64)) for _, _, trace in episodes])


# ---------------------------------------------------------------- gradient training (GRU and SNN)
def train_gradient(kind, data, out, steps, seed, threads=8, hidden=None, tau_m=.01, tau_s=.005, tag=''):
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    rng = np.random.default_rng(seed)
    train, valid = dataset(data/'training.npz'), dataset(data/'validation.npz')
    model = GRUPhase(hidden or 32) if kind == 'gru' else TorchSNN(hidden or 24, tau_m, tau_s, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3 if kind == 'gru' else 2e-3)
    def stepper():
        return TorchStep(model) if kind == 'gru' else GraphStep(model.export())
    best, history, start = None, [], time.monotonic()
    for step in range(1, steps+1):
        model.train()
        x, y = batch(train, windows(train, 16, rng))
        loss = ((model(x)[:, BURN_IN:]-y[:, BURN_IN:])**2).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        if step % 100 == 0 or step == steps:
            v = evaluate_offline(stepper, valid)
            history.append(dict(step=step, loss=float(loss), validation=v, wall_s=time.monotonic()-start))
            print(json.dumps(history[-1]), flush=True)
            if best is None or v['rms_all'] < best['validation']['rms_all']:
                best = history[-1]
                if kind == 'gru':
                    torch.save(dict(state=model.state_dict(), hidden=model.gru.hidden_size), out/f'gru{tag}.pt')
                else:
                    (out/f'snn_grad{tag}.json').write_text(json.dumps(model.export()))
    (out/f'{kind}{tag}_history.json').write_text(json.dumps(dict(history=history, best=best, steps=steps,
                                                                 hidden=hidden, tau_m=tau_m, tau_s=tau_s), indent=1))
    return best


# ---------------------------------------------------------------- NEAT (offline, delays evolvable)
_TRAIN = None


def neat_init(path):
    global _TRAIN
    _TRAIN = dataset(path)


def neat_loss(task):
    spec, picks = task
    stepper = GraphStep(spec)
    total = 0.
    for e, s in picks:
        x, y, _ = _TRAIN[e]
        stepper.reset()
        out = np.array([stepper.step(row) for row in x[s:s+WINDOW]])
        total += float(np.mean((out[BURN_IN:]-y[s+BURN_IN:s+WINDOW])**2))
    return total/len(picks)


def train_neat(data, out, generations, population, seed, workers, tag=''):
    import neat
    from . import evolve
    random.seed(seed)
    rng = np.random.default_rng(seed)
    config_dir = out/f'neat{tag}'
    config_dir.mkdir(parents=True, exist_ok=True)
    cfg = evolve.make_config(config_dir, 'snn_neat_delay', population, inputs=8)
    pop = neat.Population(cfg)
    evolve.initialize(pop, cfg, 'snn_neat_delay', seed, readout_std=.3, hidden_count=8)
    train, valid = dataset(data/'training.npz'), dataset(data/'validation.npz')
    best, history, start = None, [], time.monotonic()
    with ProcessPoolExecutor(workers, initializer=neat_init, initargs=(str(data/'training.npz'),)) as pool:
        def score(genomes, config):
            nonlocal best
            picks = windows(train, 8, rng)
            specs = {gid: evolve.export(g, 'snn_neat_delay', spike_scale=5., inputs=8) for gid, g in genomes}
            losses = list(pool.map(neat_loss, [(specs[gid], picks) for gid, _ in genomes]))
            for (gid, g), loss in zip(genomes, losses):
                g.fitness = -loss
            champion_id, champion = max(genomes, key=lambda kv: kv[1].fitness)
            row = dict(generation=len(history), loss_best=-champion.fitness, loss_mean=float(np.mean(losses)),
                       species=len(pop.species.species), complexity=evolve.complexity(specs[champion_id]),
                       wall_s=time.monotonic()-start)
            if len(history) % 10 == 9 or len(history) == 0:
                spec = specs[champion_id]
                v = evaluate_offline(lambda: GraphStep(spec), valid)
                row['validation'] = v
                if best is None or v['rms_all'] < best['validation']['rms_all']:
                    best = row
                    (out/f'snn_neat{tag}.json').write_text(json.dumps(spec))
            history.append(row)
            print(json.dumps(row), flush=True)
        pop.run(score, generations)
    (out/f'snn_neat{tag}_history.json').write_text(json.dumps(dict(history=history, best=best, seed=seed), indent=1))
    return best


# ---------------------------------------------------------------- tuned classical PLL (same recordings)
def pll_replay(trace, window, bandwidth_hz, offset):
    """Replay WingbeatPhase on a recorded pitch-gyro stream; 2nd-order loop with damping 0.7."""
    from ..wingbeat_feasibility.preliminary import WingbeatPhase
    wn = 2*math.pi*bandwidth_hz
    est = WingbeatPhase(offset, window=window, kp=2*.7*wn, ki=wn*wn)
    out = np.empty(len(trace))
    for i, row in enumerate(trace):
        est.update(float(row[0]), float(row[5]))
        out[i] = est.crank(float(row[0]))
    return out


def pll_setting(task):
    """Calibrate the constant offset on training flights, then score on validation flights."""
    window, bandwidth, train_path, valid_path = task
    train, valid = load(train_path)['episodes'], load(valid_path)['episodes']
    errors = []
    for trace in train[::4]:
        est = pll_replay(trace, window, bandwidth, 0.)
        keep = trace[:, 0] >= 1.5
        errors.append(np.angle(np.exp(1j*(est[keep]-trace[keep, 10]))))
    offset = float(np.angle(np.mean(np.exp(1j*np.concatenate(errors)))))
    score = aggregate([phase_metrics(t, pll_replay(t, window, bandwidth, offset)) for t in valid])
    return dict(window=window, bandwidth_hz=bandwidth, offset_rad=offset, validation=score)


def pll_tune(data, out, workers):
    # First sweep (15/25/40 x 0.75/1.5/3/6 Hz) picked the grid edge (15, 3 Hz); extended toward shorter windows.
    grid = [(w, b, str(data/'training.npz'), str(data/'validation.npz'))
            for w in (8, 10, 12, 15, 25) for b in (1.5, 2., 3., 4., 5.)]
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(pll_setting, grid))
    best = min(rows, key=lambda r: r['validation']['rms_all'])
    test = [phase_metrics(t, pll_replay(t, best['window'], best['bandwidth_hz'], best['offset_rad']))
            for t in load(data/'test.npz')['episodes']]
    result = dict(grid=rows, best=best, test=aggregate(test),
                  note='P0 default: window 25, bandwidth about 1.5 Hz, offset calibrated in P0')
    (out/'pll_tuned.json').write_text(json.dumps(result, indent=1))
    return result


# ---------------------------------------------------------------- held-out offline comparison
def offline_test(data, models):
    test = dataset(data/'test.npz')
    result = dict(pll=pll_offline(test))
    for name, make in models.items():
        result[name] = evaluate_offline(make, test)
    return result


def load_gru(path):
    saved = torch.load(path)
    if 'state' not in saved:  # first runs saved the bare state dict of a 32-unit GRU
        saved = dict(state=saved, hidden=32)
    gru = GRUPhase(saved['hidden'])
    gru.load_state_dict(saved['state'])
    return gru


def load_models(model_dir):
    """Every gru*.pt, snn_grad*.json and snn_neat*.json in the directory, keyed by file stem."""
    model_dir = Path(model_dir)
    models = {}
    for path in sorted(model_dir.glob('gru*.pt')):
        gru = load_gru(path)
        models[path.stem] = (lambda m: (lambda: TorchStep(m)))(gru)
    for path in sorted(list(model_dir.glob('snn_grad*.json'))+list(model_dir.glob('snn_neat*.json'))):
        if path.stem.endswith('_history'):
            continue
        spec = json.loads(path.read_text())
        models[path.stem] = (lambda s: (lambda: GraphStep(s)))(spec)
    return models


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['gru', 'snn_grad', 'snn_neat', 'offline_test', 'pll_tune'])
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--generations', type=int, default=150)
    p.add_argument('--population', type=int, default=64)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--hidden', type=int)
    p.add_argument('--tau-m', type=float, default=.01)
    p.add_argument('--tau-s', type=float, default=.005)
    p.add_argument('--tag', default='', help='suffix for the saved model and history files')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.stage in ('gru', 'snn_grad'):
        result = train_gradient(a.stage, a.data, a.output, a.steps, a.seed, a.threads, a.hidden, a.tau_m, a.tau_s,
                                a.tag)
    elif a.stage == 'snn_neat':
        result = train_neat(a.data, a.output, a.generations, a.population, a.seed, a.workers, a.tag)
    elif a.stage == 'pll_tune':
        result = pll_tune(a.data, a.output, a.workers)
    else:
        result = offline_test(a.data, load_models(a.output))
        (a.output/'offline_test.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
