"""Computational cost of the descending layers (paper, 'what does the controller cost to run'): parameter counts, operations
per second of flight and an energy estimate per operation, for the v11 SNN (descend8.Net8, measured on flown episodes)
and the MLP + PPO reference (57-64-64-3 tanh actor, one decision per 10 ms). The shared parts (reflex PID, phase pulse law,
supervisor) are identical for every arm and are not counted.

SNN per 2 ms tick (two 1 ms substeps), as Net8.tick computes it:
  input layer   8 hidden x 25 inputs x 11 delay taps, real-valued inputs -> MAC; zero inputs (rectified halves) can be skipped
  hidden LIF    2 substeps x 8 x (2 leak MAC + threshold)
  readout       binary hidden spikes through 11 delay taps into 2 spiking + 2 graded units -> accumulate only (AC); an
                event-driven readout does 4 x 11 = 44 AC per hidden spike, the dense loop 2 x 4 x 8 x 11 = 704 per tick
Energy per op: Horowitz (ISSCC 2014), 45 nm, 32-bit float: MAC 4.6 pJ, AC 0.9 pJ (the usual SNN-paper convention).

    python -m experiments.algorithms.spike_steer.compute_cost run --per-task 16 --workers 19
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from . import bench as B
from . import bench8 as E
from . import bench10 as T10
from . import descend8 as D8
from . import descend9 as D9
from .blind12 import load_arms
from .blind15 import cases15

ART = Path('artifacts/spike_steer/compute_cost')
E_MAC, E_AC = 4.6e-12, .9e-12
PPO = dict(layers=(57, 64, 64, 3), hz=100.)
_C = {}


def _count_tick(orig):
    def tick(self, x):
        xd = self.xbuf.copy()
        xd = np.roll(xd, 1, axis=1)
        xd[:, 0] = x
        nnz = int(np.count_nonzero(xd[:, self.din]))
        s0 = self.spikes
        fired, ref = orig(self, x)
        _C['ticks'] += 1
        _C['nnz_taps'] += nnz
        _C['hid_spikes'] += self.spikes-s0
        _C['out_spikes'] += int(np.count_nonzero(fired))
        return fired, ref
    return tick


def _init():
    E._init(str(E.SOURCE), .15, dict(E.SIGNS))
    D8.Net8.tick = _count_tick(D8.Net8.tick)


def job(args):
    name, task, c, spec = args
    _C.update(ticks=0, nnz_taps=0, hid_spikes=0, out_spikes=0)
    r = T10.job((name, task, c, 'learned', spec, False))
    return dict(name=name, task=task, seed=c['seed'], flown=r['flown'], **_C)


def stage_run(a):
    arms, _ = load_arms()
    snn = [(n, s) for n, k, s in arms if n.startswith('snn11_')]
    cs = {t: cases15(t, 'nominal', a.per_task) for t in D9.TASKS}
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
        rows = list(ex.map(job, [(n, t, c, s) for n, s in snn for t in D9.TASKS for c in cs[t]], chunksize=1))
    spec = snn[0][1]
    H, n_in, K = np.asarray(spec['W_in']).shape
    D = np.asarray(spec['W_out']).shape[2]
    G = np.asarray(spec['W_g']).shape[0]
    params = int(H*n_in*K+H+2*H*D+2+G*H*D+G)
    tick_hz = 1./.002                         # one tick per 2 ms
    out = dict(snn=dict(params=params, neurons=dict(hidden_lif=H, output_lif=2, graded=G), inputs=n_in, delay_taps=K, readout_taps=D,
                        state_floats=int(n_in*K+(D+1)*H+2*H+4+2*G)), per_arm={})
    for n, _ in snn:
        R = [r for r in rows if r['name'] == n]
        T = sum(r['ticks'] for r in R)
        sec = T/tick_hz
        dense_in = H*n_in*K*tick_hz
        sparse_in = H*sum(r['nnz_taps'] for r in R)/sec
        lif = 2*(2*H+2*2+2*G)*tick_hz                       # leak MACs of hidden, output, graded units
        hid_rate = sum(r['hid_spikes'] for r in R)/sec
        ac_event = hid_rate*(2+G)*D
        ac_dense = 2*(2+G)*H*D*tick_hz
        out['per_arm'][n] = dict(
            seconds_flown=round(sec, 1), episodes=len(R), input_nonzero_fraction=sum(r['nnz_taps'] for r in R)/(T*n_in*K),
            hidden_spikes_per_s=hid_rate, hidden_rate_hz_per_neuron=hid_rate/H, output_spikes_per_s=sum(r['out_spikes'] for r in R)/sec,
            mac_per_s_dense=dense_in+lif, mac_per_s_zero_skip=sparse_in+lif, ac_per_s_event=ac_event, ac_per_s_dense=ac_dense,
            power_uW_dense=1e6*((dense_in+lif)*E_MAC+ac_dense*E_AC), power_uW_event=1e6*((sparse_in+lif)*E_MAC+ac_event*E_AC))
        print(n, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in out['per_arm'][n].items()}), flush=True)
    L = PPO['layers']
    mac = sum(L[i]*L[i+1] for i in range(len(L)-1))
    out['ppo'] = dict(params_actor=int(sum(L[i]*L[i+1]+L[i+1] for i in range(len(L)-1))), mac_per_decision=mac, tanh_per_decision=sum(L[1:-1]),
                      decisions_per_s=PPO['hz'], mac_per_s=mac*PPO['hz'], power_uW=1e6*mac*PPO['hz']*E_MAC,
                      state_floats=10*n_in+L[0]+L[1]+L[2])      # 10 past feature frames (lags up to 10 ticks), observation, activations
    print('ppo', json.dumps(out['ppo']))
    ART.mkdir(parents=True, exist_ok=True)
    (ART/'compute_cost.json').write_text(json.dumps(out, indent=1))
    (ART/'rows.json').write_text(json.dumps(rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['run'])
    p.add_argument('--per-task', type=int, default=16)
    p.add_argument('--workers', type=int, default=19)
    a = p.parse_args()
    E.load_signs()
    B.DASH_M['value'] = .15
    stage_run(a)


if __name__ == '__main__':
    main()
