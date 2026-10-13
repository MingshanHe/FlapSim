"""Hand-built SNN that implements the classical pulse law with the pilot's 12 inputs. No learning."""
import json, sys, numpy as np
from experiments.algorithms.spike_steer import pilot as P, pilot_v3 as V

LAW = json.load(open('artifacts/spike_steer/v2/law_tuned.json'))['best']  # k_theta 8, k_q .4, thr .5, width 45 deg

def spec(G, tau_m, tau_s, twitch, P_gate=1.5):
    # u = -(8 theta + .4 q); inputs: theta/.1 rectified (+,-) ids -9,-10; q rectified (+,-) ids -11,-12
    # up neuron 0: fires when phase bin 3 (112.5-157.5 deg, id -4) and u > .5
    # down neuron 1: fires when phase bin 7 (292.5-337.5 deg, id -8) and u < -.5
    kt, kq, thr = LAW['k_theta']*P.PITCH_SCALE, LAW['k_q']*P.RATE_SCALE, LAW['threshold']
    edges = []
    for out, sign, cell in ((0, 1., -4), (1, -1., -8)):
        for src, w in ((-9, -kt), (-10, kt), (-11, -kq), (-12, kq)):
            edges.append(dict(source=src, target=out, weight=G*sign*w, delay=0., enabled=True))
        edges.append(dict(source=cell, target=out, weight=G*P_gate, delay=0., enabled=True))
    nodes = [dict(id=0, bias=-G*(thr+P_gate)), dict(id=1, bias=-G*(thr+P_gate))]
    return dict(kind='snn', inputs=P.INPUTS, outputs=[0, 1], tau_m=tau_m, tau_s=tau_s, spike_scale=P.SPEC['spike_scale'],
                nodes=nodes, edges=edges, calib=dict(twitch=twitch))

VARIANTS = {
    'law_tuned': None,
    'H_G10_tau5_tw3': spec(10., .005, .0025, 3),
    'H_G30_tau5_tw3': spec(30., .005, .0025, 3),
    'H_G100_tau5_tw3': spec(100., .005, .0025, 3),
    'H_G30_tau5_tw2': spec(30., .005, .0025, 2),
    'H_G10_fast_tw3': spec(10., .0005, .0005, 3),
    'H_G30_fast_tw3': spec(30., .0005, .0005, 3),
    'H_G30_fast_tw2': spec(30., .0005, .0005, 2),
}

if __name__ == '__main__':
    cases = P.cases('test', 16, 8.)
    jobs = [(name, c, s, LAW if s is None else None, False) for name, s in VARIANTS.items() for c in cases]
    with P.pool(18) as ex:
        rows = V.run_all(ex, jobs)
    json.dump(rows, open(sys.argv[1], 'w'))
    for name in VARIANTS:
        items = [r for r in rows if r['controller'] == name]
        st = V.stats(items); di = V.directions(items)
        print(f"{name:18s} flown {st['flown']:2d} J {st['J']:.2f} pitch {st['pitch_rms_deg']:.2f} h {st['height_rmse_cm']:.1f} duty {st['pulse_duty']:.4f} "
              f"up {di['need_up']['onsets']} {di['need_up']['in_window']:.2f} dn {di['need_down']['onsets']} {di['need_down']['in_window']:.2f}")
