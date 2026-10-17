"""The in-bounds comparator law-SNN (G 2.4, tau 0.5 ms, twitch 2) on the two 16-case test sets, against law_tuned."""
import json, sys, numpy as np
sys.path.insert(0, '/tmp/claude-1000/-home-robot-FlapSim-repo-FlapSim/2c45b87d-97ba-4f36-a3f1-208d5ba469e1/scratchpad')
from experiments.algorithms.spike_steer import pilot as P, pilot_v3 as V
import fragility2 as F2
law = json.load(open('artifacts/spike_steer/v2/law_tuned.json'))['best']
specs = {'comparator_inbounds': F2.inbounds_spec(F2.G, F2.P_gate, .0005, .00025, 2),
         'comparator_inbounds_tau1ms': F2.inbounds_spec(F2.G, F2.P_gate, .001, .0005, 2)}
sets = dict(old16=P.cases('test', 16, 8.), fresh16=V.consecutive(V.BASES['test3'], 16, 8.))
jobs = [(f'{label}|{name}', c, s, None, False) for label, cs in sets.items() for name, s in specs.items() for c in cs]
jobs += [(f'{label}|law_tuned', c, None, law, False) for label, cs in sets.items() for c in cs]
with P.pool(18) as ex:
    rows = V.run_all(ex, jobs)
json.dump(dict(specs=specs, rows=rows), open(sys.argv[1], 'w'))
for label in sets:
    by = {n: [r for r in rows if r['controller'] == f'{label}|{n}'] for n in list(specs)+['law_tuned']}
    for n, v in by.items():
        s = V.stats(v); d = V.directions(v)
        print(f"{label} {n:28s} flown {s['flown']:2d} J {s['J']:.2f} (ratio {s['J']/V.stats(by['law_tuned'])['J']:.3f}) pitch {s['pitch_rms_deg']:.2f} h {s['height_rmse_cm']:.1f} duty {s['pulse_duty']:.4f} up {d['need_up']['onsets']} {d['need_up']['in_window']:.2f} dn {d['need_down']['onsets']} {d['need_down']['in_window']:.2f}")
    for n in specs:
        p = V.paired(by[n], by['law_tuned']); print(f"   paired {n} vs law_tuned: dJ {p['mean_dJ']:+.3f} CI {[round(x,3) for x in p['ci95']]} wins {p['wins']}/16")
