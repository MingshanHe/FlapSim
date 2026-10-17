import json, math, sys, numpy as np
from experiments.algorithms.spike_steer import pilot as P, pilot_v3 as V
from experiments.algorithms.neuro_gates.timing import PULSE_PHASE

class Traced(V.SteerV3):
    def reset(self):
        super().reset(); self.log = []
    def step(self, env, now, t):
        pulse, onset = super().step(env, now, t)
        r = env.last_reading
        which = int(np.flatnonzero(self.fired)[0]) if (self.net is not None and onset) else -1
        self.log.append((t, float(r[1]), float(self.estimator.slow), float(self.estimator.crank(now)) % (2*math.pi), int(pulse), int(onset), which))
        return pulse, onset

def job(args):
    name, c, spec, law = args
    st = Traced(spec, law, False)
    res = P.run(P._ENV, c, st)
    return dict(name=name, seed=c['seed'], J=res['objective'], log=st.log)

if __name__ == '__main__':
    root = 'artifacts/spike_steer/v3/'
    spec = json.load(open(root+'selected_calibrated.json'))
    spec = spec.get('spec', spec)
    law = json.load(open('artifacts/spike_steer/v2/law_tuned.json'))['best']
    cases = P.cases('test', 16, 8.)[:8]
    with P.pool(16) as ex:
        out = list(ex.map(job, [(n, c, s, l) for n, s, l in (('snn', spec, None), ('law', None, law)) for c in cases], chunksize=1))
    json.dump(out, open(sys.argv[1], 'w'))
    print({o['name']+str(o['seed']): round(o['J'], 2) for o in out})
