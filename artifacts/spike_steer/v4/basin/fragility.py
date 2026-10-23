"""Basin width under NEAT's own mutation operators: hand-built comparator law-SNN (tau 0.5 ms) vs the v4 and v3 champions.
Each source -> 32 mutants (k = 1, 2, 4 mutate() rounds, v3 template rates); J on 8 stratified 6 s cases."""
import json, random, sys, tempfile, numpy as np, neat
sys.path.insert(0, 'artifacts/spike_steer/v3/substrate_check')
from experiments.algorithms.spike_steer import pilot as P, pilot_v3 as V, pilot_v4 as W
import handlaw_snn as H

def mutants(cfg, spec, k, n, seed):
    import itertools
    cfg.genome_config.node_indexer = itertools.count(max(nd['id'] for nd in spec['nodes'])+1)  # as warm_population does
    random.seed(seed); out = []
    for i in range(n):
        g = W.genome_from_spec(cfg, i, spec, direct_need=False)
        for _ in range(k):
            g.mutate(cfg.genome_config)
        out.append(dict(W.export(g, False), calib=spec.get('calib', {})))
    return out

if __name__ == '__main__':
    hand = H.spec(30., .0005, .00025, 2)                      # J 6.03 on old16
    v4 = json.load(open('artifacts/spike_steer/v4/selected_calibrated.json'))
    v3 = json.load(open('artifacts/spike_steer/v3/selected_calibrated.json'))
    cases = V.stratified(V.BASES['select'], 8, 6.)
    jobs = []
    with tempfile.TemporaryDirectory() as tmp:
        W.OUTPUT_KEYS = 2
        cfg = W.make_config(tmp, 8, 2); neat.Population(cfg)
        for name, spec in (('hand', hand), ('v4', v4), ('v3', v3)):
            jobs += [(f'{name}|k0|0', c, spec, None, False) for c in cases]
            for k in (1, 2, 4):
                for i, m in enumerate(mutants(cfg, spec, k, 32 if k == 2 else 16, 100*k)):
                    jobs += [(f'{name}|k{k}|{i}', c, m, None, False) for c in cases]
    with P.pool(18) as ex:
        rows = V.run_all(ex, jobs)
    json.dump(rows, open(sys.argv[1], 'w'))
    by = {}
    for r in rows:
        by.setdefault(r['controller'], []).append(r['objective'])
    for name in ('hand', 'v4', 'v3'):
        ref = np.mean(by[f'{name}|k0|0'])
        print(f'{name}: unmutated J {ref:.2f}')
        for k in (1, 2, 4):
            Js = np.array([np.mean(v) for c, v in by.items() if c.startswith(f'{name}|k{k}|')])
            print(f'   k={k}: n {len(Js)} median J {np.median(Js):.2f}  p25 {np.percentile(Js,25):.2f} p75 {np.percentile(Js,75):.2f}  '
                  f'within +0.5 of ref {np.mean(Js <= ref+.5):.2f}  better than ref {np.mean(Js < ref):.2f}  crashed(J>20) {np.mean(Js > 20):.2f}')
