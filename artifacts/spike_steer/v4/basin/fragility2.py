"""Basin width, fair version: the hand-built comparator scaled to lie inside NEAT's weight/bias bounds."""
import json, random, re, sys, tempfile, itertools, numpy as np, neat
sys.path.insert(0, 'artifacts/spike_steer/v3/substrate_check')
from experiments.algorithms.spike_steer import pilot as P, pilot_v3 as V, pilot_v4 as W
import handlaw_snn as H
ini = open('experiments/algorithms/spike_steer/neat_template_v3.ini').read()
wmax = float(re.search(r'weight_max_value\s*=\s*([-\d.]+)', ini).group(1))
bmax = float(re.search(r'bias_max_value\s*=\s*([-\d.]+)', ini).group(1))
# In-bounds comparator: drive = G (u - thr) + G P (gate cell) + bias with bias = -G P + 1 - G thr, so that inside the gate
# the drive crosses the LIF threshold (1) exactly at u = thr, and outside it stays below 0 for u < thr + (G P - 1)/G.
G, W_GATE = 2.4, 2.8          # input weights 1.92 / 0.96 <= 5, gate weight 2.8 <= 5, bias 1 - 2.8 - 1.2 = -3.0 >= -3; fires off-gate only for u >= 1.67
P_gate = W_GATE/G
print(f'bounds weight {wmax} bias {bmax} -> G = {G:.2f}, P_gate = {P_gate}', flush=True)

def inbounds_spec(G, P_gate, tau_m, tau_s, twitch):
    kt, kq, thr = H.LAW['k_theta']*P.PITCH_SCALE, H.LAW['k_q']*P.RATE_SCALE, H.LAW['threshold']
    edges = []
    for out, sign, cell in ((0, 1., -4), (1, -1., -8)):
        for src, w in ((-9, -kt), (-10, kt), (-11, -kq), (-12, kq)):
            edges.append(dict(source=src, target=out, weight=G*sign*w, delay=0., enabled=True))
        edges.append(dict(source=cell, target=out, weight=G*P_gate, delay=0., enabled=True))
    b = 1.-G*P_gate-G*thr  # inside the gate the drive reaches the LIF threshold (1) at u = thr
    nodes = [dict(id=0, bias=b), dict(id=1, bias=b)]
    assert all(abs(e['weight']) <= wmax for e in edges) and all(abs(n['bias']) <= bmax + 1e-9 for n in nodes), (edges, nodes)
    return dict(kind='snn', inputs=P.INPUTS, outputs=[0, 1], tau_m=tau_m, tau_s=tau_s, spike_scale=P.SPEC['spike_scale'],
                nodes=nodes, edges=edges, calib=dict(twitch=twitch))

def mutants(cfg, spec, k, n, seed):
    cfg.genome_config.node_indexer = itertools.count(max(nd['id'] for nd in spec['nodes'])+1)
    random.seed(seed); out = []
    for i in range(n):
        g = W.genome_from_spec(cfg, i, spec, direct_need=False)
        for _ in range(k):
            g.mutate(cfg.genome_config)
        out.append(dict(W.export(g, False), calib=spec.get('calib', {})))
    return out

if __name__ == '__main__':
    hand = inbounds_spec(G, P_gate, .0005, .00025, 2)
    cases = V.stratified(V.BASES['select'], 8, 6.)
    jobs = [('hand|k0|0', c, hand, None, False) for c in cases]
    with tempfile.TemporaryDirectory() as tmp:
        W.OUTPUT_KEYS = 2
        cfg = W.make_config(tmp, 8, 2); neat.Population(cfg)
        for k in (1, 2, 4):
            for i, m in enumerate(mutants(cfg, hand, k, 32 if k == 2 else 16, 100*k)):
                jobs += [(f'hand|k{k}|{i}', c, m, None, False) for c in cases]
    with P.pool(18) as ex:
        rows = V.run_all(ex, jobs)
    json.dump(dict(G=G, rows=rows), open(sys.argv[1], 'w'))
    by = {}
    for r in rows:
        by.setdefault(r['controller'], []).append(r['objective'])
    ref = np.mean(by['hand|k0|0']); print(f'hand (G={G:.1f}, tau .5 ms): unmutated J {ref:.2f}')
    for k in (1, 2, 4):
        Js = np.array([np.mean(v) for c, v in by.items() if c.startswith(f'hand|k{k}|')])
        print(f'   k={k}: n {len(Js)} median J {np.median(Js):.2f}  p25 {np.percentile(Js,25):.2f} p75 {np.percentile(Js,75):.2f}  '
              f'within +0.5 of ref {np.mean(Js <= ref+.5):.2f}  better {np.mean(Js < ref):.2f}  crashed(J>20) {np.mean(Js > 20):.2f}')
