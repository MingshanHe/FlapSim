"""Independent development search for asymmetric stable-memory initial graphs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
import math
from pathlib import Path
import time
from .residual import action_map
from .robust import MEMORY, init_worker, rollout, summarize
from .train import write_json


def graph(mapping, kp, kd, tau, alpha_offset):
    nodes = [dict(key=key, bias=0., response=1., activation='identity', aggregation='sum',
                  links=[[source, 1-rho], [key, rho]]) for key, (source, rho) in MEMORY.items()]
    for output in range(3):
        weights = {source: 0. for source in [-1, -2, -3]+list(MEMORY)}
        bias = 0.
        if output < 2:
            axis, sign, p, d, t = (1, -1., kp, kd, tau) if output == 0 else (0, 1., .3, .03, .02)
            scale = .35/(2.5*mapping['span'][output])
            weights[-axis-1] = sign*(p+d/t)*scale
            weights[3+(3 if t == .08 else 0)+axis] = -sign*d/t*scale
        if output == 0:
            bias = math.atanh(alpha_offset/mapping['span'][output])/2.5
        nodes.append(dict(key=output, bias=bias, response=1., activation='tanh', aggregation='sum', links=list(weights.items())))
    return dict(format='flapsim-neat-residual-v1', recurrent=True, inputs=[-1,-2,-3], outputs=[0,1,2],
                nodes=nodes, action_map=mapping, input_names=['roll','pitch','yaw'], input_scale_rad=.35,
                output_names=['alpha','beta','motor_speed'], priming_steps=100,
                memory_design='protected EMA, 20 and 80 ms per attitude axis')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trim', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=16)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        p.error('output must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    trim = json.loads(args.trim.read_text())
    mapping = action_map(trim)
    from .robust import cases
    development = cases([882001000], 2.)+cases([882002000], 2., True)
    verification = cases(range(882101000,882101004), 4.)+cases(range(882102000,882102004), 4., True)
    started = time.monotonic()
    specs = list(itertools.product((.1,.3,1.), (.005,.02,.05,.1), (.02,.08), (0.,.06,.12,.18)))
    payloads = [graph(mapping, *spec) for spec in specs]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=({'trim':trim},)) as pool:
        groups = list(pool.map(rollout, [(payload, development, False) for payload in payloads]))
        ranked = [dict(spec=spec, payload=payload, development=summarize(rows)) for spec,payload,rows in zip(specs,payloads,groups)]
        ranked.sort(key=lambda row: row['development']['robust_score'], reverse=True)
        for candidate in ranked[:8]:
            rows = [r for batch in pool.map(rollout, [(candidate['payload'], [case], False) for case in verification]) for r in batch]
            candidate['verification'] = summarize(rows)
    write_json(args.output/'candidates.json', ranked)
    finalists = sorted(ranked[:8], key=lambda row: row['verification']['robust_score'], reverse=True)
    for index, candidate in enumerate(finalists):
        write_json(args.output/f'seed_{index}.json', candidate['payload'])
    report = dict(candidates=len(ranked), development_cases=development, verification_cases=verification,
                  control_steps=sum(r['development']['control_steps'] for r in ranked)+sum(r['verification']['control_steps'] for r in finalists),
                  finalists=[{k:v for k,v in r.items() if k!='payload'} for r in finalists], wall_seconds=time.monotonic()-started)
    write_json(args.output/'results.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
