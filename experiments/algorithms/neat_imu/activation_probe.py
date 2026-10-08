"""Screen signed output activations, response gains, and memory time scales."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import itertools
import json
import math
from pathlib import Path
import time
from .activation import signed_map, init_worker, rollout
from .robust import MEMORY, cases, summarize
from .train import write_json


def candidate(parent, mapping, output_activation, gain_alpha, gain_beta, memory_scale, hidden_activation):
    payload = copy.deepcopy(parent)
    payload['action_map'] = mapping
    for node in payload['nodes']:
        key = node['key']
        if key in MEMORY:
            old_rho = next(w for source,w in node['links'] if source == key)
            tau = min(.15,max(.01,-.01/math.log(old_rho)*memory_scale))
            rho = math.exp(-.01/tau)
            node.update(activation='identity',bias=0.,response=1.,links=[[MEMORY[key][0],1-rho],[key,rho]])
        elif key in (0,1):
            gain = gain_alpha if key == 0 else gain_beta
            # Match the small-signal slope when comparing tanh(2.5z) with clamp(z).
            gain *= 2.5 if output_activation == 'clamped' else 1.
            node['activation'] = output_activation
            node['response'] *= gain
            node['bias'] *= gain
        elif key > 2 and hidden_activation != 'keep':
            node['activation'] = hidden_activation
    payload['memory_design'] = 'stable EMA activation/gain/time-scale development search'
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trim',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()): p.error('output must be empty')
    args.output.mkdir(parents=True,exist_ok=True)
    trim,parent=json.loads(args.trim.read_text()),json.loads(args.parent.read_text())
    mapping=signed_map(trim)
    specs=list(itertools.product(('tanh','clamped'),(.5,1.,1.5),(.5,1.,1.5),(.5,1.,1.5),('keep','clamped','sin')))
    payloads=[candidate(parent,mapping,*spec) for spec in specs]
    development=cases([893010000],2.)+cases([893020000],2.,True)
    validation=cases(range(893810000,893810006),8.)+cases(range(893820000,893820006),8.,True)
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,initargs=({'trim':trim},)) as pool:
        groups=list(pool.map(rollout,[(payload,development,False) for payload in payloads]))
        ranked=[dict(spec=spec,payload=payload,development=summarize(rows)) for spec,payload,rows in zip(specs,payloads,groups)]
        ranked.sort(key=lambda r:r['development']['robust_score'],reverse=True)
        for row in ranked[:12]:
            episodes=[r for batch in pool.map(rollout,[(row['payload'],[case],False) for case in validation]) for r in batch]
            row['validation']=summarize(episodes)
        baseline_rows=[r for batch in pool.map(rollout,[(parent,[case],False) for case in validation]) for r in batch]
    finalists=sorted(ranked[:12],key=lambda r:(r['validation']['successes'],r['validation']['robust_score']),reverse=True)
    write_json(args.output/'candidates.json',ranked)
    for i,row in enumerate(finalists): write_json(args.output/f'seed_{i}.json',row['payload'])
    write_json(args.output/'parent.json',parent)
    result=dict(candidates=len(ranked),development_cases=development,validation_cases=validation,
                baseline=summarize(baseline_rows),finalists=[{k:v for k,v in r.items() if k!='payload'} for r in finalists],
                control_steps=sum(r['development']['control_steps'] for r in ranked)+sum(r['validation']['control_steps'] for r in finalists)+sum(r['steps'] for r in baseline_rows),
                wall_seconds=time.monotonic()-started)
    write_json(args.output/'results.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
