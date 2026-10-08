"""Development-only gain and control-direction screening with six IMU inputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path
import time
import zipfile
from .activation import signed_map, gyro_network, init_worker, rollout, cases
from .robust import summarize
from .train import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trim',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16)
    args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()): p.error('output must be empty')
    args.output.mkdir(parents=True,exist_ok=True)
    trim=json.loads(args.trim.read_text()); mapping=signed_map(trim)
    specs=list(itertools.product((.05,.2,.8,2.),(.002,.01,.04,.12),(-.15,0.,.15),((-1.,1.),(1.,-1.),(-1.,-1.),(1.,1.))))
    payloads=[gyro_network(mapping,kp,kd,bias_alpha=bias,signs=signs) for kp,kd,bias,signs in specs]
    development=cases([894010000],2.)+cases([894020000],2.,True)
    validation=cases(range(893810000,893810006))+cases(range(893820000,893820006),stress=True)
    started=time.monotonic(); steps=0
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,initargs=({'trim':trim,'sensor_inputs':6},)) as pool:
        ranked=[]
        for spec,payload,rows in zip(specs,payloads,pool.map(rollout,[(payload,development,False) for payload in payloads])):
            summary=summarize(rows); steps+=summary['control_steps']
            ranked.append(dict(spec=spec,payload=payload,development=summary))
        ranked.sort(key=lambda row:row['development']['robust_score'],reverse=True)
        print(json.dumps(dict(stage='screened',best=ranked[0]['development'],spec=ranked[0]['spec'])),flush=True)
        for row in ranked[:16]:
            episodes=[r for batch in pool.map(rollout,[(row['payload'],[case],False) for case in validation]) for r in batch]
            row['validation']=summarize(episodes); steps+=row['validation']['control_steps']
    finalists=sorted(ranked[:16],key=lambda row:(row['validation']['successes'],row['validation']['robust_score']),reverse=True)
    write_json(args.output/'candidates.json',ranked)
    for i,row in enumerate(finalists): write_json(args.output/f'seed_{i}.json',row['payload'])
    result=dict(candidates=len(ranked),development_cases=development,validation_cases=validation,
                finalists=[{k:v for k,v in row.items() if k!='payload'} for row in finalists],
                control_steps=steps,wall_seconds=time.monotonic()-started)
    write_json(args.output/'results.json',result)
    files=list(Path(__file__).parent.glob('*.py'))
    write_json(args.output/'source_manifest.json',{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
    with zipfile.ZipFile(args.output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in files: archive.write(path,str(path))
    print(json.dumps(dict(stage='completed',best=finalists[0]['validation'],spec=finalists[0]['spec'],control_steps=steps,wall_seconds=result['wall_seconds'])),flush=True)


if __name__=='__main__': main()
