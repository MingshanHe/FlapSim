"""Screen measured-gyro filtering before a possible second NEAT initialization."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import zipfile
from .activation import signed_map, gyro_network, init_worker, rollout, cases
from .robust import summarize
from .train import write_json


def main():
    output=Path('artifacts/neat_imu_activation_20261007/filter_probe')
    output.mkdir(exist_ok=False)
    trim=json.loads(Path('artifacts/neat_imu_trim_20261007/calibration_fine/trim.json').read_text())
    mapping=signed_map(trim)
    specs=list(itertools.product((.05,.2,.8),(.01,.04,.12),(0.,.15),(.015,.04,.08),(-1.,1.)))
    payloads=[]
    for kp,kd,bias,tau,sign in specs:
        payload=gyro_network(mapping,kp,kd,bias_alpha=bias,signs=(sign,-1.))
        for node in payload['nodes']:
            if node['key'] in (0,1):
                node['links']=[[5-source if source in (-4,-5,-6) else source,w] for source,w in node['links']]
        rho=math.exp(-.01/tau)
        payload['nodes']=[dict(key=key,activation='identity',aggregation='sum',bias=0.,response=1.,
                               links=[[5-key,1-rho],[key,rho]]) for key in (9,10,11)]+payload['nodes']
        payload['memory_design']='stable gyro EMA development search'
        payload['gyro_memory']='stable gyro EMA development search'
        payloads.append(payload)
    development=cases([895010000],2.)+cases([895020000],2.,True)
    validation=cases(range(893810000,893810006))+cases(range(893820000,893820006),stress=True)
    files=list(Path(__file__).parent.glob('*.py'))
    write_json(output/'source_manifest.json',{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
    with zipfile.ZipFile(output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in files: archive.write(path,str(path))
    started=time.monotonic(); steps=0
    with ProcessPoolExecutor(max_workers=4,initializer=init_worker,initargs=({'trim':trim,'sensor_inputs':6},)) as pool:
        ranked=[]
        for spec,payload,rows in zip(specs,payloads,pool.map(rollout,[(p,development,False) for p in payloads])):
            summary=summarize(rows); steps+=summary['control_steps']
            ranked.append(dict(spec=spec,payload=payload,development=summary))
        ranked.sort(key=lambda row:row['development']['robust_score'],reverse=True)
        print(json.dumps(dict(stage='screened',best=ranked[0]['development'],spec=ranked[0]['spec'])),flush=True)
        for row in ranked[:8]:
            episodes=[r for batch in pool.map(rollout,[(row['payload'],[case],False) for case in validation]) for r in batch]
            row['validation']=summarize(episodes); steps+=row['validation']['control_steps']
    finalists=sorted(ranked[:8],key=lambda row:(row['validation']['successes'],row['validation']['robust_score']),reverse=True)
    write_json(output/'candidates.json',ranked)
    for i,row in enumerate(finalists): write_json(output/f'seed_{i}.json',row['payload'])
    result=dict(candidates=len(ranked),development_cases=development,validation_cases=validation,
                finalists=[{k:v for k,v in row.items() if k!='payload'} for row in finalists],control_steps=steps,
                wall_seconds=time.monotonic()-started)
    write_json(output/'results.json',result)
    print(json.dumps(dict(stage='completed',best=finalists[0]['validation'],spec=finalists[0]['spec'],steps=steps)),flush=True)


if __name__=='__main__': main()
