"""Separate reference trim, release phase and initial centroidal momentum."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from .hover_initial import HoverInitialEnv
from .activation import MappedPolicy,signed_map,gyro_network,cases
from .robust import summarize,robust_score
from .train import write_json


def evaluate(task):
    settings,payload,batch=task
    if 'model_path' in settings:
        from .mass_model import MassVariantEnv
        env=MassVariantEnv(**settings)
    else:
        env=HoverInitialEnv(**settings)
    policy=MappedPolicy(payload,signed_map(settings['trim']))
    rows=[]
    for case in batch:
        env.configure_case(case); obs,initial=env.reset(seed=case['seed']); policy.reset(obs)
        quality,tilt_squared=0.,0.
        while True:
            obs,_,terminated,truncated,info=env.step(policy.act(obs))
            tilt=np.arccos(np.clip(env.sim.data.xmat[env.body,8],-1,1))
            tilt_squared+=tilt**2
            quality+=np.exp(-(tilt/.35)**2-(info['position_error']/.25)**2)
            if terminated or truncated: break
        rows.append(dict(seed=case['seed'],case=case,steps=env.steps,elapsed=info['elapsed'],success=info['is_success'],
                         tracking_rmse_m=info['tracking_rmse'],termination=info['termination_reason'],
                         tilt_rmse_deg=float(np.rad2deg(np.sqrt(tilt_squared/env.steps))),stable_seconds=.01*quality,
                         fitness=(env.steps+2*quality)/800.,initial_state={k:v for k,v in initial.items() if k.startswith('initial_momentum') or k=='root_velocity_correction'}))
    env.close()
    return rows


if __name__=='__main__':
    root=Path('artifacts/neat_imu_hoverstart_20261008'); root.mkdir(exist_ok=False)
    trim=json.loads(Path('artifacts/neat_imu_trim_20261007/calibration_fine/trim.json').read_text())
    old=json.loads(Path('artifacts/neat_imu_activation_20261007/training/best_policy.json').read_text())
    write_json(root/'frozen_candidate.json',old)
    constant=gyro_network(signed_map(trim),0.,0.,activation='clamped')
    constant['nodes'][0]['bias']=trim['command'][1]/.3
    constant['nodes'][1]['bias']=trim['command'][2]/.3
    specs=list(itertools.product(('release','trim','momentum'),(.1,.2,.4,.8,1.)))
    development=cases([896010000],2.)+cases([896010001],2.)
    for case in development: case['degrees']=1.
    settings=[dict(trim=trim,sensor_inputs=6,initialization=mode,warmup_seconds=warmup) for mode,warmup in specs]
    tasks=[(setting,payload,development) for setting in settings for payload in (constant,old)]
    files=list(Path(__file__).parent.glob('*.py'))
    write_json(root/'source_manifest.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    with zipfile.ZipFile(root/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in files: archive.write(path,str(path))
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=4) as pool:
        results=[]
        for task,episodes in zip(tasks,pool.map(evaluate,tasks)):
            setting,payload,_=task
            row=dict(initialization=setting['initialization'],warmup=setting['warmup_seconds'],policy='constant_trim' if payload is constant else 'frozen_candidate',summary=summarize(episodes),episodes=episodes)
            results.append(row); write_json(root/'progress.json',results)
            print(json.dumps({k:v for k,v in row.items() if k!='episodes'}),flush=True)
    result=dict(development_cases=development,results=results,control_steps=sum(row['summary']['control_steps'] for row in results),wall_seconds=time.monotonic()-started)
    write_json(root/'results.json',result)
