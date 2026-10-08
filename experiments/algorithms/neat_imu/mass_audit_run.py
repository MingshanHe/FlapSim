"""Mass/COM audit: fixed-base authority, lift matching, paired free releases."""
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from .trim import measure
from .mass_model import MassVariantEnv
from .activation import MappedPolicy,signed_map,cases,gyro_network
from .robust import summarize
from .train import write_json

ROOT=Path('artifacts/neat_imu_mass_audit_20261008')
BASE=3377.708636988347


def fixed(task):
    path,command,precise=task
    return measure(command,settle=.8,duration=1.2 if precise else .6,sample_dt=.00005 if precise else .0002,model_path=path)


def fly(task):
    model,trim,payload,case,label,speed_mode=task
    env=MassVariantEnv(trim,model_path=model['path'],sensor_inputs=6,initialization='release')
    env.configure_case(case); obs,initial=env.reset(seed=case['seed'])
    policy=MappedPolicy(payload,signed_map(trim)); policy.reset(obs)
    quality,tilt_squared,trace=0.,0.,[]
    while True:
        obs,_,terminated,truncated,info=env.step(policy.act(obs))
        tilt=np.arccos(np.clip(env.sim.data.xmat[env.body,8],-1,1))
        quality+=np.exp(-(tilt/.35)**2-(info['position_error']/.25)**2)
        tilt_squared+=tilt**2
        trace.append([info['elapsed'],*np.rad2deg(info['rpy']),info['alpha'],info['beta'],*info['position'],info['motor_command']])
        if terminated or truncated: break
    record=dict(model=model['label'],policy=label,speed_mode=speed_mode,case=case,seed=case['seed'],steps=env.steps,
                elapsed=info['elapsed'],success=info['is_success'],tracking_rmse_m=info['tracking_rmse'],termination=info['termination_reason'],
                tilt_rmse_deg=float(np.rad2deg(np.sqrt(tilt_squared/env.steps))),stable_seconds=.01*quality,
                fitness=(env.steps+2*quality)/800.,trace=trace)
    env.close(); return record


def main():
    models=json.loads((ROOT/'models.json').read_text())
    if (ROOT/'fixed.json').exists(): raise RuntimeError('do not overwrite an audit')
    started=time.monotonic()
    source_files=list(Path(__file__).parent.glob('*.py'))+list(Path('flapsim').rglob('*.py'))+list(Path('flapsim').rglob('*.cpp'))
    manifest={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    write_json(ROOT/'source_manifest.json',manifest)
    with zipfile.ZipFile(ROOT/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in source_files: archive.write(path,str(path))
    conditions=[(BASE,0.,0.),(BASE,.05,0.),(BASE,-.05,0.),(BASE,0.,.05),(BASE,0.,-.05),(3700.,0.,0.)]
    tasks=[(model['path'],command,False) for model in models for command in conditions]
    with ProcessPoolExecutor(max_workers=12) as pool:
        measurements=list(pool.map(fixed,tasks))
        fixed_rows=[]; targets=[]
        for i,model in enumerate(models):
            rows=measurements[i*len(conditions):(i+1)*len(conditions)]
            derivative=(np.array(rows[1]['torque_com_Nm'])-np.array(rows[2]['torque_com_Nm']))/.1
            derivative_beta=(np.array(rows[3]['torque_com_Nm'])-np.array(rows[4]['torque_com_Nm']))/.1
            mass=model['properties']['total']['mass_g']/1000
            slope=(rows[5]['force_N'][2]-rows[0]['force_N'][2])/(3700-BASE)
            if slope<=0: raise RuntimeError('lift slope is not positive')
            speed=float(np.clip(BASE+(mass*9.81-rows[0]['force_N'][2])/slope,2800,4500))
            targets.append((model['path'],[speed,0.,0.],False))
            fixed_rows.append(dict(model=model['label'],measurements=rows,roll_pitch_authority_Nm_per_rad=np.column_stack((derivative[:2],derivative_beta[:2])).tolist(),lift_slope_N_per_command_rad_s=slope))
        write_json(ROOT/'fixed.json',fixed_rows)
        checks=list(pool.map(fixed,targets)); measurements+=checks
        refined=[]
        for model,row,check in zip(models,fixed_rows,checks):
            speed=check['command'][0]+(check['weight_N']-check['force_N'][2])/row['lift_slope_N_per_command_rad_s']
            refined.append((model['path'],[float(speed),0.,0.],True))
        matched=list(pool.map(fixed,refined)); measurements+=matched
        for model,row in zip(models,matched):
            row.update(model=model['label'],relative_vertical_lift_error=row['force_N'][2]/row['weight_N']-1.,
                       hover_attitude_rad=[0.,0.,0.],scope='Vertical lift only; not an angular-moment trim or a validated hovering state')
            write_json(ROOT/(model['label']+'_lift.json'),row)
        write_json(ROOT/'lift_matching.json',matched)
        print(json.dumps(dict(stage='fixed_complete',lift=[dict(model=r['model'],speed=r['command'][0],gf=r['lift_gf'],relative_error=r['relative_vertical_lift_error']) for r in matched])),flush=True)
        old=json.loads(Path('artifacts/neat_imu_activation_20261007/training/best_policy.json').read_text())
        write_json(ROOT/'frozen_old_policy.json',old)
        batch=cases(range(898810000,898810008),8.)
        tasks=[]
        for model,lift in zip(models,matched):
            for speed_mode in ('same_15gf_command','weight_matched'):
                trim=copy.deepcopy(lift)
                if speed_mode=='same_15gf_command': trim['command']=[BASE,0.,0.]
                constant=gyro_network(signed_map(trim),0.,0.,activation='clamped')
                policy=copy.deepcopy(old)
                # The old network's normalized +/-10% motor output is rebased,
                # not retrained; report this explicitly in the comparison.
                policy['action_map']=signed_map(trim)
                for label,payload in (('constant_neutral',constant),('old_neat_rebased',policy)):
                    tasks.extend((model,trim,payload,case,label,speed_mode) for case in batch)
        episodes=list(pool.map(fly,tasks))
    write_json(ROOT/'episodes.json',episodes)
    summaries=[]
    for model in models:
        for speed_mode in ('same_15gf_command','weight_matched'):
            for label in ('constant_neutral','old_neat_rebased'):
                group=[row for row in episodes if row['model']==model['label'] and row['policy']==label and row['speed_mode']==speed_mode]
                summaries.append(dict(model=model['label'],speed_mode=speed_mode,policy=label,summary=summarize(group)))
    result=dict(status='completed',cases=batch,summaries=summaries,
                control_steps=sum(row['steps'] for row in episodes),fixed_physics_steps=sum(row['physics_steps'] for row in measurements),
                wall_seconds=time.monotonic()-started)
    write_json(ROOT/'results.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
