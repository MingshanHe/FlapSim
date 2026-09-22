"""Isolate aerodynamic implementation and kinematic-configuration differences."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import argparse
import json
import sys
import hashlib
import numpy as np
import mujoco

FOLDER=Path(__file__).resolve().parent
ROOT=FOLDER.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments/fixbase')]
from rect_material_experiment import config_for, OUTPUT as BASE
from rect_simulation import ProfileBladeElement, mean_between
from flappy_audit_models import OUT, build_reference, StripAuditFluid, PurdueBenchFluid, cn, dcp
from run_15hz import cycle_statistics
from flapsim import Simulator
from flapsim.kinematics import point_velocity
from analyze_comparison import write_csv

PROFILE=BASE/'rect_profile.npz'


def fluid_for(name, config):
    if name in ('baseline','baseline_45deg'):
        return ProfileBladeElement(config.fluid,PROFILE)
    if name=='pressure_only':
        return StripAuditFluid(config.fluid,PROFILE,pressure=True)
    if name=='no_pitch_velocity':
        return StripAuditFluid(config.fluid,PROFILE,exclude_pitch=True)
    if name=='flappy_strip':
        return StripAuditFluid(config.fluid,PROFILE,pressure=True,exclude_pitch=True,chord_drag=False)
    if name in ('flappy_cpp','flappy_cpp_45deg','flappy_cpp_66deg'):
        return PurdueBenchFluid(config.fluid,PROFILE)
    if name=='flappy_native_density':
        return PurdueBenchFluid(config.fluid,PROFILE,density=1.18009482370369)
    raise ValueError(name)


def original(hz=15):
    return json.loads((BASE/f'simulation/rect/{hz}hz/result.json').read_text())


def initial_command(hz=15):
    return json.loads((BASE/f'simulation/rect/{hz}hz/calibration.json').read_text())[-1]['command_rad_s']


def capture():
    config=config_for();sim=Simulator(config,fluid=fluid_for('baseline',config))
    command=initial_command()
    saved=np.load(BASE/original()['source'])
    qpos=[];qvel=[];force=[]
    for i in range(4001):
        if i:
            phase=min((i-1)/2000/.15,1)
            sim.set_speed(command*(.5-.5*np.cos(np.pi*phase)));sim.step(100)
        qpos.append(sim.data.qpos.copy());qvel.append(sim.data.qvel.copy())
        force.append(sim.fluid.forces(sim.data)[sim.model.body('left_wing').id].copy())
    force=np.array(force)
    np.testing.assert_allclose(force[:,2],saved['aero_Fz_N'],rtol=0,atol=1e-14)
    np.savez_compressed(OUT/'baseline_states.npz',qpos=qpos,qvel=qvel,force=force,time_s=saved['time_s'])
    print('Baseline state capture matches saved lift.',flush=True)


def offline():
    saved=np.load(OUT/'baseline_states.npz')
    config=config_for();sim=Simulator(config,fluid=fluid_for('baseline',config))
    names=['baseline','pressure_only','no_pitch_velocity','flappy_strip','flappy_cpp','flappy_native_density']
    models={name:fluid_for(name,config) for name in names}
    for f in models.values(): f.bind(sim.model)
    body=sim.model.body('left_wing').id
    forces={k:[] for k in names};max_jac_error=0.
    for i,(qpos,qvel) in enumerate(zip(saved['qpos'],saved['qvel'])):
        sim.data.qpos[:]=qpos;sim.data.qvel[:]=qvel;mujoco.mj_forward(sim.model,sim.data)
        for name,f in models.items(): forces[name].append(f.forces(sim.data)[body].copy())
        if i%200==0:
            p=sim.data.xpos[body]+sim.data.xmat[body].reshape(3,3)@np.array([0,.05,.02])
            jac=np.zeros((3,sim.model.nv));rot=jac.copy()
            mujoco.mj_jac(sim.model,sim.data,jac,rot,p,body)
            v=point_velocity(sim.data,body,sim.model.body_rootid[body],p)
            max_jac_error=max(max_jac_error,float(np.max(abs(v-jac@qvel))))
    times=saved['time_s'];start,end=original()['averaging_interval_s']
    rows=[]
    for name,values in forces.items():
        f=np.asarray(values);forces[name]=f
        means=[mean_between(times,f[:,axis],start,end) for axis in range(6)]
        rows.append(dict(model=name,mean_lift_mN=means[2]*1000,
                         delta_from_baseline_pct=(means[2]/original()['mean_lift_N']-1)*100))
    np.savez_compressed(OUT/'same_motion_forces.npz',time_s=times,**forces)
    (OUT/'same_motion_summary.json').write_text(json.dumps(rows,indent=2))
    write_csv(OUT/'same_motion_summary.csv',rows)
    (OUT/'velocity_check.json').write_text(json.dumps(dict(max_mujoco_jacobian_velocity_error_m_s=max_jac_error),indent=2))
    print('Same motion:',json.dumps(rows), 'Jacobian error',max_jac_error,flush=True)


def run_case(name, hz=15):
    target=original(hz)['target_wing_frequency_hz']
    config=config_for();command=initial_command(hz)
    path=OUT/'coupled'/name/f'{hz}hz';path.mkdir(parents=True,exist_ok=True)
    trials=[]
    for attempt in range(1,6):
        sim=Simulator(config,fluid=fluid_for(name,config))
        if name.endswith(('45deg','66deg')):
            limit=45 if name.endswith('45deg') else 66
            sim.model.joint('left_feathering_joint').range[:]=np.radians([-limit,limit])
            mujoco.mj_setConst(sim.model,sim.data);sim.reset()
        body=sim.model.body('left_wing').id;records=[]
        for i in range(4001):
            if i:
                phase=min((i-1)/2000/.15,1)
                sim.set_speed(command*(.5-.5*np.cos(np.pi*phase)));sim.step(100)
            f=sim.fluid.forces(sim.data)[body]
            records.append([sim.data.time,sim.data.joint('motor_joint').qpos[0],
                            sim.data.joint('left_flapping_joint').qpos[0],
                            sim.data.joint('left_feathering_joint').qpos[0],*f])
        a=np.asarray(records)
        assert np.isfinite(a).all() and not np.any(sim.data.warning.number)
        keys=['time_s','motor_joint_angle_rad','flap_rad','pitch_rad','aero_Fx_N','aero_Fy_N','aero_Fz_N','aero_Mx_Nm','aero_My_Nm','aero_Mz_Nm']
        data=dict(zip(keys,a.T));stats=cycle_statistics(data,target_hz=target)
        np.savez_compressed(path/f'run_{attempt}.npz',**data)
        actual=stats['measured_wing_frequency_hz']
        trials.append(dict(command_rad_s=float(command),actual_hz=actual,mean_lift_mN=stats['mean_lift_mN']))
        (path/'calibration.json').write_text(json.dumps(trials,indent=2))
        print(name,hz,'attempt',attempt,'frequency',round(actual,5),'lift mN',round(stats['mean_lift_mN'],3),flush=True)
        if abs(actual-target)<.01:
            result=dict(name=name,hz=hz,**stats,source=str((path/f'run_{attempt}.npz').relative_to(OUT)),
                        pitch_limits_deg=np.degrees(sim.model.joint('left_feathering_joint').range).tolist(),
                        pitch_minmax_deg=np.degrees([a[a[:,0]>.6,3].min(),a[a[:,0]>.6,3].max()]).tolist(),
                        numerical_warnings=sim.data.warning.number.tolist(),
                        model_sha256=hashlib.sha256(config.path.read_bytes()).hexdigest(),
                        geometry_and_materials='Unchanged rectangular single-left-wing PET/Kapton model',
                        note='45deg and 66deg cases are diagnostic changes only, NOT the confirmed physical stop.')
            (path/'result.json').write_text(json.dumps(result,indent=2))
            return result
        slope=1/(2*np.pi*16)
        if len(trials)>1:
            slope=(actual-trials[-2]['actual_hz'])/(command-trials[-2]['command_rad_s'])
        assert slope>0
        command+=(target-actual)/slope
        assert 0<command<=2000
    raise RuntimeError(f'{name}: frequency calibration failed')


def coupled():
    cases=['pressure_only','no_pitch_velocity','flappy_cpp','flappy_native_density','baseline_45deg','flappy_cpp_45deg']
    results=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(run_case,name) for name in cases]
        for job in as_completed(jobs):
            results.append(job.result())
            (OUT/'coupled_summary.json').write_text(json.dumps(results,indent=2))


def sweep():
    """Confirm the native-core result over the original 10--15 Hz range."""
    jobspec = [('flappy_cpp', hz) for hz in range(10,15)] + [('flappy_cpp_45deg',10)]
    results=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(run_case,name,hz) for name,hz in jobspec]
        for job in as_completed(jobs):
            results.append(job.result())
            (OUT/'sweep_summary.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('build','capture','offline','coupled','sweep'): p.add_argument('--'+name,action='store_true')
    args=p.parse_args()
    if args.build: build_reference()
    if args.capture: capture()
    if args.offline: offline()
    if args.coupled: coupled()
    if args.sweep: sweep()
