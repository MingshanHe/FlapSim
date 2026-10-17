"""Held-out flight audit with conservative whole-robot geometric clearance."""
import argparse,copy,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .core import DT
from .initialization import InitializedAirborne
from ..neat_imu.train import write_json


def clearance(model,data):
    ids=np.flatnonzero(model.geom_bodyid!=0)
    vertical=data.geom_xmat[ids].reshape(-1,3,3)[:,2,:]
    boxes=model.geom_aabb[ids]
    # The rotated local AABB encloses every mesh vertex (also air proxy geoms).
    lower=data.geom_xpos[ids,2]+np.sum(vertical*boxes[:,:3],axis=1)-np.sum(np.abs(vertical)*boxes[:,3:],axis=1)
    return float(lower.min())


class AuditedAirborne(InitializedAirborne):
    def reset(self,seed,p,degrees=2.,**kwargs):
        super().reset(seed,p,degrees,**kwargs)
        self.minimum_clearance=clearance(self.sim.model,self.sim.data)
        self.audit_rows=[]
        self.audit_rng=np.random.default_rng(np.random.SeedSequence([seed,7142]))
        self.motor_gain=1.+self.audit_rng.uniform(-p.get('motor_uncertainty',0.),p.get('motor_uncertainty',0.))
        if p.get('acceleration_bias_range',0.):
            bias=self.audit_rng.uniform(-p['acceleration_bias_range'],p['acceleration_bias_range'],3)
            self.frontend.reset_imu(bias=bias,attitude_tau=p.get('sensor_tau',.02),tau=.002)
        self.measurement_delay=[]

    def control(self,p):
        # Sensor error is added to a returned reading, then restored; never feed
        # measurement noise back into the native sensor filter's hidden state.
        sensor=self.frontend.imu
        original=np.array(sensor.attitude)
        noisy=original+self.audit_rng.normal(0,np.deg2rad(p.get('attitude_noise_deg',0.)),3)
        for i in range(3):sensor.attitude[i]=noisy[i]
        try:return super().control(p)
        finally:
            for i in range(3):sensor.attitude[i]=original[i]

    def advance(self,command,p):
        command=command.copy()
        command[2]=float(np.clip(command[2]*self.motor_gain,0,5000))
        super().advance(command,p)
        low=clearance(self.sim.model,self.sim.data)
        self.minimum_clearance=min(low,self.minimum_clearance)
        position,velocity,rotation=self.env._state()
        tilt=np.arccos(np.clip(rotation[2,2],-1,1))
        self.audit_rows.append([self.steps*DT,low,tilt,position[2]-self.start[2],
                               self.frontend.imu.position[2]-(position[2]-self.start[2])])

    def rollout(self,p,seed,seconds=20.,degrees=2.,trace=False):
        result=super().rollout(p,seed,seconds,degrees,trace)
        a=np.array(self.audit_rows); settled=a[a[:,0]>=2.]
        if not len(settled):settled=a
        late=a[a[:,0]>=max(0,a[-1,0]-5)]
        tilt_rms=np.rad2deg(np.sqrt(np.mean(settled[:,2]**2)))
        height_span=float(np.ptp(late[:,3]))
        slope=float(np.polyfit(late[:,0],late[:,3],1)[0]) if len(late)>1 else 999.
        complete=result['termination']=='time_limit'
        stable=bool(complete and self.minimum_clearance>.03 and tilt_rms<10 and height_span<.2 and abs(slope)<.03)
        result.update(minimum_clearance_m=self.minimum_clearance,settled_tilt_rms_deg=float(tilt_rms),
            late_height_span_m=height_span,late_height_slope_m_s=slope,
            observer_final_height_error_m=float(a[-1,4]),airborne_stable=stable,
            strict_height_hover=bool(stable and result['height_rmse_m']<.15),
            actual_final_xy_offset_m=float(np.linalg.norm(np.array(result['final_position'])[:2]-self.start[:2])))
        if trace:result['audit_trace']=a.tolist()
        return result


ENV=None
def init(root):
    global ENV
    ENV=AuditedAirborne(root)
def evaluate(task):
    label,p,seed,seconds,degrees,trace=task
    return dict(policy=label,parameters=p,**ENV.rollout(p,seed,seconds,degrees,trace))


def summarize(rows):
    return dict(episodes=len(rows),completed=sum(r['termination']=='time_limit' for r in rows),
        stable=sum(r['airborne_stable'] for r in rows),strict_hover=sum(r['strict_height_hover'] for r in rows),
        mean_duration=float(np.mean([r['elapsed'] for r in rows])),
        mean_settled_tilt_rms_deg=float(np.mean([r['settled_tilt_rms_deg'] for r in rows])),
        mean_height_rmse_m=float(np.mean([r['height_rmse_m'] for r in rows])),
        minimum_clearance_m=min(r['minimum_clearance_m'] for r in rows),
        maximum_xy_offset_m=max(r['actual_final_xy_offset_m'] for r in rows),
        mean_final_height_estimate_error_m=float(np.mean([r['observer_final_height_error_m'] for r in rows])),
        termination_counts={k:sum(r['termination']==k for r in rows) for k in sorted({r['termination'] for r in rows})})


def register(root):
    root=Path(root)
    protocol=dict(status='registered_before_held_out_testing',mass_g=[7,8],gravity_m_s2=9.81,
        physics_dt_s=.000005,control_dt_s=DT,yaw_control=False,xy_control=False,
        initial_state='Motionless whole mechanism at 0.8 m, powered-phase joint positions retained except aligned gimbals; 100 ms motor ramp. This is an airborne release, not ground takeoff.',
        nominal_seeds=list(range(310900000,310900016)),stress_seeds=list(range(310910000,310910008)),
        validation_seeds=list(range(310300000,310300008)),seconds=20.,
        nominal_initial_angle_deg=2.,stress_initial_angle_deg=5.,
        stress={'acceleration_bias_range':.01,'motor_uncertainty':.01,'attitude_noise_deg':.02},
        stable_criteria='Complete horizon; minimum conservative geometric clearance > 0.03 m; tilt RMS after 2 s < 10 deg; last 5 s height span < 0.2 m and linear height trend magnitude < 0.03 m/s.',
        strict_height_criteria='Stable criteria and entire-episode height RMSE relative to release height < 0.15 m.',
        ground_check='Rotated local AABB lower bound over all robot geoms at 500 Hz. Mesh collisions disabled in source model, so this is geometric clearance, not a contact-physics validation.',
        sensors='Same optimistic high-rate virtual IMU front end as prior round; no true position/velocity in IMU policy. Truth-height mode is a separately labeled diagnostic.',
        video_selection='First fixed nominal seed 310900000, regardless of outcome; same seed for truth-height diagnostic if rendered.')
    write_json(root/'audit_protocol.json',protocol)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['register','validate','test'],required=True);a=p.parse_args();root=a.root
    if a.stage=='register':register(root);return
    protocol=json.loads((root/'audit_protocol.json').read_text())
    if a.stage=='validate':
        candidates=json.loads((root/'height_search.json').read_text())['rows'][:3]
        tasks=[(f'candidate_{i}',c['parameters'],s,15.,5.,False) for i,c in enumerate(candidates) for s in protocol['validation_seeds']]
    else:
        policy=json.loads((root/'selected_pid.json').read_text())['parameters'];tasks=[]
        for label,seeds,degrees,extra in [('imu_nominal',protocol['nominal_seeds'],2.,{}),
                ('imu_stress',protocol['stress_seeds'],5.,protocol['stress']),
                ('truth_height_diagnostic',protocol['nominal_seeds'][:8],2.,{'height_source':'truth'})]:
            tasks.extend((label,dict(policy,**extra),s,protocol['seconds'],degrees,s==310900000) for s in seeds)
    with ProcessPoolExecutor(max_workers=8,initializer=init,initargs=(str(root),)) as pool:
        rows=list(pool.map(evaluate,tasks))
    summaries={label:summarize([r for r in rows if r['policy']==label]) for label in sorted({r['policy'] for r in rows})}
    write_json(root/f'{a.stage}_results.json',dict(summaries=summaries,episodes=rows))
    if a.stage=='validate':
        def key(i):
            s=summaries[f'candidate_{i}']
            return (s['stable'],s['completed'],-s['mean_height_rmse_m'],-s['mean_settled_tilt_rms_deg'])
        chosen=max(range(len(candidates)),key=key)
        write_json(root/'selected_pid.json',dict(format='flapsim-pid-airborne-v1',parameters=candidates[chosen]['parameters'],
            selection='Eight validation seeds; no held-out test data used',validation=summaries[f'candidate_{chosen}']))
    print(json.dumps(summaries),flush=True)

if __name__=='__main__':main()
