"""Shared 100 Hz actor / 500 Hz PID / 200 kHz unchanged physical plant."""
from collections import deque
import json
from pathlib import Path
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from ..pid_airborne.initialization import InitializedAirborne
from ..pid_airborne.audit import clearance
from .sensors import SensorBackend,HeightSensor

INNER_DT=.002
ACTOR_DT=.01
NAMES=['roll','pitch','yaw','gyro_x','gyro_y','gyro_z','accel_x','accel_y','accel_z','height_error','vertical_speed']
DEFAULT_SOURCE='artifacts/pid_airborne_20261008'


def case(seed,seconds=4.,condition='train',phase=None):
    stress=condition in ('train','stress','ood')
    return dict(seed=int(seed),seconds=seconds,condition=condition,phase=phase,
        degrees=5. if stress else 2.,motor_uncertainty=.015 if stress else 0.,
        acceleration_bias=.01 if stress else 0.,attitude_bias_deg=.05 if stress else 0.,
        delay_steps=1 if stress else 0,attitude_noise_deg=.02,gyro_noise=.002,
        accel_noise=.02,height_noise=.002,height_delay=.02,
        lower_mass_scale=1. if condition!='ood' else 1.125,
        pulse_force=.006 if stress else 0.,pulse_torque=.000008 if stress else 0.)


class HoverEnv(gym.Env):
    def __init__(self,source=DEFAULT_SOURCE,mode='residual',sensor_mode='height',training=False,
                 seed=0,rank=0,horizon=2.,residual_scale=(.035,.035,100.)):
        self.air=InitializedAirborne(Path(source));self.sim=self.air.sim
        self.frontend=SensorBackend(self.sim);self.sim.backend=self.frontend;self.air.frontend=self.frontend
        self.p=json.loads((Path(source)/'selected_pid.json').read_text())['parameters']
        self.original_mass=self.sim.model.body_mass.copy();self.original_inertia=self.sim.model.body_inertia.copy()
        self.mode=mode;self.sensor_mode=sensor_mode;self.training=training
        self.run_seed=seed;self.rank=rank;self.horizon=horizon;self.episode_index=0
        self.residual_scale=np.array(residual_scale,dtype=float)
        self.observation_space=spaces.Box(-10.,10.,(9 if sensor_mode=='imu' else 11,),dtype=np.float32)
        self.action_space=spaces.Box(-1.,1.,(3,),dtype=np.float32)
        self.case=case(seed,horizon,'nominal');self.record=False

    def configure(self,c):self.case=dict(c)

    def reset(self,*,seed=None,options=None):
        if self.training:
            seed=int(np.random.default_rng(np.random.SeedSequence([self.run_seed,self.rank,self.episode_index,8163])).integers(100000,900000000))
            self.case=case(seed,self.horizon,'nominal' if self.episode_index%3==0 else 'train')
            self.episode_index+=1
        if seed is None:seed=self.case['seed']
        c=self.case;self.rng=np.random.default_rng(np.random.SeedSequence([seed,981]))
        model=self.sim.model;model.body_mass[:]=self.original_mass;model.body_inertia[:]=self.original_inertia
        body=self.air.env.body;factor=c.get('lower_mass_scale',1.)
        # Base is 7 g, connector 1 g. Scale both lower bodies consistently when requested.
        lower=[model.body('base_link').id,model.body('pitch_link').id]
        for bid in lower:model.body_mass[bid]*=factor;model.body_inertia[bid]*=factor
        mujoco.mj_setConst(model,self.sim.data)
        self.air.reset(seed,self.p,c.get('degrees',2.),phase=c.get('phase'))
        if c.get('phase') is not None:self.air.env.phase_index=c['phase']
        self.frontend.reset_sensor(self.rng.uniform(-1,1,3)*c.get('acceleration_bias',0.),
            self.rng.uniform(-1,1,3)*np.deg2rad(c.get('attitude_bias_deg',0.)),self.p['sensor_tau'])
        self.height=HeightSensor(self.air.start[2],self.rng,c.get('height_noise',.002),c.get('height_delay',.02))
        self.motor_gain=1+self.rng.uniform(-c.get('motor_uncertainty',0.),c.get('motor_uncertainty',0.))
        self.pulse_sign=self.rng.choice([-1.,1.],3)
        self.delay=deque();self.steps=0;self.total_reward=0.;self.previous_action=np.zeros(3)
        self.samples=[];self.trace=[];self.minimum_clearance=clearance(model,self.sim.data)
        self.sensor_clipping_start=0;self.last_reading=self.reading();self.reading_queue=deque([self.last_reading.copy()]*(c.get('delay_steps',0)+1))
        self.command=np.r_[self.p['trim'],self.p['motor']]
        return self.observation(),dict(phase=self.air.env.phase_index)

    def reading(self):
        s=self.frontend.sensor;c=self.case
        angle=np.array(s.attitude)+self.rng.normal(0,np.deg2rad(c.get('attitude_noise_deg',0)),3)
        gyro=np.array(s.gyro)+self.rng.normal(0,c.get('gyro_noise',0),3)
        force=np.array(s.force)+self.rng.normal(0,c.get('accel_noise',0),3)
        if self.sensor_mode=='imu':z,vz=s.position[2],s.velocity[2]
        elif self.sensor_mode=='truth':
            pos,vel,_=self.air.env._state();z,vz=pos[2]-self.air.start[2],vel[2]
        else:z,vz=self.height.h-self.height.origin,self.height.v
        return np.r_[angle,gyro,force,z,vz]

    def observation(self):
        scale=np.array([.35,.35,np.pi,10.,10.,10.,9.81,9.81,9.81,.2,1.])
        return np.clip(self.last_reading[:self.observation_space.shape[0]]/scale[:self.observation_space.shape[0]],-10,10).astype(np.float32)

    def pid(self):
        a=self.air;p=self.p;rpy=self.last_reading[:3]
        derivative=np.arctan2(np.sin(rpy[:2]-a.last_rpy[:2]),np.cos(rpy[:2]-a.last_rpy[:2]))/INNER_DT
        a.last_rpy=rpy.copy();a.rate+=(1-np.exp(-INNER_DT/p['derivative_tau']))*(derivative-a.rate)
        error=-rpy[:2];candidate=np.clip(a.integral+error*INNER_DT,-.3,.3)
        effort=np.array(p['kp'])*error-np.array(p['kd'])*a.rate+np.array(p['ki'])*candidate
        angles=np.array(p['trim'])+effort[::-1]
        a.integral=np.where(np.abs(angles[::-1])<.3,candidate,a.integral)
        z,vz=self.last_reading[9:11];a.z_integral=float(np.clip(a.z_integral-z*INNER_DT,-.5,.5))
        kp,kd,ki=p['height_gains'];motor=p['motor']-kp*z-kd*vz+ki*a.z_integral
        return np.r_[np.clip(angles,-.3,.3),np.clip(motor,3115.224079,3807.496097)]

    @staticmethod
    def normalize_command(command):
        return np.r_[np.asarray(command[:2])/.3,(command[2]-3461.360088)/346.136009]

    def step(self,action):
        action=np.clip(np.asarray(action,dtype=float),-1,1)
        if action.shape!=(3,) or not np.isfinite(action).all():raise ValueError('Expected three finite actions')
        terminated=False;reason='time_limit';reward=0.
        for k in range(5):
            base=self.pid()
            if k==0:teacher_action=self.normalize_command(base).astype(np.float32)
            if self.mode=='direct':command=np.r_[.3*action[:2],3461.360088+346.136009*action[2]]
            else:command=base+(np.zeros(3) if self.mode=='pid' else action*self.residual_scale)
            command[:2]=np.clip(command[:2],-.3,.3);command[2]=np.clip(command[2],3115.224079,3807.496097)
            self.command=command.copy();command[2]*=self.motor_gain
            t=self.air.steps*INNER_DT
            self.sim.external_body_forces[:]=0.
            if .6<=t%2.<.8:
                self.sim.external_body_forces[self.air.env.body,:3]=self.pulse_sign*np.array([1,1,0])*self.case.get('pulse_force',0.)
                self.sim.external_body_forces[self.air.env.body,3:5]=self.pulse_sign[:2]*self.case.get('pulse_torque',0.)
            self.air.advance(command,self.p)
            pos,vel,R=self.air.env._state();t=self.air.steps*INNER_DT
            self.height.step(t,pos[2],R[2,2],np.array(self.frontend.sensor.attitude))
            read=self.reading();self.reading_queue.append(read)
            while len(self.reading_queue)>self.case.get('delay_steps',0)+1:self.reading_queue.popleft()
            self.last_reading=self.reading_queue[0].copy()
            tilt=math.acos(float(np.clip(R[2,2],-1,1)));z=pos[2]-self.air.start[2]
            low=clearance(self.sim.model,self.sim.data);self.minimum_clearance=min(low,self.minimum_clearance)
            self.samples.append([t,tilt,z,low,float(np.linalg.norm(pos[:2]-self.air.start[:2]))])
            reward+=INNER_DT*(2.+math.exp(-(tilt/.15)**2)+math.exp(-(z/.12)**2))
            if self.record:self.trace.append([t,*pos,*vel,tilt,*self.command,*self.last_reading,*action,*self.sim.data.qpos])
            if not np.isfinite(self.sim.data.qpos).all() or any(w.number for w in self.sim.data.warning):reason='numerical'
            elif low<=.03:reason='clearance'
            elif pos[2]<.12:reason='low_height'
            elif abs(z)>1.:reason='height_escape'
            elif tilt>1.4:reason='tilt'
            if reason!='time_limit':terminated=True;reward-=1.;break
        reward-=ACTOR_DT*.02*float(np.sum((action-self.previous_action)**2))
        self.previous_action=action.copy();self.steps+=1;self.total_reward+=reward
        truncated=self.air.steps*INNER_DT>=self.case['seconds']-1e-9 and not terminated
        info={'teacher_action':teacher_action}
        if terminated or truncated:info['episode']=self.metrics(reason)
        return self.observation(),float(reward),terminated,truncated,info

    def metrics(self,reason='time_limit'):
        a=np.asarray(self.samples);settled=a[a[:,0]>=min(2.,a[-1,0]/2)]
        late=a[a[:,0]>=max(0,a[-1,0]-5)]
        tilt=float(np.rad2deg(np.sqrt(np.mean(settled[:,1]**2))))
        hrms=float(np.sqrt(np.mean(a[:,2]**2)))
        slope=float(np.polyfit(late[:,0],late[:,2],1)[0]) if len(late)>1 else 999.
        success=bool(reason=='time_limit' and a[-1,0]>=self.case['seconds']-1e-6 and self.minimum_clearance>.03 and tilt<10 and hrms<.15 and np.ptp(late[:,2])<.2 and abs(slope)<.03)
        return dict(seed=self.case['seed'],case=self.case,steps=self.steps,physics_steps=self.air.steps*400,elapsed=float(a[-1,0]),
            termination=reason,success=success,return_sum=self.total_reward,tilt_rms_deg=tilt,height_rmse_m=hrms,
            height_slope_m_s=slope,minimum_clearance_m=self.minimum_clearance,xy_offset_m=float(a[-1,4]),
            accel_clipped_fraction=float(self.frontend.sensor.clipped_acc/(self.air.steps*400*3)),
            estimated_height_error_m=float(self.last_reading[9]-a[-1,2]),phase=self.air.env.phase_index)

    def close(self):pass


def rollout(env,policy,c,trace=False):
    env.configure(c);env.record=trace;obs,_=env.reset(seed=c['seed']);policy.reset()
    while True:
        action=policy.act(obs);obs,_,done,truncated,info=env.step(action)
        if done or truncated:break
    result=info['episode']
    if trace:result['trace']=env.trace
    return result


class Zero:
    def reset(self):pass
    def act(self,obs):return np.zeros(3)
