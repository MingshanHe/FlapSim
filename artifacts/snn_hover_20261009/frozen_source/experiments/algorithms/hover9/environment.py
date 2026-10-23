"""Common residual controller: nine IMU channels and six causal history features."""
import json
from pathlib import Path
import numpy as np
from gymnasium import spaces
from .plant import Plant
from .imu import attach
from ..neat_imu.activation import cases
from ..neat_imu.residual import ResidualPolicy

NAMES=['roll','pitch','yaw','gyro_x','gyro_y','gyro_z','accel_x','accel_y','accel_z',
       'estimated_x','estimated_y','estimated_z','estimated_vx','estimated_vy','estimated_vz']


def training_case(seed,fraction,stress):
    case=cases([seed],4. if fraction<.5 else 8.,stress)[0]
    case['degrees']=2. if fraction<.3 else 5.
    if stress:
        case.update(noise_degrees=.1,noise_gyro_rad_s=.01,motor_uncertainty=.01,
                    angular_rate=.1,delay_steps=0,acceleration_bias=.005,attitude_bias_degrees=.05)
    return case


class HistoryEnv(Plant):
    """A shared frozen six-IMU attitude controller, with learned actuator corrections.

    The actor gets no simulator position, velocity, joint states, or flap phase.
    Translational estimates are reset to zero once at release, never during flight.
    """
    def __init__(self,root,training=False,**kwargs):
        super().__init__(root,training=False,**kwargs)
        self.learn=training
        self.frontend=attach(self.sim)
        self.inner=ResidualPolicy(json.loads((Path(root)/'attitude_prior.json').read_text()))
        self.observation_space=spaces.Box(-10.,10.,(15,),dtype=np.float32)
        self.ablation=None

    def features(self,raw6):
        s=self.frontend.imu
        current=np.r_[np.asarray(s.specific_force)/9.81,np.asarray(s.position)/.5,np.asarray(s.velocity)/2.]
        values=self.last_extra.copy() if self.case.get('delay_steps',0) else current.copy()
        self.last_extra=current
        values[:3]+=self.sensor_rng.normal(0,self.case.get('accel_noise',0)/9.81,3)
        if self.ablation=='zero_history':values[3:]=0.
        if self.ablation=='zero_accel_history':values[:]=0.
        return np.clip(np.r_[raw6,values],-10,10).astype(np.float32)

    def reset(self,*,seed=None,options=None):
        if self.learn:
            rng=np.random.default_rng(np.random.SeedSequence([self.run_seed,self.rank,self.episode_index,9631]))
            seed=int(rng.integers(10000,800000000))
            self.configure_case(training_case(seed,min(1.,self.train_steps*self.workers/self.budget),self.episode_index%2==1))
            self.episode_index+=1
        if seed is None:seed=self.case['seed']
        raw,info=super().reset(seed=seed,options=options)
        rng=np.random.default_rng(np.random.SeedSequence([seed,9431]))
        bias=rng.uniform(-1,1,3)*self.case.get('acceleration_bias',0.)
        angle_error=rng.uniform(-1,1,3)*np.deg2rad(self.case.get('attitude_bias_degrees',0.))
        self.frontend.reset_imu(bias=bias,attitude_error=angle_error)
        self.last_extra=np.r_[np.asarray(self.frontend.imu.specific_force)/9.81,np.zeros(6)]
        self.inner.reset(raw);self.raw6=raw.copy()
        self.observer_error_squared=0.
        return self.features(raw),info

    def step(self,action):
        action=np.asarray(action,dtype=float)
        if action.shape!=(3,) or not np.isfinite(action).all():raise ValueError('three finite residual actions required')
        action=np.clip(action,-1,1)
        prior=self.inner.act(self.raw6)
        # +/-2 normalized corrections can reach the entire actuator range from any prior.
        physical=np.clip(np.r_[prior[:2]+2.*action[:2],action[2]],-1,1)
        raw,reward,done,truncated,info=super().step(physical)
        self.raw6=raw.copy();self.train_steps+=int(self.learn)
        p_hat=np.array(self.frontend.imu.position);v_hat=np.array(self.frontend.imu.velocity)
        p_true=self.sim.data.xpos[self.body]-self.start_position
        error=float(np.linalg.norm(p_hat-p_true))
        self.observer_error_squared+=error**2
        info.update(estimated_position=p_hat,estimated_velocity=v_hat,relative_position=p_true.copy(),
                    observer_error_m=error,observer_rmse_m=float(np.sqrt(self.observer_error_squared/self.steps)),
                    outer_action=action.copy(),physical_action=physical.copy())
        return self.features(raw),reward,done,truncated,info

