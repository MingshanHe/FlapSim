"""Add controlled recovery scenarios without changing the frozen plant/PID."""
from collections import deque
import math
import mujoco
import numpy as np
from ..topology_hover.environment import HoverEnv as FrozenEnv, case, rollout, Zero
from ..topology_hover.sensors import HeightSensor
from ..pid_airborne.audit import clearance


def horizon(progress):return 2. if progress<256000 else 4. if progress<512000 else 8.


class HoverEnv(FrozenEnv):
    def __init__(self,*args,training=False,**kwargs):
        super().__init__(*args,training=False,**kwargs)
        self.train_enabled=training;self.training_progress=0
        self._advance=self.air.advance
        self.air.advance=self.advance_instrumented

    def set_training_progress(self,steps):self.training_progress=steps

    def reset(self,*,seed=None,options=None):
        if self.train_enabled:
            rng=np.random.default_rng(np.random.SeedSequence([self.run_seed,self.rank,self.episode_index,527]))
            seed=int(rng.integers(360000000,400000000))
            self.configure(case(seed,horizon(self.training_progress),'nominal' if self.episode_index%3==0 else 'train'))
            self.episode_index+=1
        obs,info=super().reset(seed=seed,options=options)
        if 'tilt_deg' in self.case:
            # A free-root rigid rotation preserves the articulated constraints.
            e=self.air.env;qpos=self.sim.data.qpos.copy();qvel=self.sim.data.qvel.copy()
            angle=np.deg2rad(self.case['tilt_deg']);phi=np.deg2rad(self.case.get('direction_deg',0.))
            axis=np.array([np.cos(phi),np.sin(phi),0.])
            qpos[e.free_qpos+3:e.free_qpos+7]=np.r_[np.cos(angle/2),axis*np.sin(angle/2)]
            qvel[:]=0.
            self.sim.reset(qpos=qpos,qvel=qvel)
            self.sim.set_gimbal(pitch=float(e.reference_angles[0]),roll=float(e.reference_angles[1]));self.sim.set_speed(0.)
            s=self.frontend.sensor
            self.frontend.reset_sensor(np.array(s.bias),np.array(s.angle_bias),self.p['sensor_tau'])
            self.air.start=self.sim.data.xpos[e.body].copy()
            self.air.last_rpy=np.array(self.frontend.sensor.attitude)
            self.height=HeightSensor(self.air.start[2],self.rng,self.case.get('height_noise',.002),self.case.get('height_delay',.02))
            self.last_reading=self.reading();self.reading_queue=deque([self.last_reading.copy()]*(self.case.get('delay_steps',0)+1))
            self.minimum_clearance=clearance(self.sim.model,self.sim.data);obs=self.observation()
        self.saturation_counts=np.zeros(3,dtype=int);self.range_missing=0;self.peak_rate=0.
        self.initial_tilt_deg=float(np.rad2deg(np.arccos(np.clip(self.air.env._state()[2][2,2],-1,1))))
        return obs,info

    def advance_instrumented(self,command,p):
        t=self.air.steps*.002;c=self.case
        if c.get('impulse_start',5.)<=t<c.get('impulse_start',5.)+c.get('impulse_duration',.1) and c.get('impulse_torque',0):
            phi=np.deg2rad(c.get('direction_deg',0.))
            self.sim.external_body_forces[self.air.env.body,3:5]+=c['impulse_torque']*np.array([np.cos(phi),np.sin(phi)])
        self.saturation_counts+=np.r_[np.abs(command[:2])>=.3-1e-9,command[2]<=3115.224079+1e-6 or command[2]>=3807.496097-1e-6]
        self._advance(command,p)
        self.range_missing+=int(self.sim.data.xmat[self.air.env.body,8]<=.7)
        self.peak_rate=max(self.peak_rate,float(np.linalg.norm(self.sim.data.cvel[self.air.env.body,:3])))

    def metrics(self,reason='time_limit'):
        r=super().metrics(reason);a=np.asarray(self.samples);late=a[a[:,0]>=max(0,a[-1,0]-5)]
        r.update(height_late_peak_to_peak_m=float(np.ptp(late[:,2])),initial_tilt_deg=self.initial_tilt_deg,
            maximum_tilt_deg=float(np.rad2deg(a[:,1].max())),maximum_height_loss_m=float(max(0,-a[:,2].min())),
            saturation_fraction=(self.saturation_counts/self.air.steps).tolist(),height_sensor_out_of_angle_fraction=self.range_missing/self.air.steps,
            peak_angular_speed_rad_s=self.peak_rate)
        origin=self.case.get('impulse_start',5.)+self.case.get('impulse_duration',.1) if self.case.get('impulse_torque',0) else 0.
        window=250;hold=500;recovery=None
        if len(a)>=window+hold:
            squares=np.r_[0,np.cumsum(a[:,1]**2)]
            rms=np.sqrt((squares[window:]-squares[:-window])/window)
            good=rms<np.deg2rad(10.);count=np.r_[0,np.cumsum(good)]
            held=(count[hold+1:]-count[:-(hold+1)])==hold+1
            for i in np.flatnonzero(held):
                t=a[i+window-1,0]
                if t>=origin+.5-1e-9:
                    recovery=float(t-origin);break
        r['recovery_time_s']=recovery
        height_ok=r['height_rmse_m']<.15 and r['height_late_peak_to_peak_m']<.2 and abs(r['height_slope_m_s'])<.03
        r['recovery_success']=bool(recovery is not None and recovery<=5. and reason=='time_limit' and a[-1,0]>=self.case['seconds']-1e-6 and r['minimum_clearance_m']>.03 and height_ok)
        return r
