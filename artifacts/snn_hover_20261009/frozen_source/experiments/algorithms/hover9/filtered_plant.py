"""Anti-aliased base-link attitude/gyro with IMU-only inertial history."""
import numpy as np
from .plant import Plant
from .imu import attach


class FilteredPlant(Plant):
    def __init__(self,*args,attitude_tau=.01,**kwargs):
        super().__init__(*args,**kwargs)
        self.frontend=attach(self.sim);self.attitude_tau=attitude_tau

    def sensor(self,obs):
        # The inherited environment's instantaneous sensor path is unused.
        return obs

    def measured(self):
        s=self.frontend.imu
        obs=np.r_[np.asarray(s.attitude)/.35,np.asarray(s.gyro)/10.]
        sigma=np.r_[np.full(3,np.deg2rad(self.case.get('noise_degrees',0))/.35),
                    np.full(3,self.case.get('noise_gyro_rad_s',0)/10.)]
        current=obs.copy()
        if self.case.get('delay_steps',0):obs=self.last_measured.copy()
        self.last_measured=current
        return (obs+self.sensor_rng.normal(0,sigma)).astype(np.float32)

    def reset(self,*,seed=None,options=None):
        _,info=super().reset(seed=seed,options=options)
        self.frontend.reset_imu(attitude_tau=self.attitude_tau)
        self.last_measured=np.r_[np.asarray(self.frontend.imu.attitude)/.35,np.asarray(self.frontend.imu.gyro)/10.]
        return self.measured(),info

    def step(self,action):
        _,reward,done,truncated,info=super().step(action)
        return self.measured(),reward,done,truncated,info
