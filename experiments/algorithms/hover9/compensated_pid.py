"""PID using a command-history estimate of gimbal inertial reaction.

The base IMU sees the opposite reaction when the upper assembly is moved. A
virtual attitude adds a fraction of the predicted relative gimbal displacement.
No actual joint state, global position, or hidden simulator velocity is read.
"""
import numpy as np
from .plant import PID


class CompensatedPID(PID):
    def reset(self,obs):
        super().reset(obs)
        self.neutral=np.array([.013661637965125593,.0008819029388009655])
        self.predicted_gimbal=self.neutral.copy()
        self.predicted_rate=np.zeros(2)

    def act(self,obs,target=None,motor=0.):
        proxy=np.asarray(obs,dtype=float).copy()
        kappa=self.p[9:11]
        proxy[:2]+=kappa*(self.predicted_gimbal-self.neutral)[::-1]/.35
        proxy[3:5]+=kappa*self.predicted_rate[::-1]/10.
        action=super().act(proxy,target,motor)
        previous=self.predicted_gimbal.copy()
        self.predicted_gimbal+=(1-np.exp(-self.dt/.02))*(action[:2]*.3-self.predicted_gimbal)
        self.predicted_rate=(self.predicted_gimbal-previous)/self.dt
        return action
