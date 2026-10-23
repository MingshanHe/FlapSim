"""Unmodified 7/8 g physics with an explicit relative-start reference."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from ..neat_imu.compare_env import CompareEnv
from ..neat_imu.mass_model import make_variant
from ..neat_imu.train import write_json
from flapsim.rl.trajectories import Trajectory


def prepare(root, source):
    root,source=Path(root),Path(source)
    root.mkdir(parents=True,exist_ok=True)
    (root/'calibration').mkdir(exist_ok=True)
    for name in ('calibration/trim.json','phase_bank.npz'):
        shutil.copyfile(source/name,root/name)
    make_variant(root/'robot_upper7_lower8.xml',7.,8.)
    tree=ET.parse(root/'robot_upper7_lower8.xml'); model=tree.getroot()
    ET.SubElement(model.find(".//body[@name='base_link']"),'site',name='history_imu',pos='0 0 0',size='.0001')
    sensors=model.find('sensor')
    if sensors is None:sensors=ET.SubElement(model,'sensor')
    ET.SubElement(sensors,'accelerometer',name='history_accel',site='history_imu')
    ET.SubElement(sensors,'gyro',name='history_gyro',site='history_imu')
    tree.write(root/'robot_upper7_lower8.xml',encoding='utf-8',xml_declaration=True)
    write_json(root/'setup.json',dict(source=str(source),upper_g=7,lower_g=8,
        reference='Actual initial base_link position, used for reward/evaluation only',
        initial_velocity='base_link origin initially at rest; internal moving phase and angular momentum initialization retained',
        physical_outputs='alpha, beta +/-0.3 rad; motor 3461.360088271976 rad/s +/-10%'))


class Plant(CompareEnv):
    def reset(self,*,seed=None,options=None):
        obs,info=super().reset(seed=seed,options=options)
        data=self.sim.data
        data.qvel[self.free_dof:self.free_dof+3]=0.
        if self.case.get('initial_velocity',0.):
            rng=np.random.default_rng(np.random.SeedSequence([seed,9164]))
            data.qvel[self.free_dof:self.free_dof+3]=rng.uniform(-self.case['initial_velocity'],self.case['initial_velocity'],3)
        mujoco.mj_forward(self.sim.model,data)
        self.start_position=data.xpos[self.body].copy()
        self.reference=Trajectory('hover',tuple(self.start_position))
        self.previous_observation=self._observation().copy()
        obs=self.sensor(self.previous_observation)
        return obs,dict(info,start_position=self.start_position.tolist())


class PID:
    """Two-axis filtered attitude PID; plant retains the 20 ms servo target filter."""
    def __init__(self,parameters,reference=(0.,0.),dt=.01):
        self.p=np.asarray(parameters,dtype=float);self.reference=np.array(reference);self.dt=dt
    def reset(self,obs):
        self.angle=np.asarray(obs[:2])* .35
        self.rate=np.asarray(obs[3:5])*10.
        self.integral=np.zeros(2)
    def act(self,obs,target=None,motor=0.):
        target=self.reference if target is None else np.asarray(target)
        blend=1-np.exp(-self.dt/self.p[8])
        self.angle+=blend*(np.asarray(obs[:2])*.35-self.angle)
        self.rate+=blend*(np.asarray(obs[3:5])*10.-self.rate)
        error=np.clip(target-self.angle,-1.,1.)
        self.integral=np.clip(self.integral+error*self.dt,-.2,.2)
        command=self.p[:2]*error-self.p[2:4]*self.rate+self.p[4:6]*self.integral
        # parameters use roll/pitch order; physical actuators use alpha/beta.
        physical=np.array([command[1]+self.p[6],command[0]+self.p[7]])
        saturation=np.abs(physical)>.3
        self.integral[saturation[::-1]]-=error[saturation[::-1]]*self.dt
        return np.r_[np.clip(physical/.3,-1,1),np.clip(motor,-1,1)]
