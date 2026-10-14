"""Finite-bandwidth virtual IMU and delayed scalar height sensor."""
import ctypes as ct
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import mujoco
import numpy as np
from flapsim.rl.native import NativeCPUBackend, pointer
from ..neat_imu.environment import rotation_to_rpy


class SensorState(ct.Structure):
    _fields_ = [(k,ct.c_int) for k in ('site','accel_id','gyro_id','counter','period')]
    _fields_ += [(k,ct.c_double) for k in ('accel_limit','gyro_limit','filter_tau','attitude_tau')]
    _fields_ += [(k,ct.c_double*3) for k in ('acc1','acc2','gyro1','gyro2','force','gyro','attitude','velocity','position','bias','angle_bias')]
    _fields_ += [(k,ct.c_longlong) for k in ('clipped_acc','clipped_gyro','samples')]


class SensorBackend(NativeCPUBackend):
    @property
    def imu(self):
        return self.sensor

    def reset_imu(self,bias=(0,0,0),attitude_error=(0,0,0),tau=.002,attitude_tau=.13045658515789563):
        self.reset_sensor(bias,attitude_error,attitude_tau)

    def __init__(self,sim):
        super().__init__(sim.model,sim.actuator,sim.controller,sim.fluid,controls=sim._gimbal_command)
        source=Path(__file__).with_name('sensor_step.cpp')
        native=Path(__file__).resolve().parents[3]/'flapsim/rl/native_step.cpp'
        package=Path(mujoco.__file__).parent; binary=next(package.glob('libmujoco.so.*'))
        key=hashlib.sha256(source.read_bytes()+native.read_bytes()+str(binary).encode()).hexdigest()[:20]
        cache=Path.home()/'.cache/flapsim'; cache.mkdir(parents=True,exist_ok=True)
        target=cache/f'topology-sensor-{key}.so'
        if not target.exists():
            fd,tmp=tempfile.mkstemp(suffix='.so',dir=cache);os.close(fd)
            try:
                subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(source),'-I'+str(package/'include'),str(binary),'-Wl,-rpath,'+str(package),'-o',tmp],check=True,capture_output=True)
                os.replace(tmp,target)
            finally: Path(tmp).unlink(missing_ok=True)
        self.sensor_lib=ct.CDLL(str(target))
        self.sensor_lib.fs_advance_sensor.argtypes=self.lib.fs_advance.argtypes+[ct.POINTER(SensorState)]
        self.sensor_lib.fs_advance_sensor.restype=None
        mujoco.mj_copyData(self.data,sim.model,sim.data)
        self.controller_state=sim.backend.controller_state
        self.reset_sensor()

    def reset_sensor(self,bias=(0,0,0),angle_bias=(0,0,0),attitude_tau=.13045658515789563,accel_g=32.):
        self.sensor=SensorState();s=self.sensor
        s.site=self.model.site('history_imu').id
        s.accel_id=self.model.sensor('history_accel').id;s.gyro_id=self.model.sensor('history_gyro').id
        s.period=round(.001/self.model.opt.timestep)
        s.accel_limit=accel_g*9.81;s.gyro_limit=np.deg2rad(2000.)
        s.filter_tau=1/(2*np.pi*80.);s.attitude_tau=attitude_tau
        R=self.data.site_xmat[s.site].reshape(3,3)
        gravity=R.T@np.array([0,0,9.81])
        s.acc1[:]=gravity;s.acc2[:]=gravity;s.force[:]=gravity
        s.attitude[:]=rotation_to_rpy(R)+np.asarray(angle_bias)
        s.bias[:]=bias;s.angle_bias[:]=angle_bias

    def advance(self,steps,command,joint_forces,body_forces,controls):
        c=self.fluid.config
        arrays=[np.ascontiguousarray(a,dtype=np.float64) for a in (joint_forces,body_forces,controls)]
        self.sensor_lib.fs_advance_sensor(self.model._address,self.data._address,steps,ct.byref(self.motor),float(command[0]),self.voltage_mode,self.wings,len(self.wings),c.density,c.chord_drag,c.rotational_drag,pointer(self.wind),*(pointer(a) for a in arrays),ct.byref(self.sensor))


class HeightSensor:
    """50 Hz nadir range, delayed 20 ms; alpha-beta observer uses no true velocity."""
    def __init__(self,initial_height,rng,noise=.002,delay=.02):
        self.h=float(initial_height);self.v=0.;self.origin=self.h
        self.rng=rng;self.noise=noise;self.delay=delay;self.pending=[];self.age=0.
    def step(self,time,true_z,true_cosine,measured_rpy,dt=.002):
        self.h+=self.v*dt;self.age+=dt
        if round(time/dt)%10==0 and true_cosine>.7:
            # The environment truth is used only to synthesize a physical sensor measurement.
            distance=true_z/true_cosine
            if .05<distance<2.:
                measured_range=distance+self.rng.normal(0,self.noise)
                z=measured_range*np.cos(measured_rpy[0])*np.cos(measured_rpy[1])
                self.pending.append((time+self.delay,z,self.delay))
        while self.pending and self.pending[0][0]<=time+1e-9:
            _,z,delay=self.pending.pop(0)
            innovation=z-(self.h-delay*self.v)
            self.h+=.30*innovation;self.v+=.035/.02*innovation;self.age=0.
        return self.h-self.origin,self.v
