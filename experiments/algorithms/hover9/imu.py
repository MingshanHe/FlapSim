"""Causal virtual IMU preintegration; positions are estimates, never simulator reads."""
import ctypes as ct
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import mujoco
import numpy as np
from flapsim.kinematics import euler_matrix
from flapsim.rl.native import NativeCPUBackend,Motor,D,pointer


class ImuState(ct.Structure):
    _fields_=[('site',ct.c_int),('acceleration_sensor',ct.c_int),('gyro_sensor',ct.c_int),
              ('acceleration_bias',ct.c_double*3),('attitude_error',ct.c_double*9),
              ('acceleration',ct.c_double*3),('velocity',ct.c_double*3),
              ('position',ct.c_double*3),('specific_force',ct.c_double*3),
              ('gyro',ct.c_double*3),('attitude',ct.c_double*3),('tau',ct.c_double),('attitude_tau',ct.c_double)]


class ImuBackend(NativeCPUBackend):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        source=Path(__file__).with_name('imu_step.cpp')
        original=Path(__file__).resolve().parents[3]/'flapsim/rl/native_step.cpp'
        package=Path(mujoco.__file__).parent; binary=next(package.glob('libmujoco.so.*'))
        key=hashlib.sha256(source.read_bytes()+original.read_bytes()+str(binary).encode()).hexdigest()[:20]
        cache=Path.home()/'.cache/flapsim';cache.mkdir(parents=True,exist_ok=True)
        target=cache/f'imu-{key}.so'
        if not target.exists():
            fd,temporary=tempfile.mkstemp(suffix='.so',dir=cache);os.close(fd)
            try:
                subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(source),
                    '-I'+str(package/'include'),str(binary),'-Wl,-rpath,'+str(package),'-o',temporary],check=True,capture_output=True)
                os.replace(temporary,target)
            finally:Path(temporary).unlink(missing_ok=True)
        lib=ct.CDLL(str(target))
        lib.fs_advance_imu.argtypes=self.lib.fs_advance.argtypes+[ct.POINTER(ImuState)]
        lib.fs_advance_imu.restype=None;self.imu_lib=lib
        self.imu=ImuState();self.reset_imu()

    def reset_imu(self,bias=(0.,0.,0.),attitude_error=(0.,0.,0.),tau=.002,attitude_tau=.01):
        self.imu=ImuState()
        self.imu.site=self.model.site('history_imu').id
        self.imu.acceleration_sensor=self.model.sensor('history_accel').id
        self.imu.gyro_sensor=self.model.sensor('history_gyro').id
        self.imu.acceleration_bias[:]=bias
        self.imu.attitude_error[:]=euler_matrix(attitude_error).ravel()
        self.imu.tau=tau
        self.imu.attitude_tau=attitude_tau
        r=self.data.site_xmat[self.imu.site].reshape(3,3)
        self.imu.specific_force[:]=r.T@np.array([0.,0.,9.81])
        from ..neat_imu.environment import rotation_to_rpy
        self.imu.attitude[:]=rotation_to_rpy(r)
        self.imu.gyro[:]=self.data.sensor('history_gyro').data

    def advance(self,steps,command,joint_forces,body_forces,controls):
        c=self.fluid.config
        arrays=[np.ascontiguousarray(a,dtype=np.float64) for a in (joint_forces,body_forces,controls)]
        self.imu_lib.fs_advance_imu(self.model._address,self.data._address,steps,
            ct.byref(self.motor),float(command[0]),self.voltage_mode,self.wings,len(self.wings),
            c.density,c.chord_drag,c.rotational_drag,pointer(self.wind),*(pointer(a) for a in arrays),ct.byref(self.imu))


def attach(sim):
    original=sim.backend
    backend=ImuBackend(sim.model,sim.actuator,sim.controller,sim.fluid,controls=sim._gimbal_command)
    mujoco.mj_copyData(backend.data,sim.model,original.data)
    backend.controller_state=original.controller_state
    sim.backend=backend
    backend.reset_imu()
    return backend
