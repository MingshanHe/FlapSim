"""Optional C++ acceleration of the unchanged CPU physics loop (Linux/WSL via g++, macOS via clang++)."""
import ctypes as ct
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import mujoco
import numpy as np
from flapsim.backends import CPUBackend
from flapsim.actuator import DCMotor
from flapsim.controller import SpeedController
from flapsim.fluid.blade_element import BladeElement

D = ct.POINTER(ct.c_double)
V3 = ct.c_double * 3


class Wing(ct.Structure):
    _fields_ = [('body', ct.c_int), ('root', ct.c_int), ('count', ct.c_int),
                ('points', D), ('areas', D), ('normal', V3), ('chord', V3),
                ('span', V3), ('rotational_area', ct.c_double)]


class Motor(ct.Structure):
    _fields_ = [('dof', ct.c_int)] + [(key, ct.c_double) for key in
                ('resistance', 'kt', 'ke', 'i0', 'voltage', 'current', 'torque', 'velocity', 'gain')]


def pointer(a):
    return a.ctypes.data_as(D)


def library():
    if sys.platform not in ('linux', 'darwin'):
        raise RuntimeError('native physics requires Linux/WSL (g++) or macOS (clang++); disable native_physics otherwise')
    darwin = sys.platform == 'darwin'
    source = Path(__file__).with_name('native_step.cpp')
    package = Path(mujoco.__file__).parent
    binary = next(package.glob('libmujoco.*.dylib' if darwin else 'libmujoco.so.*'), None)
    if binary is None:
        raise RuntimeError(f'MuJoCo shared library not found in {package}')
    key = hashlib.sha256(source.read_bytes()+str(binary).encode()+mujoco.__version__.encode()).hexdigest()[:20]
    cache = Path(os.environ.get('XDG_CACHE_HOME', Path.home()/'.cache'))/'flapsim'
    cache.mkdir(parents=True, exist_ok=True)
    target = cache/f'native-{key}.so'
    if not target.exists():
        fd, temporary = tempfile.mkstemp(suffix='.so', dir=cache)
        os.close(fd)
        try:
            compiler = ['clang++', '-ffp-contract=off'] if darwin else ['g++']   # no FMA contraction: keep macOS numerics close to x86 g++
            subprocess.run(compiler+['-O3', '-std=c++17', '-shared', '-fPIC', str(source),
                            '-I'+str(package/'include'), str(binary),
                            '-Wl,-rpath,'+str(package), '-o', temporary], check=True, capture_output=True)
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    lib = ct.CDLL(str(target))
    if lib.fs_version() != mujoco.mj_version():
        raise RuntimeError('MuJoCo C/Python ABI mismatch')
    lib.fs_advance.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int, ct.POINTER(Motor),
                              ct.c_double, ct.c_int, ct.POINTER(Wing), ct.c_int,
                              ct.c_double, ct.c_double, ct.c_double, D, D, D, D]
    lib.fs_advance.restype = None
    return lib


class NativeCPUBackend(CPUBackend):
    def __init__(self, *args, voltage_mode=False, **kwargs):
        super().__init__(*args, **kwargs)
        if (not isinstance(self.actuator, DCMotor) or type(self.controller) is not SpeedController
                or type(self.fluid) is not BladeElement or len(self.actuator.dof_ids) != 1):
            raise ValueError('native physics supports one DC motor, SpeedController and BladeElement')
        self.lib = library()
        self.voltage_mode = voltage_mode
        c = self.actuator.config
        self.motor = Motor(int(self.actuator.dof_ids[0]), c.resistance, c.torque_constant,
                           c.back_emf, c.no_load_current, c.max_voltage, c.max_current,
                           c.max_torque, c.max_velocity, c.speed_gain)
        self.arrays = []
        records = []
        for wing, body, root, points, _, areas, rotational_area in self.fluid.elements:
            points, areas = np.ascontiguousarray(points), np.ascontiguousarray(areas)
            self.arrays.extend([points, areas])
            records.append(Wing(body, root, len(areas), pointer(points), pointer(areas),
                                V3(*wing.normal_axis), V3(*wing.chord_axis),
                                V3(*wing.span_axis), rotational_area))
        self.wings = (Wing*len(records))(*records)
        self.wind = np.asarray(self.fluid.config.wind, dtype=np.float64)

    def advance(self, steps, command, joint_forces, body_forces, controls):
        c = self.fluid.config
        arrays = [np.ascontiguousarray(a, dtype=np.float64) for a in
                  (joint_forces, body_forces, controls)]
        self.lib.fs_advance(self.model._address, self.data._address, steps,
                            ct.byref(self.motor), float(command[0]), self.voltage_mode,
                            self.wings, len(self.wings), c.density, c.chord_drag,
                            c.rotational_drag, pointer(self.wind), *(pointer(a) for a in arrays))


def accelerate(sim, *, voltage_mode=False):
    """Attach before stepping; preserve the full MjData and controller state."""
    original = sim.backend
    if type(original) is not CPUBackend:
        raise ValueError('acceleration requires the original CPU backend')
    sim.backend = NativeCPUBackend(sim.model, sim.actuator, sim.controller, sim.fluid,
                                   controls=sim._gimbal_command, voltage_mode=voltage_mode)
    mujoco.mj_copyData(sim.backend.data, sim.model, original.data)
    sim.backend.controller_state = original.controller_state
    return sim
