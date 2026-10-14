"""Split-step variant, verified separately against the original CPU integrator."""
import ctypes as ct
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import mujoco
from .native import NativeCPUBackend


def implementation_sha256():
    root = Path(__file__).parent
    return hashlib.sha256(b''.join((root/name).read_bytes() for name in
                                  ('native_fast.py', 'native_fast_step.cpp', 'native_step.cpp'))).hexdigest()


class FastCPUBackend(NativeCPUBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.model.opt.integrator not in (mujoco.mjtIntegrator.mjINT_EULER,
                                             mujoco.mjtIntegrator.mjINT_IMPLICITFAST):
            raise ValueError('split stepping requires Euler or implicitfast')
        package = Path(mujoco.__file__).parent
        binary = next(package.glob('libmujoco.so.*'))
        key = hashlib.sha256((implementation_sha256()+str(binary)+mujoco.__version__).encode()).hexdigest()[:20]
        cache = Path(os.environ.get('XDG_CACHE_HOME', Path.home()/'.cache'))/'flapsim'
        target = cache/f'split-{key}.so'
        if not target.exists():
            fd, temporary = tempfile.mkstemp(suffix='.so', dir=cache)
            os.close(fd)
            try:
                subprocess.run(['g++', '-O3', '-std=c++17', '-shared', '-fPIC',
                                str(Path(__file__).with_name('native_fast_step.cpp')),
                                '-I'+str(package/'include'), str(binary),
                                '-Wl,-rpath,'+str(package), '-o', temporary],
                               check=True, capture_output=True)
                os.replace(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)
        lib = ct.CDLL(str(target))
        if lib.fs_version() != mujoco.mj_version():
            raise RuntimeError('MuJoCo C/Python ABI mismatch')
        lib.fs_advance.argtypes = self.lib.fs_advance.argtypes
        lib.fs_advance.restype = None
        self.lib = lib


def accelerate_fast(sim):
    original = sim.backend
    if not isinstance(original, NativeCPUBackend):
        raise ValueError('enable native physics before split stepping')
    sim.backend = FastCPUBackend(sim.model, sim.actuator, sim.controller, sim.fluid,
                                  controls=sim._gimbal_command)
    mujoco.mj_copyData(sim.backend.data, sim.model, original.data)
    sim.backend.controller_state = original.controller_state
    return sim
