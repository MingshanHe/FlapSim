"""Small, explicit configuration objects. All quantities use SI units."""
from dataclasses import dataclass, field
from pathlib import Path
import math


def positive(name, value, *, zero=False):
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f'{name} must be finite and {"non-negative" if zero else "positive"}')


@dataclass(frozen=True)
class SceneConfig:
    timestep: float = 5e-6
    fixed_base: bool = True
    position: tuple = (0.0, 0.0, 0.5)
    orientation: tuple | None = None  # wxyz; None preserves the model's mounting pose
    gravity: tuple = (0.0, 0.0, -9.81)
    ground: bool = True
    integrator: str = 'auto'

    def __post_init__(self):
        positive('timestep', self.timestep)
        if self.orientation is not None:
            if (len(self.orientation) != 4 or not all(math.isfinite(v) for v in self.orientation)
                    or not math.isclose(sum(v*v for v in self.orientation), 1.0, abs_tol=1e-6)):
                raise ValueError('orientation must be a unit wxyz quaternion')
        for name in ('position', 'gravity'):
            values = getattr(self, name)
            if len(values) != 3 or not all(math.isfinite(v) for v in values):
                raise ValueError(f'{name} must have three finite values')
        if self.integrator not in ('auto', 'euler', 'implicitfast', 'rk4'):
            raise ValueError('unknown integrator')


@dataclass(frozen=True)
class ActuatorConfig:
    kind: str = 'dcmotor'
    joints: tuple = ('motor_joint',)
    max_velocity: float = 2000.0
    max_torque: float = 0.002
    speed_gain: float = 2e-6
    resistance: float = 4.0
    torque_constant: float = 0.001
    back_emf: float = 0.001
    max_voltage: float = 5.0
    max_current: float = 2.0
    dea_coefficient: float = 1e-8
    stiffness: float = 0.0
    damping: float = 0.0

    def __post_init__(self):
        if self.kind not in ('dcmotor', 'bldcmotor', 'dea'):
            raise ValueError('actuator kind must be dcmotor, bldcmotor or dea')
        if not self.joints or len(set(self.joints)) != len(self.joints):
            raise ValueError('actuator joints must be nonempty and unique')
        for name in ('max_velocity', 'max_torque', 'speed_gain', 'resistance',
                     'torque_constant', 'max_voltage', 'max_current', 'dea_coefficient'):
            positive(name, getattr(self, name))
        for name in ('back_emf', 'stiffness', 'damping'):
            positive(name, getattr(self, name), zero=True)


@dataclass(frozen=True)
class WingConfig:
    body: str
    span: float = 0.083
    chord: float = 0.03325
    origin: tuple = (0.0, 0.0, 0.0)
    normal_axis: tuple = (1.0, 0.0, 0.0)
    span_axis: tuple = (0.0, 1.0, 0.0)
    chord_axis: tuple = (0.0, 0.0, 1.0)
    chord_offset: float = 0.015375

    def __post_init__(self):
        import numpy as np
        positive('span', self.span)
        positive('chord', self.chord)
        axes = np.asarray([self.normal_axis, self.span_axis, self.chord_axis])
        if axes.shape != (3, 3) or not np.allclose(axes @ axes.T, np.eye(3)):
            raise ValueError('wing axes must be orthonormal body-local vectors')
        if len(self.origin) != 3 or not np.isfinite(self.origin).all():
            raise ValueError('wing origin must contain three finite values')
        if not math.isfinite(self.chord_offset):
            raise ValueError('chord_offset must be finite')


@dataclass(frozen=True)
class FluidConfig:
    kind: str = 'blade_element'
    builtin_model: str = 'inertia'  # shared CPU/MJX model; ellipsoid is native-only
    density: float = 1.225
    viscosity: float = 1.81e-5
    wind: tuple = (0.0, 0.0, 0.0)
    strips: int = 12
    chord_drag: float = 0.08
    rotational_drag: float = 5.0
    wings: tuple = field(default_factory=lambda: (
        WingConfig('left_wing'), WingConfig('right_wing'),
    ))

    def __post_init__(self):
        if self.kind not in ('blade_element', 'mujoco', 'none'):
            raise ValueError('fluid kind must be blade_element, mujoco or none')
        if self.builtin_model not in ('inertia', 'ellipsoid'):
            raise ValueError('builtin_model must be inertia or ellipsoid')
        for name in ('density', 'viscosity', 'chord_drag', 'rotational_drag'):
            positive(name, getattr(self, name), zero=True)
        if not isinstance(self.strips, int) or isinstance(self.strips, bool) or self.strips < 1:
            raise ValueError('strips must be a positive integer')
        if len(self.wind) != 3 or not all(math.isfinite(v) for v in self.wind):
            raise ValueError('wind must have three finite values')
        if self.kind != 'none' and not self.wings:
            raise ValueError('at least one wing is required')
        if len({w.body for w in self.wings}) != len(self.wings):
            raise ValueError('wing bodies must be unique')


@dataclass(frozen=True)
class ControllerConfig:
    kind: str = 'speed'
    speed: tuple = (100.0,)
    target: tuple = (0.0, 0.0, 0.5, 0.0, 0.0, 0.0)
    kp: tuple = (1.0, 1.0, 2.0, 0.01, 0.01, 0.01)
    ki: tuple = (0.0, 0.0, 0.1, 0.0, 0.0, 0.0)
    kd: tuple = (0.5, 0.5, 0.5, 0.002, 0.002, 0.002)
    integral_limit: float = 0.5
    # Local calibration: delta shaft velocity = matrix @ body wrench.
    allocation: tuple | None = None

    def __post_init__(self):
        if self.kind not in ('speed', 'hover', 'move'):
            raise ValueError('controller kind must be speed, hover or move')
        for name in ('target', 'kp', 'ki', 'kd'):
            values = getattr(self, name)
            if len(values) != 6 or not all(math.isfinite(v) for v in values):
                raise ValueError(f'{name} must contain six finite values')
        if not self.speed or not all(math.isfinite(v) for v in self.speed):
            raise ValueError('speed must contain finite shaft velocities')
        positive('integral_limit', self.integral_limit, zero=True)


@dataclass(frozen=True)
class SimulationConfig:
    robot: str = 'IST_robot'
    model_path: str | Path | None = None
    root_body: str = 'base_link'
    backend: str = 'cpu'
    device: str = 'cpu'
    scene: SceneConfig = field(default_factory=SceneConfig)
    actuator: ActuatorConfig = field(default_factory=ActuatorConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    fluid: FluidConfig = field(default_factory=FluidConfig)

    def __post_init__(self):
        if self.backend not in ('cpu', 'mjx') or self.device not in ('cpu', 'gpu'):
            raise ValueError('backend must be cpu/mjx; device must be cpu/gpu')
        if self.backend == 'cpu' and self.device != 'cpu':
            raise ValueError('GPU requires backend="mjx"')
        if len(self.controller.speed) != len(self.actuator.joints):
            raise ValueError('one speed command is required per actuator joint')

    @property
    def path(self):
        if self.model_path is not None:
            return Path(self.model_path).expanduser().resolve()
        suffix = 'xml' if self.robot == 'IST_robot' else 'urdf'
        path = Path(__file__).parents[1] / 'robot' / self.robot / 'urdf' / f'{self.robot}.{suffix}'
        if not path.is_file():
            raise ValueError(f'unknown robot {self.robot!r}; provide model_path')
        return path
