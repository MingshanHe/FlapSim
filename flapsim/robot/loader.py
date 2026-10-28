"""Load a robot from a manifest, a bare URDF/MJCF file, or an ``MjSpec``."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import mujoco

from flapsim.wings import Wing

from .manifest import ManifestError, ModelSpec, RobotManifest
from .patches import apply_model_spec
from .scene import add_scene

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets" / "robots"
MODEL_SUFFIXES = (".urdf", ".xml", ".mjcf")
MANIFEST_SUFFIXES = (".yaml", ".yml", ".json")


@lru_cache(maxsize=1)
def builtin_robots() -> dict[str, Path]:
    """Map every built-in robot name and alias to its manifest file."""
    robots: dict[str, Path] = {}
    if not ASSETS_DIR.is_dir():
        return robots
    for manifest_path in sorted(ASSETS_DIR.glob("*/robot.yaml")):
        robots[manifest_path.parent.name] = manifest_path
        try:
            manifest = RobotManifest.from_file(manifest_path)
        except ManifestError:
            continue
        for name in manifest.names:
            robots.setdefault(name, manifest_path)
    return robots


def resolve_manifest(source: str | Path | RobotManifest) -> RobotManifest:
    """Accept a manifest, a manifest path, a model path or a built-in name."""
    if isinstance(source, RobotManifest):
        return source
    if not isinstance(source, str | Path):
        raise TypeError(f"cannot load a robot from {type(source).__name__}")
    text = str(source)
    path = Path(text).expanduser()
    suffix = path.suffix.lower()
    if path.is_file():
        if suffix in MANIFEST_SUFFIXES:
            return RobotManifest.from_file(path)
        if suffix in MODEL_SUFFIXES:
            return RobotManifest.for_model_file(path)
        raise ManifestError(
            f"{path} is neither a manifest ({MANIFEST_SUFFIXES}) nor a model file "
            f"({MODEL_SUFFIXES})"
        )
    if path.is_dir() and (path / "robot.yaml").is_file():
        return RobotManifest.from_file(path / "robot.yaml")
    robots = builtin_robots()
    if text in robots:
        return RobotManifest.from_file(robots[text])
    available = sorted({p.parent.name for p in robots.values()})
    raise ManifestError(
        f"cannot resolve robot {text!r}: not a manifest, a model file or a "
        f"built-in robot. Built-in robots: {available}"
    )


@dataclass
class RobotModel:
    """A compiled MuJoCo model with its manifest and wing objects."""

    manifest: RobotManifest
    spec: mujoco.MjSpec
    model: mujoco.MjModel
    data: mujoco.MjData
    root_body: int
    torso_body: int
    wings: list[Wing]
    base_type: str

    @property
    def fixed_base(self) -> bool:
        return self.base_type == "fixed"

    @property
    def root_body_name(self) -> str:
        return self.model.body(self.root_body).name

    @property
    def torso_body_name(self) -> str:
        return self.model.body(self.torso_body).name

    def wing(self, name: str) -> Wing:
        for wing in self.wings:
            if wing.name == name:
                return wing
        raise KeyError(
            f"no wing named {name!r}; wings are {[w.name for w in self.wings]}"
        )

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def geometry_warnings(self) -> list[str]:
        """Frame and joint-sign warnings for every wing at the current pose."""
        messages: list[str] = []
        for wing in self.wings:
            messages.extend(wing.geometry_warnings(self.model, self.data))
        return messages

    def __repr__(self) -> str:
        return (
            f"RobotModel({self.manifest.name!r}, base={self.base_type}, "
            f"nbody={self.model.nbody}, njnt={self.model.njnt}, "
            f"wings={len(self.wings)})"
        )


def build_wings(
    manifest: RobotManifest, model: mujoco.MjModel, data: mujoco.MjData
) -> list[Wing]:
    """Instantiate wings, sources before the wings that mirror them."""
    built: dict[str, Wing] = {}
    for spec in manifest.wings:
        if spec.mirror_of is None:
            built[spec.name] = Wing(spec, model, data, rho=manifest.air_density)
    for spec in manifest.wings:
        if spec.mirror_of is not None:
            built[spec.name] = Wing(
                spec,
                model,
                data,
                rho=manifest.air_density,
                mirror_source=built[spec.mirror_of],
            )
    return [built[spec.name] for spec in manifest.wings]


def load(
    source: str | Path | RobotManifest | mujoco.MjSpec,
    *,
    manifest: RobotManifest | None = None,
    base: str | None = None,
    position=None,
    timestep: float | None = None,
    scene: bool = True,
    check: bool = True,
) -> RobotModel:
    """Build a :class:`RobotModel`.

    ``source`` is a built-in robot name, a manifest file, a bare URDF/MJCF
    file, a :class:`RobotManifest`, or an :class:`mujoco.MjSpec` (then pass
    its ``manifest`` separately or get a wing-less default).  ``base``
    overrides the manifest's base type (``"fixed"`` or ``"free"``),
    ``position`` the spawn position and ``timestep`` the integrator step.
    With ``check`` the wing frames and joint signs are sanity-checked at the
    neutral pose and problems are reported as warnings.
    """
    if isinstance(source, mujoco.MjSpec):
        spec = source
        if manifest is None:
            manifest = RobotManifest(name=spec.modelname or "robot", model=ModelSpec())
    else:
        if manifest is not None:
            raise TypeError("pass `manifest` only together with an MjSpec")
        manifest = resolve_manifest(source)
        if manifest.model.file is None:
            raise ManifestError(f"manifest {manifest.name!r} has no model.file")
        spec = mujoco.MjSpec.from_file(str(manifest.model.file))

    base_type = base or manifest.model.base.type
    if base_type not in ("fixed", "free"):
        raise ManifestError(f"base must be 'fixed' or 'free', got {base_type!r}")
    model_spec = manifest.model
    if position is not None:
        model_spec = replace(
            model_spec,
            base=replace(model_spec.base, position=tuple(float(v) for v in position)),
        )
    if timestep is not None:
        model_spec = replace(model_spec, timestep=float(timestep))
    manifest = replace(manifest, model=model_spec)

    root = apply_model_spec(spec, manifest, base_type=base_type)
    if scene:
        add_scene(spec, target_body=root.name)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    if any(wing.mass_model is not None for wing in manifest.wings):
        # Wing mass models are given in wing coordinates; converting them
        # needs the neutral-pose frames, so compile once, patch, recompile.
        for wing in build_wings(manifest, model, data):
            properties = wing.body_mass_properties()
            if properties is None:
                continue
            mass, com, full = properties
            body = spec.body(wing.spec.body)
            body.mass = mass
            body.ipos = com.tolist()
            body.inertia = [0.0, 0.0, 0.0]
            body.fullinertia = full.tolist()
            body.explicitinertial = True
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

    root_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root.name))
    if manifest.torso is not None:
        torso_id = int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, manifest.torso.body)
        )
        if torso_id < 0:
            raise ManifestError(f"torso body {manifest.torso.body!r} not found")
    else:
        torso_id = root_id
    wings = build_wings(manifest, model, data)
    robot = RobotModel(manifest, spec, model, data, root_id, torso_id, wings, base_type)
    if check:
        for message in robot.geometry_warnings():
            warnings.warn(message, stacklevel=2)
    return robot


__all__ = [
    "ASSETS_DIR",
    "RobotModel",
    "build_wings",
    "builtin_robots",
    "load",
    "resolve_manifest",
]
