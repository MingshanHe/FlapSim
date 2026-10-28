"""Robot description and import: manifests, model patches, loaders, builders."""

from .builders import flat_plate_flapper, flat_plate_simulation
from .inspect import describe_model, model_summary
from .loader import RobotModel, builtin_robots, load, resolve_manifest
from .manifest import (
    BaseSpec,
    BodyPatch,
    EqualityPatch,
    JointPatch,
    JointRef,
    ManifestError,
    ModelPatches,
    ModelSpec,
    Planform,
    RobotManifest,
    TorsoSpec,
    WingFrame,
    WingMass,
    WingSpec,
)

__all__ = [
    "BaseSpec",
    "BodyPatch",
    "EqualityPatch",
    "JointPatch",
    "JointRef",
    "ManifestError",
    "ModelPatches",
    "ModelSpec",
    "Planform",
    "RobotManifest",
    "RobotModel",
    "TorsoSpec",
    "WingFrame",
    "WingMass",
    "WingSpec",
    "builtin_robots",
    "describe_model",
    "flat_plate_flapper",
    "flat_plate_simulation",
    "load",
    "model_summary",
    "resolve_manifest",
]
