"""Robot manifest: what a URDF or MJCF file cannot say about a flapping robot.

A manifest is a small YAML or JSON document that lives next to the model file.
It names the torso and the wings, gives each wing its aerodynamic frame, its
stroke and feathering joints and its planform, and lists the model patches
(joint dynamics, equality constraints, mass overrides) that MuJoCo needs but
URDF cannot express.  :meth:`RobotManifest.from_file` parses and validates it
and :func:`flapsim.robot.load` turns it into a compiled MuJoCo model with
:class:`~flapsim.wings.Wing` objects attached.

Conventions
-----------
* Vectors are expressed in the frame of the body they belong to.
* ``frame.span`` points from the wing root to the tip along the feathering
  axis; ``frame.chord`` points from the leading edge (the hinge line) to the
  trailing edge.  FlapSim derives the surface normal from them, so the side
  of the hinge on which the wing lies is never ambiguous.
* ``sign`` on a joint reference converts the MuJoCo joint coordinate into the
  aerodynamic convention: ``psi = sign * qpos``.
* ``when`` on a patch restricts it to ``"fixed"`` or ``"free"`` base models;
  the default ``"always"`` applies it to both.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "flapsim.robot/1"
SIDE_SIGNS = {"left": 1.0, "right": -1.0}
WHEN_VALUES = ("always", "fixed", "free")
BASE_TYPES = ("fixed", "free")
PROFILES = ("rectangular", "elliptic", "polyline", "integrals")
INTEGRAL_NAMES = ("r00", "r11", "r22", "r33", "z_cp0", "z_cp1", "z_cp2", "z_rd")
OPTIONAL_INTEGRAL_NAMES = ("c4",)
EQUALITY_TYPES = ("joint", "weld", "connect")
INTEGRATORS = ("euler", "rk4", "implicit", "implicitfast")


class ManifestError(ValueError):
    """Raised when a robot manifest is malformed or contradicts itself."""


# --------------------------------------------------------------------------- #
# Parsing helpers                                                             #
# --------------------------------------------------------------------------- #


def _mapping(data: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise ManifestError(f"{where} must be a mapping, got {type(data).__name__}")
    return data


def _check_keys(data: Mapping[str, Any], allowed: Sequence[str], where: str) -> None:
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ManifestError(
            f"{where}: unknown field(s) {unknown}; allowed fields are {sorted(allowed)}"
        )


def _vector(value: Any, length: int | None, where: str) -> tuple[float, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ManifestError(f"{where} must be a list of numbers")
    try:
        out = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{where} must be a list of numbers") from exc
    if length is not None and len(out) != length:
        raise ManifestError(f"{where} must have {length} entries, got {len(out)}")
    if not all(np.isfinite(out)):
        raise ManifestError(f"{where} must be finite")
    return out


def _number(
    value: Any, where: str, *, positive: bool = False, nonnegative: bool = False
) -> float:
    if isinstance(value, bool):
        raise ManifestError(f"{where} must be a number")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{where} must be a number, got {value!r}") from exc
    if not np.isfinite(out):
        raise ManifestError(f"{where} must be finite")
    if positive and out <= 0.0:
        raise ManifestError(f"{where} must be positive, got {out}")
    if nonnegative and out < 0.0:
        raise ManifestError(f"{where} must be non-negative, got {out}")
    return out


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{where} must be a non-empty string")
    return value


def _choice(value: Any, choices: Sequence[str], where: str) -> str:
    if value not in choices:
        raise ManifestError(f"{where} must be one of {list(choices)}, got {value!r}")
    return value


def _sign(value: Any, where: str) -> float:
    out = _number(value, where)
    if out not in (-1.0, 1.0):
        raise ManifestError(f"{where} must be +1 or -1, got {out}")
    return out


def _when(data: Mapping[str, Any], where: str) -> str:
    return _choice(data.get("when", "always"), WHEN_VALUES, f"{where}.when")


# --------------------------------------------------------------------------- #
# Model patches                                                               #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class JointPatch:
    """Joint properties URDF cannot express (all SI, ``None`` leaves MuJoCo's)."""

    stiffness: float | None = None
    damping: float | None = None
    armature: float | None = None
    springref: float | None = None
    frictionloss: float | None = None
    range: tuple[float, float] | None = None
    solref_limit: tuple[float, float] | None = None
    solimp_limit: tuple[float, ...] | None = None
    when: str = "always"

    _KEYS = (
        "stiffness",
        "damping",
        "armature",
        "springref",
        "frictionloss",
        "range",
        "solref_limit",
        "solimp_limit",
        "when",
    )

    @classmethod
    def from_dict(cls, data: Any, where: str) -> JointPatch:
        data = _mapping(data, where)
        _check_keys(data, cls._KEYS, where)
        kwargs: dict[str, Any] = {}
        for key in ("stiffness", "damping", "armature", "frictionloss"):
            if key in data:
                kwargs[key] = _number(data[key], f"{where}.{key}", nonnegative=True)
        if "springref" in data:
            kwargs["springref"] = _number(data["springref"], f"{where}.springref")
        if "range" in data:
            lo, hi = _vector(data["range"], 2, f"{where}.range")
            if lo >= hi:
                raise ManifestError(f"{where}.range must be increasing")
            kwargs["range"] = (lo, hi)
        if "solref_limit" in data:
            kwargs["solref_limit"] = _vector(
                data["solref_limit"], 2, f"{where}.solref_limit"
            )
        if "solimp_limit" in data:
            kwargs["solimp_limit"] = _vector(
                data["solimp_limit"], 5, f"{where}.solimp_limit"
            )
        kwargs["when"] = _when(data, where)
        return cls(**kwargs)


@dataclass(frozen=True)
class BodyPatch:
    """Mass-property overrides for one body (or ``"*"`` for all others)."""

    mass: float | None = None
    com: tuple[float, float, float] | None = None
    fullinertia: tuple[float, ...] | None = None
    diaginertia: tuple[float, float, float] | None = None
    when: str = "always"

    _KEYS = ("mass", "com", "fullinertia", "diaginertia", "when")

    @classmethod
    def from_dict(cls, data: Any, where: str) -> BodyPatch:
        data = _mapping(data, where)
        _check_keys(data, cls._KEYS, where)
        kwargs: dict[str, Any] = {}
        if "mass" in data:
            kwargs["mass"] = _number(data["mass"], f"{where}.mass", positive=True)
        if "com" in data:
            kwargs["com"] = _vector(data["com"], 3, f"{where}.com")
        if "fullinertia" in data and "diaginertia" in data:
            raise ManifestError(f"{where}: give fullinertia or diaginertia, not both")
        if "fullinertia" in data:
            kwargs["fullinertia"] = _vector(
                data["fullinertia"], 6, f"{where}.fullinertia"
            )
        if "diaginertia" in data:
            kwargs["diaginertia"] = _vector(
                data["diaginertia"], 3, f"{where}.diaginertia"
            )
        kwargs["when"] = _when(data, where)
        return cls(**kwargs)


@dataclass(frozen=True)
class EqualityPatch:
    """An equality constraint between two joints or two bodies."""

    type: str
    targets: tuple[str, str]
    name: str | None = None
    polycoef: tuple[float, ...] = (0.0, 1.0, 0.0, 0.0, 0.0)
    anchor: tuple[float, float, float] | None = None
    solref: tuple[float, float] | None = None
    when: str = "always"

    _KEYS = (
        "type",
        "targets",
        "joints",
        "bodies",
        "name",
        "polycoef",
        "anchor",
        "solref",
        "when",
    )

    @classmethod
    def from_dict(cls, data: Any, where: str) -> EqualityPatch:
        data = _mapping(data, where)
        _check_keys(data, cls._KEYS, where)
        kind = _choice(data.get("type"), EQUALITY_TYPES, f"{where}.type")
        raw_targets = data.get("targets", data.get("joints", data.get("bodies")))
        if raw_targets is None:
            raise ManifestError(f"{where} needs 'targets' (two joint or body names)")
        if (
            isinstance(raw_targets, str)
            or not isinstance(raw_targets, Sequence)
            or len(raw_targets) != 2
        ):
            raise ManifestError(f"{where}.targets must list exactly two names")
        targets = (_string(raw_targets[0], where), _string(raw_targets[1], where))
        kwargs: dict[str, Any] = {"type": kind, "targets": targets}
        if "name" in data:
            kwargs["name"] = _string(data["name"], f"{where}.name")
        if "polycoef" in data:
            coef = _vector(data["polycoef"], None, f"{where}.polycoef")
            if not 1 <= len(coef) <= 5:
                raise ManifestError(f"{where}.polycoef takes 1 to 5 coefficients")
            kwargs["polycoef"] = coef + (0.0,) * (5 - len(coef))
        if "anchor" in data:
            kwargs["anchor"] = _vector(data["anchor"], 3, f"{where}.anchor")
        if kind == "connect" and "anchor" not in data:
            raise ManifestError(f"{where}: a connect equality needs an anchor")
        if "solref" in data:
            kwargs["solref"] = _vector(data["solref"], 2, f"{where}.solref")
        kwargs["when"] = _when(data, where)
        return cls(**kwargs)


@dataclass(frozen=True)
class ModelPatches:
    joints: dict[str, JointPatch] = field(default_factory=dict)
    bodies: dict[str, BodyPatch] = field(default_factory=dict)
    equalities: tuple[EqualityPatch, ...] = ()
    exclude_self_collision: bool = False
    exclude_pairs: tuple[tuple[str, str], ...] = ()

    _KEYS = (
        "joints",
        "bodies",
        "equalities",
        "exclude_self_collision",
        "exclude_pairs",
    )

    @classmethod
    def from_dict(cls, data: Any, where: str) -> ModelPatches:
        data = _mapping(data, where)
        _check_keys(data, cls._KEYS, where)
        joints = {
            _string(name, f"{where}.joints"): JointPatch.from_dict(
                patch, f"{where}.joints.{name}"
            )
            for name, patch in _mapping(
                data.get("joints", {}), f"{where}.joints"
            ).items()
        }
        bodies = {
            _string(name, f"{where}.bodies"): BodyPatch.from_dict(
                patch, f"{where}.bodies.{name}"
            )
            for name, patch in _mapping(
                data.get("bodies", {}), f"{where}.bodies"
            ).items()
        }
        raw_eq = data.get("equalities", [])
        if not isinstance(raw_eq, Sequence) or isinstance(raw_eq, str):
            raise ManifestError(f"{where}.equalities must be a list")
        equalities = tuple(
            EqualityPatch.from_dict(item, f"{where}.equalities[{index}]")
            for index, item in enumerate(raw_eq)
        )
        exclude = data.get("exclude_self_collision", False)
        if not isinstance(exclude, bool):
            raise ManifestError(f"{where}.exclude_self_collision must be true or false")
        raw_pairs = data.get("exclude_pairs", [])
        pairs = []
        for index, pair in enumerate(raw_pairs):
            if isinstance(pair, str) or len(pair) != 2:
                raise ManifestError(
                    f"{where}.exclude_pairs[{index}] must list two bodies"
                )
            pairs.append((_string(pair[0], where), _string(pair[1], where)))
        return cls(joints, bodies, equalities, exclude, tuple(pairs))


# --------------------------------------------------------------------------- #
# Model section                                                               #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BaseSpec:
    type: str = "fixed"
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    joint_name: str = "floating_base"

    _KEYS = ("type", "position", "quaternion", "joint_name")

    @classmethod
    def from_dict(cls, data: Any, where: str) -> BaseSpec:
        data = _mapping(data, where)
        _check_keys(data, cls._KEYS, where)
        kwargs: dict[str, Any] = {
            "type": _choice(data.get("type", "fixed"), BASE_TYPES, f"{where}.type")
        }
        if "position" in data:
            kwargs["position"] = _vector(data["position"], 3, f"{where}.position")
        if "quaternion" in data:
            quat = np.array(_vector(data["quaternion"], 4, f"{where}.quaternion"))
            norm = float(np.linalg.norm(quat))
            if norm < 1e-9:
                raise ManifestError(f"{where}.quaternion must be non-zero")
            kwargs["quaternion"] = tuple(float(v) for v in quat / norm)
        if "joint_name" in data:
            kwargs["joint_name"] = _string(data["joint_name"], f"{where}.joint_name")
        return cls(**kwargs)


@dataclass(frozen=True)
class ModelSpec:
    file: Path | None = None
    root_body: str | None = None
    base: BaseSpec = field(default_factory=BaseSpec)
    timestep: float | None = None
    integrator: str | None = None
    patches: ModelPatches = field(default_factory=ModelPatches)

    _KEYS = ("file", "root_body", "base", "timestep", "integrator", "patches")

    @classmethod
    def from_dict(cls, data: Any, where: str, base_dir: Path | None) -> ModelSpec:
        data = _mapping(data, where)
        _check_keys(data, cls._KEYS, where)
        kwargs: dict[str, Any] = {}
        if "file" in data:
            path = Path(_string(data["file"], f"{where}.file")).expanduser()
            if not path.is_absolute() and base_dir is not None:
                path = base_dir / path
            kwargs["file"] = path.resolve()
        if "root_body" in data:
            kwargs["root_body"] = _string(data["root_body"], f"{where}.root_body")
        if "base" in data:
            kwargs["base"] = BaseSpec.from_dict(data["base"], f"{where}.base")
        if "timestep" in data:
            kwargs["timestep"] = _number(
                data["timestep"], f"{where}.timestep", positive=True
            )
        if "integrator" in data:
            kwargs["integrator"] = _choice(
                data["integrator"], INTEGRATORS, f"{where}.integrator"
            )
        if "patches" in data:
            kwargs["patches"] = ModelPatches.from_dict(
                data["patches"], f"{where}.patches"
            )
        return cls(**kwargs)


# --------------------------------------------------------------------------- #
# Wings                                                                       #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class JointRef:
    name: str
    sign: float = 1.0

    @classmethod
    def from_dict(cls, data: Any, where: str) -> JointRef:
        if isinstance(data, str):
            return cls(_string(data, where))
        data = _mapping(data, where)
        _check_keys(data, ("name", "sign"), where)
        return cls(
            _string(data.get("name"), f"{where}.name"),
            _sign(data.get("sign", 1.0), f"{where}.sign"),
        )


@dataclass(frozen=True)
class Planform:
    """Wing planform: how to obtain the blade-element integrals.

    ``profile`` selects the source: an analytic shape (``rectangular`` or
    ``elliptic``, needs ``length`` and ``mean_chord``), a chord ``polyline``
    of ``(r, c)`` pairs in metres from root to tip, or explicit ``integrals``.
    ``pitch_axis_fraction`` is the chordwise position of the feathering axis
    measured from the leading edge as a fraction of the chord (hinge inside
    the plate); ``leading_edge_offset`` is the distance in metres from the
    feathering axis to the leading edge when the plate starts beyond the
    hinge (a root spar).  Use one or the other.
    """

    profile: str = "rectangular"
    length: float | None = None
    mean_chord: float | None = None
    chord_polyline: tuple[tuple[float, float], ...] = ()
    integrals: dict[str, float] | None = None
    pitch_axis_fraction: float = 0.0
    leading_edge_offset: float = 0.0

    _KEYS = (
        "profile",
        "length",
        "mean_chord",
        "chord_polyline",
        "integrals",
        "pitch_axis_fraction",
        "leading_edge_offset",
    )

    @classmethod
    def from_dict(cls, data: Any, where: str) -> Planform:
        data = _mapping(data, where)
        _check_keys(data, cls._KEYS, where)
        profile = _choice(
            data.get("profile", "rectangular"), PROFILES, f"{where}.profile"
        )
        kwargs: dict[str, Any] = {"profile": profile}
        if "length" in data:
            kwargs["length"] = _number(data["length"], f"{where}.length", positive=True)
        if "mean_chord" in data:
            kwargs["mean_chord"] = _number(
                data["mean_chord"], f"{where}.mean_chord", positive=True
            )
        if "chord_polyline" in data:
            points = []
            for index, point in enumerate(data["chord_polyline"]):
                r, c = _vector(point, 2, f"{where}.chord_polyline[{index}]")
                points.append((r, c))
            kwargs["chord_polyline"] = tuple(points)
        if "integrals" in data:
            raw = _mapping(data["integrals"], f"{where}.integrals")
            _check_keys(
                raw, INTEGRAL_NAMES + OPTIONAL_INTEGRAL_NAMES, f"{where}.integrals"
            )
            missing = sorted(set(INTEGRAL_NAMES) - set(raw))
            if missing:
                raise ManifestError(f"{where}.integrals is missing {missing}")
            kwargs["integrals"] = {
                name: _number(raw[name], f"{where}.integrals.{name}")
                for name in INTEGRAL_NAMES + OPTIONAL_INTEGRAL_NAMES
                if name in raw
            }
        if "pitch_axis_fraction" in data:
            frac = _number(data["pitch_axis_fraction"], f"{where}.pitch_axis_fraction")
            if not 0.0 <= frac <= 1.0:
                raise ManifestError(f"{where}.pitch_axis_fraction must lie in [0, 1]")
            kwargs["pitch_axis_fraction"] = frac
        if "leading_edge_offset" in data:
            kwargs["leading_edge_offset"] = _number(
                data["leading_edge_offset"],
                f"{where}.leading_edge_offset",
                nonnegative=True,
            )
        if kwargs.get("pitch_axis_fraction", 0.0) and kwargs.get(
            "leading_edge_offset", 0.0
        ):
            raise ManifestError(
                f"{where}: give pitch_axis_fraction or leading_edge_offset, not both"
            )
        planform = cls(**kwargs)
        planform.validate(where)
        return planform

    def validate(self, where: str = "planform") -> None:
        if self.profile in ("rectangular", "elliptic", "integrals"):
            if self.length is None or self.mean_chord is None:
                raise ManifestError(
                    f"{where}: profile {self.profile!r} needs length and mean_chord"
                )
        if self.profile == "integrals" and self.integrals is None:
            raise ManifestError(
                f"{where}: profile 'integrals' needs the integrals block"
            )
        if self.profile == "polyline" and len(self.chord_polyline) < 2:
            raise ManifestError(
                f"{where}: profile 'polyline' needs at least two points"
            )


@dataclass(frozen=True)
class WingFrame:
    """Aerodynamic frame of a wing in its body frame, or the name of a site.

    ``span`` points from root to tip along the feathering axis and ``chord``
    from the leading edge (hinge line) to the trailing edge.  For a site,
    the site's ``+X`` is the chord direction and ``+Y`` the span direction.
    """

    span: tuple[float, float, float] | None = None
    chord: tuple[float, float, float] | None = None
    site: str | None = None

    @classmethod
    def from_dict(cls, data: Any, where: str) -> WingFrame:
        data = _mapping(data, where)
        _check_keys(data, ("span", "chord", "normal", "site"), where)
        if "site" in data:
            if len(data) > 1:
                raise ManifestError(f"{where}: give either a site or span/chord axes")
            return cls(site=_string(data["site"], f"{where}.site"))
        if "span" not in data or "chord" not in data:
            raise ManifestError(f"{where} needs 'span' and 'chord' (or a 'site')")
        span = np.array(_vector(data["span"], 3, f"{where}.span"))
        chord = np.array(_vector(data["chord"], 3, f"{where}.chord"))
        if np.linalg.norm(span) < 1e-12 or np.linalg.norm(chord) < 1e-12:
            raise ManifestError(f"{where}: span and chord must be non-zero")
        span = span / np.linalg.norm(span)
        chord = chord / np.linalg.norm(chord)
        if abs(float(span @ chord)) > 1e-6:
            raise ManifestError(f"{where}: span and chord must be perpendicular")
        if "normal" in data:
            normal = np.array(_vector(data["normal"], 3, f"{where}.normal"))
            normal = normal / np.linalg.norm(normal)
            if abs(abs(float(normal @ np.cross(span, chord))) - 1.0) > 1e-6:
                raise ManifestError(
                    f"{where}: normal must be perpendicular to span and chord "
                    "(it is derived from them; listing it only checks consistency)"
                )
        return cls(
            span=tuple(float(v) for v in span), chord=tuple(float(v) for v in chord)
        )


@dataclass(frozen=True)
class WingMass:
    """Mass properties of a wing body, given in wing coordinates.

    Wing coordinates are ``(normal, span, chord)`` with span positive from
    root to tip and chord positive from the leading to the trailing edge, so
    mirrored wings share the same numbers.

    * ``type: thin_plate``: a uniform plate covering ``length x mean_chord``.
    * ``type: spar_plate``: a leading-edge spar (rod along the span) holding
      ``spar_fraction`` of the mass plus a uniform membrane plate holding the
      rest.  This is the usual mass distribution of a membrane wing and the
      one that gives physically correct passive feathering.
    * ``type: explicit``: ``com`` and ``diaginertia`` about the wing axes
      through the centre of mass.
    """

    mass: float
    type: str = "spar_plate"
    spar_fraction: float = 0.8
    com: tuple[float, float, float] | None = None
    diaginertia: tuple[float, float, float] | None = None

    _KEYS = ("type", "mass", "spar_fraction", "com", "diaginertia")
    TYPES = ("thin_plate", "spar_plate", "explicit")

    @classmethod
    def from_dict(cls, data: Any, where: str) -> WingMass:
        data = _mapping(data, where)
        _check_keys(data, cls._KEYS, where)
        kind = _choice(data.get("type", "spar_plate"), cls.TYPES, f"{where}.type")
        mass = _number(data.get("mass"), f"{where}.mass", positive=True)
        kwargs: dict[str, Any] = {"mass": mass, "type": kind}
        if kind == "explicit":
            if "com" not in data or "diaginertia" not in data:
                raise ManifestError(f"{where}: explicit mass needs com and diaginertia")
            kwargs["com"] = _vector(data["com"], 3, f"{where}.com")
            diag = _vector(data["diaginertia"], 3, f"{where}.diaginertia")
            if min(diag) <= 0.0:
                raise ManifestError(f"{where}.diaginertia must be positive")
            a, b, c = sorted(diag)
            if a + b < c:
                raise ManifestError(
                    f"{where}.diaginertia violates the triangle inequality "
                    f"(two moments must sum to at least the third): {list(diag)}"
                )
            kwargs["diaginertia"] = diag
        elif "com" in data or "diaginertia" in data:
            raise ManifestError(f"{where}: {kind} takes no com/diaginertia")
        if "spar_fraction" in data:
            if kind != "spar_plate":
                raise ManifestError(
                    f"{where}: spar_fraction applies to spar_plate only"
                )
            fraction = _number(data["spar_fraction"], f"{where}.spar_fraction")
            if not 0.0 <= fraction < 1.0:
                raise ManifestError(f"{where}.spar_fraction must lie in [0, 1)")
            kwargs["spar_fraction"] = fraction
        return cls(**kwargs)


@dataclass(frozen=True)
class WingSpec:
    name: str
    body: str
    planform: Planform
    side: str | None = None
    frame: WingFrame | None = None
    stroke_joint: JointRef | None = None
    feathering_joint: JointRef | None = None
    mirror_of: str | None = None
    shoulder_offset: float = 0.0
    stroke_plane_offset: float = 0.0
    mass_model: WingMass | None = None

    _KEYS = (
        "name",
        "body",
        "side",
        "frame",
        "stroke_joint",
        "feathering_joint",
        "planform",
        "mirror_of",
        "shoulder_offset",
        "stroke_plane_offset",
        "mass_model",
    )

    @property
    def side_sign(self) -> float:
        return SIDE_SIGNS.get(self.side or "left", 1.0)

    @classmethod
    def from_dict(cls, data: Any, where: str) -> WingSpec:
        data = _mapping(data, where)
        _check_keys(data, cls._KEYS, where)
        name = _string(data.get("name"), f"{where}.name")
        where = f"{where}({name})"
        kwargs: dict[str, Any] = {
            "name": name,
            "body": _string(data.get("body"), f"{where}.body"),
            "planform": Planform.from_dict(
                data.get("planform", {}), f"{where}.planform"
            ),
        }
        if "side" in data:
            kwargs["side"] = _choice(data["side"], tuple(SIDE_SIGNS), f"{where}.side")
        if "frame" in data:
            kwargs["frame"] = WingFrame.from_dict(data["frame"], f"{where}.frame")
        for key in ("stroke_joint", "feathering_joint"):
            if key in data and data[key] is not None:
                kwargs[key] = JointRef.from_dict(data[key], f"{where}.{key}")
        if "mirror_of" in data:
            kwargs["mirror_of"] = _string(data["mirror_of"], f"{where}.mirror_of")
        for key in ("shoulder_offset", "stroke_plane_offset"):
            if key in data:
                kwargs[key] = _number(data[key], f"{where}.{key}")
        if "mass_model" in data and data["mass_model"] is not None:
            kwargs["mass_model"] = WingMass.from_dict(
                data["mass_model"], f"{where}.mass_model"
            )
        if kwargs.get("frame") is None and kwargs.get("mirror_of") is None:
            raise ManifestError(f"{where} needs a frame or mirror_of")
        if kwargs.get("frame") is not None and kwargs.get("mirror_of") is not None:
            raise ManifestError(f"{where}: give a frame or mirror_of, not both")
        return cls(**kwargs)


@dataclass(frozen=True)
class TorsoSpec:
    body: str

    @classmethod
    def from_dict(cls, data: Any, where: str) -> TorsoSpec:
        if isinstance(data, str):
            return cls(_string(data, where))
        data = _mapping(data, where)
        _check_keys(data, ("body",), where)
        return cls(_string(data.get("body"), f"{where}.body"))


# --------------------------------------------------------------------------- #
# Manifest                                                                    #
# --------------------------------------------------------------------------- #


def _plain(value: Any) -> Any:
    """Convert paths and tuples to JSON/YAML friendly values, dropping ``None``."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple | list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items() if v is not None}
    return value


@dataclass(frozen=True)
class RobotManifest:
    name: str
    model: ModelSpec = field(default_factory=ModelSpec)
    wings: tuple[WingSpec, ...] = ()
    torso: TorsoSpec | None = None
    aliases: tuple[str, ...] = ()
    air_density: float = 1.225
    extras: dict[str, Any] = field(default_factory=dict)
    source: Path | None = None
    root: Path | None = None
    schema: str = SCHEMA

    _KEYS = (
        "schema",
        "name",
        "aliases",
        "model",
        "torso",
        "wings",
        "air_density",
        "extras",
    )

    def __post_init__(self) -> None:
        names = [wing.name for wing in self.wings]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ManifestError(f"duplicate wing names: {duplicates}")
        for wing in self.wings:
            if wing.mirror_of is not None:
                if wing.mirror_of == wing.name or wing.mirror_of not in names:
                    raise ManifestError(
                        f"wing {wing.name!r}: mirror_of must name another wing"
                    )
                source = self.wing(wing.mirror_of)
                if source.mirror_of is not None:
                    raise ManifestError(
                        f"wing {wing.name!r}: mirror_of must point at a wing "
                        "with a frame"
                    )

    # -- construction -------------------------------------------------------

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        base_dir: str | Path | None = None,
        source: str | Path | None = None,
    ) -> RobotManifest:
        data = _mapping(data, "manifest")
        _check_keys(data, cls._KEYS, "manifest")
        schema = data.get("schema", SCHEMA)
        if schema != SCHEMA:
            raise ManifestError(
                f"unsupported manifest schema {schema!r}; expected {SCHEMA!r}"
            )
        base = Path(base_dir).resolve() if base_dir is not None else None
        kwargs: dict[str, Any] = {
            "name": _string(data.get("name"), "manifest.name"),
            "schema": schema,
            "source": Path(source).resolve() if source is not None else None,
            "root": base,
        }
        if "aliases" in data:
            raw = data["aliases"]
            if isinstance(raw, str) or not isinstance(raw, Sequence):
                raise ManifestError("manifest.aliases must be a list of names")
            kwargs["aliases"] = tuple(_string(a, "manifest.aliases") for a in raw)
        if "model" in data:
            kwargs["model"] = ModelSpec.from_dict(data["model"], "manifest.model", base)
        if "torso" in data:
            kwargs["torso"] = TorsoSpec.from_dict(data["torso"], "manifest.torso")
        raw_wings = data.get("wings", [])
        if isinstance(raw_wings, str) or not isinstance(raw_wings, Sequence):
            raise ManifestError("manifest.wings must be a list")
        kwargs["wings"] = tuple(
            WingSpec.from_dict(item, f"manifest.wings[{index}]")
            for index, item in enumerate(raw_wings)
        )
        if "air_density" in data:
            kwargs["air_density"] = _number(
                data["air_density"], "manifest.air_density", positive=True
            )
        if "extras" in data:
            kwargs["extras"] = dict(_mapping(data["extras"], "manifest.extras"))
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: str | Path) -> RobotManifest:
        path = Path(path).expanduser().resolve()
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
        if path.suffix.lower() == ".json":
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"invalid JSON in {path}: {exc}") from exc
        else:
            import yaml

            try:
                data = yaml.safe_load(text)
            except yaml.YAMLError as exc:
                raise ManifestError(f"invalid YAML in {path}: {exc}") from exc
        if data is None:
            raise ManifestError(f"manifest {path} is empty")
        manifest = cls.from_dict(data, base_dir=path.parent, source=path)
        if manifest.model.file is None:
            raise ManifestError(f"manifest {path} has no model.file")
        if not manifest.model.file.is_file():
            raise ManifestError(f"model file does not exist: {manifest.model.file}")
        return manifest

    @classmethod
    def for_model_file(cls, path: str | Path) -> RobotManifest:
        """A wing-less manifest for a bare URDF or MJCF file."""
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise ManifestError(f"model file does not exist: {path}")
        return cls(name=path.stem, model=ModelSpec(file=path), source=path)

    # -- queries ------------------------------------------------------------

    @property
    def names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)

    @property
    def wing_names(self) -> tuple[str, ...]:
        return tuple(wing.name for wing in self.wings)

    def wing(self, name: str) -> WingSpec:
        for wing in self.wings:
            if wing.name == name:
                return wing
        raise KeyError(f"no wing named {name!r}; wings are {list(self.wing_names)}")

    @property
    def base_dir(self) -> Path | None:
        if self.root is not None:
            return self.root
        if self.source is not None:
            return self.source.parent
        if self.model.file is not None:
            return self.model.file.parent
        return None

    def resolve(self, path: str | Path) -> Path:
        """Resolve a path from ``extras`` relative to the manifest location."""
        candidate = Path(path).expanduser()
        if candidate.is_absolute() or self.base_dir is None:
            return candidate
        return (self.base_dir / candidate).resolve()

    def with_base(self, base_type: str) -> RobotManifest:
        base_type = _choice(base_type, BASE_TYPES, "base")
        return replace(
            self,
            model=replace(self.model, base=replace(self.model.base, type=base_type)),
        )

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("source", None)
        data.pop("root", None)
        return _plain(data)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        data = self.to_dict()
        if path.suffix.lower() == ".json":
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        else:
            import yaml

            path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return path


__all__ = [
    "SCHEMA",
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
    "TorsoSpec",
    "WingFrame",
    "WingMass",
    "WingSpec",
]
