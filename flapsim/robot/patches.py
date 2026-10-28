"""Apply a manifest's ``model`` section to a :class:`mujoco.MjSpec`."""

from __future__ import annotations

import mujoco
import numpy as np

from .manifest import BodyPatch, ManifestError, ModelPatches, RobotManifest

INTEGRATORS = {
    "euler": mujoco.mjtIntegrator.mjINT_EULER,
    "rk4": mujoco.mjtIntegrator.mjINT_RK4,
    "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}
EQUALITY_TYPES = {
    "joint": mujoco.mjtEq.mjEQ_JOINT,
    "weld": mujoco.mjtEq.mjEQ_WELD,
    "connect": mujoco.mjtEq.mjEQ_CONNECT,
}


def find_root_body(spec: mujoco.MjSpec, name: str | None = None) -> mujoco.MjsBody:
    """Return the robot's root body (a direct child of the world body)."""
    children = list(spec.worldbody.bodies)
    if name is not None:
        for body in children:
            if body.name == name:
                return body
        available = [body.name for body in children]
        raise ManifestError(
            f"root body {name!r} is not a top-level body; "
            f"top-level bodies are {available}"
        )
    if len(children) != 1:
        raise ManifestError(
            f"the model has {len(children)} top-level bodies; set model.root_body"
        )
    return children[0]


def set_scalar(obj: object, attribute: str, value: float) -> None:
    """Assign a scalar to an MjSpec field that may be stored as an array."""
    current = getattr(obj, attribute)
    if isinstance(current, np.ndarray) and current.size > 1:
        array = np.zeros_like(current)
        array[0] = value
        setattr(obj, attribute, array)
    else:
        setattr(obj, attribute, value)


def _active(when: str, base_type: str) -> bool:
    return when == "always" or when == base_type


def _joint(spec: mujoco.MjSpec, name: str) -> mujoco.MjsJoint:
    joint = spec.joint(name)
    if joint is None:
        raise ManifestError(
            f"joint {name!r} not found; joints are {[j.name for j in spec.joints]}"
        )
    return joint


def _body(spec: mujoco.MjSpec, name: str) -> mujoco.MjsBody:
    body = spec.body(name)
    if body is None:
        raise ManifestError(
            f"body {name!r} not found; bodies are {[b.name for b in spec.bodies]}"
        )
    return body


def _apply_body_patch(body: mujoco.MjsBody, patch: BodyPatch) -> None:
    if patch.mass is not None:
        body.mass = patch.mass
    if patch.com is not None:
        body.ipos = list(patch.com)
    # MuJoCo rejects a body that carries both inertia forms, so write the
    # patch in whichever form the body already uses (URDF imports use the
    # full tensor, MJCF bodies usually the diagonal).
    uses_full = bool(np.any(np.asarray(body.fullinertia) != 0.0))
    if patch.fullinertia is not None:
        full = list(patch.fullinertia)
        if uses_full or not np.any(np.asarray(body.inertia) != 0.0):
            body.fullinertia = full
        elif all(abs(v) < 1e-300 for v in full[3:]):
            body.inertia = full[:3]
        else:
            raise ManifestError(
                "cannot set a full inertia tensor with products of inertia on a body "
                "that defines a diagonal inertia; give 'diaginertia' instead"
            )
    if patch.diaginertia is not None:
        diag = list(patch.diaginertia)
        if uses_full:
            body.fullinertia = diag + [0.0, 0.0, 0.0]
        else:
            body.inertia = diag
    body.explicitinertial = True


def apply_patches(
    spec: mujoco.MjSpec,
    patches: ModelPatches,
    *,
    base_type: str,
    root_name: str,
) -> None:
    for name, patch in patches.joints.items():
        if not _active(patch.when, base_type):
            continue
        joint = _joint(spec, name)
        for attribute in (
            "stiffness",
            "damping",
            "armature",
            "springref",
            "frictionloss",
        ):
            value = getattr(patch, attribute)
            if value is not None:
                set_scalar(joint, attribute, value)
        if patch.range is not None:
            joint.range = list(patch.range)
            joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        if patch.solref_limit is not None:
            joint.solref_limit = list(patch.solref_limit)
        if patch.solimp_limit is not None:
            joint.solimp_limit = list(patch.solimp_limit)

    body_names = [body.name for body in spec.bodies if body.name != "world"]
    explicit = {name for name in patches.bodies if name != "*"}
    for name, patch in patches.bodies.items():
        if not _active(patch.when, base_type):
            continue
        if name == "*":
            targets = [n for n in body_names if n not in explicit and n != root_name]
        else:
            targets = [name]
        for target in targets:
            _apply_body_patch(_body(spec, target), patch)

    for index, equality in enumerate(patches.equalities):
        if not _active(equality.when, base_type):
            continue
        first, second = equality.targets
        if equality.type == "joint":
            _joint(spec, first)
            _joint(spec, second)
        else:
            _body(spec, first)
            _body(spec, second)
        # mjNEQDATA grew from 11 to 12 between MuJoCo 3.x releases; size it
        # from the installed library so newer wheels keep working.
        data = np.zeros(int(getattr(mujoco, "mjNEQDATA", 11)))
        if equality.type == "joint":
            data[: len(equality.polycoef)] = equality.polycoef
        elif equality.type == "connect":
            data[:3] = equality.anchor
        kwargs = {
            "name": equality.name or f"{equality.type}_{first}_{second}_{index}",
            "type": EQUALITY_TYPES[equality.type],
            "name1": first,
            "name2": second,
            "data": data.tolist(),
        }
        if equality.solref is not None:
            kwargs["solref"] = list(equality.solref)
        spec.add_equality(**kwargs)

    if patches.exclude_self_collision:
        for index, first in enumerate(body_names):
            for second in body_names[index + 1 :]:
                spec.add_exclude(bodyname1=first, bodyname2=second)
    for first, second in patches.exclude_pairs:
        _body(spec, first)
        _body(spec, second)
        spec.add_exclude(bodyname1=first, bodyname2=second)


def apply_model_spec(
    spec: mujoco.MjSpec, manifest: RobotManifest, *, base_type: str
) -> mujoco.MjsBody:
    """Position the root, add the free joint if asked, apply patches and options."""
    model = manifest.model
    root = find_root_body(spec, model.root_body)
    root.pos = list(model.base.position)
    if tuple(model.base.quaternion) != (1.0, 0.0, 0.0, 0.0):
        root.quat = list(model.base.quaternion)
    if base_type == "free":
        existing = list(root.joints)
        if any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in existing):
            pass  # the model already floats
        elif existing:
            raise ManifestError(
                f"cannot add a free joint: root body {root.name!r} already has joints "
                f"{[j.name for j in existing]}. Give the robot a joint-less root body."
            )
        else:
            root.add_freejoint(name=model.base.joint_name)
    if model.timestep is not None:
        spec.option.timestep = model.timestep
    if model.integrator is not None:
        spec.option.integrator = INTEGRATORS[model.integrator]
    apply_patches(spec, model.patches, base_type=base_type, root_name=root.name)
    return root


__all__ = ["apply_model_spec", "apply_patches", "find_root_body", "set_scalar"]
