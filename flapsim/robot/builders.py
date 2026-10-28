"""Parametric robots built in Python, for users without CAD and for tests."""

from __future__ import annotations

import mujoco

from .manifest import (
    BaseSpec,
    JointRef,
    ModelPatches,
    ModelSpec,
    Planform,
    RobotManifest,
    TorsoSpec,
    WingFrame,
    WingMass,
    WingSpec,
)
from .patches import set_scalar


def flat_plate_flapper(
    *,
    span: float = 0.07,
    chord: float = 0.02,
    thickness: float = 5.0e-4,
    shoulder: float = 0.012,
    body_mass: float = 0.03,
    wing_mass: float = 1.0e-4,
    arm_mass: float = 2.0e-4,
    stroke_range: float = 1.5,
    feathering_range: float = 0.785398,
    feathering_stiffness: float = 2.0e-7,
    feathering_damping: float = 1.0e-9,
    feathering_armature: float = 1.0e-10,
    spar_fraction: float = 0.8,
    timestep: float = 5.0e-5,
    position=(0.0, 0.0, 0.5),
    name: str = "flat_plate_flapper",
) -> tuple[mujoco.MjSpec, RobotManifest]:
    """A two-wing flapper with rectangular flat-plate wings.

    The torso sits at the origin and the wings span the body ``x`` axis.  At
    the neutral pose each plate is vertical: its normal points along ``-y``
    (the stroke direction) and its chord hangs down (``-z``) from the
    leading-edge hinge.  This is the quasi-steady model's convention, in which
    a feathering angle of zero means the plate meets the stroke flow at 90
    degrees.  The stroke joints rotate about ``+z`` (left) and ``-z`` (right),
    so equal joint angles give mirrored motion; the passive feathering joints
    rotate about ``+x`` for both wings.  Returns the ``MjSpec`` and a matching
    manifest; pass both to :func:`flapsim.robot.load`.  The wing mass is
    distributed as a leading-edge spar (``spar_fraction``) plus a membrane
    plate through the manifest's mass model.
    """
    if span <= 0.0 or chord <= 0.0 or thickness <= 0.0:
        raise ValueError("span, chord and thickness must be positive")
    torso_mass = body_mass - 2.0 * (wing_mass + arm_mass)
    if torso_mass <= 0.0:
        raise ValueError("body_mass must exceed the mass of the arms and wings")

    spec = mujoco.MjSpec()
    spec.modelname = name
    spec.compiler.degree = False  # ranges below are in radians
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    torso = spec.worldbody.add_body(name="torso", pos=list(position))
    torso.add_geom(
        name="torso_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.006, 0.006, 0.02],
        mass=torso_mass,
        rgba=[0.3, 0.3, 0.35, 1.0],
    )

    for side, sign in (("left", 1.0), ("right", -1.0)):
        arm = torso.add_body(name=f"{side}_arm", pos=[sign * shoulder, 0.0, 0.0])
        stroke = arm.add_joint(
            name=f"{side}_stroke",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[0.0, 0.0, sign],
            range=[-stroke_range, stroke_range],
        )
        stroke.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        arm.add_geom(
            name=f"{side}_arm_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.003, 0.0, 0.0],
            mass=arm_mass,
            rgba=[0.6, 0.6, 0.6, 1.0],
        )
        wing = arm.add_body(name=f"{side}_wing", pos=[0.0, 0.0, 0.0])
        joint = wing.add_joint(
            name=f"{side}_feathering",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[1.0, 0.0, 0.0],
            range=[-feathering_range, feathering_range],
        )
        joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        # stiff limits: the default solver constants let a 0.1 g wing swing
        # far past its stops
        joint.solref_limit = [4.0e-4, 1.0]
        joint.solimp_limit = [0.97, 0.999, 5.0e-4, 0.5, 2.0]
        set_scalar(joint, "stiffness", feathering_stiffness)
        set_scalar(joint, "damping", feathering_damping)
        set_scalar(joint, "armature", feathering_armature)
        wing.add_geom(
            name=f"{side}_plate",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.5 * span, 0.5 * thickness, 0.5 * chord],
            pos=[sign * 0.5 * span, 0.0, -0.5 * chord],
            mass=wing_mass,
            rgba=[0.75, 0.8, 0.95, 0.7],
        )

    planform = Planform(profile="rectangular", length=span, mean_chord=chord)
    # A membrane wing carries most of its mass in the leading-edge spar; with
    # the plate's own centroid (mid-chord) the passive feathering responds in
    # anti-phase and the wing produces negative lift.
    mass_model = WingMass(
        mass=wing_mass, type="spar_plate", spar_fraction=spar_fraction
    )
    manifest = RobotManifest(
        name=name,
        model=ModelSpec(
            root_body="torso",
            base=BaseSpec(type="fixed", position=tuple(float(v) for v in position)),
            timestep=timestep,
            integrator="implicitfast",
            patches=ModelPatches(exclude_self_collision=True),
        ),
        torso=TorsoSpec(body="torso"),
        wings=(
            WingSpec(
                name="left",
                body="left_wing",
                side="left",
                frame=WingFrame(span=(1.0, 0.0, 0.0), chord=(0.0, 0.0, -1.0)),
                stroke_joint=JointRef("left_stroke", 1.0),
                feathering_joint=JointRef("left_feathering", 1.0),
                planform=planform,
                shoulder_offset=shoulder,
                mass_model=mass_model,
            ),
            WingSpec(
                name="right",
                body="right_wing",
                side="right",
                frame=WingFrame(span=(-1.0, 0.0, 0.0), chord=(0.0, 0.0, -1.0)),
                stroke_joint=JointRef("right_stroke", 1.0),
                feathering_joint=JointRef("right_feathering", 1.0),
                planform=planform,
                shoulder_offset=shoulder,
                mass_model=mass_model,
            ),
        ),
        extras={
            "stroke_joints": ["left_stroke", "right_stroke"],
            "flap": {"frequency_hz": 30.0, "amplitude_rad": 1.2},
        },
    )
    return spec, manifest


def flat_plate_simulation(
    *,
    frequency_hz: float = 30.0,
    amplitude_rad: float = 1.2,
    base: str = "fixed",
    aero: str | None = "aero/quasi_steady",
    **builder_kwargs,
):
    """A :class:`~flapsim.core.Simulation` of :func:`flat_plate_flapper`.

    The stroke joints are driven sinusoidally (equal angles give mirrored
    motion) and the quasi-steady aero plugin is attached unless ``aero`` is
    ``None``.
    """
    from flapsim.core import Simulation

    spec, manifest = flat_plate_flapper(**builder_kwargs)
    from flapsim.robot.loader import load

    sim = Simulation(load(spec, manifest=manifest, base=base))
    sim.add(
        {
            "type": "actuator/sinusoidal_stroke",
            "id": "stroke",
            "joints": manifest.extras["stroke_joints"],
            "frequency_hz": frequency_hz,
            "amplitude_rad": amplitude_rad,
        }
    )
    if aero is not None:
        sim.add(aero)
    return sim


__all__ = ["flat_plate_flapper", "flat_plate_simulation"]
