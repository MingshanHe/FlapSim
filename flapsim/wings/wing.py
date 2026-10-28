"""Runtime wing: a manifest :class:`WingSpec` resolved against a MuJoCo model.

The aerodynamic frame of a wing has ``+X`` along the surface normal, ``+Y``
along the feathering axis and ``+Z = X x Y``; the chord extends from the
hinge along ``-Z``.  From the manifest's ``span`` (root to tip) and ``chord``
(leading to trailing edge) vectors: ``Y = side * span``, ``Z = -chord`` and
``X = Y x Z``.  Mirrored wings therefore share one frame orientation at the
neutral pose (for a right wing ``+Y`` is the negative root-to-tip direction)
and the ``side`` factor in the aerodynamic formulas puts the centre of
pressure back on the wing.  This is the convention of the quasi-steady model
in :mod:`flapsim.fluid.aero`.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from flapsim.robot.manifest import ManifestError, WingSpec

from .planform import resolve_planform, wing_params

PLATE_THICKNESS_FRACTION = 0.01


@dataclass
class WingState:
    """Kinematic state of one wing read from ``MjData``."""

    name: str
    side: float
    psi: float
    psi_dot: float
    theta: float
    theta_dot: float
    R_world_from_aero: np.ndarray
    origin_world: np.ndarray
    com_world: np.ndarray
    body_id: int


def _quat_to_mat(quat) -> np.ndarray:
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(quat, dtype=float))
    return out.reshape(3, 3)


class Wing:
    def __init__(
        self,
        spec: WingSpec,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        rho: float = 1.225,
        mirror_source: Wing | None = None,
    ) -> None:
        self.spec = spec
        self.name = spec.name
        self.side = float(spec.side_sign)
        self.body_id = self._body_id(model, spec.body)
        self.stroke = self._joint(model, spec.stroke_joint)
        self.feathering = self._joint(model, spec.feathering_joint)
        self.length, self.mean_chord, self.integrals = resolve_planform(spec.planform)
        self.params = wing_params(
            self.length,
            self.mean_chord,
            self.integrals,
            rho=rho,
            d0=spec.shoulder_offset,
            ds=spec.stroke_plane_offset,
            x_le=spec.planform.leading_edge_offset,
            x0=spec.planform.pitch_axis_fraction,
        )
        self.aero_from_body = self._aero_from_body(model, data, mirror_source)

    # -- resolution ---------------------------------------------------------

    @staticmethod
    def _body_id(model: mujoco.MjModel, name: str) -> int:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body < 0:
            names = [model.body(i).name for i in range(model.nbody)]
            raise ManifestError(f"wing body {name!r} not found; bodies are {names}")
        return int(body)

    @staticmethod
    def _joint(model: mujoco.MjModel, ref) -> tuple[int, int, float] | None:
        if ref is None:
            return None
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ref.name)
        if joint < 0:
            names = [model.joint(i).name for i in range(model.njnt)]
            raise ManifestError(
                f"wing joint {ref.name!r} not found; joints are {names}"
            )
        if int(model.jnt_type[joint]) not in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            raise ManifestError(
                f"wing joint {ref.name!r} must be a hinge or slide joint"
            )
        return (
            int(model.jnt_qposadr[joint]),
            int(model.jnt_dofadr[joint]),
            float(ref.sign),
        )

    def _aero_from_body(
        self, model: mujoco.MjModel, data: mujoco.MjData, mirror_source: Wing | None
    ) -> np.ndarray:
        spec = self.spec
        if spec.mirror_of is not None:
            if mirror_source is None or mirror_source.name != spec.mirror_of:
                raise ManifestError(
                    f"wing {self.name!r} mirrors {spec.mirror_of!r}, "
                    "which must be built first"
                )
            # Share the source wing's aero frame orientation at the neutral pose.
            source_world = data.xmat[mirror_source.body_id].reshape(3, 3)
            neutral = source_world @ mirror_source.aero_from_body
            own_world = data.xmat[self.body_id].reshape(3, 3)
            return own_world.T @ neutral
        frame = spec.frame
        if frame.site is not None:
            site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, frame.site)
            if site < 0:
                raise ManifestError(f"wing site {frame.site!r} not found")
            if int(model.site_bodyid[site]) != self.body_id:
                raise ManifestError(
                    f"wing site {frame.site!r} must belong to body {spec.body!r}"
                )
            site_mat = _quat_to_mat(model.site_quat[site])
            chord, span = site_mat[:, 0], site_mat[:, 1]
        else:
            span = np.asarray(frame.span, dtype=float)
            chord = np.asarray(frame.chord, dtype=float)
        y_axis = self.side * span / np.linalg.norm(span)
        z_axis = -chord / np.linalg.norm(chord)
        x_axis = np.cross(y_axis, z_axis)
        return np.column_stack((x_axis, y_axis, z_axis))

    # -- mass model ---------------------------------------------------------

    def body_mass_properties(self) -> tuple[float, np.ndarray, np.ndarray] | None:
        """``(mass, com_body, fullinertia_body)`` from the manifest mass model.

        Wing coordinates ``(normal, span, chord)`` map onto the aero axes as
        ``(X, side * Y, -Z)``; the inertia is rotated into the body frame and
        returned as ``[Ixx, Iyy, Izz, Ixy, Ixz, Iyz]``.
        """
        model = self.spec.mass_model
        if model is None:
            return None
        m = float(model.mass)
        R, c = self.length, self.mean_chord
        # A perfectly planar lamina sits exactly on MuJoCo's triangle
        # inequality (I_normal = I_span + I_chord); give the plate a thickness
        # of one percent of the chord (0.2 mm for a 20 mm chord).
        t2 = (PLATE_THICKNESS_FRACTION * c) ** 2
        if model.type == "thin_plate":
            com_wing = np.array([0.0, 0.5 * R, 0.5 * c])
            diag_wing = np.array(
                [
                    m * (R**2 + c**2) / 12.0,
                    m * (c**2 + t2) / 12.0,
                    m * (R**2 + t2) / 12.0,
                ]
            )
        elif model.type == "spar_plate":
            # leading-edge rod (spar) plus uniform membrane plate
            m_spar = model.spar_fraction * m
            m_plate = m - m_spar
            com_chord = 0.5 * c * m_plate / m
            com_wing = np.array([0.0, 0.5 * R, com_chord])
            d_spar = com_chord  # spar sits at chord 0
            d_plate = 0.5 * c - com_chord
            shift = m_spar * d_spar**2 + m_plate * d_plate**2
            i_normal = m_spar * R**2 / 12.0 + m_plate * (R**2 + c**2) / 12.0 + shift
            i_span = m_plate * (c**2 + t2) / 12.0 + m_spar * t2 / 12.0 + shift
            i_chord = m_spar * (R**2 + t2) / 12.0 + m_plate * (R**2 + t2) / 12.0
            diag_wing = np.array([i_normal, i_span, i_chord])
        else:
            com_wing = np.asarray(model.com, dtype=float)
            diag_wing = np.asarray(model.diaginertia, dtype=float)
        # wing (normal, span, chord) -> aero (X, Y, Z)
        wing_to_aero = np.diag([1.0, self.side, -1.0])
        rotation = self.aero_from_body @ wing_to_aero
        com_body = rotation @ com_wing
        inertia_body = rotation @ np.diag(diag_wing) @ rotation.T
        full = np.array(
            [
                inertia_body[0, 0],
                inertia_body[1, 1],
                inertia_body[2, 2],
                inertia_body[0, 1],
                inertia_body[0, 2],
                inertia_body[1, 2],
            ]
        )
        return m, com_body, full

    # -- self-checks --------------------------------------------------------

    def geometry_warnings(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> list[str]:
        """Plain-language warnings when the frame or joint signs look reversed.

        Checks, at the current (neutral) pose: the wing's centre of mass lies
        on the chord side of the hinge (``-Z``), a positive stroke rate moves
        the wing against ``+X`` (so the relative flow is along ``+X``), and a
        positive feathering rate rotates the wing about ``+Y``.
        """
        warnings: list[str] = []
        rotation = data.xmat[self.body_id].reshape(3, 3) @ self.aero_from_body
        com_offset = data.xipos[self.body_id] - data.xpos[self.body_id]
        if float(com_offset @ rotation[:, 2]) > 1e-6:
            warnings.append(
                f"wing {self.name!r}: its centre of mass lies on the +Z side of the "
                "hinge, so 'chord' probably points the wrong way (it must point from "
                "the leading edge towards the trailing edge)."
            )
        for label, joint, axis_sign in (
            ("stroke_joint", self.stroke, self.side),
            ("feathering_joint", self.feathering, 1.0),
        ):
            if joint is None:
                continue
            _, dof, sign = joint
            qvel = np.zeros(model.nv)
            qvel[dof] = sign  # positive aerodynamic rate
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, self.body_id, velocity, 0
            )
            saved = data.qvel.copy()
            data.qvel[:] = qvel
            mujoco.mj_forward(model, data)
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, self.body_id, velocity, 0
            )
            data.qvel[:] = saved
            mujoco.mj_forward(model, data)
            omega_world = velocity[:3]
            if label == "stroke_joint":
                expected_axis = axis_sign * rotation[:, 2]
                what = "a right-handed rotation about side * Z (the stroke axis)"
            else:
                expected_axis = rotation[:, 1]
                what = "a right-handed rotation about +Y (the feathering axis)"
            if float(omega_world @ expected_axis) < 0.0:
                warnings.append(
                    f"wing {self.name!r}: positive {label} rate is not {what}; "
                    f"flip the joint's 'sign'."
                )
        return warnings

    # -- runtime ------------------------------------------------------------

    def state(self, data: mujoco.MjData) -> WingState:
        psi = psi_dot = theta = theta_dot = 0.0
        if self.stroke is not None:
            qpos, dof, sign = self.stroke
            psi = sign * float(data.qpos[qpos])
            psi_dot = sign * float(data.qvel[dof])
        if self.feathering is not None:
            qpos, dof, sign = self.feathering
            theta = sign * float(data.qpos[qpos])
            theta_dot = sign * float(data.qvel[dof])
        rotation = data.xmat[self.body_id].reshape(3, 3) @ self.aero_from_body
        return WingState(
            name=self.name,
            side=self.side,
            psi=psi,
            psi_dot=psi_dot,
            theta=theta,
            theta_dot=theta_dot,
            R_world_from_aero=rotation,
            origin_world=data.xpos[self.body_id].copy(),
            com_world=data.xipos[self.body_id].copy(),
            body_id=self.body_id,
        )

    def apply_wrench(self, data: mujoco.MjData, force_world, torque_world) -> None:
        """Write a world-frame wrench about the body centre of mass."""
        data.xfrc_applied[self.body_id, :3] = force_world
        data.xfrc_applied[self.body_id, 3:] = torque_world

    def clear(self, data: mujoco.MjData) -> None:
        data.xfrc_applied[self.body_id] = 0.0

    def __repr__(self) -> str:
        return (
            f"Wing({self.name!r}, body={self.spec.body!r}, side={self.side:+.0f}, "
            f"R={self.length:.4f} m, c={self.mean_chord:.4f} m)"
        )


__all__ = ["Wing", "WingState"]
