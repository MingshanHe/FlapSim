"""A minimal MuJoCo environment: load, display, and step one model."""

from __future__ import annotations

import time
from pathlib import Path

import mujoco

from .scene import add_scene


class SimulationEnvironment:
    def __init__(
        self,
        urdf_path: str | Path,
        *,
        position: tuple[float, float, float] = (0.0, 0.0, 0.5),
        timestep: float = 5e-5,
        fixed_base: bool = True,
    ) -> None:
        spec = mujoco.MjSpec.from_file(str(Path(urdf_path).resolve()))

        root_bodies = list(spec.worldbody.bodies)
        if len(root_bodies) != 1:
            raise ValueError("The URDF must contain exactly one root body")

        root_body = root_bodies[0]
        root_body.pos = position
        if not fixed_base:
            root_body.add_freejoint(name="floating_base")
            # The floating hover model uses cycle-averaged forces, so its
            # mass model must also be a single equivalent rigid body. Keep
            # child geoms/joints for visualization but remove their dynamic
            # inertia to avoid energy injection from kinematic flapping.
            root_body.mass = 0.03
            root_body.ipos = [0.0, 0.0, 0.0]
            root_body.fullinertia = [
                5.0e-5,
                5.0e-5,
                2.0e-5,
                0.0,
                0.0,
                0.0,
            ]
            for body in spec.bodies:
                if body.name not in ("world", root_body.name):
                    # MuJoCo requires every moving body to retain positive
                    # inertia.  These values are numerical placeholders only;
                    # the complete 30 g rigid-body mass/inertia is assigned to
                    # base_link above.
                    body.mass = 1.0e-9
                    body.fullinertia = [
                        1.0e-12,
                        1.0e-12,
                        1.0e-12,
                        0.0,
                        0.0,
                        0.0,
                    ]
        spec.option.timestep = timestep
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

        # MuJoCo passive-feathering parameters matching FlapX. URDF supports
        # damping/friction, while stiffness, springref and armature are added
        # here through MjSpec because they are not standard URDF fields.
        for joint_name in (
            "left_wing_feathering",
            "right_wing_feathering",
        ):
            joint = spec.joint(joint_name)
            joint.stiffness = [2.0e-7, 0.0, 0.0]
            joint.springref = 0.0
            joint.armature = 1.0e-10
            joint.solref_limit = [0.0004, 1.0]
            joint.solimp_limit = [0.97, 0.999, 0.0005, 0.5, 2.0]

        # The URDF's mirrored meshes use slightly different local frames.
        # Equal feathering coordinates produce mirrored world-space wing
        # orientations, so constrain the two passive joints to qL == qR.
        spec.add_equality(
            name="mirror_feathering",
            type=mujoco.mjtEq.mjEQ_JOINT,
            name1="left_wing_feathering",
            name2="right_wing_feathering",
            data=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            solref=[0.0004, 1.0],
        )

        # Imported collision meshes overlap slightly at the joints. Disable
        # robot self-collision while preserving collision with the world plane.
        robot_body_names = [body.name for body in spec.bodies if body.name != "world"]
        for index, body1 in enumerate(robot_body_names):
            for body2 in robot_body_names[index + 1 :]:
                spec.add_exclude(bodyname1=body1, bodyname2=body2)

        add_scene(spec, target_body=root_body.name)

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.fixed_base = fixed_base
        self.control_pairs = []
        self.force_models = []
        self.supervisors = []
        mujoco.mj_forward(self.model, self.data)

    def attach_control(self, command_source, actuator) -> None:
        self.control_pairs.append((command_source, actuator))

    def attach_force_model(self, force_model) -> None:
        self.force_models.append(force_model)

    def attach_supervisor(self, supervisor) -> None:
        self.supervisors.append(supervisor)

    def step(self) -> None:
        self.data.qfrc_applied[:] = 0.0
        for supervisor in self.supervisors:
            supervisor.update(self.model, self.data)
        for command_source, actuator in self.control_pairs:
            actuator.apply(self.model, self.data, command_source.targets)
        for force_model in self.force_models:
            force_model.apply_forces()
        mujoco.mj_step(self.model, self.data)
        forwarded = False
        for command_source, actuator in self.control_pairs:
            if hasattr(actuator, "post_step"):
                actuator.post_step(self.model, self.data, command_source.targets)
                forwarded = True
        if forwarded:
            mujoco.mj_forward(self.model, self.data)

    def run(self, *, viewer: bool = True, duration: float | None = None) -> int:
        """Run until duration expires or the viewer is closed."""
        steps = 0
        end_time = None if duration is None else self.data.time + duration

        if not viewer:
            if end_time is None:
                raise ValueError("headless mode requires a duration")
            while self.data.time < end_time:
                self.step()
                steps += 1
            return steps

        import mujoco.viewer

        with mujoco.viewer.launch_passive(self.model, self.data) as window:
            while window.is_running() and (
                end_time is None or self.data.time < end_time
            ):
                wall_start = time.perf_counter()
                frame_end = self.data.time + 1.0 / 60.0
                while self.data.time < frame_end:
                    self.step()
                    steps += 1
                window.sync()
                remaining = 1.0 / 60.0 - (time.perf_counter() - wall_start)
                if remaining > 0.0:
                    time.sleep(remaining)
        return steps
