"""Compatibility wrapper: the quasi-steady model applied without plugins.

``MuJoCoAeroAdapter(robot).apply_forces()`` evaluates the quasi-steady
blade-element model on every wing of a :class:`~flapsim.robot.RobotModel`
and writes the wrenches, exactly like the ``aero/quasi_steady`` plugin, for
scripts that step MuJoCo themselves.  New code should prefer
:class:`flapsim.aero.AeroStack`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flapsim.config import AeroOptions

if TYPE_CHECKING:
    from flapsim.robot.loader import RobotModel


class MuJoCoAeroAdapter:
    def __init__(self, robot: RobotModel, options: AeroOptions | None = None) -> None:
        from flapsim.aero.base import AeroEngine

        self.engine = AeroEngine(
            robot, ["quasi_steady"], options=options or AeroOptions()
        )
        self.robot = robot
        self.model = robot.model
        self.data = robot.data
        self.wings = self.engine.wings

    @property
    def options(self) -> AeroOptions:
        return self.engine.options

    @options.setter
    def options(self, value: AeroOptions) -> None:
        self.engine.options = value

    @property
    def diagnostics(self) -> dict:
        return self.engine.diagnostics

    def apply_forces(self) -> None:
        self.engine.apply()

    def reset(self) -> None:
        self.engine.reset()
