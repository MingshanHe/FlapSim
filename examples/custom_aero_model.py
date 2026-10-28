"""Write your own aerodynamic model and stack it with the built-in ones.

The model below adds a crude leading-edge-vortex augmentation: a fraction of
the quasi-steady normal force proportional to ``sin(2 alpha)``.  It registers
under the name ``lev_bonus`` and is then used from a plain config.

    python examples/custom_aero_model.py
"""

from __future__ import annotations

import math

import numpy as np

from flapsim import Simulation
from flapsim.aero import AeroModel, WingLoads, register_model
from flapsim.fluid.aero import wing_loads_from_kinematics


@register_model
class LevBonus(AeroModel):
    name = "lev_bonus"

    def __init__(self, *, gain: float = 0.2) -> None:
        super().__init__(gain=gain)
        self.gain = float(gain)

    def compute(self, wing, state, kin, flow, options):
        base = wing_loads_from_kinematics(wing.params, kin, options)
        bonus = self.gain * abs(math.sin(2.0 * kin["alpha"])) * base["F_N"]
        point = np.array([0.0, state.side * kin["span_CoP"], -0.5 * wing.mean_chord])
        return WingLoads(
            self.name, np.array([bonus, 0.0, 0.0]), point, np.zeros(3), {"bonus": bonus}
        )


def main() -> None:
    sim = Simulation("SingleThrustFlapper")
    sim.add(
        {
            "type": "actuator/sinusoidal_stroke",
            "id": "stroke",
            "joints": ["left_flapping_joint", "right_wing_flapping"],
            "directions": [1, -1],
            "frequency_hz": 35.0,
            "amplitude_rad": 1.5,
        }
    )
    sim.add(
        {
            "type": "aero/stack",
            "models": ["quasi_steady", {"type": "lev_bonus", "gain": 0.2}],
        }
    )
    sim.add(
        {
            "type": "recorder/memory",
            "signals": ["aero/F_world", "wings/left/*/F_N"],
            "every": 1e-3,
        }
    )
    sim.run(0.5)
    rec = sim["recorder"]
    print(f"mean lift {rec['aero/F_world'][:, 2].mean():.4f} N")
    print(
        f"quasi-steady share {rec['wings/left/quasi_steady/F_N'].mean():+.4f} N, "
        f"bonus share {rec['wings/left/lev_bonus/F_N'].mean():+.4f} N"
    )


if __name__ == "__main__":
    main()
