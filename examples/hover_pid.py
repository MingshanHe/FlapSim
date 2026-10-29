"""Hover the example robot with the cycle-averaged model and the PID controller.

python examples/hover_pid.py            # headless, prints altitude statistics
python examples/hover_pid.py --viewer   # watch it in the MuJoCo viewer
"""

from __future__ import annotations

import argparse

from flapsim import Simulation
from flapsim.robot import builtin_robots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--target", type=float, default=0.5, help="hover height [m]")
    args = parser.parse_args()

    setup = builtin_robots()["SingleThrustFlapper"].parent / "hover.yaml"
    sim = Simulation.from_config(setup)
    sim["hover"].set_target(z=args.target)
    sim.add(
        {
            "type": "recorder/memory",
            "signals": ["hover/z", "hover/frequency_hz"],
            "every": 0.01,
        }
    )

    sim.run(args.duration, viewer=args.viewer)

    recorder = sim["recorder"]
    z = recorder["hover/z"]
    settled = z[len(z) // 2 :]
    print(
        f"height after {args.duration:.1f} s: {z[-1]:.3f} m "
        f"(target {args.target:.3f} m)"
    )
    print(f"second half: mean {settled.mean():.3f} m, std {settled.std() * 1e3:.1f} mm")
    print(f"stroke frequency: mean {recorder['hover/frequency_hz'].mean():.2f} Hz")
    if not (abs(settled.mean() - args.target) < 0.05):
        raise SystemExit("the controller did not hold the target height")


if __name__ == "__main__":
    main()
