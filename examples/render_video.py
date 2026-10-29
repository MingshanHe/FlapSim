"""Render the flapping robot to a GIF with force arrows (offscreen).

MUJOCO_GL=osmesa python examples/render_video.py --output flapping.gif --slow 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flapsim import Simulation
from flapsim.io import ForceArrows, record_video
from flapsim.robot import builtin_robots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("flapping.gif"))
    parser.add_argument("--duration", type=float, default=0.1, help="simulated seconds")
    parser.add_argument("--slow", type=float, default=20.0, help="slow-motion factor")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    sim = Simulation.from_config(
        builtin_robots()["SingleThrustFlapper"].parent / "sim.yaml"
    )
    path = record_video(
        sim,
        args.output,
        duration=args.duration,
        fps=args.fps,
        width=480,
        height=360,
        overlay=ForceArrows(scale=0.3),
        realtime_factor=1.0 / args.slow,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
