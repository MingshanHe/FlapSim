"""FlapSim command-line entry point."""

from __future__ import annotations

import argparse
import sys

# Keep the source tree clean when FlapSim is launched from this entry point.
sys.dont_write_bytecode = True

from flapsim.sim_core import SimCore


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float)
    parser.add_argument(
        "--fix",
        type=parse_bool,
        default=True,
        metavar="{true,false}",
        help="fix base_link at z=0.5 or use a floating base",
    )
    args = parser.parse_args()

    if args.headless and args.duration is None:
        parser.error("--headless requires --duration")

    simulation = SimCore(args.robot, fix=args.fix)
    simulation.run(viewer=not args.headless, duration=args.duration)


if __name__ == "__main__":
    main()
