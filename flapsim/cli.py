"""FlapSim command line: ``flapsim run``, ``flapsim inspect``."""

from __future__ import annotations

import argparse
import json
import sys


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in ("true", "1", "yes"):
        return True
    if normalized in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flapsim",
        description="Simulate flapping-wing robots on MuJoCo.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a robot with the classic controllers")
    run.add_argument(
        "robot",
        nargs="?",
        default="SingleThrustFlapper",
        help="built-in robot name, manifest file or model file",
    )
    run.add_argument(
        "--base",
        choices=["fixed", "free"],
        default=None,
        help="override the manifest's base type",
    )
    run.add_argument("--headless", action="store_true")
    run.add_argument(
        "--duration", type=float, help="simulated seconds (required headless)"
    )
    run.add_argument("--frequency", type=float, help="flapping frequency [Hz]")
    run.add_argument("--amplitude", type=float, help="stroke amplitude [rad]")

    inspect = sub.add_parser("inspect", help="list bodies, joints, sites and wings")
    inspect.add_argument(
        "source", help="built-in robot name, manifest file or model file"
    )
    inspect.add_argument("--base", choices=["fixed", "free"], default=None)
    inspect.add_argument(
        "--json", action="store_true", help="print JSON instead of a table"
    )
    return parser


def cmd_inspect(args: argparse.Namespace) -> int:
    from flapsim.robot import describe_model, load, model_summary

    robot = load(args.source, base=args.base, scene=False)
    if args.json:
        print(
            json.dumps(
                model_summary(robot.model, robot.manifest, robot.wings), indent=2
            )
        )
    else:
        print(describe_model(robot.model, robot.manifest, robot.wings))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from flapsim.sim_core import SimCore

    if args.headless and args.duration is None:
        print("error: --headless requires --duration", file=sys.stderr)
        return 2
    fix = True if args.base is None else args.base == "fixed"
    sim = SimCore(args.robot, fix=fix)
    if args.frequency is not None or args.amplitude is not None:
        sim.set_thrust_command(frequency=args.frequency, amplitude=args.amplitude)
    sim.run(viewer=not args.headless, duration=args.duration)
    return 0


def main(argv: list[str] | None = None) -> int:
    sys.dont_write_bytecode = True
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        return cmd_inspect(args)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
