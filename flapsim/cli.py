"""FlapSim command line: ``flapsim run``, ``flapsim inspect``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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

    run = sub.add_parser(
        "run", help="run a simulation config or a robot's shipped setup"
    )
    run.add_argument(
        "source",
        nargs="?",
        default="SingleThrustFlapper",
        help="simulation config (flapsim.sim/1), built-in robot name, manifest "
        "or model file",
    )
    run.add_argument(
        "--config",
        default=None,
        help="setup name next to the robot manifest (sim, hover)",
    )
    run.add_argument(
        "--base", choices=["fixed", "free"], default=None, help="override the base type"
    )
    run.add_argument("--headless", action="store_true")
    run.add_argument(
        "--viewer", action="store_true", help="open the MuJoCo viewer (default)"
    )
    run.add_argument(
        "--duration", type=float, help="simulated seconds (required headless)"
    )
    run.add_argument(
        "--frequency", type=float, help="stroke frequency [Hz] for plugin 'stroke'"
    )
    run.add_argument(
        "--amplitude", type=float, help="stroke amplitude [rad] for plugin 'stroke'"
    )
    run.add_argument(
        "--describe",
        action="store_true",
        help="print the assembled simulation and exit",
    )

    plugins = sub.add_parser("plugins", help="list available plugins")
    plugins.add_argument("--kind", default=None)

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


def _find_setup(source: str, config_name: str | None) -> str | Path:
    from flapsim.core.config import is_sim_config
    from flapsim.robot import resolve_manifest

    path = Path(source).expanduser()
    if is_sim_config(path):
        return path
    manifest = resolve_manifest(source)
    name = config_name or "sim"
    if manifest.base_dir is None:
        raise SystemExit(f"error: {source} has no directory to look for {name}.yaml in")
    candidate = manifest.base_dir / f"{name}.yaml"
    if not candidate.is_file():
        raise SystemExit(
            f"error: no {name}.yaml next to {manifest.model.file}; write a simulation "
            "config (schema flapsim.sim/1) and pass its path"
        )
    return candidate


def cmd_run(args: argparse.Namespace) -> int:
    from flapsim.core import Simulation

    if args.headless and args.duration is None:
        print("error: --headless requires --duration", file=sys.stderr)
        return 2
    config_name = args.config
    if config_name is None and args.base == "free":
        config_name = "hover"
    setup = _find_setup(args.source, config_name)
    sim = Simulation.from_config(setup, base=args.base)
    if args.frequency is not None or args.amplitude is not None:
        stroke = sim.find("stroke")
        if stroke is None:
            print(
                "error: no plugin with id 'stroke' to apply --frequency/--amplitude",
                file=sys.stderr,
            )
            return 2
        stroke.set_command(frequency_hz=args.frequency, amplitude_rad=args.amplitude)
    if args.describe:
        print(json.dumps(sim.describe(), indent=2, default=str))
        return 0
    run_config = getattr(sim, "run_config", {})
    duration = (
        args.duration if args.duration is not None else run_config.get("duration")
    )
    viewer = not args.headless
    if not viewer and duration is None:
        print("error: headless runs need --duration", file=sys.stderr)
        return 2
    sim.run(duration, viewer=viewer, realtime=bool(run_config.get("realtime", True)))
    return 0


def cmd_plugins(args: argparse.Namespace) -> int:
    from flapsim.plugins import available, get

    listing = available(args.kind) if args.kind else available()
    if args.kind:
        listing = {args.kind: listing}
    for kind, names in listing.items():
        for name in names:
            cls = get(kind, name)
            doc = (cls.__doc__ or "").strip().splitlines()
            print(f"{kind}/{name:<22s} {doc[0] if doc else ''}")
    return 0


def main(argv: list[str] | None = None) -> int:
    sys.dont_write_bytecode = True
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        return cmd_inspect(args)
    if args.command == "plugins":
        return cmd_plugins(args)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
