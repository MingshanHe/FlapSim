"""Command-line package entry point."""
import argparse
from dataclasses import replace
from pathlib import Path
from . import Simulator, load_config


def main():
    parser = argparse.ArgumentParser(description='FlapSim flapping-wing simulation')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--robot', default='IST_robot')
    parser.add_argument('--backend', choices=['cpu', 'mjx'])
    parser.add_argument('--device', choices=['cpu', 'gpu'])
    parser.add_argument('--fluid', choices=['blade_element', 'mujoco', 'none'])
    parser.add_argument('--builtin-fluid', choices=['inertia', 'ellipsoid'])
    parser.add_argument('--actuator', choices=['dcmotor', 'bldcmotor', 'dea'])
    parser.add_argument('--controller', choices=['speed', 'hover', 'move'])
    parser.add_argument('--speed', nargs='+', type=float, help='shaft angular velocities [rad/s]')
    parser.add_argument('--fix', choices=['true', 'false'])
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--duration', type=float)
    args = parser.parse_args()
    if args.headless and args.duration is None:
        parser.error('--headless requires --duration')
    path = args.config or Path(__file__).parent / 'config' / 'robots' / f'{args.robot}.json'
    config = load_config(path)
    backend = args.backend or config.backend
    device = args.device or config.device
    if args.device == 'gpu' and args.backend is None:
        backend = 'mjx'
    config = replace(config, backend=backend, device=device,
                     fluid=replace(config.fluid, kind=args.fluid or config.fluid.kind,
                                   builtin_model=args.builtin_fluid or config.fluid.builtin_model),
                     actuator=replace(config.actuator, kind=args.actuator or config.actuator.kind),
                     controller=replace(config.controller,
                                        kind=args.controller or config.controller.kind,
                                        speed=tuple(args.speed) if args.speed else config.controller.speed),
                     scene=replace(config.scene, fixed_base=(args.fix == 'true')
                                   if args.fix is not None else config.scene.fixed_base))
    sim = Simulator(config)
    steps = sim.run(viewer=not args.headless, duration=args.duration)
    print(f'{config.robot}: {steps} steps, t={float(sim.data.time):.6f}s, '
          f'{config.backend}/{config.device}, {config.fluid.kind}')
