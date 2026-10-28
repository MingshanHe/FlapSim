"""Mean lift of the example robot versus stroke frequency (virtual load cell).

python examples/lift_sweep.py --frequencies 20 30 40 50
"""

from __future__ import annotations

import argparse

from flapsim import Simulation
from flapsim.aero.identify import lift_sweep
from flapsim.robot import builtin_robots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frequencies", type=float, nargs="+", default=[20.0, 30.0, 40.0, 50.0]
    )
    parser.add_argument("--amplitude", type=float, default=1.5)
    parser.add_argument("--cycles", type=float, default=10.0)
    args = parser.parse_args()

    setup = builtin_robots()["SingleThrustFlapper"].parent / "sim.yaml"
    sweep = lift_sweep(
        lambda: Simulation.from_config(setup),
        args.frequencies,
        amplitude_rad=args.amplitude,
        cycles=args.cycles,
    )
    print("frequency [Hz]  mean lift [N]  k f^2 [N]")
    for f, force in zip(sweep.frequencies, sweep.mean_forces, strict=True):
        print(f"{f:12.1f}  {force[2]:12.4f}  {sweep.lift_coefficient * f**2:10.4f}")
    print(
        f"lift coefficient k = {sweep.lift_coefficient:.4e} N/Hz^2, "
        f"fit rms {sweep.fit_rms_relative:.2%}"
    )


if __name__ == "__main__":
    main()
