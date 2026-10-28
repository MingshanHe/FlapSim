"""Log IMU and range-finder readings of the flapping robot and plot them.

python examples/sensors.py --output sensors.csv
python examples/sensors.py --plot          # needs matplotlib
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flapsim import Simulation
from flapsim.robot import builtin_robots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("sensors.csv"))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--free", action="store_true", help="floating base (hover setup)"
    )
    args = parser.parse_args()

    robot_dir = builtin_robots()["SingleThrustFlapper"].parent
    sim = Simulation.from_config(
        robot_dir / ("hover.yaml" if args.free else "sim.yaml")
    )
    sim.add({"type": "sensor/imu", "id": "imu", "rate_hz": 500.0})
    sim.add(
        {"type": "sensor/rangefinder", "id": "tof", "rate_hz": 500.0, "method": "ray"}
    )
    sim.add({"type": "sensor/wrench", "id": "loadcell", "average_s": 0.02})
    sim.add(
        {
            "type": "recorder/memory",
            "signals": ["imu/*", "tof/*", "loadcell/force", "aero/F_world"],
            "every": 0.002,
        }
    )
    sim.run(args.duration)
    recorder = sim["recorder"]
    recorder.to_csv(args.output)
    accel = recorder["imu/accel"]
    print(f"wrote {args.output} with {len(recorder)} samples")
    print(f"accelerometer mean [m/s^2]: {accel.mean(axis=0).round(3).tolist()}")
    print(f"range: mean {recorder['tof/range'].mean():.3f} m")
    load = recorder["loadcell/force"].mean(axis=0).round(4).tolist()
    print(f"load cell force mean [N]: {load}")

    if args.plot:
        import matplotlib.pyplot as plt

        t = recorder.times
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
        axes[0].plot(t, recorder["imu/gyro"])
        axes[0].set_ylabel("gyro [rad/s]")
        axes[1].plot(t, accel)
        axes[1].set_ylabel("accel [m/s^2]")
        axes[2].plot(t, recorder["tof/range"])
        axes[2].set_ylabel("range [m]")
        axes[2].set_xlabel("time [s]")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
