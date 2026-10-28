"""Parameter study: lift of the flat-plate flapper versus wing length and frequency.

python examples/wing_length_study.py --workers 4 --output study.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from flapsim.robot import flat_plate_simulation
from flapsim.study import parameter_study


def evaluate(span: float, frequency_hz: float) -> dict[str, float]:
    sim = flat_plate_simulation(span=span, frequency_hz=frequency_hz, amplitude_rad=1.2)
    sim.advance(0.2)
    total = np.zeros(3)
    steps = 0
    end = sim.time + 5.0 / frequency_hz
    while sim.time < end:
        sim.step()
        total += sim["aero"].diagnostics["F_world"]
        steps += 1
    mean = total / steps
    return {"mean_lift_n": float(mean[2]), "lift_per_span": float(mean[2] / span)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spans", type=float, nargs="+", default=[0.05, 0.07, 0.09])
    parser.add_argument("--frequencies", type=float, nargs="+", default=[25.0, 35.0])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = parameter_study(
        evaluate,
        {"span": args.spans, "frequency_hz": args.frequencies},
        workers=args.workers,
    )
    for row in result.rows():
        if row["error"]:
            print(f"span {row['span']:.3f} m, {row['frequency_hz']:.0f} Hz: failed")
        else:
            print(
                f"span {row['span']:.3f} m, {row['frequency_hz']:.0f} Hz: "
                f"lift {row['mean_lift_n']:.4f} N"
            )
    case, metrics = result.best("mean_lift_n")
    print(f"best: {case} -> {metrics['mean_lift_n']:.4f} N")
    if args.output is not None:
        result.to_csv(args.output)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
