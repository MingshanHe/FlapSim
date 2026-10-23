import json
from pathlib import Path

import numpy as np

from flapsim.rl.trajectories import Trajectory


def test_existing_references_unchanged_by_ellipse_extension():
    samples = json.loads(Path(__file__).with_name('trajectory_reference_v1.json').read_text())
    for row in samples:
        trajectory = Trajectory(row['name'], center=(.1, -.2, .8), radius=.23,
                                period=7.3, height=.11, phase=.37)
        np.testing.assert_allclose(trajectory.sample(row['time']), row['sample'],
                                   rtol=0, atol=1e-15)


def test_ellipse_closed_path_and_analytic_derivatives():
    trajectory = Trajectory('ellipse', radius=.2, period=8.)
    t, dt = 1.234, 1e-5
    position, velocity, acceleration = trajectory.sample(t)
    before, after = trajectory.sample(t-dt), trajectory.sample(t+dt)
    np.testing.assert_allclose(velocity, (after[0]-before[0])/(2*dt), atol=1e-10)
    np.testing.assert_allclose(acceleration, (after[1]-before[1])/(2*dt), atol=1e-10)
    np.testing.assert_allclose(trajectory.sample(t+8.), (position, velocity, acceleration), atol=1e-14)
