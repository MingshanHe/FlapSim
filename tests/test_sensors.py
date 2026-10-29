"""Sensor conditioning, IMU, range finder, encoder and load cell."""

from __future__ import annotations

import math

import numpy as np
import pytest

from flapsim.core import Simulation
from flapsim.sensors import (
    IMU,
    BodyWrench,
    Conditioner,
    JointEncoder,
    RangeFinder,
    rotation_to_rpy,
)


def test_conditioner_delay_noise_bias_and_quantisation():
    cond = Conditioner(size=1, rate_hz=100.0, delay_s=0.03, noise_std=0.0, seed=1)
    cond.reset(np.array([0.0]))
    readings = [cond.update([1.0], t)[0] for t in np.arange(0.0, 0.1, 0.01)]
    # 3 samples of delay: the step appears at the 4th sample
    assert readings[:3] == [0.0, 0.0, 0.0] and readings[3] == 1.0

    noisy = Conditioner(size=2, noise_std=[0.1, 0.0], bias_range=[0.0, 0.5], seed=2)
    noisy.reset(np.zeros(2))
    values = np.array([noisy.update([0.0, 0.0], t) for t in np.linspace(0, 1, 2000)])
    assert values[:, 0].std() == pytest.approx(0.1, rel=0.1)
    assert np.all(values[:, 1] == values[0, 1]) and abs(values[0, 1]) <= 0.5
    bias_before = noisy.bias.copy()
    noisy.reset(np.zeros(2))
    assert not np.allclose(noisy.bias, bias_before)  # a new bias per episode

    quant = Conditioner(size=1, resolution=0.25)
    quant.reset(np.array([0.0]))
    assert quant.update([0.6], 0.0)[0] == 0.5

    lowpass = Conditioner(size=1, rate_hz=1000.0, tau=0.01)
    lowpass.reset(np.array([0.0]))
    out = [lowpass.update([1.0], t)[0] for t in np.arange(0.001, 0.011, 0.001)]
    assert out[0] == pytest.approx(1.0 - math.exp(-0.1), rel=1e-6)
    assert out[-1] == pytest.approx(1.0 - math.exp(-1.0), rel=1e-6)


def test_imu_at_rest_reads_gravity_and_attitude():
    sim = Simulation("SingleThrustFlapper", check=False)
    imu = sim.add(IMU(rate_hz=500.0, seed=3))
    sim.advance(0.05)
    accel = sim.signals["imu/accel"]
    gyro = sim.signals["imu/gyro"]
    euler = sim.signals["imu/euler"]
    assert accel[2] == pytest.approx(9.81, abs=0.1)
    assert np.linalg.norm(accel[:2]) < 0.1
    assert np.linalg.norm(gyro) < 0.02
    true = rotation_to_rpy(sim.data.xmat[sim.robot.root_body].reshape(3, 3))
    np.testing.assert_allclose(euler, true, atol=math.radians(0.2))
    assert imu.describe()["type"] == "sensor/imu"
    # the IMU samples at its own rate, not every physics step
    sim.step()
    held = sim.signals["imu/accel"]
    sim.step()
    assert np.all(sim.signals["imu/accel"] == held)


def test_imu_accelerometer_sees_free_fall():
    sim = Simulation("SingleThrustFlapper", base="free", check=False)
    sim.add(IMU(rate_hz=1000.0, accel_noise=0.0, accel_bias_range=0.0, seed=0))
    sim.advance(0.05)
    accel = sim.signals["imu/accel"]
    # a freely falling body reads (almost) zero specific force
    assert np.linalg.norm(accel) < 0.5


def test_rangefinder_plane_and_ray_agree_on_the_ground_plane():
    sim = Simulation("SingleThrustFlapper", check=False)
    plane = sim.add(
        RangeFinder(id="plane", method="plane", noise=0.0, delay_s=0.0, rate_hz=1000.0)
    )
    ray = sim.add(
        RangeFinder(id="ray", method="ray", noise=0.0, delay_s=0.0, rate_hz=1000.0)
    )
    sim.advance(0.01)
    height = float(sim.data.xpos[sim.robot.root_body][2])
    assert sim.signals["plane/range"] == pytest.approx(height, abs=1e-3)
    assert sim.signals["ray/range"] == pytest.approx(
        sim.signals["plane/range"], abs=2e-3
    )
    assert plane.true_range() == pytest.approx(ray.true_range(), abs=2e-3)
    with pytest.raises(ValueError, match="method must be"):
        RangeFinder(method="sonar")


def test_rangefinder_delay_and_rate_while_falling():
    sim = Simulation("SingleThrustFlapper", base="free", check=False)
    sim.add(RangeFinder(rate_hz=500.0, delay_s=0.02, noise=0.0))
    sim.advance(0.2)
    true = sim["tof"].true_range()
    measured = sim.signals["tof/range"]
    # the body falls, so the delayed reading lags behind the true range
    assert measured > true
    assert sim.signals["tof/range_rate"] < 0.0


def test_joint_encoder_quantises_positions():
    sim = Simulation("SingleThrustFlapper", check=False)
    sim.add(
        {
            "type": "actuator/sinusoidal_stroke",
            "id": "stroke",
            "joints": ["left_flapping_joint"],
            "frequency_hz": 10.0,
            "amplitude_rad": 1.0,
        }
    )
    sim.add(JointEncoder(joints=["left_flapping_joint"], counts_per_rev=360))
    sim.advance(0.0125)
    reading = sim.signals["enc/left_flapping_joint/position"]
    step = 2 * math.pi / 360
    assert reading == pytest.approx(round(reading / step) * step, abs=1e-12)
    assert (
        abs(reading - sim.data.qpos[sim.model.joint("left_flapping_joint").qposadr[0]])
        <= step
    )
    assert "enc/left_flapping_joint/velocity" in sim.signals


def test_load_cell_reads_the_robot_weight_at_rest():
    sim = Simulation("SingleThrustFlapper", check=False)
    sim.add(BodyWrench(average_s=0.0))
    sim.advance(0.02)
    force = sim.signals["loadcell/force"]
    weight = float(np.sum(sim.model.body_mass[1:])) * 9.81
    assert abs(force[2]) == pytest.approx(weight, rel=0.02)
