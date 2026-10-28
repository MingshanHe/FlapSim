"""Actuator commands, limits and the memory recorder."""

from __future__ import annotations

import math

import numpy as np
import pytest

from flapsim.actuators import PDPosition, SinusoidalStroke
from flapsim.core import Simulation
from flapsim.io import MemoryRecorder


@pytest.fixture
def sim():
    return Simulation("SingleThrustFlapper", check=False)


def test_sinusoidal_stroke_tracks_commands_and_limits(sim):
    stroke = sim.add(
        SinusoidalStroke(
            joints=["left_flapping_joint", "right_wing_flapping"],
            directions=[1, -1],
            frequency_hz=20.0,
            amplitude_rad=1.0,
        )
    )
    sim.advance(0.0125)  # quarter period: sin = 1
    left = sim.data.qpos[sim.model.joint("left_flapping_joint").qposadr[0]]
    right = sim.data.qpos[sim.model.joint("right_wing_flapping").qposadr[0]]
    assert left == pytest.approx(1.0, abs=0.01) and right == pytest.approx(
        -1.0, abs=0.01
    )
    with pytest.raises(ValueError, match="exceeds the joint limit"):
        stroke.set_command(amplitude_rad=2.0)
    sim.signals.publish("cmd/stroke/frequency_hz", 30.0)
    sim.step()
    assert stroke.frequency_hz == 30.0 and sim.signals["stroke/frequency_hz"] == 30.0
    assert 0.0 <= sim.signals["stroke/phase"] < 2 * math.pi


def test_sinusoidal_stroke_validation():
    with pytest.raises(ValueError, match="directions must match"):
        SinusoidalStroke(joints=["a", "b"], directions=[1])
    with pytest.raises(ValueError, match="mode must be"):
        SinusoidalStroke(joints=["a"], mode="wobbly")
    sim = Simulation("SingleThrustFlapper", check=False)
    with pytest.raises(ValueError, match="not found"):
        sim.add(SinusoidalStroke(joints=["nope"]))


def test_pd_position_holds_targets_and_clips_commands(sim):
    gimbal = sim.add(
        PDPosition(
            joints={
                "pitch_joint": {
                    "role": "pitch",
                    "kp": 0.5,
                    "kd": 0.05,
                    "max_torque": 0.5,
                },
                "roll_joint": {
                    "role": "roll",
                    "kp": 0.5,
                    "kd": 0.05,
                    "max_torque": 0.5,
                },
            }
        )
    )
    gimbal.set_role_target("pitch", 0.2)
    sim.advance(0.5)
    pitch = sim.data.qpos[sim.model.joint("pitch_joint").qposadr[0]]
    assert pitch == pytest.approx(0.2, abs=0.02)
    with pytest.raises(ValueError, match="outside"):
        gimbal.set_target("pitch_joint", 1.0)
    sim.signals.publish("cmd/gimbal/roll_joint", 5.0)
    sim.step()
    assert gimbal.targets["roll_joint"] == pytest.approx(
        0.3
    )  # clipped to the joint range
    assert sim.signals["gimbal/roll_joint"] == pytest.approx(0.3)
    with pytest.raises(KeyError, match="no joint with role"):
        gimbal.joint_for_role("yaw")


def test_pd_position_kinematic_mode():
    sim = Simulation("SingleThrustFlapper", base="free", check=False)
    gimbal = sim.add(
        PDPosition(joints={"pitch_joint": {"target": 0.1}}, mode="kinematic")
    )
    sim.advance(0.002)
    assert sim.data.qpos[sim.model.joint("pitch_joint").qposadr[0]] == pytest.approx(
        0.1
    )
    gimbal.set_target("pitch_joint", -0.1)
    sim.step()
    assert sim.data.qpos[sim.model.joint("pitch_joint").qposadr[0]] == pytest.approx(
        -0.1
    )


def test_memory_recorder_sampling_and_export(sim, tmp_path):
    sim.add("aero/quasi_steady")
    recorder = sim.add(
        MemoryRecorder(signals=["time", "aero/F_world", "wings/left/F_N"], every=1e-3)
    )
    sim.advance(0.0105)
    assert 10 <= len(recorder) <= 12
    assert recorder["aero/F_world"].shape == (len(recorder), 3)
    assert recorder["wings/left/F_N"].shape == (len(recorder),)
    np.testing.assert_allclose(np.diff(recorder.times), 1e-3, atol=1e-4)
    npz = recorder.to_npz(tmp_path / "out.npz")
    loaded = np.load(npz)
    assert "aero__F_world" in loaded
    csv_path = recorder.to_csv(tmp_path / "out.csv")
    header = csv_path.read_text().splitlines()[0].split(",")
    assert header[:2] == ["time", "aero/F_world[0]"]
    recorder.reset()
    assert len(recorder) == 0
    with pytest.raises(KeyError, match="was not recorded"):
        recorder["nothing"]
