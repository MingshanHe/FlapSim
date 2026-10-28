"""DC motor crank, servo, setpoint schedule and callback controller."""

from __future__ import annotations

import math

import pytest

from flapsim.actuators import DCMotorCrank, Servo
from flapsim.controllers import CallbackController, SetpointSchedule
from flapsim.core import Simulation

STROKE_JOINTS = ["left_flapping_joint", "right_wing_flapping"]


def test_dc_motor_reaches_no_load_speed_without_load():
    sim = Simulation("SingleThrustFlapper", check=False)
    motor = sim.add(
        DCMotorCrank(
            joints=STROKE_JOINTS,
            directions=[1, -1],
            amplitude_rad=1.5,
            voltage_v=2.0,
            kt=1e-3,
            resistance_ohm=1.0,
            inertia=1e-8,
            viscous=0.0,
            load="none",
            gear_ratio=10.0,
        )
    )
    sim.advance(0.2)
    assert motor.omega == pytest.approx(motor.no_load_speed(), rel=0.01)
    assert sim.signals["motor/frequency_hz"] == pytest.approx(
        2000.0 / (2 * math.pi * 10.0), rel=0.01
    )
    assert sim.signals["motor/current_a"] == pytest.approx(0.0, abs=0.02)
    q = sim.data.qpos[sim.model.joint("left_flapping_joint").qposadr[0]]
    assert abs(q) <= 1.5 + 1e-9


def test_dc_motor_slows_under_aerodynamic_load_and_follows_voltage_commands():
    def spin(load: str) -> float:
        sim = Simulation("SingleThrustFlapper")
        sim.add(
            DCMotorCrank(
                joints=STROKE_JOINTS,
                directions=[1, -1],
                amplitude_rad=1.5,
                voltage_v=3.0,
                kt=2e-3,
                resistance_ohm=2.0,
                inertia=5e-8,
                viscous=0.0,
                load=load,
                gear_ratio=10.0,
                initial_frequency_hz=20.0,
            )
        )
        sim.add("aero/quasi_steady")
        sim.advance(0.3)
        return sim["motor"].frequency_hz, sim.signals["motor/power_w"]

    free, _ = spin("none")
    loaded, power = spin("aero")
    assert loaded < free * 0.95
    assert power > 0.0

    sim = Simulation("SingleThrustFlapper", check=False)
    motor = sim.add(
        DCMotorCrank(
            joints=STROKE_JOINTS, directions=[1, -1], load="none", max_voltage=5.0
        )
    )
    sim.signals.publish("cmd/motor/voltage_v", 12.0)
    sim.step()
    assert motor.voltage == 5.0  # clipped
    with pytest.raises(ValueError, match="load must be"):
        DCMotorCrank(joints=STROKE_JOINTS, load="magnetic")


def test_servo_rate_limits_a_step():
    sim = Simulation("SingleThrustFlapper", check=False)
    servo = sim.add(
        Servo(
            joints={"pitch_joint": {"kp": 1.0, "kd": 0.05, "max_torque": 1.0}},
            rate_limit_rad_s=1.0,
            id="servo",
        )
    )
    sim.signals.publish("cmd/servo/pitch_joint", 0.25)
    sim.advance(0.1)
    assert servo.targets["pitch_joint"] == pytest.approx(0.1, abs=2e-4)
    assert sim.signals["servo/pitch_joint/desired"] == 0.25
    sim.advance(0.2)
    assert servo.targets["pitch_joint"] == pytest.approx(0.25, abs=1e-6)
    lagged = Servo(
        joints={"pitch_joint": {}}, time_constant_s=0.05, rate_limit_rad_s=100.0
    )
    assert lagged.time_constant == 0.05
    with pytest.raises(ValueError):
        Servo(joints={"pitch_joint": {}}, rate_limit_rad_s=-1.0)


def test_setpoint_schedule_steps_and_interpolates():
    sim = Simulation("SingleThrustFlapper", check=False)
    stroke = sim.add(
        {
            "type": "actuator/sinusoidal_stroke",
            "id": "stroke",
            "joints": STROKE_JOINTS,
            "directions": [1, -1],
            "frequency_hz": 20.0,
            "amplitude_rad": 1.0,
        }
    )
    schedule = sim.add(
        SetpointSchedule(
            target="stroke",
            schedule={"frequency_hz": [[0.01, 30.0], [0.03, 40.0]]},
            interpolate=True,
        )
    )
    sim.advance(0.005)
    assert stroke.frequency_hz == 20.0  # before the first entry: default
    sim.advance(0.015)  # t = 0.02: halfway between 30 and 40
    assert stroke.frequency_hz == pytest.approx(35.0, abs=0.3)
    sim.advance(0.02)
    assert stroke.frequency_hz == pytest.approx(40.0)
    assert schedule.value_at("frequency_hz", 0.0) is None
    stepped = SetpointSchedule(target="x", schedule={"a": [[0.0, 1.0], [1.0, 2.0]]})
    assert stepped.value_at("a", 0.5) == 1.0 and stepped.value_at("a", 1.0) == 2.0
    with pytest.raises(ValueError, match="non-decreasing"):
        SetpointSchedule(target="x", schedule={"a": [[1.0, 1.0], [0.0, 2.0]]})


def test_callback_controller_publishes_commands():
    sim = Simulation("SingleThrustFlapper", check=False)
    stroke = sim.add(
        {
            "type": "actuator/sinusoidal_stroke",
            "id": "stroke",
            "joints": STROKE_JOINTS,
            "directions": [1, -1],
            "frequency_hz": 20.0,
            "amplitude_rad": 1.0,
        }
    )

    def ramp(simulation, controller):
        controller.state["calls"] = controller.state.get("calls", 0) + 1
        return {
            "stroke/frequency_hz": 20.0 + 100.0 * simulation.time,
            "cmd/stroke/amplitude_rad": 1.2,
        }

    ctl = sim.add(CallbackController(callback=ramp, rate_hz=100.0))
    sim.advance(0.1)
    assert ctl.state["calls"] == pytest.approx(10, abs=1)
    assert stroke.frequency_hz == pytest.approx(30.0, abs=1.5)
    assert stroke.amplitude_rad == 1.2
    with pytest.raises(TypeError):
        CallbackController(callback="not callable")
