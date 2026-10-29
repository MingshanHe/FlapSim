"""Simulation configs: files, overrides, the shipped setups and the CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from flapsim.cli import main
from flapsim.core import Simulation
from flapsim.core.config import SimConfigError, is_sim_config, load_sim_config
from flapsim.robot import builtin_robots

ROBOT_DIR = builtin_robots()["SingleThrustFlapper"].parent
ROOT = Path(__file__).resolve().parents[1]


def test_shipped_fixed_setup_runs_and_publishes_wing_signals():
    sim = Simulation.from_config(ROBOT_DIR / "sim.yaml")
    assert [p.id for p in sim.plugins] == ["stroke", "gimbal", "aero"]
    assert sim.fixed_base and sim.run_config["duration"] == 2.0
    sim.advance(0.01)
    assert "wings/left/alpha" in sim.signals and "aero/F_world" in sim.signals
    assert sim.signals["stroke/frequency_hz"] == 35.0
    assert sim["gimbal"].joint_for_role("pitch") == "pitch_joint"


def test_shipped_hover_setup_runs_with_free_base():
    sim = Simulation.from_config(ROBOT_DIR / "hover.yaml")
    assert not sim.fixed_base
    assert [p.id for p in sim.plugins] == ["stroke", "gimbal", "aero", "hover"]
    sim.advance(0.1)
    assert sim.signals["aero/lift"] > 0.0
    assert "cmd/stroke/frequency_hz" in sim.signals
    assert sim.data.xpos[sim.robot.root_body][2] > 0.3


def test_base_override_and_plugin_overrides():
    sim = Simulation.from_config(
        ROBOT_DIR / "sim.yaml", overrides=[{"id": "stroke", "frequency_hz": 20.0}]
    )
    assert sim["stroke"].frequency_hz == 20.0
    with pytest.raises(SimConfigError, match="unknown plugin id"):
        Simulation.from_config(ROBOT_DIR / "sim.yaml", overrides=[{"id": "nope"}])


def test_config_validation(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema: flapsim.sim/1\nplugins: []\n")
    with pytest.raises(SimConfigError, match="needs a 'robot'"):
        load_sim_config(bad)
    bad.write_text("schema: flapsim.sim/1\nrobot: SingleThrustFlapper\nbogus: 1\n")
    with pytest.raises(SimConfigError, match="unknown top-level"):
        load_sim_config(bad)
    bad.write_text(
        "schema: flapsim.sim/1\nrobot: SingleThrustFlapper\nplugins: [{id: x}]\n"
    )
    with pytest.raises(SimConfigError, match="needs a 'type'"):
        load_sim_config(bad)
    assert is_sim_config(ROBOT_DIR / "sim.yaml")
    assert not is_sim_config(ROBOT_DIR / "robot.yaml")


def test_config_with_builtin_robot_name_and_relative_paths(tmp_path):
    cfg = tmp_path / "my.yaml"
    cfg.write_text(
        """schema: flapsim.sim/1
robot: SingleThrustFlapper
plugins:
  - type: actuator/sinusoidal_stroke
    joints: [left_flapping_joint, right_wing_flapping]
    directions: [1, -1]
    frequency_hz: 30
    amplitude_rad: 1.0
  - type: aero/quasi_steady
  - type: recorder/memory
    signals: [aero/F_world, wings/*/alpha]
    every: 1.0e-3
run: {duration: 0.02}
"""
    )
    sim = Simulation.from_config(cfg)
    sim.run(sim.run_config["duration"])
    recorder = sim["recorder"]
    assert len(recorder) >= 20
    assert recorder["aero/F_world"].shape[1] == 3
    assert "wings/right/alpha" in recorder.names()


def test_cli_run_describe_and_plugins(capsys):
    assert main(["run", "SingleThrustFlapper", "--describe"]) == 0
    description = json.loads(capsys.readouterr().out)
    assert description["robot"] == "SingleThrustFlapper"
    assert main(["plugins", "--kind", "aero"]) == 0
    out = capsys.readouterr().out
    assert "aero/quasi_steady" in out and "aero/cycle_averaged" in out


def test_cli_run_headless_hover_via_base_flag():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flapsim.cli",
            "run",
            "SingleThrustFlapper",
            "--base",
            "free",
            "--headless",
            "--duration",
            "0.01",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_cli_run_config_file_headless():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flapsim.cli",
            "run",
            str(ROBOT_DIR / "sim.yaml"),
            "--headless",
            "--duration",
            "0.005",
            "--frequency",
            "25",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
