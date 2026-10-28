"""Headless end-to-end runs for both base modes and the command line."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from flapsim import SimCore

ROOT = Path(__file__).resolve().parents[1]


def test_fixed_base_runs_headless():
    sim = SimCore("SingleThrustFlapper", fix=True)
    steps = sim.run(viewer=False, duration=0.01)
    assert 199 <= steps <= 201
    assert "F_world" in sim.aero.diagnostics


def test_floating_base_runs_headless():
    sim = SimCore("SingleThrustFlapper", fix=False)
    steps = sim.run(viewer=False, duration=0.01)
    assert 199 <= steps <= 201
    assert sim.aero.diagnostics["mode"] == "cycle_averaged_hover"
    assert sim.hovering_controller is not None


def test_unknown_robot_is_rejected():
    with pytest.raises(ValueError, match="Unknown robot"):
        SimCore("NoSuchRobot")


def test_cli_headless_run():
    result = subprocess.run(
        [
            sys.executable,
            "flapsim.py",
            "--robot",
            "SingleThrustFlapper",
            "--headless",
            "--duration",
            "0.005",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_cli_headless_requires_duration():
    result = subprocess.run(
        [sys.executable, "flapsim.py", "--robot", "SingleThrustFlapper", "--headless"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "--headless requires --duration" in result.stderr
