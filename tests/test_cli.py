"""Command line: inspect and run."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from flapsim.cli import main
from flapsim.robot import describe_model, load, model_summary

ROOT = Path(__file__).resolve().parents[1]


def test_inspect_text_lists_wings(capsys):
    assert main(["inspect", "SingleThrustFlapper"]) == 0
    out = capsys.readouterr().out
    assert "left_wing_feathering" in out
    assert "wings:" in out
    assert "stroke=left_flapping_joint" in out


def test_inspect_json(capsys):
    assert main(["inspect", "SingleThrustFlapper", "--json", "--base", "free"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["model"] == "SingleThrustFlapper"
    assert any(j["type"] == "free" for j in summary["joints"])
    assert summary["wings"][1]["side"] == "right"


def test_describe_bare_model_suggests_a_manifest():
    robot = load(
        ROOT
        / "flapsim/assets/robots/single_thrust_flapper/urdf/SingleThrustFlapper.urdf"
    )
    text = describe_model(robot.model, robot.manifest, robot.wings)
    assert "no wings declared" in text
    assert model_summary(robot.model)["nq"] == 6


def test_run_headless_via_module():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flapsim.cli",
            "run",
            "SingleThrustFlapper",
            "--headless",
            "--duration",
            "0.005",
            "--frequency",
            "30",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_run_headless_requires_duration():
    assert main(["run", "--headless"]) == 2
