"""Plugin registry, discovery, lifecycle and the simulation step pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from flapsim.core import Simulation
from flapsim.plugins import Plugin, available, build, get, parse_type, register
from flapsim.plugins.registry import PluginError, discover_entry_points


class Tracer(Plugin):
    kind = "test"
    name = "tracer"
    provides = ("calls",)

    def __init__(self, *, log, id=None, **config):
        super().__init__(id=id, **config)
        self.log = log

    def setup(self):
        self.log.append((self.id, "setup"))

    def reset(self):
        self.log.append((self.id, "reset"))

    def pre_control(self):
        self.log.append((self.id, "pre_control"))

    def control(self):
        self.log.append((self.id, "control"))

    def actuate(self):
        self.log.append((self.id, "actuate"))

    def forces(self):
        self.log.append((self.id, "forces"))

    def post_step(self):
        self.log.append((self.id, "post_step"))
        return False

    def record(self):
        self.log.append((self.id, "record"))
        self.publish("calls", len(self.log))


def test_builtin_plugins_are_registered():
    kinds = available()
    assert "aero/quasi_steady" in [
        f"{k}/{n}" for k, names in kinds.items() for n in names
    ]
    assert get("actuator/sinusoidal_stroke").name == "sinusoidal_stroke"
    assert get("recorder", "memory").kind == "recorder"
    with pytest.raises(PluginError, match="unknown plugin"):
        get("aero/no_such_model")
    with pytest.raises(PluginError, match="kind/name"):
        parse_type("quasi_steady")


def test_register_rejects_duplicates_and_non_plugins():
    @register
    class Dummy(Plugin):
        kind = "test"
        name = "dummy_once"

    register(Dummy)  # same class again is fine

    class Other(Plugin):
        kind = "test"
        name = "dummy_once"

    with pytest.raises(PluginError, match="already registered"):
        register(Other)

    class NoName(Plugin):
        kind = "test"

    with pytest.raises(PluginError, match="no 'name'"):
        register(NoName)
    with pytest.raises(PluginError):
        register(object)


def test_phase_order_and_lifecycle():
    log = []
    sim = Simulation("SingleThrustFlapper", check=False)
    a = sim.add(Tracer(log=log, id="a"))
    b = sim.add(Tracer(log=log, id="b"))
    assert log == [("a", "setup"), ("b", "setup")]
    log.clear()
    sim.step()
    phases = [entry[1] for entry in log if entry[0] == "a"]
    assert phases == [
        "pre_control",
        "control",
        "actuate",
        "forces",
        "post_step",
        "record",
    ]
    # within a phase, plugins run in insertion order
    assert (
        log.index(("a", "control"))
        < log.index(("b", "control"))
        < log.index(("a", "actuate"))
    )
    assert sim.signals["a/calls"] > 0 and sim.signals["time"] == pytest.approx(
        sim.timestep
    )
    assert sim.steps == 1

    log.clear()
    sim.reset()
    assert ("a", "reset") in log and sim.steps == 0 and sim.time == 0.0
    assert "a/calls" not in sim.signals

    with pytest.raises(ValueError, match="already attached"):
        sim.add(Tracer(log=log, id="a"))
    sim.remove("a")
    assert a.sim is None and "a" not in sim and b in sim.plugins
    with pytest.raises(KeyError):
        sim.get("a")


def test_build_from_type_string_and_mapping():
    stroke = build(
        {
            "type": "actuator/sinusoidal_stroke",
            "id": "s",
            "joints": ["left_flapping_joint"],
            "frequency_hz": 10.0,
            "amplitude_rad": 0.5,
        }
    )
    assert stroke.id == "s" and stroke.frequency_hz == 10.0
    recorder = build("recorder/memory", signals=["time"])
    assert recorder.patterns == ["time"]
    with pytest.raises(PluginError, match="needs a 'type'"):
        build({"id": "x"})


def test_entry_point_discovery(monkeypatch):
    import flapsim.plugins.registry as registry

    class FromEntryPoint(Plugin):
        kind = "test"
        name = "from_entry_point"

    class FakeEntryPoint:
        name = "fake"

        def load(self):
            return FromEntryPoint

    monkeypatch.setattr(
        registry.metadata, "entry_points", lambda group=None: [FakeEntryPoint()]
    )
    loaded = discover_entry_points(force=True)
    assert "fake" in loaded
    assert get("test/from_entry_point") is FromEntryPoint


def test_signal_bus_patterns_and_snapshot():
    sim = Simulation("SingleThrustFlapper", check=False)
    sim.signals.publish("wings/left/alpha", 0.5)
    sim.signals.publish("wings/right/alpha", -0.5)
    sim.signals.publish("aero/F_world", [0.0, 0.0, 1.0])
    assert sim.signals.matching("wings/*/alpha") == [
        "wings/left/alpha",
        "wings/right/alpha",
    ]
    snapshot = sim.signals.snapshot(["aero/F_world"])
    np.testing.assert_allclose(snapshot["aero/F_world"], [0.0, 0.0, 1.0])
    with pytest.raises(KeyError, match="no signal"):
        sim.signals["nope"]


def test_describe_is_json_serialisable():
    import json

    sim = Simulation("SingleThrustFlapper", check=False)
    sim.add("aero/quasi_steady")
    sim.add(build("recorder/memory", signals=["aero/*"], every=1e-3))
    description = sim.describe()
    json.dumps(description)
    assert [p["id"] for p in description["plugins"]] == ["aero", "recorder"]
