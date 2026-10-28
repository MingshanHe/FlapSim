"""Every built-in plugin with an example config passes the conformance check."""

from __future__ import annotations

import pytest

from flapsim.plugins import Plugin
from flapsim.testing import check_plugin, example_plugin_factories

FACTORIES = example_plugin_factories()
FREE_BASE = {"controller/hover_pid", "aero/cycle_averaged"}


@pytest.mark.parametrize("type_name", sorted(FACTORIES))
def test_builtin_plugin_conforms(type_name):
    base = "free" if type_name in FREE_BASE else None
    report = check_plugin(FACTORIES[type_name], base=base, steps=20)
    assert report["type"] == type_name


def test_conformance_catches_missing_provides():
    class Liar(Plugin):
        kind = "test"
        name = "liar"
        provides = ("never_published",)

    with pytest.raises(AssertionError, match="did not publish"):
        check_plugin(Liar, steps=2)
