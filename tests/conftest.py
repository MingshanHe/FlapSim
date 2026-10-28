"""Shared fixtures for the FlapSim test suite."""

from __future__ import annotations

import sys

import pytest

# Keep the source tree free of bytecode when tests import the package.
sys.dont_write_bytecode = True


@pytest.fixture(scope="session")
def fixed_base_sim():
    """A fixed-base SingleThrustFlapper with the quasi-steady aero model."""
    from flapsim import SimCore

    return SimCore("SingleThrustFlapper", fix=True)
