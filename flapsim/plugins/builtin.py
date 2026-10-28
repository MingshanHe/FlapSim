"""Import every built-in plugin module so that they register themselves."""

from flapsim import actuators, aero, controllers, io  # noqa: F401

__all__: list[str] = []
