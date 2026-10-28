# Simulation and plugins

::: flapsim.core.simulation.Simulation

::: flapsim.core.signals.SignalBus

::: flapsim.core.config
    options:
      members: [load_sim_config, build_simulation, SimConfigError]

::: flapsim.plugins.base
    options:
      members: [Plugin, ForcePlugin, AeroPlugin, AtmospherePlugin, ActuatorPlugin, SensorPlugin, ControllerPlugin, RecorderPlugin]

::: flapsim.plugins.registry
    options:
      members: [register, get, available, build, discover_entry_points]

::: flapsim.testing.conformance
