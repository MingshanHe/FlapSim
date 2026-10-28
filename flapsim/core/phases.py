"""Step phases, in execution order.

``pre_control`` (sensors read the state), ``control`` (controllers publish
commands), ``actuate`` (actuators drive joints), ``forces`` (force models write
``xfrc_applied``), then ``mj_step``, ``post_step`` (kinematic actuators
re-impose their targets; returns whether ``mj_forward`` is needed) and
``record`` (recorders sample signals).
"""

PHASES = ("pre_control", "control", "actuate", "forces", "post_step", "record")

__all__ = ["PHASES"]
