"""Position/attitude hold using the same PID and actuator interface as movement."""
from .move import MoveController


class HoveringController(MoveController):
    """A fixed SixDOFCommand target; no automatic waypoint changes."""
