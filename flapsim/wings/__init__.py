"""Wings as first-class objects: planform integrals and runtime wing state."""

from .planform import (
    RECTANGULAR,
    MeshPlanform,
    PlanformIntegrals,
    PolygonPlanform,
    from_chord_distribution,
    from_mesh,
    from_polygon,
    from_polyline,
    from_profile,
    resolve_planform,
    wing_params,
)
from .wing import Wing, WingState

__all__ = [
    "RECTANGULAR",
    "MeshPlanform",
    "PlanformIntegrals",
    "PolygonPlanform",
    "Wing",
    "WingState",
    "from_chord_distribution",
    "from_mesh",
    "from_polygon",
    "from_polyline",
    "from_profile",
    "resolve_planform",
    "wing_params",
]
