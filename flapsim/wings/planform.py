"""Wing planform integrals for blade-element aerodynamics.

The quasi-steady model needs chord-weighted radial moments of the planform.
With the non-dimensional span coordinate ``r_hat = r / R`` and chord
``c_hat = c(r) / c_mean`` they are

* ``r_kk  = integral c_hat * r_hat**k``       for k = 0..3 (normal force),
* ``z_cpk = integral c_hat**2 * r_hat**k``    for k = 0..2 (pitching moment),
* ``z_rd  = integral (|y_te| y_te**3 - |y_le| y_le**3) / c_mean**4``
  (rotational damping), where ``y`` is the chordwise distance from the
  feathering axis to the trailing and leading edges.

For a rectangular wing hinged at its leading edge all of them are
``1 / (k + 1)`` and ``z_rd = 1``.  Several constructors are offered: analytic
profiles, a chord polyline, a planform polygon and a mesh outline.
"""

from __future__ import annotations

import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from flapsim.config import WingParams


@dataclass(frozen=True)
class PlanformIntegrals:
    r00: float
    r11: float
    r22: float
    r33: float
    z_cp0: float
    z_cp1: float
    z_cp2: float
    z_rd: float
    c4: float = 1.0  # integral of c_hat**4 (added-mass rotational inertia)

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, float]) -> PlanformIntegrals:
        kwargs = {
            name: float(values[name])
            for name in cls.__dataclass_fields__
            if name in values
        }
        return cls(**kwargs)

    def replace(self, **changes: float) -> PlanformIntegrals:
        return PlanformIntegrals(**{**self.as_dict(), **changes})


RECTANGULAR = PlanformIntegrals(
    1.0, 0.5, 1.0 / 3.0, 0.25, 1.0, 0.5, 1.0 / 3.0, 1.0, 1.0
)


def _damping_factor(
    pitch_axis_fraction: float, le_offset_over_chord: float = 0.0
) -> float:
    """``(|y1| y1^3 - |y0| y0^3) / c^4`` for a chord spanning ``[y0, y1]``.

    ``y`` is the chordwise distance from the hinge: ``y0 = x_le - x0 c`` at the
    leading edge and ``y1 = x_le + (1 - x0) c`` at the trailing edge.
    """
    x0 = float(pitch_axis_fraction)
    if not 0.0 <= x0 <= 1.0:
        raise ValueError("pitch_axis_fraction must lie in [0, 1]")
    xle = float(le_offset_over_chord)
    if xle < 0.0:
        raise ValueError("leading edge offset must be non-negative")
    y0 = xle - x0
    y1 = xle + 1.0 - x0
    return abs(y1) * y1**3 - abs(y0) * y0**3


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def from_chord_distribution(
    r_hat, c_hat, *, pitch_axis_fraction: float = 0.0, samples: int = 20001
) -> PlanformIntegrals:
    """Integrals of a sampled chord distribution ``c(r_hat)`` on ``[0, 1]``.

    ``c_hat`` may carry any unit; it is normalised to a unit mean chord.  The
    points define a piecewise-linear chord, which is resampled on ``samples``
    points before trapezoid integration so that coarse polylines integrate
    accurately.
    """
    r = np.asarray(r_hat, dtype=float)
    c = np.asarray(c_hat, dtype=float)
    if r.ndim != 1 or r.shape != c.shape or r.size < 2:
        raise ValueError("r_hat and c_hat must be 1-D arrays of equal length >= 2")
    if np.any(np.diff(r) <= 0.0):
        raise ValueError("r_hat must be strictly increasing")
    if r[0] < -1e-12 or abs(r[-1] - 1.0) > 1e-9:
        raise ValueError("r_hat must start at or after 0 and end at 1")
    if np.any(c < 0.0):
        raise ValueError("chord values must be non-negative")
    if r[0] > 0.0:
        r = np.r_[0.0, r]
        c = np.r_[0.0, c]
    grid = np.linspace(0.0, 1.0, samples)
    c = np.interp(grid, r, c)
    r = grid
    mean = _trapezoid(c, r)
    if mean <= 0.0:
        raise ValueError("the chord distribution has zero area")
    c = c / mean
    radial = [_trapezoid(c * r**k, r) for k in range(4)]
    moment = [_trapezoid(c**2 * r**k, r) for k in range(3)]
    c4 = _trapezoid(c**4, r)
    z_rd = _damping_factor(pitch_axis_fraction) * c4
    return PlanformIntegrals(*radial, *moment, z_rd, c4)


def from_profile(
    profile: str, *, pitch_axis_fraction: float = 0.0, samples: int = 20001
) -> PlanformIntegrals:
    """Integrals of an analytic planform: ``rectangular`` or ``elliptic``."""
    if profile == "rectangular":
        return RECTANGULAR.replace(z_rd=_damping_factor(pitch_axis_fraction))
    if profile == "elliptic":
        r = np.linspace(0.0, 1.0, samples)
        c = (4.0 / math.pi) * np.sqrt(np.clip(1.0 - r**2, 0.0, None))
        return from_chord_distribution(
            r, c, pitch_axis_fraction=pitch_axis_fraction, samples=samples
        )
    raise ValueError(f"unknown planform profile {profile!r}")


def from_polyline(
    points, *, pitch_axis_fraction: float = 0.0
) -> tuple[float, float, PlanformIntegrals]:
    """Integrals of a chord polyline ``[(r, c), ...]`` in metres, root to tip.

    Returns ``(length, mean_chord, integrals)``.
    """
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) < 2:
        raise ValueError("points must be an (N, 2) array of (r, c) pairs, N >= 2")
    r, c = array[:, 0], array[:, 1]
    if r[0] < 0.0 or np.any(np.diff(r) <= 0.0):
        raise ValueError("span coordinates must be non-negative and increasing")
    length = float(r[-1])
    if length <= 0.0:
        raise ValueError("wing length must be positive")
    r_hat = r / length
    if r_hat[0] > 0.0:
        r_hat = np.r_[0.0, r_hat]
        c = np.r_[0.0, c]
    mean_chord = _trapezoid(c, r_hat * length) / length
    integrals = from_chord_distribution(
        r_hat, c, pitch_axis_fraction=pitch_axis_fraction
    )
    return length, mean_chord, integrals


@dataclass(frozen=True)
class PolygonPlanform:
    length: float
    mean_chord: float
    area: float
    integrals: PlanformIntegrals
    axis_offset_terms: tuple[float, float, float]
    pitch_axis_exact: bool


def _vertical_intervals(poly: np.ndarray, r: float, eps: float = 1e-12):
    """Chord intervals of a simple polygon at span coordinate ``r`` (even-odd)."""
    ys = []
    for (r0, y0), (r1, y1) in zip(poly, np.roll(poly, -1, axis=0), strict=True):
        if abs(r1 - r0) <= eps:
            continue
        if min(r0, r1) <= r < max(r0, r1):
            t = (r - r0) / (r1 - r0)
            ys.append(y0 + t * (y1 - y0))
    ys.sort()
    if len(ys) % 2:
        raise ValueError(f"polygon is self-intersecting near span coordinate {r}")
    return list(zip(ys[0::2], ys[1::2], strict=True))


def from_polygon(
    polygon,
    hinge_origin,
    span_direction,
    chord_direction,
    *,
    quadrature_order: int = 24,
    axis_tolerance: float = 0.01,
) -> PolygonPlanform:
    """Integrals of a planar planform polygon given in metres.

    ``polygon`` is an ``(N, 2)`` outline in any planar coordinates,
    ``hinge_origin`` the wing root on the feathering axis, ``span_direction``
    the axis towards the tip and ``chord_direction`` the axis from the leading
    edge towards the trailing edge.  The quasi-steady moment formula assumes
    the feathering axis runs along the leading edge; ``pitch_axis_exact`` in
    the result reports whether the outline satisfies that within
    ``axis_tolerance``.
    """
    poly = np.asarray(polygon, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
        raise ValueError("polygon must have shape (N, 2) with N >= 3")
    if not np.isfinite(poly).all():
        raise ValueError("polygon contains NaN or infinity")
    if np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    e_r = np.asarray(span_direction, dtype=float)
    e_y = np.asarray(chord_direction, dtype=float)
    if np.linalg.norm(e_r) == 0.0 or np.linalg.norm(e_y) == 0.0:
        raise ValueError("span and chord directions must be non-zero")
    e_r = e_r / np.linalg.norm(e_r)
    e_y = e_y / np.linalg.norm(e_y)
    if abs(float(e_r @ e_y)) > 1e-8:
        raise ValueError("span_direction and chord_direction must be perpendicular")
    rel = poly - np.asarray(hinge_origin, dtype=float)
    local = np.column_stack((rel @ e_r, rel @ e_y))
    tol = 1e-10 * max(1.0, float(np.ptp(local[:, 0])))
    if local[:, 0].min() < -tol:
        raise ValueError("part of the wing lies behind the hinge origin along the span")
    length = float(local[:, 0].max())
    if length <= 0.0:
        raise ValueError("wing length must be positive")

    breaks = np.unique(np.clip(local[:, 0], 0.0, length))
    if breaks[0] > 0.0:
        breaks = np.r_[0.0, breaks]
    gx, gw = np.polynomial.legendre.leggauss(quadrature_order)
    rows = []
    for lo, hi in zip(breaks[:-1], breaks[1:], strict=True):
        if hi - lo <= tol:
            continue
        for x, w in zip(
            0.5 * (hi - lo) * gx + 0.5 * (hi + lo), 0.5 * (hi - lo) * gw, strict=True
        ):
            intervals = _vertical_intervals(local, float(x))
            chord = sum(y1 - y0 for y0, y1 in intervals)
            rows.append((float(x), float(w), intervals, chord))
    if not rows:
        raise ValueError("no wing area found at r >= 0")

    def integrate(fn):
        return sum(w * fn(r, ints, c) for r, w, ints, c in rows)

    area = integrate(lambda r, ints, c: c)
    cbar = area / length
    radial = [
        integrate(lambda r, ints, c, k=k: c * r**k) / (cbar * length ** (k + 1))
        for k in range(4)
    ]
    moment = [
        integrate(lambda r, ints, c, k=k: c * c * r**k) / (cbar**2 * length ** (k + 1))
        for k in range(3)
    ]
    z_rd = integrate(
        lambda r, ints, c: sum(abs(y1) * y1**3 - abs(y0) * y0**3 for y0, y1 in ints)
    ) / (cbar**4 * length)
    c4 = integrate(lambda r, ints, c: c**4) / (cbar**4 * length)
    offsets = tuple(
        integrate(lambda r, ints, c, k=k: sum(y0 * (y1 - y0) for y0, y1 in ints) * r**k)
        / (cbar**2 * length ** (k + 1))
        for k in range(3)
    )
    exact = max(abs(v) for v in offsets) <= axis_tolerance * max(max(moment), 1e-15)
    return PolygonPlanform(
        length,
        cbar,
        area,
        PlanformIntegrals(*radial, *moment, z_rd, c4),
        offsets,
        exact,
    )


@dataclass(frozen=True)
class MeshPlanform:
    length: float
    mean_chord: float
    area: float
    integrals: PlanformIntegrals
    polyline: tuple[tuple[float, float], ...]


def read_stl_triangles(path: str | Path) -> np.ndarray:
    """Return the triangles of a binary or ASCII STL file as ``(N, 3, 3)``."""
    raw = Path(path).read_bytes()
    is_binary = False
    if len(raw) >= 84:
        (count,) = struct.unpack_from("<I", raw, 80)
        is_binary = len(raw) == 84 + 50 * count
    if is_binary:
        body = np.frombuffer(raw, dtype=np.uint8, offset=84)
        records = body.reshape(count, 50)
        floats = records[:, :48].copy().view("<f4").reshape(count, 12)
        return floats[:, 3:].reshape(-1, 3, 3).astype(float)
    else:
        text = raw.decode("ascii", errors="ignore").split()
        vertices = []
        for index, token in enumerate(text):
            if token == "vertex":
                vertices.append([float(v) for v in text[index + 1 : index + 4]])
        if not vertices or len(vertices) % 3:
            raise ValueError(f"{path} is neither a binary nor an ASCII STL file")
        return np.asarray(vertices, dtype=float).reshape(-1, 3, 3)


def read_stl_vertices(path: str | Path) -> np.ndarray:
    """Return the unique vertices of a binary or ASCII STL file as ``(N, 3)``."""
    return np.unique(read_stl_triangles(path).reshape(-1, 3), axis=0)


def _sample_triangle_edges(triangles: np.ndarray, samples_per_edge: int) -> np.ndarray:
    """Points along every triangle edge, so coarse meshes still fill the bins."""
    t = np.linspace(0.0, 1.0, samples_per_edge)[:, None, None]
    points = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        start = triangles[:, a, :][None, :, :]
        end = triangles[:, b, :][None, :, :]
        points.append((start * (1.0 - t) + end * t).reshape(-1, 3))
    return np.unique(np.concatenate(points), axis=0)


def from_mesh(
    path: str | Path,
    *,
    span_axis,
    chord_axis,
    hinge_origin=(0.0, 0.0, 0.0),
    bins: int = 64,
    scale: float = 1.0,
    samples_per_edge: int = 64,
) -> MeshPlanform:
    """Approximate planform integrals from a wing mesh (STL).

    Vertices are projected onto the ``span_axis``/``chord_axis`` plane and
    binned along the span; each bin's chord is the extent of its vertices in
    the chord direction.  ``chord_axis`` points from the leading edge towards
    the trailing edge and ``hinge_origin`` lies on the feathering axis.
    ``scale`` converts the mesh unit to metres (``1e-3`` for millimetre
    meshes).  The result is an approximation whose quality depends on the
    mesh resolution and on ``bins``.
    """
    vertices = (
        _sample_triangle_edges(read_stl_triangles(path), samples_per_edge) * scale
    )
    e_r = np.asarray(span_axis, dtype=float)
    e_y = np.asarray(chord_axis, dtype=float)
    e_r = e_r / np.linalg.norm(e_r)
    e_y = e_y / np.linalg.norm(e_y)
    if abs(float(e_r @ e_y)) > 1e-8:
        raise ValueError("span_axis and chord_axis must be perpendicular")
    rel = vertices - np.asarray(hinge_origin, dtype=float) * scale
    r = rel @ e_r
    y = rel @ e_y
    keep = r >= 0.0
    if keep.sum() < 3:
        raise ValueError("fewer than three mesh vertices lie at r >= 0")
    r, y = r[keep], y[keep]
    length = float(r.max())
    edges = np.linspace(0.0, length, bins + 1)
    centres, chords, damping = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (r >= lo) & (r <= hi)
        if mask.sum() < 2:
            continue
        y_lo, y_hi = float(y[mask].min()), float(y[mask].max())
        centres.append(0.5 * (lo + hi))
        chords.append(y_hi - y_lo)
        # integral of 4|y|^3 across the chord, see the module docstring
        damping.append(abs(y_hi) * y_hi**3 - abs(y_lo) * y_lo**3)
    if len(centres) < 2:
        raise ValueError("not enough populated span bins; lower `bins`")
    centres = np.array(centres)
    chords = np.array(chords)
    damping = np.array(damping)
    r_hat = np.r_[0.0, centres / length, 1.0]
    c = np.r_[chords[0], chords, chords[-1]]
    dmp = np.r_[damping[0], damping, damping[-1]]
    area = _trapezoid(c, r_hat * length)
    mean_chord = area / length
    integrals = from_chord_distribution(r_hat, c)
    integrals = integrals.replace(z_rd=_trapezoid(dmp, r_hat) / mean_chord**4)
    polyline = tuple((float(a), float(b)) for a, b in zip(centres, chords, strict=True))
    return MeshPlanform(length, mean_chord, area, integrals, polyline)


def resolve_planform(planform) -> tuple[float, float, PlanformIntegrals]:
    """Turn a manifest :class:`~flapsim.robot.manifest.Planform` into numbers."""
    x0 = planform.pitch_axis_fraction
    x_le = planform.leading_edge_offset
    if planform.profile == "integrals":
        return (
            float(planform.length),
            float(planform.mean_chord),
            PlanformIntegrals.from_dict(planform.integrals),
        )
    if planform.profile == "polyline":
        length, mean_chord, integrals = from_polyline(
            planform.chord_polyline, pitch_axis_fraction=x0
        )
        if planform.length is not None:
            length = float(planform.length)
        if planform.mean_chord is not None:
            mean_chord = float(planform.mean_chord)
    else:
        length = float(planform.length)
        mean_chord = float(planform.mean_chord)
        integrals = from_profile(planform.profile, pitch_axis_fraction=x0)
    if x_le > 0.0:
        # Rotational damping grows quickly with the hinge offset; the offset is
        # scaled by the mean chord for analytic and polyline profiles.
        factor = _damping_factor(x0, x_le / mean_chord) / _damping_factor(x0)
        integrals = integrals.replace(z_rd=integrals.z_rd * factor)
    return length, mean_chord, integrals


def wing_params(
    length: float,
    mean_chord: float,
    integrals: PlanformIntegrals,
    *,
    rho: float = 1.225,
    d0: float = 0.0,
    ds: float = 0.0,
    x_le: float = 0.0,
    x0: float = 0.0,
) -> WingParams:
    """Assemble the flat parameter object used by the quasi-steady model."""
    return WingParams(
        R=float(length),
        c=float(mean_chord),
        r33=integrals.r33,
        r22=integrals.r22,
        r11=integrals.r11,
        r00=integrals.r00,
        z_cp2=integrals.z_cp2,
        z_cp1=integrals.z_cp1,
        z_cp0=integrals.z_cp0,
        z_rd=integrals.z_rd,
        rho=float(rho),
        d0=float(d0),
        ds=float(ds),
        x_le=float(x_le),
        x0=float(x0),
    )


__all__ = [
    "RECTANGULAR",
    "MeshPlanform",
    "PlanformIntegrals",
    "PolygonPlanform",
    "from_chord_distribution",
    "from_mesh",
    "from_polygon",
    "from_polyline",
    "from_profile",
    "read_stl_triangles",
    "read_stl_vertices",
    "resolve_planform",
    "wing_params",
]
