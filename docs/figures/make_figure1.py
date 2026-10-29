"""Figure 1: one-canvas visual overview of FlapSim in MuJoCo.

A large hero render of the flapping-wing robot (translucent housing showing
the gear train, stroke sweep, wing-tip paths, blade-element forces) carries
the physics callouts; four floating cards show the top view of the stroke,
free flight, wind and gusts, and a parametric robot built without CAD.
Reproduce from the repository root with

    MUJOCO_GL=osmesa PYTHONPATH=. python docs/figures/make_figure1.py

Writes ``figure1.png`` (600 dpi), ``figure1.pdf`` and the individual renders
to ``docs/assets/figure1``.
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib import font_manager  # noqa: E402

for _font in Path("/usr/share/fonts/opentype/inter").glob("Inter*.otf"):
    font_manager.fontManager.addfont(str(_font))

import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

from flapsim.core import Simulation  # noqa: E402
from flapsim.core.config import load_sim_config  # noqa: E402
from flapsim.io import add_arrow, add_marker  # noqa: E402
from flapsim.robot import RobotManifest, builtin_robots, flat_plate_simulation, load  # noqa: E402
from flapsim.wings.planform import read_stl_triangles  # noqa: E402

ROBOT_DIR = builtin_robots()["SingleThrustFlapper"].parent
FAMILY = (
    "Inter"
    if any(f.name == "Inter" for f in font_manager.fontManager.ttflist)
    else "DejaVu Sans"
)
plt.rcParams.update(
    {
        "font.family": FAMILY,
        "font.size": 7,
        "mathtext.fontset": "custom",
        "mathtext.it": f"{FAMILY}:italic",
        "mathtext.rm": FAMILY,
        "figure.facecolor": "white",
        "pdf.fonttype": 42,
    }
)
INK, INK2, RULE = "#1d1d1b", "#55544f", "#d6d5cf"

# --------------------------------------------------------------- palette ----
# Robot parts (RGBA 0-1), after the CAD renders: translucent housing so the
# gear train shows, saturated gears and linkage, slate mount, amber wings.
HOUSING = (0.78, 0.82, 0.88, 0.24)
LINK_PLATE = (0.106, 0.604, 0.667, 1.0)  # teal accent
CRANK_GEAR = (0.86, 0.52, 0.10, 1.0)  # deep amber
GEAR_A = (0.98, 0.76, 0.25, 1.0)  # light amber
GEAR_B = (0.08, 0.45, 0.50, 1.0)  # dark teal
SHAFT = (0.58, 0.61, 0.66, 1.0)
GIMBAL = (0.106, 0.604, 0.667, 1.0)  # teal accent
ARM = (0.88, 0.55, 0.12, 1.0)  # deep amber crank arms
MOUNT = (0.40, 0.44, 0.50, 1.0)  # slate gray
WING = (0.97, 0.66, 0.12, 0.74)  # translucent amber
WING_GHOST_ALPHA = 0.12
TORSO = (0.40, 0.44, 0.50, 1.0)
FORCE = (0.83, 0.12, 0.13, 1.0)
TIP_PATH = (0.47, 0.25, 0.06, 0.95)
TRAIL = (0.106, 0.604, 0.667, 0.95)  # teal flight path
TARGET = (0.95, 0.62, 0.15, 0.55)  # amber waypoints
WIND = (0.45, 0.52, 0.62, 0.9)  # slate wind arrows
HIDDEN_GROUP = 4

# Parts of the hub mesh (link2), identified by their centre along the shaft
# (local y, millimetres).  The largest part is the housing.
HUB_PARTS = [
    (-44.7, LINK_PLATE, "hub_link_plate"),
    (-42.7, CRANK_GEAR, "hub_crank_gear"),
    (-38.6, GEAR_A, "hub_gear_a"),
    (-33.5, GEAR_B, "hub_gear_b"),
    (-31.3, SHAFT, "hub_collar"),
    (-26.3, GEAR_A, "hub_gear_c"),
]


# ------------------------------------------------------------ robot build ----
def _write_stl(path: Path, triangles: np.ndarray) -> None:
    with path.open("wb") as handle:
        handle.write(b"\0" * 80)
        handle.write(struct.pack("<I", len(triangles)))
        for tri in triangles.astype(np.float32):
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            norm = np.linalg.norm(normal)
            normal = normal / norm if norm > 0 else normal
            handle.write(struct.pack("<3f", *normal))
            handle.write(tri.tobytes())
            handle.write(b"\0\0")


def split_components(path: Path) -> list[np.ndarray]:
    """Triangles of each connected component of an STL mesh."""
    tris = read_stl_triangles(path)
    flat = np.round(tris.reshape(-1, 3), 6)
    _, inverse = np.unique(flat, axis=0, return_inverse=True)
    ids = inverse.reshape(-1, 3)
    parent = np.arange(ids.max() + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b, c in ids:
        ra = find(a)
        parent[find(b)] = ra
        parent[find(c)] = ra
    roots = np.array([find(t[0]) for t in ids])
    return [tris[roots == r] for r in np.unique(roots)]


def colourful_robot(base: str, position=None):
    """The example robot with its hub mesh split into coloured parts."""
    manifest = RobotManifest.from_file(ROBOT_DIR / "robot.yaml")
    spec = mujoco.MjSpec.from_file(str(manifest.model.file))
    cache = Path(tempfile.mkdtemp(prefix="flapsim_fig1_"))
    hub = spec.body("link2")
    for geom in list(hub.geoms):
        if geom.contype == 0 and geom.conaffinity == 0:  # the visual copy
            spec.delete(geom)
    for index, part in enumerate(split_components(ROBOT_DIR / "meshes" / "link2.STL")):
        centre_y = part.reshape(-1, 3)[:, 1].mean() * 1e3
        if len(part) > 4000:
            colour, name = HOUSING, "hub_housing"
        else:
            _, colour, name = min(HUB_PARTS, key=lambda p: abs(p[0] - centre_y))
            name = f"{name}_{index}"
        stl = cache / f"{name}.stl"
        _write_stl(stl, part)
        spec.add_mesh(name=name, file=str(stl))
        hub.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=name,
            contype=0,
            conaffinity=0,
            group=1,
            rgba=list(colour),
            mass=0.0,
        )
    return load(spec, manifest=manifest, base=base, position=position)


def paint(model):
    """White sky, light grid floor, soft light, per-part colours."""
    for t in range(model.ntex):
        adr = model.tex_adr[t]
        w, h, c = model.tex_width[t], model.tex_height[t], model.tex_nchannel[t]
        n = w * h * c
        name = model.texture(t).name
        if name == "skybox":
            model.tex_data[adr : adr + n] = 255
        elif name == "ground_texture":
            img = np.full((h, w, c), 255, dtype=np.uint8)
            minor = np.array([232, 233, 235], dtype=np.uint8)[:c]
            major = np.array([200, 203, 208], dtype=np.uint8)[:c]
            cells, every = 8, 4
            step = w // cells
            for k in range(cells):
                colour, width = (major, 2) if k % every == 0 else (minor, 1)
                img[:, k * step : k * step + width] = colour
                img[k * step : k * step + width, :] = colour
            model.tex_data[adr : adr + n] = img.reshape(-1)
    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    if ground >= 0 and model.geom_matid[ground] >= 0:
        mat = model.geom_matid[ground]
        model.mat_rgba[mat] = (1, 1, 1, 1)
        model.mat_reflectance[mat] = 0.0
        model.mat_texrepeat[mat] = (1.0 / GRID_TEXTURE_M, 1.0 / GRID_TEXTURE_M)
    by_body = {"base_link": MOUNT, "link1": GIMBAL, "torso": TORSO}
    for b in range(1, model.nbody):
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == b]
        visual = [
            g
            for g in geoms
            if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0
        ]
        body = model.body(b).name
        for g in geoms:
            if visual and g not in visual:
                model.geom_group[g] = HIDDEN_GROUP  # coincident collision copy
                continue
            if model.geom(g).name.startswith("hub_"):
                continue  # coloured when built
            if "wing" in body:
                model.geom_rgba[g] = WING
            elif "arm" in body:
                model.geom_rgba[g] = ARM
            else:
                model.geom_rgba[g] = by_body.get(body, SHAFT)
    model.vis.headlight.ambient[:] = 0.58
    model.vis.headlight.diffuse[:] = 0.38
    model.vis.headlight.specular[:] = 0.06
    model.light_diffuse[:] = 0.4
    model.light_specular[:] = 0.1
    model.vis.quality.offsamples = 8
    model.vis.quality.shadowsize = 8192


def simulation(setup: str, base: str, position=None):
    sim = Simulation(colourful_robot(base, position))
    for plugin in load_sim_config(ROBOT_DIR / setup)["plugins"]:
        sim.add(plugin)
    paint(sim.model)
    return sim


# ---------------------------------------------------------------- render ----
class Camera:
    def __init__(self, lookat, distance, azimuth, elevation):
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = lookat
        self.cam.distance = distance
        self.cam.azimuth = azimuth
        self.cam.elevation = elevation

    def projector(self, model, width, height):
        az, el = math.radians(self.cam.azimuth), math.radians(self.cam.elevation)
        forward = np.array(
            [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
        )
        pos = np.array(self.cam.lookat, dtype=float) - self.cam.distance * forward
        right = np.array([math.sin(az), -math.cos(az), 0.0])
        up = np.cross(right, forward)
        focal = (height / 2.0) / math.tan(math.radians(model.vis.global_.fovy) / 2.0)

        def project(point):
            d = np.asarray(point, dtype=float) - pos
            z = float(d @ forward)
            return width / 2 + focal * float(d @ right) / z, height / 2 - focal * float(
                d @ up
            ) / z

        return project


def render(
    sim, camera, width, height, overlays=(), *, hide_ground=False, hide_robot=False
):
    model = sim.model
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    renderer = mujoco.Renderer(model, height=height, width=width, max_geom=30000)
    option = mujoco.MjvOption()
    option.geomgroup[HIDDEN_GROUP] = 0
    if hide_robot:  # robot visual geoms live in groups 1 and 2
        option.geomgroup[1] = 0
        option.geomgroup[2] = 0
    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    if hide_ground and ground >= 0:
        model.geom_group[ground] = 5
        option.geomgroup[5] = 0
    renderer.update_scene(sim.data, camera=camera.cam, scene_option=option)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    if hide_ground and ground >= 0:
        model.geom_group[ground] = 0
    for overlay in overlays:
        overlay(renderer.scene)
    image = renderer.render().copy()
    renderer.close()
    return image, camera.projector(model, width, height)


GEOM_FIELDS = (
    "type",
    "dataid",
    "objtype",
    "objid",
    "category",
    "matid",
    "texcoord",
    "segid",
    "size",
    "pos",
    "mat",
    "rgba",
    "emission",
    "specular",
    "shininess",
    "reflectance",
)


def ghost_overlay(model, snapshots, body_names, alpha):
    """Translucent copies of some bodies at other instants."""
    data = mujoco.MjData(model)
    option, perturb = mujoco.MjvOption(), mujoco.MjvPerturb()
    option.geomgroup[HIDDEN_GROUP] = 0
    bodies = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in body_names}

    def overlay(scene):
        for qpos in snapshots:
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            start = scene.ngeom
            mujoco.mjv_addGeoms(
                model, data, option, perturb, mujoco.mjtCatBit.mjCAT_DYNAMIC, scene
            )
            keep = start
            for j in range(start, scene.ngeom):
                src = scene.geoms[j]
                if (
                    src.objtype == mujoco.mjtObj.mjOBJ_GEOM
                    and model.geom_bodyid[src.objid] in bodies
                ):
                    if keep != j:
                        dst = scene.geoms[keep]
                        for field in GEOM_FIELDS:
                            setattr(dst, field, getattr(src, field))
                    scene.geoms[keep].rgba[3] = alpha
                    keep += 1
            scene.ngeom = keep

    return overlay


def polyline_overlay(points, width, rgba, *, closed=False):
    def overlay(scene):
        pts = list(points) + ([points[0]] if closed else [])
        for a, b in zip(pts[:-1], pts[1:], strict=True):
            if scene.ngeom >= scene.maxgeom:
                return
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).flatten(),
                np.asarray(rgba, dtype=np.float32),
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                width,
                np.asarray(a, float),
                np.asarray(b, float),
            )
            scene.ngeom += 1

    return overlay


def force_overlay(sim, *, scale=0.15, width=0.0018, cop_radius=0.0026):
    forces = [
        (
            np.array(sim.signals[f"wings/{w.name}/cop_world"]),
            np.array(sim.signals[f"wings/{w.name}/F_world"]),
        )
        for w in sim.robot.wings
    ]

    def overlay(scene):
        for cop, force in forces:
            add_marker(scene, cop, cop_radius, (0.12, 0.12, 0.12, 1.0))
            add_arrow(scene, cop, cop + scale * force, width, FORCE)

    return overlay


def wing_tip(sim, wing):
    state = wing.state(sim.data)
    return state.origin_world + state.R_world_from_aero @ np.array(
        [0.0, wing.side * wing.length, 0.0]
    )


def stroke_capture(make_sim, frequency_hz, *, ghosts=4, settle=0.3, freeze_phase=0.05):
    sim = make_sim()
    sim.advance(settle)
    period = 1.0 / frequency_hz
    start = sim.time
    snapshots, tips = [], {w.name: [] for w in sim.robot.wings}
    ghost_times = [start + 0.5 * period * k / max(ghosts - 1, 1) for k in range(ghosts)]
    next_ghost = 0
    while sim.time < start + period:
        sim.step()
        if next_ghost < len(ghost_times) and sim.time >= ghost_times[next_ghost]:
            snapshots.append(sim.data.qpos.copy())
            next_ghost += 1
        for wing in sim.robot.wings:
            tips[wing.name].append(wing_tip(sim, wing))
    while sim.time < start + period * (1.0 + freeze_phase):
        sim.step()
    return sim, snapshots, {k: np.array(v) for k, v in tips.items()}


def stroke_overlays(sim, ghosts, tips, *, tip_width=0.0006, force_scale=0.13):
    return [
        ghost_overlay(
            sim.model, ghosts, [w.spec.body for w in sim.robot.wings], WING_GHOST_ALPHA
        ),
        *[
            polyline_overlay(path[::3], tip_width, TIP_PATH, closed=True)
            for path in tips.values()
        ],
        force_overlay(sim, scale=force_scale, width=0.0016),
    ]


def fixed_base_sim(frequency_hz=40.0, position=None):
    sim = simulation("sim.yaml", "fixed", position)
    sim["stroke"].set_command(frequency_hz=frequency_hz)
    return sim


WAYPOINTS = [(0.0, 0.0, 0.5), (0.2, 0.0, 0.55), (0.2, 0.2, 0.5), (0.0, 0.2, 0.45)]


def free_flight(duration=26.0, interval=6.0):
    sim = simulation("hover.yaml", "free")
    sim["hover"].set_waypoints(WAYPOINTS, interval=interval)
    track = []
    next_sample = 0.0
    while sim.time < duration:
        sim.step()
        if sim.time >= next_sample:
            track.append(sim.data.xpos[sim.robot.root_body].copy())
            next_sample += 0.15
    return sim, np.array(track)


# ----------------------------------------------------------- composition ----
FIG_W, FIG_H = 7.16, 3.45  # inches, IEEE double column
PX_PER_IN = 600
BACKING = dict(
    boxstyle="round,pad=0.18,rounding_size=0.25", fc="white", ec="none", alpha=0.82
)
GRID_TEXTURE_M = 0.4  # one floor texture tile: 8 cells of 5 cm, major line every 20 cm
GRID_CELL_M = GRID_TEXTURE_M / 8
# teal (the accent) running into deep indigo: cool against the amber wings
WORDMARK_GRADIENT = [(22, 163, 172), (38, 118, 178), (72, 78, 168)]
LOUPE = "#1b9aaa"  # teal ring linking the magnifier to the hub


def callout(ax, xy, text, xytext, *, ha="left", va="center", size=6.7):
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        textcoords="data",
        fontsize=size,
        color=INK,
        ha=ha,
        va=va,
        linespacing=1.18,
        bbox=BACKING,
        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.55, shrinkA=2, shrinkB=0),
    )
    ax.plot(*xy, "o", ms=3.0, color=INK, mec="white", mew=0.8, zorder=5)


def card(fig, rect, image, title, note=None):
    """A floating card: soft shadow, white frame, image, small title."""
    x, y, w, h = rect
    shadow = FancyBboxPatch(
        (x + 0.0025, y - 0.004),
        w,
        h,
        boxstyle="round,pad=0,rounding_size=0.012",
        transform=fig.transFigure,
        fc="#000000",
        ec="none",
        alpha=0.07,
        zorder=3,
    )
    frame = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0,rounding_size=0.012",
        transform=fig.transFigure,
        fc="white",
        ec=RULE,
        lw=0.6,
        zorder=4,
    )
    fig.add_artist(shadow)
    fig.add_artist(frame)
    pad_x, pad_top, pad_bottom = 0.006, 0.042, 0.008
    ax = fig.add_axes(
        (x + pad_x, y + pad_bottom, w - 2 * pad_x, h - pad_top - pad_bottom), zorder=5
    )
    ax.imshow(image, interpolation="lanczos", aspect="auto")
    ax.set_axis_off()
    fig.text(
        x + 0.009,
        y + h - 0.011,
        title,
        fontsize=6.9,
        fontweight="semibold",
        color=INK,
        ha="left",
        va="top",
        zorder=6,
    )
    if note:
        fig.text(
            x + w - 0.009,
            y + h - 0.012,
            note,
            fontsize=5.7,
            color=INK2,
            ha="right",
            va="top",
            zorder=6,
        )
    return ax


def card_pixels(rect, *, pad_top=0.042, pad_bottom=0.008, pad_x=0.006):
    _, _, w, h = rect
    return (
        int(round((w - 2 * pad_x) * FIG_W * PX_PER_IN)),
        int(round((h - pad_top - pad_bottom) * FIG_H * PX_PER_IN)),
    )


def soft_mask(image, background, *, threshold=6.0, ramp=26.0):
    """Alpha in [0, 1] where ``image`` differs from ``background``."""
    diff = np.abs(image.astype(float) - background.astype(float)).max(axis=2)
    return np.clip((diff - threshold) / ramp, 0.0, 1.0)


def _homography(dst, src):
    """Coefficients mapping output pixels to source pixels (PIL PERSPECTIVE)."""
    rows, rhs = [], []
    for (x, y), (X, Y) in zip(dst, src, strict=True):
        rows.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        rows.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
        rhs.extend([X, Y])
    return tuple(np.linalg.solve(np.array(rows, float), np.array(rhs, float)))


def floor_point(camera, model, width, height, u, v):
    """Where the ray through pixel (u, v) meets the floor z = 0."""
    az, el = math.radians(camera.cam.azimuth), math.radians(camera.cam.elevation)
    forward = np.array(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
    )
    pos = np.array(camera.cam.lookat, dtype=float) - camera.cam.distance * forward
    right = np.array([math.sin(az), -math.cos(az), 0.0])
    up = np.cross(right, forward)
    focal = (height / 2.0) / math.tan(math.radians(model.vis.global_.fovy) / 2.0)
    direction = (
        forward + (u - width / 2) / focal * right - (v - height / 2) / focal * up
    )
    if direction[2] >= 0.0:
        raise ValueError("that image row does not see the floor")
    return pos + (-pos[2] / direction[2]) * direction


def snap(value, step=GRID_CELL_M):
    return round(value / step) * step


def floor_decal(
    camera,
    model,
    width,
    height,
    text,
    *,
    x_range,
    y_range,
    gradient,
    fill_alpha=0.92,
    shadow=(0.022, 0.05, 0.03, 0.22, (40, 46, 56)),
    font_path="/usr/share/fonts/opentype/inter/InterDisplay-Black.otf",
):
    """RGBA image of ``text`` painted on the floor inside a world rectangle.

    ``x_range`` runs left to right across the view, ``y_range`` from the near
    to the far edge (the camera looks along +y), both in metres on z = 0, so
    the letters share the grid's vanishing point and their edges lie on grid
    lines when the ranges are snapped to the grid.  The letters are filled
    with a left-to-right ``gradient`` (list of RGB stops) and cast a soft
    shadow on the floor: ``shadow = (dx, dy, blur, alpha, rgb)`` with offsets
    and blur as fractions of the text height (dy > 0 towards the viewer).
    """
    from PIL import ImageDraw, ImageFilter, ImageFont

    project = camera.projector(model, width, height)
    (x0, x1), (y_near, y_far) = x_range, y_range
    corners = [(x0, y_far, 0.0), (x1, y_far, 0.0), (x1, y_near, 0.0), (x0, y_near, 0.0)]
    dst = [project(c) for c in corners]

    font = ImageFont.truetype(font_path, 720)
    left, top, right_px, bottom = font.getbbox(text)
    pad = 60
    size = (right_px - left + 2 * pad, bottom - top + 2 * pad)
    bw, bh = size
    glyphs = Image.new("L", size, 0)
    ImageDraw.Draw(glyphs).text((pad - left, pad - top), text, font=font, fill=255)
    dx, dy, blur, shadow_alpha, shadow_rgb = shadow
    text_h = bottom - top
    shadow_img = Image.new("L", size, 0)
    shadow_img.paste(glyphs, (int(dx * text_h), int(dy * text_h)))
    shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(blur * text_h))

    src = [(pad, pad), (bw - pad, pad), (bw - pad, bh - pad), (pad, bh - pad)]
    coeffs = _homography(dst, src)

    def warp(img):
        warped = img.transform(
            (width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC
        )
        return np.asarray(warped, dtype=float) / 255.0

    # gradient across the word, in bitmap coordinates so it follows the letters
    stops = np.asarray(gradient, dtype=float)
    t = np.linspace(0.0, 1.0, bw)
    positions = np.linspace(0.0, 1.0, len(stops))
    ramp = np.stack([np.interp(t, positions, stops[:, k]) for k in range(3)], axis=1)
    ramp_img = np.repeat(ramp[None, :, :], bh, axis=0).astype(np.uint8)
    colour = np.stack(
        [
            warp(Image.fromarray(np.ascontiguousarray(ramp_img[..., k]))) * 255.0
            for k in range(3)
        ],
        axis=-1,
    )

    a_text = warp(glyphs) * fill_alpha
    a_shadow = warp(shadow_img) * shadow_alpha * (1.0 - a_text)
    alpha = np.clip(a_text + a_shadow, 0.0, 1.0)
    rgb = a_text[..., None] * colour + a_shadow[..., None] * np.array(shadow_rgb, float)
    rgb = rgb / np.maximum(alpha[..., None], 1e-6)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    rgba[..., 3] = (alpha * 255).astype(np.uint8)
    return rgba


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/assets/figure1"))
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    W, H = int(FIG_W * PX_PER_IN), int(FIG_H * PX_PER_IN)

    # ---------------- hero, rendered twice: with and without the robot ----
    # the robot stands on the floor so the camera sees the nearby grid
    sim, ghosts, tips = stroke_capture(
        lambda: fixed_base_sim(40.0, position=(0.0, 0.0, 0.0)), 40.0
    )
    # looking straight along +y, on the grid line x = 0.05 m: cross lines of the
    # grid are horizontal and the lengthwise lines meet on the figure's centre
    axis_x = GRID_CELL_M
    camera = Camera(lookat=(axis_x, 0.0, 0.0), distance=0.36, azimuth=90, elevation=-28)
    hero, project = render(sim, camera, W, H, stroke_overlays(sim, ghosts, tips))
    backdrop, _ = render(sim, camera, W, H, hide_robot=True)
    alpha = soft_mask(hero, backdrop)
    Image.fromarray(hero).save(out / "render_hero.png")

    # wordmark painted on the floor, between the backdrop and the robot; the
    # three layers are composited at the output resolution (no resampling)
    # the wordmark spans six grid cells across (30 cm, x = -0.15..0.15 m, so it
    # sits left of the figure centre) and two in depth, in front of the mount
    x_range = (
        -3 * GRID_CELL_M,
        3 * GRID_CELL_M,
    )  # six cells, left of the figure centre
    y_range = (-2 * GRID_CELL_M, 0.0)
    decal = floor_decal(
        camera,
        sim.model,
        W,
        H,
        "FlapSim",
        x_range=x_range,
        y_range=y_range,
        gradient=WORDMARK_GRADIENT,
    )
    a_text = decal[..., 3:4].astype(float) / 255.0
    a_robot = alpha[..., None]
    canvas = (
        backdrop.astype(float) * (1 - a_text) + decal[..., :3].astype(float) * a_text
    )
    canvas = canvas * (1 - a_robot) + hero.astype(float) * a_robot
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    Image.fromarray(canvas).save(out / "render_hero_composite.png")
    ax = fig.add_axes((0, 0, 1, 1), zorder=2)
    ax.imshow(canvas, interpolation="none")
    ax.set_axis_off()
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)

    model, data = sim.model, sim.data
    names = [w.name for w in sim.robot.wings]
    u_of = {n: project(sim.signals[f"wings/{n}/cop_world"])[0] for n in names}
    left_name, right_name = sorted(names, key=lambda n: u_of[n])
    lw, rw = sim.robot.wing(left_name), sim.robot.wing(right_name)
    rs = rw.state(data)
    plate = rs.origin_world + rs.R_world_from_aero @ np.array(
        [0.0, rw.side * rw.length * 0.74, -0.66 * rw.mean_chord]
    )
    tip_path = tips[left_name]
    tip = tip_path[int(np.argmin([project(q)[0] for q in tip_path]))]
    hub = data.xanchor[model.joint(lw.spec.stroke_joint.name).id]
    link2 = model.body("link2").id
    gear_point = data.xpos[link2] + data.xmat[link2].reshape(3, 3) @ np.array(
        [0.0, -0.0386, 0.0]
    )
    gimbal = data.xanchor[model.joint("roll_joint").id]
    mount = data.xpos[sim.robot.root_body] + [0.0, 0.0, 0.014]
    frequency = sim["stroke"].frequency_hz
    amplitude = sim["stroke"].amplitude_rad

    callout(
        ax,
        project(sim.signals[f"wings/{left_name}/cop_world"]),
        "blade-element force at the\ncentre of pressure, every 50 µs",
        (W * 0.025, H * 0.075),
    )
    callout(ax, project(tip), "wing-tip path\nover one stroke", (W * 0.025, H * 0.19))
    hub_uv = project(hub)
    callout(
        ax,
        hub_uv,
        f"stroke $\\psi(t) = {amplitude:.1f}\\,\\sin(2\\pi\\,{frequency:.0f}\\,t)$",
        (W * 0.39, H * 0.04),
        ha="center",
    )
    callout(
        ax,
        project(plate),
        "rigid wing, passive feathering $\\theta$\non a spring hinge (±45° stops)",
        (W * 0.735, H * 0.09),
        ha="right",
    )
    callout(
        ax,
        project(gimbal),
        "2-axis gimbal\n(thrust vectoring)",
        (W * 0.025, H * 0.3),
        ha="left",
    )
    callout(
        ax,
        project(mount),
        "fixed mount\n(virtual load cell)",
        (W * 0.025, H * 0.41),
        ha="left",
    )

    # ---------------- cards ----------------
    right_col, col_w = 0.753, 0.237
    bottom_h = 0.262
    cards = {
        "top": (right_col, 0.51, col_w, 0.47),
        "flight": (right_col, 0.02, col_w, 0.47),
        "flat": (0.018, 0.02, 0.2, bottom_h),
        "wind": (0.228, 0.02, 0.2, bottom_h),
    }

    rect = cards["top"]
    w_px, h_px = card_pixels(rect)
    tsim, tghosts, ttips = stroke_capture(
        fixed_base_sim, 40.0, ghosts=7, freeze_phase=0.62
    )
    top_cam = Camera(
        lookat=(0.0, 0.0, 0.585), distance=0.29, azimuth=90, elevation=-89.5
    )
    image, _ = render(
        tsim,
        top_cam,
        w_px * 2,
        h_px * 2,
        stroke_overlays(tsim, tghosts, ttips, tip_width=0.0008, force_scale=0.15),
        hide_ground=True,
    )
    Image.fromarray(image).save(out / "render_top.png")
    card(
        fig,
        rect,
        image,
        "Top view of the stroke",
        f"±{math.degrees(amplitude):.0f}°, {frequency:.0f} Hz",
    )

    rect = cards["flight"]
    w_px, h_px = card_pixels(rect)
    fsim, track = free_flight()

    def markers(scene):
        for point in track:
            add_marker(scene, point, 0.0042, TRAIL)
        for target in WAYPOINTS:
            add_marker(scene, target, 0.013, TARGET)

    flight_cam = Camera(
        lookat=(0.1, 0.1, 0.5), distance=0.58, azimuth=130, elevation=-27
    )
    image, _ = render(fsim, flight_cam, w_px * 2, h_px * 2, [markers])
    Image.fromarray(image).save(out / "render_flight.png")
    card(fig, rect, image, "Free flight", "hover PID, waypoints")

    rect = cards["wind"]
    w_px, h_px = card_pixels(rect)
    wind = {
        "type": "atmosphere/standard",
        "wind": [1.5, 0.0, 0.0],
        "gust": {
            "type": "one_minus_cos",
            "amplitude": 1.0,
            "start": 0.25,
            "duration": 0.2,
            "direction": [1.0, 0.0, 0.3],
        },
    }

    def make_wind():
        s = fixed_base_sim(40.0)
        s.remove("aero")
        s.add(wind)
        s.add(
            {
                "type": "aero/stack",
                "id": "aero",
                "models": ["quasi_steady", "rotational_lift", "added_mass"],
            }
        )
        return s

    wsim, _, _ = stroke_capture(make_wind, 40.0, ghosts=1)
    direction = np.array(wsim.signals["atmosphere/wind_world"])
    direction = direction / np.linalg.norm(direction)

    def field(scene):
        for yy in np.linspace(-0.08, 0.08, 3):
            for zz in np.linspace(0.5, 0.62, 3):
                start = np.array([-0.19, yy, zz])
                add_arrow(scene, start, start + 0.075 * direction, 0.0026, WIND)

    wind_cam = Camera(
        lookat=(-0.06, 0.0, 0.555), distance=0.29, azimuth=120, elevation=-12
    )
    image, _ = render(
        wsim,
        wind_cam,
        w_px * 2,
        h_px * 2,
        [field, force_overlay(wsim, scale=0.15, width=0.0019)],
    )
    Image.fromarray(image).save(out / "render_wind.png")
    card(fig, rect, image, "Wind and gusts", "1.5 m/s + gust")

    rect = cards["flat"]
    w_px, h_px = card_pixels(rect)

    def make_flat():
        s = flat_plate_simulation(frequency_hz=30.0, amplitude_rad=1.2)
        paint(s.model)
        return s

    psim, pghosts, ptips = stroke_capture(make_flat, 30.0, ghosts=4, freeze_phase=0.62)
    root = psim.data.xpos[psim.robot.root_body]
    flat_cam = Camera(
        lookat=root + [0.0, 0.0, -0.004], distance=0.19, azimuth=128, elevation=-20
    )
    image, _ = render(
        psim,
        flat_cam,
        w_px * 2,
        h_px * 2,
        stroke_overlays(psim, pghosts, ptips, tip_width=0.0007, force_scale=0.15),
    )
    Image.fromarray(image).save(out / "render_flat_plate.png")
    card(fig, rect, image, "Parametric robot", "no CAD")

    # ---------------- magnifier on the gear train ----------------
    gear_uv = project(gear_point)
    zoom_target = data.xpos[link2] + data.xmat[link2].reshape(3, 3) @ np.array(
        [0.0, -0.039, 0.0]
    )
    zoom_cam = Camera(lookat=zoom_target, distance=0.052, azimuth=100, elevation=-22)
    saved = model.geom_rgba.copy()
    for w in sim.robot.wings:
        for g in range(model.ngeom):
            if model.geom_bodyid[g] == w.body_id:
                model.geom_rgba[g, 3] = 0.18
    zoom, _ = render(sim, zoom_cam, 1100, 1100, hide_ground=True)
    model.geom_rgba[:] = saved
    Image.fromarray(zoom).save(out / "render_gears.png")
    loupe_in = 0.9
    lx, ly = 0.442, 0.022
    lw_f, lh_f = loupe_in / FIG_W, loupe_in / FIG_H
    lax = fig.add_axes((lx, ly, lw_f, lh_f), zorder=5)
    clip = Circle((550, 550), 545, transform=lax.transData)
    shown = lax.imshow(zoom, interpolation="lanczos")
    shown.set_clip_path(clip)
    lax.add_patch(Circle((550, 550), 545, fill=False, ec=LOUPE, lw=1.6))
    lax.set_xlim(0, 1100)
    lax.set_ylim(1100, 0)
    lax.set_axis_off()
    # matching teal rings tie the magnifier to the hub (no line across the text)
    ax.add_patch(Circle(gear_uv, W * 0.017, fill=False, ec=LOUPE, lw=1.4, zorder=6))
    fig.text(
        lx + lw_f + 0.006,
        ly + lh_f - 0.004,
        "gear train (teal ring)\nseen through the\ntranslucent housing",
        fontsize=6.3,
        color=INK,
        ha="left",
        va="top",
        linespacing=1.15,
        zorder=6,
        bbox=BACKING,
    )
    fig.text(
        lx + lw_f + 0.006,
        ly + 0.006,
        "URDF robot, 30 g\nfaint wings: earlier instants\n"
        "red arrows: wing force at CoP",
        fontsize=5.4,
        color=INK2,
        ha="left",
        va="bottom",
        linespacing=1.3,
        zorder=6,
        bbox=BACKING,
    )

    fig.savefig(out / "figure1.png", dpi=600, facecolor="white")
    fig.savefig(out / "figure1.pdf", dpi=600, facecolor="white")
    print(f"wrote {out}/figure1.png and figure1.pdf")


if __name__ == "__main__":
    main()
