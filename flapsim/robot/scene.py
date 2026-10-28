"""Create the minimal MuJoCo scene surrounding the imported URDF."""

from __future__ import annotations

import mujoco


def add_scene(spec: mujoco.MjSpec, target_body: str) -> None:
    # MuJoCo Menagerie-style blue background and checker ground plane.
    spec.add_texture(
        name="skybox",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.3, 0.5, 0.7],
        rgb2=[0.0, 0.0, 0.0],
        width=512,
        height=3072,
    )
    spec.add_texture(
        name="ground_texture",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        rgb1=[0.2, 0.3, 0.4],
        rgb2=[0.1, 0.2, 0.3],
        markrgb=[0.8, 0.8, 0.8],
        width=300,
        height=300,
    )
    spec.add_material(
        name="ground_material",
        # Texture slot 0 is USER; the visible color texture belongs to RGB (1).
        textures=["", "ground_texture"],
        texuniform=True,
        texrepeat=[5.0, 5.0],
        reflectance=0.2,
        rgba=[1.0, 1.0, 1.0, 1.0],
    )
    spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.1],
        material="ground_material",
        friction=[0.8, 0.005, 0.0001],
    )
    spec.worldbody.add_light(
        name="light",
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=[0.0, 0.0, 3.0],
        dir=[0.0, 0.0, -1.0],
    )
    spec.worldbody.add_camera(
        name="camera",
        mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODYCOM,
        targetbody=target_body,
        pos=[0.8, -0.8, 0.6],
    )
