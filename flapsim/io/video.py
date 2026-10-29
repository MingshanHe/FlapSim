"""Offscreen rendering to image sequences, GIF or MP4."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import mujoco
import numpy as np


def render_frames(
    sim,
    *,
    duration: float,
    fps: float = 30.0,
    width: int = 640,
    height: int = 480,
    camera: str | int | None = "camera",
    overlay: Callable | None = None,
    realtime_factor: float = 1.0,
) -> Iterator[np.ndarray]:
    """Step ``sim`` and yield RGB frames (``height x width x 3`` uint8).

    ``realtime_factor`` below one slows the video down (more simulated steps
    per frame are skipped).  Needs an offscreen OpenGL backend
    (``MUJOCO_GL=osmesa``, ``egl`` or a display for ``glfw``).
    """
    model = sim.model
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        sim_per_frame = realtime_factor / fps
        end = sim.time + duration
        while sim.time < end:
            sim.advance(sim_per_frame)
            if camera is None:
                renderer.update_scene(sim.data)
            else:
                renderer.update_scene(sim.data, camera=camera)
            if overlay is not None:
                overlay(sim, renderer.scene)
            yield renderer.render().copy()
    finally:
        renderer.close()


def record_video(
    sim,
    path: str | Path,
    *,
    duration: float,
    fps: float = 30.0,
    width: int = 640,
    height: int = 480,
    camera: str | int | None = "camera",
    overlay: Callable | None = None,
    realtime_factor: float = 1.0,
) -> Path:
    """Write a ``.mp4`` (imageio with ffmpeg), ``.gif`` (Pillow) or a directory
    of ``.png`` frames, chosen by the path suffix."""
    path = Path(path)
    frames = render_frames(
        sim,
        duration=duration,
        fps=fps,
        width=width,
        height=height,
        camera=camera,
        overlay=overlay,
        realtime_factor=realtime_factor,
    )
    suffix = path.suffix.lower()
    if suffix == ".mp4":
        try:
            import imageio.v3 as iio
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "MP4 output needs imageio with ffmpeg support "
                "(pip install imageio[ffmpeg])"
            ) from exc
        iio.imwrite(path, list(frames), fps=fps, codec="libx264", quality=8)
        return path
    if suffix == ".gif":
        from PIL import Image

        images = [Image.fromarray(frame) for frame in frames]
        if not images:
            raise ValueError("no frames rendered")
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=int(round(1000.0 / fps)),
            loop=0,
        )
        return path
    path.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    for index, frame in enumerate(frames):
        Image.fromarray(frame).save(path / f"frame_{index:05d}.png")
    return path


__all__ = ["record_video", "render_frames"]
