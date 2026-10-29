"""Describe a compiled MuJoCo model so a user can write a manifest quickly."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

JOINT_TYPES = {
    mujoco.mjtJoint.mjJNT_FREE: "free",
    mujoco.mjtJoint.mjJNT_BALL: "ball",
    mujoco.mjtJoint.mjJNT_SLIDE: "slide",
    mujoco.mjtJoint.mjJNT_HINGE: "hinge",
}


def model_summary(model: mujoco.MjModel, manifest=None, wings=None) -> dict[str, Any]:
    """A JSON-serialisable description of bodies, joints, sites and wings."""
    bodies = []
    for body in range(model.nbody):
        bodies.append(
            {
                "id": body,
                "name": model.body(body).name,
                "parent": model.body(int(model.body_parentid[body])).name
                if body
                else None,
                "mass": float(model.body_mass[body]),
                "geoms": int(model.body_geomnum[body]),
            }
        )
    joints = []
    for joint in range(model.njnt):
        kind = JOINT_TYPES.get(mujoco.mjtJoint(int(model.jnt_type[joint])), "?")
        entry = {
            "id": joint,
            "name": model.joint(joint).name,
            "type": kind,
            "body": model.body(int(model.jnt_bodyid[joint])).name,
            "qpos_address": int(model.jnt_qposadr[joint]),
            "dof_address": int(model.jnt_dofadr[joint]),
        }
        if kind in ("hinge", "slide"):
            entry["axis"] = [round(float(v), 6) for v in model.jnt_axis[joint]]
            entry["range"] = (
                [float(v) for v in model.jnt_range[joint]]
                if model.jnt_limited[joint]
                else None
            )
        joints.append(entry)
    sites = [
        {
            "name": model.site(site).name,
            "body": model.body(int(model.site_bodyid[site])).name,
        }
        for site in range(model.nsite)
    ]
    summary: dict[str, Any] = {
        "model": manifest.name if manifest is not None else "model",
        "timestep": float(model.opt.timestep),
        "total_mass": float(np.sum(model.body_mass[1:])),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "bodies": bodies,
        "joints": joints,
        "sites": sites,
        "equalities": [model.equality(i).name for i in range(model.neq)],
    }
    if wings:
        summary["wings"] = [
            {
                "name": wing.name,
                "body": wing.spec.body,
                "side": "left" if wing.side > 0 else "right",
                "length": wing.length,
                "mean_chord": wing.mean_chord,
                "stroke_joint": wing.spec.stroke_joint.name
                if wing.spec.stroke_joint
                else None,
                "feathering_joint": (
                    wing.spec.feathering_joint.name
                    if wing.spec.feathering_joint
                    else None
                ),
                "integrals": wing.integrals.as_dict(),
            }
            for wing in wings
        ]
    return summary


def describe_model(model: mujoco.MjModel, manifest=None, wings=None) -> str:
    """A human-readable table of the model, as printed by ``flapsim inspect``."""
    summary = model_summary(model, manifest, wings)
    lines = [
        f"model: {summary['model']}   total mass: "
        f"{summary['total_mass'] * 1e3:.3f} g   "
        f"nq={summary['nq']} nv={summary['nv']}   "
        f"timestep={summary['timestep']:g} s",
        "",
        "bodies (id  name  parent  mass[g]  geoms):",
    ]
    for body in summary["bodies"]:
        lines.append(
            f"  {body['id']:3d}  {body['name']:<24s} {str(body['parent']):<20s} "
            f"{body['mass'] * 1e3:10.4f}  {body['geoms']:3d}"
        )
    lines += ["", "joints (id  name  type  body  axis  range[rad or m]):"]
    for joint in summary["joints"]:
        axis = joint.get("axis")
        rng = joint.get("range")
        axis_text = (
            "" if axis is None else "[" + " ".join(f"{v:+.3f}" for v in axis) + "]"
        )
        range_text = (
            ""
            if rng is None
            else f"[{rng[0]:+.4f}, {rng[1]:+.4f}]"
            if rng
            else "unlimited"
        )
        lines.append(
            f"  {joint['id']:3d}  {joint['name']:<24s} {joint['type']:<6s} "
            f"{joint['body']:<16s} {axis_text:<24s} {range_text}"
        )
    if summary["sites"]:
        lines += [
            "",
            "sites: "
            + ", ".join(f"{s['name']} ({s['body']})" for s in summary["sites"]),
        ]
    if summary["equalities"]:
        lines += ["", "equalities: " + ", ".join(summary["equalities"])]
    if "wings" in summary:
        lines += ["", "wings:"]
        for wing in summary["wings"]:
            lines.append(
                f"  {wing['name']:<8s} body={wing['body']:<16s} "
                f"side={wing['side']:<5s} "
                f"R={wing['length']:.4f} m  c={wing['mean_chord']:.4f} m  "
                f"stroke={wing['stroke_joint']}  feathering={wing['feathering_joint']}"
            )
    else:
        lines += [
            "",
            "no wings declared: write a manifest (robot.yaml) that names the wing "
            "bodies, their normal/span axes and their stroke/feathering joints.",
        ]
    return "\n".join(lines)


__all__ = ["describe_model", "model_summary"]
