"""Quasi-steady numerical blade elements, shared by NumPy and JAX.

Each strip uses actual body kinematics and ambient wind. Forces oppose normal
and chordwise motion, and their moment arms are taken about the wing COM.
No kinematic prescription or hard-coded left/right force sign is used.
"""
import numpy as np
from .base import FluidModel
from flapsim.kinematics import point_velocity


class BladeElement(FluidModel):
    def bind(self, model):
        super().bind(model)
        self.elements = []
        for wing in self.config.wings:
            body = model.body(wing.body).id
            radius = (np.arange(self.config.strips) + .5) * wing.span / self.config.strips
            if wing.planform:
                profile = np.asarray(wing.planform)
                leading = np.interp(radius / wing.span, profile[:, 0], profile[:, 1])
                trailing = np.interp(radius / wing.span, profile[:, 0], profile[:, 2])
                chords, offsets = trailing-leading, (leading+trailing)/2
            else:
                chords = np.full(self.config.strips, wing.chord)
                offsets = np.full(self.config.strips, wing.chord_offset)
            points = (np.asarray(wing.origin) + radius[:, None] * np.asarray(wing.span_axis)
                      + offsets[:, None] * np.asarray(wing.chord_axis))
            areas = wing.span / self.config.strips * chords
            rotational_area = float(wing.span / self.config.strips * np.sum(chords**4))
            selector = np.eye(model.nbody)[body]
            self.elements.append((wing, body, int(model.body_rootid[body]), points, selector,
                                  areas, rotational_area))

    def forces(self, data, xp=np):
        result = xp.zeros((self.nbody, 6))
        c = self.config
        for wing, body, root, local_points, selector, areas, rotational_area in self.elements:
            rotation = data.xmat[body].reshape(3, 3)
            points = xp.asarray(local_points) @ rotation.T + data.xpos[body]
            velocity = point_velocity(data, body, root, points, xp) - xp.asarray(c.wind)
            normal = rotation @ xp.asarray(wing.normal_axis)
            chord = rotation @ xp.asarray(wing.chord_axis)
            span = rotation @ xp.asarray(wing.span_axis)
            vn, vc = velocity @ normal, velocity @ chord
            # Symmetric plate: reversing the chord direction must not reverse drag.
            alpha = xp.arctan2(vn, xp.abs(vc) + 1e-12)
            cn = (1.8*xp.sin(2*alpha)*xp.cos(alpha) + 1.95*xp.sin(alpha)
                  - 1.5*xp.cos(2*alpha)*xp.sin(alpha))
            area = xp.asarray(areas)
            fn = -.5 * c.density * area * (vn**2 + vc**2) * cn
            fc = -.5 * c.density * area * c.chord_drag * xp.abs(vc) * vc
            force = fn[:, None]*normal + fc[:, None]*chord
            moment = xp.sum(xp.cross(points - data.xipos[body], force), axis=0)
            pitch_rate = data.cvel[body, :3] @ span
            damping = (-.125 * c.density * c.rotational_drag * rotational_area
                       * xp.abs(pitch_rate) * pitch_rate)
            wrench = xp.concatenate((xp.sum(force, axis=0), moment + damping*span))
            result = result + xp.asarray(selector)[:, None] * wrench
        return result
