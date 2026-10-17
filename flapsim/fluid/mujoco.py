"""Delegate fluid forces to the engine: inertia-box or native ellipsoid."""
import mujoco
import numpy as np
from .base import NoFluid


class MuJoCoFluid(NoFluid):
    def configure(self, spec):
        c = self.config
        spec.option.density = c.density
        spec.option.viscosity = c.viscosity
        spec.option.wind = c.wind
        if c.builtin_model == 'inertia':
            # MJX-JAX 3.8 implements only the inertia-box model. Explicitly
            # select the same model on CPU instead of silently changing physics.
            for geom in spec.geoms:
                geom.fluid_ellipsoid = 0
            return
        # Preserve source ellipsoid coefficients if already supplied.
        for wing in c.wings:
            body = spec.body(wing.body)
            if body is None:
                raise ValueError(f'wing body not found: {wing.body}')
            if any(g.fluid_ellipsoid > 0 for g in body.geoms):
                continue
            axes = np.column_stack((wing.normal_axis, wing.span_axis, wing.chord_axis))
            if np.linalg.det(axes) < 0:
                axes[:, 2] *= -1
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, axes.ravel())
            position = (np.asarray(wing.origin) + .5*wing.span*np.asarray(wing.span_axis)
                        + wing.chord_offset*np.asarray(wing.chord_axis))
            body.add_geom(name=f'{wing.body}_flapsim_air', type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                          pos=position, quat=quat, size=[.00125, wing.span/2, wing.chord/2],
                          mass=0, contype=0, conaffinity=0, rgba=[0, 0, 0, 0],
                          fluid_ellipsoid=1, fluid_coefs=[.5, .25, 1.5, 1, 1])
