"""Scene construction only; simulation stepping belongs to Simulator."""
import mujoco


def configure_scene(spec, config, root_name):
    root = spec.body(root_name)
    if root is None:
        raise ValueError(f'root body not found: {root_name!r}')
    if root_name not in [body.name for body in spec.worldbody.bodies]:
        raise ValueError('root_body must be a top-level body')
    root.pos = config.position
    if config.orientation is not None:
        root.quat = config.orientation
    free = [j for j in root.joints if j.type == mujoco.mjtJoint.mjJNT_FREE]
    if config.fixed_base:
        for joint in free:
            spec.delete(joint)
    elif not free:
        root.add_freejoint(name='floating_base')
    spec.option.timestep = config.timestep
    spec.option.gravity = config.gravity
    if config.ground:
        spec.worldbody.add_geom(name='flapsim_ground', type=mujoco.mjtGeom.mjGEOM_PLANE,
                                size=[0, 0, .1], rgba=[.2, .3, .4, 1],
                                contype=1, conaffinity=1)
    spec.worldbody.add_light(name='flapsim_light', pos=[0, 0, 3], dir=[0, 0, -1])


def build_model(config, fluid):
    """Preserve CAD inertia and source constraints; remove source servos only."""
    if (config.backend == 'mjx' and config.fluid.kind == 'mujoco'
            and config.fluid.builtin_model == 'ellipsoid'):
        raise ValueError('MJX-JAX ellipsoid fluid is unsupported; use builtin_model="inertia", '
                         'blade_element, or backend="cpu"')
    spec = mujoco.MjSpec.from_file(str(config.path))
    spec.compiler.fusestatic = False
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for key in list(spec.keys):
        spec.delete(key)
    configure_scene(spec, config.scene, config.root_body)
    integrator = config.scene.integrator
    if integrator == 'auto':
        integrator = 'euler' if config.backend == 'mjx' and config.fluid.kind == 'mujoco' else 'implicitfast'
    if config.backend == 'mjx' and config.fluid.kind == 'mujoco' and integrator == 'implicitfast':
        raise ValueError('MJX built-in fluid requires euler or rk4')
    spec.option.integrator = {
        'euler': mujoco.mjtIntegrator.mjINT_EULER,
        'rk4': mujoco.mjtIntegrator.mjINT_RK4,
        'implicitfast': mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    }[integrator]
    fluid.configure(spec)
    return spec.compile()
