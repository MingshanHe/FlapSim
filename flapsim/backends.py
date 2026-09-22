"""Native CPU and optional MJX stepping, separate from the fluid model."""
from abc import ABC, abstractmethod
import mujoco
import numpy as np


class Backend(ABC):
    def __init__(self, model, actuator, controller, fluid):
        self.model, self.actuator = model, actuator
        self.controller, self.fluid = controller, fluid

    @abstractmethod
    def reset(self, qpos=None, qvel=None):
        pass

    @abstractmethod
    def advance(self, steps, command, joint_forces, body_forces):
        pass

    @abstractmethod
    def snapshot(self):
        """Return an independent CPU MjData suitable for rendering/inspection."""


class CPUBackend(Backend):
    def __init__(self, *args):
        super().__init__(*args)
        self.data = mujoco.MjData(self.model)
        self.reset()

    def reset(self, qpos=None, qvel=None):
        mujoco.mj_resetData(self.model, self.data)
        if qpos is not None:
            self.data.qpos[:] = qpos
        if qvel is not None:
            self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)
        self.controller_state = self.controller.initial_state()

    def advance(self, steps, command, joint_forces, body_forces):
        for _ in range(steps):
            target, self.controller_state = self.controller.update(
                self.data, self.controller_state, command)
            self.data.qfrc_applied[:] = joint_forces + self.actuator.forces(self.data, target)
            self.data.xfrc_applied[:] = body_forces + self.fluid.forces(self.data)
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)

    def snapshot(self):
        result = mujoco.MjData(self.model)
        mujoco.mj_copyData(result, self.model, self.data)
        return result


class MJXBackend(Backend):
    """JIT the complete controller -> actuator -> fluid -> physics loop.

    No CPU force callbacks or per-step host transfers. The Python classes are
    traced once; arrays and controller state remain on the chosen device.
    """
    def __init__(self, *args, device='gpu'):
        super().__init__(*args)
        try:
            import jax
            import jax.numpy as jnp
            from mujoco import mjx
        except ImportError as exc:
            raise ImportError('MJX needs the optional dependencies: pip install "flapsim[mjx]"') from exc
        self.jax, self.xp, self.mjx = jax, jnp, mjx
        try:
            self.device = jax.devices(device)[0]
        except (RuntimeError, IndexError) as exc:
            raise RuntimeError(f'MJX device {device!r} is unavailable; install a matching JAX runtime') from exc
        self.device_model = mjx.put_model(self.model, device=self.device, impl='jax')

        def advance(data, state, steps, command, joint_forces, body_forces):
            def one_step(_, carry):
                data, state = carry
                target, state = self.controller.update(data, state, command, jnp)
                data = data.replace(
                    qfrc_applied=joint_forces + self.actuator.forces(data, target, jnp),
                    xfrc_applied=body_forces + self.fluid.forces(data, jnp))
                data = mjx.step(self.device_model, data)
                return mjx.forward(self.device_model, data), state
            return jax.lax.fori_loop(0, steps, one_step, (data, state))

        self._advance = jax.jit(advance)
        self.reset()

    def reset(self, qpos=None, qvel=None):
        data = mujoco.MjData(self.model)
        if qpos is not None:
            data.qpos[:] = qpos
        if qvel is not None:
            data.qvel[:] = qvel
        mujoco.mj_forward(self.model, data)
        self.data = self.mjx.put_data(self.model, data, device=self.device, impl='jax')
        self.controller_state = self.jax.device_put(self.controller.initial_state(), self.device)

    def advance(self, steps, command, joint_forces, body_forces):
        arrays = [self.jax.device_put(v, self.device) for v in
                  (np.asarray(steps, dtype=np.int32), command, joint_forces, body_forces)]
        self.data, self.controller_state = self._advance(self.data, self.controller_state, *arrays)

    def snapshot(self):
        return self.mjx.get_data(self.model, self.data)
