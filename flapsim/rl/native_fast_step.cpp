// Reuse the force laws verbatim. Split stepping avoids recomputing position and
// velocity dynamics twice per substep. Only Euler/implicitfast are supported.
#include <mujoco/mujoco.h>
#define fs_advance fs_split_advance
#define mj_step mj_step2
#define mj_forward mj_step1
#include "native_step.cpp"
#undef fs_advance
#undef mj_step
#undef mj_forward

extern "C" void fs_advance(const mjModel* m, mjData* d, int steps,
    const Motor* motor, double command, int voltage_mode, const Wing* wings, int nwings,
    double density, double chord_drag, double rotational_drag, const double* wind,
    const double* joint_forces, const double* body_forces, const double* controls) {
  mj_step1(m,d);
  fs_split_advance(m,d,steps,motor,command,voltage_mode,wings,nwings,
                  density,chord_drag,rotational_drag,wind,joint_forces,body_forces,controls);
  mj_forward(m,d);
}
