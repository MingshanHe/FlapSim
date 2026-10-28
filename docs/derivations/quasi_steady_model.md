# Quasi-steady blade-element model: conventions and equations

This page states the conventions FlapSim uses for wings and summarises the
quasi-steady model implemented in `flapsim/fluid/aero.py`. The complete
derivation with closed-form integrals is in the Chinese original,
[docs/zh/derivations/quasi_steady_aero.md](../zh/derivations/quasi_steady_aero.md);
an English translation is planned.

## Wing aero frame

Every wing has an aerodynamic frame attached to its MuJoCo body:

| Axis | Meaning |
|---|---|
| `+Y` | feathering axis, `side * span` where `span` points root to tip |
| `-Z` | chord direction from the leading-edge hinge to the trailing edge |
| `+X` | surface normal, `X = Y x Z` |

A manifest gives `span` and `chord`; the normal is derived. `side` is `+1` for
a left wing and `-1` for its mirror image, so mirrored wings share one frame
orientation at the neutral pose and the `side` factor inside the formulas
places the centre of pressure on each wing. Positive stroke angle `psi` is a
right-handed rotation about `side * Z`; positive feathering angle `theta` is a
right-handed rotation about `+Y` for both wings. The manifest's joint `sign`
fields convert MuJoCo joint coordinates into these conventions, and
`flapsim.robot.load` checks them at the neutral pose and warns when a sign
or the chord direction looks reversed.

## Equations

With body velocities `(u, v, w)` and rates `(p, q, r)` expressed in the torso
frame, the relative flow at span coordinate `r_w` has a component `U_i`
along `+X` and `U_o` along `+Z`, both affine in `r_w`:

```
U_i(r_w) = U_i1 r_w + U_i0,     U_o(r_w) = U_o1 r_w + U_o0
```

(the coefficients are the long expressions in `wing_aero`). The angle of
attack is evaluated at the fixed spanwise centre of pressure
`r_cp = R r33 / r22`:

```
alpha = theta + sign(U_i*) pi/2 - atan(U_o* / U_i*)
C_N(alpha) = 1.8 sin(2 alpha) cos(alpha) + 1.95 sin(alpha) - 1.5 cos(2 alpha) sin(alpha)
d_cp(alpha) = 0.46 - 0.332 cos(alpha) - 0.037 cos(3 alpha) - 0.013 cos(5 alpha)
```

`C_N` is odd in `alpha`, so the normal force carries the sign of the flow;
`d_cp` is even. With `U^2 = a_U2 r_w^2 + a_U1 r_w + a_U0` the closed-form
blade-element integrals give

```
F_N    =  1/2 rho c C_N (a_U2 R^3 r22 + a_U1 R^2 r11 + a_U0 R r00)
M_aero = -1/2 rho c^2 C_N d_cp (a_U2 R^3 z_cp2 + a_U1 R^2 z_cp1 + a_U0 R z_cp0)
M_rd   = -1/8 rho |theta_dot| theta_dot (5 R c^4 z_rd)
```

where `r_kk` and `z_cpk` are the planform integrals from
`flapsim.wings.planform`. `M_aero` is not applied as a separate torque: it is
converted into the chordwise centre of pressure `chord_CoP = -M_aero / F_N`
(always positive), and the single force `F_N` along `+X` is applied at
`[0, side * span_CoP, -chord_CoP]`. The torque about the body centre of mass
is `r x F` plus the rotational damping `M_rd` about `+Y`. The torque on the
feathering hinge therefore always turns the trailing edge downstream, which
is what makes passive feathering work.

## Hinge geometry

The feathering axis need not coincide with the leading edge. `planform.
pitch_axis_fraction` places the hinge inside the chord at that fraction from
the leading edge; `planform.leading_edge_offset` places the leading edge that
many metres beyond the hinge (a root spar). Both shift the chordwise centre
of pressure, `chord_CoP = x_le + (d_cp - x0) c_eff`, and the rotational
damping integral `z_rd`, which grows with the fourth power of the distance
from the hinge.

## Passive feathering and the wing mass model

Passive feathering is a driven oscillator: the aerodynamic moment turns the
trailing edge downstream, rotational damping and the hinge spring resist,
and the wing's inertia, including the inertial torque from the stroke
acceleration acting on a centre of mass offset from the hinge, carries the
motion through reversal. When inertia dominates, the response is in
anti-phase with the moment: the trailing edge leads, the angle of attack is
inverted and the mean lift is negative. CAD meshes exported as uniform solids
often put the wing centroid far behind the hinge and produce exactly this.
A membrane wing carries most of its mass in the leading-edge spar, so
manifests should declare the mass distribution explicitly with
`wings[].mass_model` (`spar_plate` with a spar fraction, `thin_plate`, or
`explicit` values in wing coordinates). The example robot uses
`spar_plate` with 80 percent of 62.6 mg in the spar; with the URDF's own
centroid, 22 mm aft, it does not fly.

## Correction relative to the pre-platform implementation

The earlier implementation multiplied `M_aero` by `sign(alpha)`. Since `C_N`
is odd, that made `M_aero = -|...|` independent of the flow direction, so the
centre of pressure switched to the opposite side of the hinge at every stroke
reversal and the force and torque were no longer those of one force acting
at one point. A translational force flip in the MuJoCo adapter compensated
the resulting lift sign for the example robot, at the cost of an
inconsistent force/torque pair. On 2026-10-28 the moment was restored to
`-d_cp c F_N` as written in the derivation, the flip was removed, and the
wing frame was redefined through `span` and `chord` so the side of the hinge
on which the plate lies is explicit. The regression values in
`tests/test_lift_sweep.py` were re-measured after this change.
