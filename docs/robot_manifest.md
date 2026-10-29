# Import your robot

FlapSim loads URDF and MJCF through MuJoCo's `MjSpec`. What those formats
cannot say about a flapping robot goes into a manifest, `robot.yaml`, next to
the model file.

## 1. Inspect the model

```bash
flapsim inspect my_robot.urdf
```

prints every body with its parent and mass, every joint with its type, axis
and range, and the sites. Note the wing bodies, the stroke joints and the
passive feathering joints.

## 2. Write the manifest

```yaml
schema: flapsim.robot/1
name: MyFlapper
aliases: [my_flapper]

model:
  file: urdf/my_robot.urdf        # relative to this file; .xml for MJCF
  root_body: base_link            # a direct child of the world body
  base: {type: fixed, position: [0.0, 0.0, 0.5]}
  timestep: 5.0e-5
  integrator: implicitfast
  patches:                        # what URDF cannot express
    joints:
      left_feathering:  {stiffness: 2.0e-7, armature: 1.0e-10,
                         solref_limit: [4.0e-4, 1.0]}
      right_feathering: {stiffness: 2.0e-7, armature: 1.0e-10,
                         solref_limit: [4.0e-4, 1.0]}
    equalities:
      - {type: joint, targets: [left_feathering, right_feathering], polycoef: [0, 1]}
    bodies:
      base_link: {when: free, mass: 0.03, com: [0, 0, 0],
                  fullinertia: [5.0e-5, 5.0e-5, 2.0e-5, 0, 0, 0]}
    exclude_self_collision: true

torso: {body: torso}              # body whose velocity the aero models use
air_density: 1.225

wings:
  - name: left
    body: left_wing
    side: left
    frame:
      span: [0.0, 0.0, -1.0]      # root -> tip, along the feathering axis (body frame)
      chord: [0.0, -1.0, 0.0]     # leading edge (hinge) -> trailing edge
    stroke_joint: {name: left_stroke, sign: 1}
    feathering_joint: {name: left_feathering, sign: 1}
    planform: {profile: rectangular, length: 0.07, mean_chord: 0.02}
    mass_model: {type: spar_plate, mass: 6.26e-5, spar_fraction: 0.8}
    shoulder_offset: 0.014
  - name: right
    body: right_wing
    side: right
    mirror_of: left               # shares the left frame at the neutral pose
    stroke_joint: {name: right_stroke, sign: -1}
    feathering_joint: {name: right_feathering, sign: 1}
    planform: {profile: rectangular, length: 0.07, mean_chord: 0.02}
    mass_model: {type: spar_plate, mass: 6.26e-5, spar_fraction: 0.8}
    shoulder_offset: 0.014
```

### Wing frame

`span` points from the wing root to the tip along the feathering axis;
`chord` points from the leading edge (the hinge line) towards the trailing
edge. Both are unit-free vectors in the wing body's frame. FlapSim derives
the surface normal, so the side of the hinge on which the plate lies is never
ambiguous. A `site` whose `+X` is the chord and `+Y` the span works too. A
mirrored wing can simply say `mirror_of: <other wing>`.

### Joint signs

Positive stroke angle is a right-handed rotation about the stroke axis
(`side * Z` of the aero frame), positive feathering angle a right-handed
rotation about `+Y` for both wings. `sign` converts the MuJoCo joint
coordinate. If you get it wrong the loader warns:

```
wing 'left': positive stroke_joint rate is not a right-handed rotation about
side * Z (the stroke axis); flip the joint's 'sign'.
```

It also warns when the wing's centre of mass lies on the wrong side of the
hinge for the chosen `chord`.

### Planform

* `rectangular` or `elliptic` with `length` and `mean_chord`;
* `polyline`: `chord_polyline: [[r, c], ...]` in metres from root to tip;
* `integrals`: the eight blade-element integrals directly;
* `pitch_axis_fraction` (hinge inside the chord) or `leading_edge_offset`
  (root spar between hinge and leading edge).

`flapsim.wings.from_polygon` and `from_mesh` compute the integrals from an
outline or an STL file.

### Mass model

CAD exports often place a wing's centroid far behind the hinge. For a
membrane wing that is unphysical, and it inverts the passive feathering
phase (negative lift). Declare `mass_model` in wing coordinates (normal,
span, chord), so mirrored wings share the same numbers:
`spar_plate` (a leading-edge spar carrying `spar_fraction` of the mass plus a
membrane plate), `thin_plate`, or `explicit` (`com`, `diaginertia`).

## 3. Load and check

```python
from flapsim.robot import load

robot = load("path/to/robot.yaml")  # or base="free"
print(robot.wings, robot.geometry_warnings())
```

```bash
flapsim inspect path/to/robot.yaml          # lists the resolved wings too
```

## 4. Give it a setup file

Copy `sim.yaml` from the example robot, change the joint names, and run
`flapsim run path/to/sim.yaml`. Robots without CAD can start from
`flapsim.robot.flat_plate_flapper()`, which builds a two-wing flat-plate
robot and its manifest in Python.
