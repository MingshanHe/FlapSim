# Command line

```
flapsim run      [source] [--config NAME] [--base fixed|free] [--headless]
                 [--duration S] [--frequency HZ] [--amplitude RAD]
                 [--overlay forces|none] [--describe]
flapsim inspect  <source> [--base fixed|free] [--json]
flapsim plugins  [--kind KIND]
flapsim sweep    [source] [--frequencies ...] [--reference HZ] [--amplitude RAD]
                 [--gimbal-angle RAD] [--settle S] [--cycles N] [--no-damping]
                 [--output params.yaml]
flapsim record   [source] --output clip.gif|clip.mp4|frames_dir [--duration S]
                 [--fps N] [--width W] [--height H] [--camera NAME] [--slow F]
```

`source` is a built-in robot name (`SingleThrustFlapper`), a robot manifest,
a bare URDF/MJCF file, or a `flapsim.sim/1` setup file. For a robot, `run`
uses the `sim.yaml` next to its manifest, or `<NAME>.yaml` with `--config`
(`--base free` picks `hover.yaml`).

`sweep` runs the fixed-base setup as a virtual wind tunnel and writes a
`flapsim.cycle_averaged/1` parameter file for `aero/cycle_averaged`.

`python flapsim.py --robot NAME [--fix false] [--headless --duration S]` is
the pre-platform entry point and still works.
