"""Inspect supplied STEP solids in their original millimetre coordinates."""
from pathlib import Path
import json
import numpy as np
import trimesh
from OCP.STEPControl import STEPControl_Reader
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.StlAPI import StlAPI_Writer
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID

FOLDER = Path(__file__).resolve().parent
OUTPUT = FOLDER / 'rect_validation' / 'geometry'


def read_step(path):
    reader = STEPControl_Reader()
    assert int(reader.ReadFile(str(path))) == 1
    reader.TransferRoots()
    return reader.OneShape()


def properties(shape):
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, g)
    return dict(volume_mm3=g.Mass(), com_mm=g.CentreOfMass().Coord(),
                inertia_mm5=[[g.MatrixOfInertia().Value(i, j) for j in range(1, 4)] for i in range(1, 4)])


def inspect():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), layout='constrained')
    results = {}
    for ax, (kind, path) in zip(axes, [('curve', FOLDER/'0831_curve_capton/Curve.STEP'),
                                     ('rect', FOLDER/'0831_rect_capton/Rectangular.STEP')]):
        shape = read_step(path)
        BRepMesh_IncrementalMesh(shape, .025, False, .15, True)
        exp = TopExp_Explorer(shape, TopAbs_SOLID)
        items = []
        while exp.More():
            solid = exp.Current()
            out = OUTPUT/f'{kind}_solid_{len(items)}.stl'
            assert StlAPI_Writer().Write(solid, str(out))
            mesh = trimesh.load(out)
            item = properties(solid)
            item['bounds_mm'] = mesh.bounds.tolist()
            item['mesh'] = out.name
            items.append(item)
            ax.triplot(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.faces, lw=.15,
                       label=f'solid {len(items)-1}: {item["volume_mm3"]:.2f} mm3')
            exp.Next()
        results[kind] = dict(total=properties(shape), solids=items)
        ax.set(title=f'{kind}: STEP XY projection', xlabel='STEP X [mm]', ylabel='STEP Y [mm]', aspect='equal')
        ax.legend(fontsize=8)
    mesh = trimesh.load(FOLDER.parents[1]/'flapsim/robot/IST_robot/meshes/left_wing.stl')
    axes[2].triplot(mesh.vertices[:, 1]*1000, mesh.vertices[:, 2]*1000, mesh.faces, lw=.15)
    axes[2].set(title='Existing curve mesh: body YZ projection', xlabel='Body Y [mm]', ylabel='Body Z [mm]', aspect='equal')
    fig.savefig(OUTPUT/'step_inspection.png', dpi=150)
    (OUTPUT/'step_properties.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    inspect()
