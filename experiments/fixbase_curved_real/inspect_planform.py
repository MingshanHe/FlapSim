"""Projected STL planform without counting overlapping front/back triangles twice."""
from pathlib import Path
import numpy as np


def planform(path, bins=1200):
    raw = Path(path).read_bytes()
    count = int.from_bytes(raw[80:84], 'little')
    dtype = np.dtype([('normal', '<f4', (3,)), ('v', '<f4', (3, 3)), ('attr', '<u2')])
    triangles = np.frombuffer(raw, dtype=dtype, count=count, offset=84)['v'].astype(float)
    y = triangles[:, :, 1]
    z = triangles[:, :, 2]
    # Positive span is the aerodynamic region used by the existing model.
    edges_y = np.stack((y, np.roll(y, -1, axis=1)), axis=-1)
    edges_z = np.stack((z, np.roll(z, -1, axis=1)), axis=-1)
    dy = edges_y[:, :, 1]-edges_y[:, :, 0]
    start, end = max(0, y.min()), y.max()
    dr = (end-start)/bins
    radius = start+(np.arange(bins)+.5)*dr
    widths, centers = [], []
    for r in radius:
        crossing = (edges_y.min(axis=-1) <= r) & (edges_y.max(axis=-1) > r) & (np.abs(dy) > 1e-14)
        frac = np.divide(r-edges_y[:, :, 0], dy, out=np.zeros_like(dy), where=np.abs(dy)>1e-14)
        zi = edges_z[:, :, 0]+frac*(edges_z[:, :, 1]-edges_z[:, :, 0])
        low = np.min(np.where(crossing, zi, np.inf), axis=1)
        high = np.max(np.where(crossing, zi, -np.inf), axis=1)
        intervals = sorted(zip(low[high > low], high[high > low]))
        merged = []
        for a, b in intervals:
            if merged and a <= merged[-1][1]+1e-10:
                merged[-1][1] = max(b, merged[-1][1])
            else:
                merged.append([a,b])
        width = sum(b-a for a,b in merged)
        widths.append(width)
        centers.append(sum((b*b-a*a)/2 for a,b in merged)/width if width else 0.)
    return radius, np.array(widths), np.array(centers), dr


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    r,c,z,dr = planform(root/'flapsim/robot/IST_robot/meshes/left_wing.stl')
    rectangle_area=.083*.03325
    rectangle_i2=.03325*.083**3/3
    print('area m2 / rectangle:', c.sum()*dr, rectangle_area, c.sum()*dr/rectangle_area)
    print('spanwise second area moment m4 / rectangle:', (c*r*r).sum()*dr, rectangle_i2, (c*r*r).sum()*dr/rectangle_i2)
    print('mean centroid chordwise m:', (c*z).sum()/c.sum())
