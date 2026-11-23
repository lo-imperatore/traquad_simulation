#!/usr/bin/env python3
"""Track colliders of the IsaacLab "nose" asset (traquad_3rollers_r14mm_nose.usd, Legs3+ policies) as STL meshes.

Same construction as ``flatten_traquad_usd.py --track_nose`` (isaac_model/traquad_repo_flat): the box collider of each
track body (bottom 12 mm above the ground, vertical front face) is cut along the belt line, the line tangent to the front
ground wheel (wheel 3) and the idler (wheel 4) on the ground side, and a second convex prism (belt_nose) fills the belt
line between the two wheels. In IsaacLab both colliders are frictionless (they stand for the driven belt).

All points are in the track body frame (= the URDF link frame of body_<side>_<end>, same for front and hind tracks).

    python3 make_track_nose.py      # writes track_box_nose_{left,right}.stl and track_belt_nose_{left,right}.stl
"""
import os
import struct

import numpy as np
from scipy.spatial import ConvexHull

HERE = os.path.dirname(os.path.abspath(__file__))
WHEEL_RADIUS = 0.015

# box colliders of the track bodies (traquad_simulation isaac_sim/assets/traquad/payloads/base.usda, Cube size 1)
BOX = {
    'left': dict(translate=(0.025, -0.0119, -0.0174), scale=(0.05, 0.1417, 0.0869)),
    'right': dict(translate=(-0.025, 0.0174, -0.0119), scale=(0.05, 0.0869, 0.1417)),
}
# wheel 3 (front ground wheel) and 4 (idler) joints in the body frame: xyz, rpy (track.xacro)
WHEELS = {
    'left': {3: ((0, -0.0510313, -0.0583795), (-6.12323e-17, -1.5708, 0)),
             4: ((0, -0.0766777, -0.0417374), (-6.12323e-17, -1.5708, 0))},
    'right': {3: ((-2.08167e-17, 0.0583795, -0.0510312), (-1.5708, 0, -1.5708)),
              4: ((-3.46945e-17, 0.0417374, -0.0766777), (-1.5708, 0, -1.5708))},
}


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def colliders(side):
    t, sc = np.array(BOX[side]['translate']), np.array(BOX[side]['scale'])
    c3, c4 = (np.array(WHEELS[side][k][0]) for k in (3, 4))
    axis = rpy_matrix(*WHEELS[side][4][1]) @ np.array([0.0, 0.0, 1.0])
    axis /= np.linalg.norm(axis)
    corners = np.array([t + sc * np.array([x, y, z]) / 2 for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    # belt line: tangent to the two circles (same radius) on the side away from the box centre
    d = (c4 - c3) / np.linalg.norm(c4 - c3)
    n = np.cross(axis, d)
    if np.dot(n, t - c3) > 0:
        n = -n
    s_c = (corners - c3) @ n - WHEEL_RADIUS  # > 0: outside the belt
    pts = [p for p, v in zip(corners, s_c) if v <= 0]
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(np.abs(corners[i] - corners[j]) > 1e-9) == 1 and s_c[i] * s_c[j] < 0:
                pts.append(corners[i] + (corners[j] - corners[i]) * s_c[i] / (s_c[i] - s_c[j]))
    box = np.array(pts)
    width, mid = float(np.abs(axis) @ sc), float(axis @ t)
    nose = np.array([q - axis * (axis @ q) + axis * (mid + w)
                     for c in (c3, c4) for q in (c, c + WHEEL_RADIUS * n) for w in (-width / 2, width / 2)])
    return box, nose, int(np.sum(s_c > 0)), n


def write_stl(path, points):
    hull = ConvexHull(points)
    centre = points.mean(axis=0)
    with open(path, 'wb') as f:
        f.write(b'convex collider from make_track_nose.py'.ljust(80, b' '))
        f.write(struct.pack('<I', len(hull.simplices)))
        for tri in hull.simplices:
            a, b, c = points[tri]
            normal = np.cross(b - a, c - a)
            if np.dot(normal, a - centre) < 0:  # outward winding
                b, c = c, b
                normal = -normal
            normal /= np.linalg.norm(normal)
            f.write(struct.pack('<12fH', *normal, *a, *b, *c, 0))


if __name__ == '__main__':
    for side in ('left', 'right'):
        box, nose, cut, n = colliders(side)
        write_stl(os.path.join(HERE, f'track_box_nose_{side}.stl'), box)
        write_stl(os.path.join(HERE, f'track_belt_nose_{side}.stl'), nose)
        lo = box.min(axis=0)
        print(f'{side}: {cut} box corner(s) cut, box {len(box)} points, belt line normal {np.round(n, 3)}, '
              f'nose z/y range {np.round(nose.min(axis=0), 4)} .. {np.round(nose.max(axis=0), 4)}')
