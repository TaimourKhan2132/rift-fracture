"""Make the Day 3 test shape: a wall with a rectangular window, as a closed triangle mesh.

Run from the repo root, inside the rift-teacher environment:

    python scripts/make_wall.py

Writes outputs/day3/wall.obj (outputs/ is gitignored).

Sizes are in metres: a 4 m x 3 m wall, 0.2 m thick, with a 1.2 m x 1.0 m window
whose bottom edge is 1.0 m above the floor. fracture-modes rescales every mesh
to fit a unit cube, so only the proportions matter.

How it is built: one box for the wall, one box for the window, then a mesh
boolean "difference" (wall minus window) cuts the hole. The result is watertight
(no gaps) with one hole through it, which this script checks before saving.
"""

import argparse
from pathlib import Path

import igl
import numpy as np
from gpytoolbox.copyleft import mesh_boolean


def box(lo, hi):
    """Closed triangle mesh of an axis-aligned box. Faces point outward."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    V = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],  # back corners (z0)
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],  # front corners (z1)
    ], dtype=float)
    F = np.array([
        [0, 2, 1], [0, 3, 2],  # back   (-z)
        [4, 5, 6], [4, 6, 7],  # front  (+z)
        [0, 1, 5], [0, 5, 4],  # bottom (-y)
        [3, 7, 6], [3, 6, 2],  # top    (+y)
        [0, 4, 7], [0, 7, 3],  # left   (-x)
        [1, 2, 6], [1, 6, 5],  # right  (+x)
    ], dtype=np.int32)
    return V, F


def signed_volume(V, F):
    """Volume enclosed by a closed mesh (positive when faces point outward)."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)


def genus(V, F):
    """Number of holes through a closed mesh: 0 for a box, 1 for a wall with one window.

    Uses Euler's formula V - E + F = 2 - 2g. In a closed triangle mesh every edge
    is shared by exactly two triangles, so E = 3F/2.
    """
    used = np.unique(F).size
    euler = used - F.shape[0] * 3 // 2 + F.shape[0]
    return (2 - euler) // 2


def make_wall(width=4.0, height=3.0, thickness=0.2,
              win_width=1.2, win_height=1.0, sill=1.0):
    wall_V, wall_F = box([0, 0, 0], [width, height, thickness])
    # The window box sticks out past both faces of the wall so the cut is clean.
    x0 = (width - win_width) / 2
    win_V, win_F = box([x0, sill, -thickness], [x0 + win_width, sill + win_height, 2 * thickness])
    V, F = mesh_boolean(wall_V, wall_F, win_V, win_F, boolean_type="difference")
    return V, F.astype(np.int32)


def main():
    parser = argparse.ArgumentParser(description="Make a wall-with-window test mesh.")
    parser.add_argument("--out", type=Path, default=Path("outputs/day3/wall.obj"))
    args = parser.parse_args()

    V, F = make_wall()

    # Sanity checks: closed, one hole, and the volume we expect.
    expected = 4.0 * 3.0 * 0.2 - 1.2 * 1.0 * 0.2
    vol = signed_volume(V, F)
    g = genus(V, F)
    boundary_edges = igl.boundary_facets(F)  # edges used by only one triangle
    if len(boundary_edges) != 0:
        raise SystemExit(f"Mesh is not closed: {len(boundary_edges)} open edges")
    if g != 1:
        raise SystemExit(f"Expected exactly one hole (the window), got genus {g}")
    if abs(vol - expected) > 1e-6:
        raise SystemExit(f"Volume {vol:.4f} m^3, expected {expected:.4f} m^3")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    igl.write_obj(str(args.out), V, F)
    print(f"Wrote {args.out}: {V.shape[0]} vertices, {F.shape[0]} triangles, "
          f"volume {vol:.2f} m^3, genus {g} (one window)")


if __name__ == "__main__":
    main()
