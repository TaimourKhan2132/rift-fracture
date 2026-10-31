"""Shared helpers for running the fracture-modes teacher on one shape.

Used by the Day 4-6 scripts (sweep_threshold, cell_graph, direction_check,
second_hit, repeat_check, label_only_check). Not meant to be run on its own.

Words used here:
  fine mesh   the shape as the game would draw it (here: the wall)
  cage        a coarser closed surface around the fine mesh; physics runs inside it
  tets        tetrahedra filling the cage; the smallest unit fracture-modes works with
  pieces      groups of tets that the modes allow to separate (fixed per shape)
  fragments   what you get after one impact: pieces still stuck together form one fragment
  threshold   break rule: two neighbouring pieces stay joined while the difference in
              how far the impact pushes them is below the threshold
  target damage  a way to set the threshold so it means the same strength on every
              shape: the median random hit breaks off a fixed share of the volume (Day 6d)
"""

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FM_DIR = REPO_ROOT / "third_party" / "fracture-modes"


# ---------------------------------------------------------------- building the teacher


def load_shape(path):
    """Read a triangle mesh and rescale it to fit a unit cube, as fracture-modes does."""
    import gpytoolbox
    import igl

    v, f = igl.read_triangle_mesh(str(path))
    return gpytoolbox.normalize_points(v), f


def fix_mode_solver_start():
    """Give fracture-modes' eigenvalue solver a fixed starting vector (Day 6e).

    fracture-modes starts its mode solve with scipy's eigsh and passes no starting
    vector, so ARPACK draws a random one, and that random state carries over between
    calls in the same Python process. When two eigenvalues are (nearly) equal, as on
    a symmetric wall, the answer then depends on how many shapes were built before
    this one: the same tets gave 476 pieces in one build and 475 in the next. A fixed
    start makes every build of the same tets identical. Only the starting guess
    changes, not the method. Their code is not copied or edited: we replace the
    eigsh name inside their module at runtime.
    """
    module = sys.modules.get("fracture_utility.compute_fracture_modes")
    if module is None or not hasattr(module, "eigsh"):
        print("Warning: fracture-modes' mode solver not found; starting vector not fixed.")
        return
    if getattr(module.eigsh, "rift_fixed_start", False):
        return  # already patched in this process
    original = module.eigsh

    def eigsh_fixed_start(A, *args, **kwargs):
        if kwargs.get("v0") is None:
            kwargs["v0"] = np.random.default_rng(0).standard_normal(A.shape[0])
        return original(A, *args, **kwargs)

    eigsh_fixed_start.rift_fixed_start = True
    module.eigsh = eigsh_fixed_start


def build_teacher(v_fine, f_fine, cage, d, modes, grid_size=256, minratio=1.5,
                  fm_dir=DEFAULT_FM_DIR, display_mesh=True):
    """Run every once-per-shape step. Returns a dict with the fracture_modes object and timings.

    Settings default to fracture-modes' own dataset generator (generate_fractures.py).

    display_mesh=False skips cutting the fine mesh into per-piece meshes. The break
    decision never uses them (they are only for drawing fragments), and on a mesh
    with thousands of triangles this step takes most of the time (Day 6d). Labels
    are unchanged (checked by scripts/label_only_check.py); pictures that colour the
    fine mesh need display_mesh=True.

    The mode solver always gets a fixed starting vector (fix_mode_solver_start), so
    the same shape gives the same teacher in any process and in any order.
    """
    sys.path.insert(0, str(fm_dir))
    import tetgen
    from gpytoolbox.copyleft import lazy_cage
    import fracture_utility as fracture
    fix_mode_solver_start()

    timings = {}
    t = time.perf_counter()
    cage_v, cage_f = lazy_cage(v_fine, f_fine, num_faces=cage, grid_size=grid_size)
    timings["cage_s"] = time.perf_counter() - t

    t = time.perf_counter()
    nodes, elements = tetgen.TetGen(cage_v, cage_f).tetrahedralize(minratio=minratio)
    timings["tetgen_s"] = time.perf_counter() - t

    fm = fracture.fracture_modes(nodes, elements)
    params = fracture.fracture_modes_parameters(num_modes=modes, verbose=False, d=d)
    t = time.perf_counter()
    fm.compute_modes(parameters=params)
    timings["modes_s"] = time.perf_counter() - t

    t = time.perf_counter()
    if display_mesh:
        fm.impact_precomputation(v_fine=v_fine, f_fine=f_fine)
    else:
        fm.impact_precomputation()
    timings["precompute_s"] = time.perf_counter() - t

    return {"fm": fm, "nodes": nodes, "elements": elements, "cage_v": cage_v,
            "cage_f": cage_f, "d": d, "timings": timings}


def tet_volumes_and_centroids(nodes, elements):
    import igl

    vol = np.abs(igl.volume(nodes, elements))
    centroids = nodes[elements].mean(axis=1)
    return vol, centroids


# ---------------------------------------------------------------- impacts


def sample_impacts(v_fine, f_fine, n, seed):
    """Random hit points on the visible surface, each aimed straight into the surface.

    Points are spread evenly by area, so big faces get proportionally more hits.
    Uses our own random generator so the same seed always gives the same impacts.
    Returns (points (n,3), directions (n,3)).
    """
    rng = np.random.default_rng(seed)
    a, b, c = v_fine[f_fine[:, 0]], v_fine[f_fine[:, 1]], v_fine[f_fine[:, 2]]
    cross = np.cross(b - a, c - a)
    area = np.linalg.norm(cross, axis=1) / 2
    normals = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-12)
    # If the mesh's faces point inward, flip so "normal" always points outward.
    signed_volume = np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0
    if signed_volume < 0:
        normals = -normals

    face = rng.choice(len(f_fine), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)  # standard trick for uniform points in a triangle
    points = ((1 - s)[:, None] * a[face] + (s * (1 - r2))[:, None] * b[face]
              + (s * r2)[:, None] * c[face])
    directions = -normals[face]  # into the surface
    return points, directions


def project_impact(fm, point, direction3, d, threshold, project_on_modes=False):
    """Call fracture-modes' own impact_projection (the authoritative result).

    project_on_modes=False (fracture-modes' default): the impact is fitted per piece
    by least squares. Every axis is treated identically, so a unit impact direction
    has NO effect on which pieces break (only its length would).
    project_on_modes=True: the impact is projected onto the fracture modes
    themselves, which can make the break depend on the impact direction.
    """
    direction = direction3 if d == 3 else np.array([1.0])
    fm.impact_projection(contact_point=point, direction=direction, threshold=threshold,
                         project_on_modes=project_on_modes)


def piece_distances(fm):
    """How differently the last impact pushed each pair of neighbouring pieces.

    Same formula as fracture_modes.impact_projection. It does not depend on the
    threshold, so one impact can be tested against many thresholds cheaply.
    """
    dim = fm.modes.shape[0] // fm.elements.shape[0]
    disp = np.reshape(fm.projected_impact, (-1, dim), order="F")
    nb = fm.piece_neighbors
    return np.linalg.norm(disp[nb[:, 0]] - disp[nb[:, 1]], axis=1)


def decide(fm, distances, threshold):
    """Break decision for one threshold, same rule as fracture-modes.

    Neighbouring pieces stay joined while distance < threshold; joined pieces form
    one fragment. Returns (number of fragments, fragment label per piece).
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    keep = fm.piece_neighbors[distances < threshold]
    n = fm.precomputed_num_pieces
    adj = csr_matrix((np.ones(len(keep)), (keep[:, 0], keep[:, 1])), shape=(n, n))
    return connected_components(adj, directed=False)


def hit_distances(fm, point, direction3, d):
    """Piece-pair distances for one hit (they do not depend on the threshold)."""
    project_impact(fm, point, direction3, d, 1.0)
    return piece_distances(fm)


def piece_volumes(T):
    """Volume of each piece (sum of its tets)."""
    fm = T["fm"]
    tet_vol, _ = tet_volumes_and_centroids(T["nodes"], T["elements"])
    tet_piece = np.asarray(fm.piece_to_tet_matrix.argmax(axis=1)).ravel()
    return np.bincount(tet_piece, weights=tet_vol, minlength=fm.precomputed_num_pieces)


def damage_share(fm, distances, piece_vol, threshold):
    """Share of the volume that breaks off: everything outside the largest fragment."""
    _, lab = decide(fm, distances, threshold)
    frag_vol = np.bincount(lab, weights=piece_vol)
    return 1.0 - frag_vol.max() / piece_vol.sum()


def median_damage(fm, distances_list, piece_vol, threshold):
    return float(np.median([damage_share(fm, dist, piece_vol, threshold)
                            for dist in distances_list]))


def target_damage_threshold(fm, distances_list, piece_vol, target):
    """Threshold at which the median hit breaks off `target` of the volume (Day 6d).

    A higher threshold keeps more pieces joined, so damage only falls as the threshold
    rises. Bisection on log(threshold) finds where the median crosses the target.
    Returns (threshold, median damage at that threshold).
    """
    pooled = np.concatenate(distances_list)
    pooled = pooled[pooled > 0]
    lo, hi = np.log(pooled.min() * 0.5), np.log(pooled.max() * 2.0)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if median_damage(fm, distances_list, piece_vol, np.exp(mid)) > target:
            lo = mid  # too much damage: raise the threshold
        else:
            hi = mid
    threshold = float(np.exp(hi))
    return threshold, median_damage(fm, distances_list, piece_vol, threshold)


# ---------------------------------------------------------------- pictures


def fine_triangle_labels(fm):
    """Fragment label of each fine-mesh triangle after the last impact.

    After impact_precomputation, every fine triangle belongs to exactly one piece,
    so its first vertex's label is the triangle's label (as in write_segmented_output).
    """
    return np.asarray(fm.fine_vertex_labels_after_impact)[fm.fine_triangles[:, 0]].astype(int)


def to_display(P):
    """Mesh files are y-up; matplotlib 3D is z-up. Rotate so shapes stand upright
    (x stays x, y becomes up, the front (+z) faces the default camera)."""
    P = np.asarray(P, dtype=float)
    return np.stack([P[..., 0], -P[..., 2], P[..., 1]], axis=-1)


def render_break(ax, V, F, tri_labels, point=None, direction=None, explode=0.12, title=""):
    """Draw one broken shape on a 3D matplotlib axis, pieces pushed apart slightly.

    The largest fragment is grey; broken-off fragments get colours. The red dot
    is the impact point, the red line the impact direction.
    """
    from matplotlib import colormaps
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    V = to_display(V)
    point = None if point is None else to_display(point)
    direction = None if direction is None else to_display(direction)
    labels = np.unique(tri_labels)
    sizes = {l: np.sum(tri_labels == l) for l in labels}
    biggest = max(sizes, key=sizes.get)
    palette = colormaps["tab10"]
    centre = V.mean(axis=0)
    colour_i = 0
    for l in labels:
        tris = F[tri_labels == l]
        piece_centre = V[np.unique(tris)].mean(axis=0)
        offset = (piece_centre - centre) * explode
        if l == biggest:
            colour = (0.78, 0.78, 0.78, 1.0)
        else:
            colour = palette(colour_i % 10)
            colour_i += 1
        poly = Poly3DCollection(V[tris] + offset, facecolors=colour,
                                edgecolors=(0, 0, 0, 0.15), linewidths=0.2, zorder=1)
        ax.add_collection3d(poly)
    if point is not None:
        ax.computed_zorder = False  # draw the hit marker on top, even on the far side
        if direction is not None:
            start = point - direction * 0.15
            ax.plot(*np.stack([start, point]).T, color="red", linewidth=2.5, zorder=9)
        ax.scatter(*point, color="red", s=70, edgecolors="black", linewidths=0.8,
                   depthshade=False, zorder=10)
    lo, hi = V.min(axis=0), V.max(axis=0)
    pad = (hi - lo).max() * (explode + 0.1)
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_zlim(lo[2] - pad, hi[2] + pad)
    ax.set_box_aspect(np.maximum(hi - lo + 2 * pad, 1e-3))
    ax.view_init(elev=22, azim=-62)
    ax.set_axis_off()
    ax.set_title(title, fontsize=8)
