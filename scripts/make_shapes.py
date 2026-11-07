"""Make the RIFT-Arch base shapes: walls, L-corners, arches, slabs, beams and pillars.

Every shape has an integer ID. The ID alone decides the shape: its family comes
from a fixed 20-slot cycle (so any 20 consecutive IDs hold the same family mix) and
its sizes from a random seed made from (MASTER_SEED, ID, attempt). Anyone with this
file can rebuild the exact same shapes.

How one shape is made:
  1. Draw sizes for its family. Sizes are in metres, y is up. fracture-modes scales
     every shape so its longest side is 1, so only proportions reach the teacher;
     thin parts are therefore limited as a share of the longest side (MIN_PART).
  2. Describe the solid as a distance formula: for any point, how far it is from
     the surface (negative inside). Boxes, cylinders and arch outlines combine
     with min (join) and max (cut). No mesh booleans: they left zero-area
     triangles where a round pillar met its base, and the re-mesher then hung.
  3. Extract the surface from that formula on a fine grid (marching cubes, half
     a triangle edge apart), then re-mesh it to triangles of one size (EDGE) so
     every shape, intact or damaged later, reaches the teacher with the same
     triangle density (Day 6d: the teacher's answer depends on the triangle layout).
  4. Check it: closed, one piece, the expected number of holes, no slivers, and
     the same solid as the formula (volume and an inside/outside test at 20,000
     random points). A shape that fails is drawn again with new sizes (up to 3
     times) and every failure is recorded, never dropped silently. Each draw runs
     in a helper process with a time limit, so one stuck shape cannot stall a night.

Usage (Miniforge Prompt, rift-teacher active, repo root):
    python scripts/make_shapes.py --ids 0-19        the pilot: one full family cycle
    python scripts/make_shapes.py --ids 0-1999      the full set (skips IDs already done)

Outputs (data/ and outputs/ are gitignored; share data/ through Google Drive):
    data/rift-arch/shapes/<id>.obj     one mesh per shape, in metres
    data/rift-arch/shapes.jsonl        one line per shape: family, seed, sizes, checks
    outputs/shapes/                    families.png, sheets/, summary.json, log_table.md
"""

import argparse
import hashlib
import json
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # find make_wall.py and ui.py
from make_wall import genus, signed_volume
import ui

GENERATOR_VERSION = 1  # bump whenever a change would alter any shape
MASTER_SEED = 2026
EDGE = 0.015           # target triangle edge length, in the unit box fracture-modes uses
MIN_PART = 3 * EDGE    # thinnest part allowed (same units): about 3 triangles across
GRID = EDGE / 2        # spacing of the grid the surface is extracted on (same units)
ATTEMPTS = 3           # draws per ID before it is recorded as failed
DRAW_TIMEOUT_S = 180   # a draw taking longer than this is stopped and counts as failed

# Family mix: 20 slots, repeated. Walls are the most common breakable in games.
CYCLE = ["wall", "l_corner", "arch", "slab", "wall", "beam", "pillar_square", "arch",
         "wall", "pillar_round", "l_corner", "slab", "wall", "beam", "arch",
         "pillar_square", "wall", "l_corner", "pillar_round", "slab"]
FAMILIES = sorted(set(CYCLE))

DATA_DIR = Path("data/rift-arch")
OUT_DIR = Path("outputs/shapes")


# ---------------------------------------------------------------- distance formulas
# Each takes points P (n x 3, metres) and returns the signed distance to the surface:
# negative inside, positive outside. Joins use min, cuts use max(a, -b). After a join
# or cut the value is only roughly a distance, but its zero (the surface) is exact,
# and that is all the surface extraction needs.


def sd_box(P, lo, hi):
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    q = np.abs(P - (lo + hi) / 2) - (hi - lo) / 2
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)


def sd_rect(u, v, u0, u1, v0, v1):
    """2D rectangle [u0, u1] x [v0, v1]."""
    qu, qv = np.abs(u - (u0 + u1) / 2) - (u1 - u0) / 2, np.abs(v - (v0 + v1) / 2) - (v1 - v0) / 2
    return np.hypot(np.maximum(qu, 0), np.maximum(qv, 0)) + np.minimum(np.maximum(qu, qv), 0)


def sd_disc(u, v, cu, cv, r):
    """2D disc of radius r around (cu, cv)."""
    return np.hypot(u - cu, v - cv) - r


def extrude(d2, w, w0, w1):
    """A 2D outline (its distance d2) made solid between w0 and w1 along the third axis."""
    return np.maximum(d2, np.abs(w - (w0 + w1) / 2) - (w1 - w0) / 2)


def cut(d, *cutters):
    for c in cutters:
        d = np.maximum(d, -c)
    return d


class Shape:
    """What a family builder returns: the formula, its box, and what to expect."""

    def __init__(self, sdf, lo, hi, holes, sizes, describe):
        self.sdf, self.lo, self.hi = sdf, np.asarray(lo, float), np.asarray(hi, float)
        self.holes, self.sizes, self.describe = holes, sizes, describe


def place_along(rng, length, widths, margin):
    """Left edges for openings of these widths along [0, length], each at least
    `margin` from its neighbours and from both ends. None if they do not fit."""
    free = length - sum(widths) - margin * (len(widths) + 1)
    if free < 0:
        return None
    gaps = margin + free * rng.dirichlet(np.ones(len(widths) + 1))
    lefts, x = [], gaps[0]
    for w, g in zip(widths, gaps[1:]):
        lefts.append(x)
        x += w + g
    return lefts


def thickness(rng, longest, top=0.11):
    """A thin dimension, drawn as a share of the longest side (what the teacher sees)."""
    return rng.uniform(MIN_PART, top) * longest


def r3(x):
    return [round(float(v), 3) for v in x] if np.ndim(x) else round(float(x), 3)


def clean(d):
    """Sizes as plain JSON numbers (rounded to the millimetre)."""
    if isinstance(d, dict):
        return {k: clean(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [clean(v) for v in d]
    if isinstance(d, (bool, np.bool_)):
        return bool(d)
    if isinstance(d, (int, float, np.integer, np.floating)):
        return round(float(d), 3)
    return str(d)


# ---------------------------------------------------------------- families


def make_openings(rng, length, height, margin, count, door_ok=True):
    """Openings along a wall: list of (kind, left, width, bottom, top), bottom None for a door."""
    for n in range(count, 0, -1):
        door = door_ok and rng.random() < 0.3
        kinds = ["door" if (door and i == 0) else "window" for i in range(n)]
        rng.shuffle(kinds)
        widths = [rng.uniform(0.8, 1.3) if k == "door" else rng.uniform(0.5, 1.6) for k in kinds]
        lefts = place_along(rng, length, widths, margin)
        if lefts is None:
            continue
        out = []
        for k, x, w in zip(kinds, lefts, widths):
            if k == "door":
                top = min(rng.uniform(1.9, 2.4), height - margin)
                if top >= 1.6:
                    out.append(("door", x, w, None, top))
                continue
            sill = rng.uniform(max(margin, 0.5), 1.2)
            top = min(sill + rng.uniform(0.5, 1.6), height - margin)
            if top - sill >= 0.4:
                out.append(("window", x, w, sill, top))
        return out
    return []


def describe_openings(openings):
    if not openings:
        return "no openings"
    n_win = sum(o[0] == "window" for o in openings)
    n_door = len(openings) - n_win
    parts = [f"{n_win} window{'s' * (n_win != 1)}"] if n_win else []
    parts += [f"{n_door} door"] if n_door else []
    return ", ".join(parts)


def wall(rng):
    W, H = rng.uniform(2.0, 7.0), rng.uniform(2.4, 4.0)
    longest = max(W, H)
    T = thickness(rng, longest)
    margin = max(0.3, MIN_PART * longest)
    openings = make_openings(rng, W, H, margin, int(rng.integers(0, 4)))

    def sdf(P):
        return cut(sd_box(P, [0, 0, 0], [W, H, T]),
                   *[sd_box(P, [x, -T if k == "door" else lo, -T], [x + w, hi, 2 * T])
                     for k, x, w, lo, hi in openings])

    sizes = {"width": W, "height": H, "thickness": T,
             "openings": [{"kind": k, "left": x, "width": w, "bottom": lo or 0.0, "top": hi}
                          for k, x, w, lo, hi in openings]}
    holes = sum(o[0] == "window" for o in openings)  # a door is a notch, not a hole
    return Shape(sdf, [0, 0, 0], [W, H, T], holes, sizes, describe_openings(openings))


def l_corner(rng):
    La, Lb, H = rng.uniform(1.5, 5.0), rng.uniform(1.5, 5.0), rng.uniform(2.4, 4.0)
    longest = max(La, Lb, H)
    T = thickness(rng, longest)
    margin = max(0.3, MIN_PART * longest)
    windows = []
    for arm, length in (("x", La), ("z", Lb)):  # one window per arm, half the time
        if rng.random() < 0.5:
            continue
        free = length - T - margin  # the window stays clear of the corner block
        w = min(rng.uniform(0.5, 1.4), free - margin)
        sill = rng.uniform(max(margin, 0.5), 1.2)
        top = min(sill + rng.uniform(0.5, 1.4), H - margin)
        if w < 0.4 or top - sill < 0.4:
            continue
        start = T + margin + rng.uniform(0, free - margin - w)
        windows.append({"arm": arm, "start": start, "width": w, "bottom": sill, "top": top})

    def sdf(P):
        cutters = [sd_box(P, [T, -1, T], [La + 1, H + 1, Lb + 1])]  # hollow out the L
        for wd in windows:
            a, b = wd["start"], wd["start"] + wd["width"]
            if wd["arm"] == "x":
                cutters.append(sd_box(P, [a, wd["bottom"], -T], [b, wd["top"], 2 * T]))
            else:
                cutters.append(sd_box(P, [-T, wd["bottom"], a], [2 * T, wd["top"], b]))
        return cut(sd_box(P, [0, 0, 0], [La, H, Lb]), *cutters)

    sizes = {"arm_x": La, "arm_z": Lb, "height": H, "thickness": T, "windows": windows}
    return Shape(sdf, [0, 0, 0], [La, H, Lb], len(windows), sizes,
                 f"{len(windows)} window{'s' * (len(windows) != 1)}")


def arch(rng):
    """A free-standing arch: two piers and a curved opening, flat or ring-shaped on top."""
    style = str(rng.choice(["semicircular", "segmental", "pointed"], p=[0.45, 0.25, 0.30]))
    top = str(rng.choice(["flat", "ring"]))
    S, P, Hs = rng.uniform(1.2, 3.5), rng.uniform(0.35, 0.9), rng.uniform(1.0, 3.0)
    cx, width = P + S / 2, 2 * P + S  # middle of the opening; total width
    if style == "semicircular":
        rise, ring_top = Hs + S / 2, Hs + S / 2 + P
    elif style == "segmental":  # flatter: part of a circle wider than the opening
        half = np.radians(rng.uniform(50, 75))  # half the angle the arc spans
        R = S / 2 / np.sin(half)
        yc = Hs - R * np.cos(half)  # the circle's centre sits below the springing line
        rise, ring_top = yc + R, yc + R + P
    else:  # pointed: two circles, each through one springing point, meeting at the top
        r = S * rng.uniform(0.6, 1.0)
        a = r - S / 2  # how far each circle's centre sits from the middle
        rise, ring_top = Hs + np.sqrt(r ** 2 - a ** 2), Hs + np.sqrt((r + P) ** 2 - a ** 2)
    crown = rng.uniform(max(0.25, 0.5 * P), max(0.3, 1.2 * P))
    height = rise + crown if top == "flat" else ring_top
    T = rng.uniform(max(0.3, MIN_PART * max(width, height)), 0.9)

    def curve(u, v, grow):
        """The arch curve above the springing line, `grow` metres outward."""
        if style == "semicircular":
            return sd_disc(u, v, cx, Hs, S / 2 + grow)
        if style == "segmental":
            return np.maximum(sd_disc(u, v, cx, yc, R + grow), Hs - v)
        return np.maximum.reduce([sd_disc(u, v, P + S - r, Hs, r + grow),
                                  sd_disc(u, v, P + r, Hs, r + grow), Hs - v])

    def sdf(Q):
        u, v, w = Q[:, 0], Q[:, 1], Q[:, 2]
        opening = np.minimum(sd_rect(u, v, P, P + S, -1.0, Hs), curve(u, v, 0.0))
        if top == "flat":
            outline = sd_rect(u, v, 0, width, -1.0, height)
        else:
            outline = np.minimum(sd_rect(u, v, 0, width, -1.0, Hs),
                                 np.maximum(curve(u, v, P), np.abs(u - cx) - width / 2))
        solid = extrude(np.maximum(outline, -v), w, 0, T)  # -v: nothing below the ground
        return np.maximum(solid, -opening)

    sizes = {"span": S, "pier_width": P, "spring_height": Hs, "thickness": T,
             "style": style, "top": top, "height": height}
    return Shape(sdf, [0, 0, 0], [width, height, T], 0, sizes, f"{style}, {top} top")


def slab(rng):
    Lx = rng.uniform(2.0, 6.0)
    Lz = rng.uniform(1.5, Lx)
    T = thickness(rng, Lx, top=0.09)
    margin = max(0.3, MIN_PART * Lx)
    holes = []
    for _ in range(int(rng.choice([0, 1, 2], p=[0.4, 0.35, 0.25]))):
        for _try in range(30):
            if rng.random() < 0.5:
                w, d, kind = rng.uniform(0.4, 1.5), rng.uniform(0.4, 1.5), "square"
            else:  # radius at least 0.045 of the slab, so the hole stays round after re-meshing
                w = d = 2 * rng.uniform(max(0.15, 0.045 * Lx), max(0.5, 0.06 * Lx))
                kind = "round"
            if w > Lx - 2 * margin or d > Lz - 2 * margin:  # too big for this slab
                continue
            x0, z0 = rng.uniform(margin, Lx - margin - w), rng.uniform(margin, Lz - margin - d)
            clash = any(x0 < h["x"] + h["w"] + margin and h["x"] < x0 + w + margin and
                        z0 < h["z"] + h["d"] + margin and h["z"] < z0 + d + margin for h in holes)
            if not clash:
                holes.append({"kind": kind, "x": x0, "z": z0, "w": w, "d": d})
                break

    def sdf(P):
        cutters = []
        for h in holes:
            if h["kind"] == "square":
                cutters.append(sd_box(P, [h["x"], -T, h["z"]], [h["x"] + h["w"], 2 * T, h["z"] + h["d"]]))
            else:  # a vertical cylinder through the slab
                cutters.append(sd_disc(P[:, 0], P[:, 2], h["x"] + h["w"] / 2, h["z"] + h["d"] / 2, h["w"] / 2))
        return cut(sd_box(P, [0, 0, 0], [Lx, T, Lz]), *cutters)

    sizes = {"length": Lx, "width": Lz, "thickness": T, "holes": holes}
    return Shape(sdf, [0, 0, 0], [Lx, T, Lz], len(holes), sizes,
                 f"{len(holes)} hole{'s' * (len(holes) != 1)}")


def beam(rng):
    L = rng.uniform(3.0, 7.0)
    profile = str(rng.choice(["rectangle", "T", "I"], p=[0.5, 0.3, 0.2]))
    part = MIN_PART * L
    if profile == "rectangle":
        D, Bf = rng.uniform(0.07, 0.16) * L, rng.uniform(0.05, 0.10) * L
        sizes, cutters = {"length": L, "depth": D, "width": Bf}, []
    else:
        flanges = 1 if profile == "T" else 2
        web = rng.uniform(part, 1.6 * part)
        tf = rng.uniform(part, 1.3 * part)  # flange thickness
        D = tf * flanges + rng.uniform(1.2, 2.5) * part
        Bf = web + 2 * rng.uniform(part, 2 * part)  # flange width
        lo_y = tf if flanges == 2 else -1.0  # a T has its one flange on top
        side = (Bf - web) / 2
        cutters = [([-1, lo_y, -1], [L + 1, D - tf, side]),
                   ([-1, lo_y, Bf - side], [L + 1, D - tf, Bf + 1])]
        sizes = {"length": L, "depth": D, "flange_width": Bf, "flange_thickness": tf, "web": web}
    sizes["profile"] = profile

    def sdf(P):
        return cut(sd_box(P, [0, 0, 0], [L, D, Bf]), *[sd_box(P, a, b) for a, b in cutters])

    return Shape(sdf, [0, 0, 0], [L, D, Bf], 0, sizes, f"{profile} profile")


def pillar(rng, round_shaft):
    H = rng.uniform(2.4, 4.5)
    # Shaft width (or diameter) as a share of the height. Round shafts stay at least
    # 0.09 wide so about 19 triangles go round them and they still look round.
    a = rng.uniform(0.09 if round_shaft else 0.06, 0.2) * H
    part = max(MIN_PART * H, 0.1)
    ends = {}
    for end in ("base", "capital"):
        if rng.random() < 0.5:
            ends[end] = {"width": a + 2 * rng.uniform(part, 2.5 * part),
                         "height": rng.uniform(max(part, 0.15), max(part, 0.15) + 0.3),
                         "round": bool(round_shaft and rng.random() < 0.5)}

    def block(P, w, y0, y1, is_round):
        if is_round:
            return extrude(sd_disc(P[:, 0], P[:, 2], 0, 0, w / 2), P[:, 1], y0, y1)
        return sd_box(P, [-w / 2, y0, -w / 2], [w / 2, y1, w / 2])

    def sdf(P):
        d = block(P, a, 0, H, round_shaft)
        for end, e in ends.items():
            y0 = 0.0 if end == "base" else H - e["height"]
            d = np.minimum(d, block(P, e["width"], y0, y0 + e["height"], e["round"]))
        return d

    w = max([a] + [e["width"] for e in ends.values()])
    sizes = {"height": H, "shaft": a, **ends}
    return Shape(sdf, [-w / 2, 0, -w / 2], [w / 2, H, w / 2], 0, sizes,
                 " and ".join(ends) if ends else "plain")


BUILDERS = {"wall": wall, "l_corner": l_corner, "arch": arch, "slab": slab, "beam": beam,
            "pillar_square": lambda rng: pillar(rng, False),
            "pillar_round": lambda rng: pillar(rng, True)}


# ---------------------------------------------------------------- surface, re-mesh, checks


def surface(shape):
    """Triangle surface of the formula's solid, from a grid GRID apart (unit-box units)."""
    import igl
    from skimage.measure import marching_cubes
    scale = (shape.hi - shape.lo).max()
    h = GRID * scale
    # Grid offset by an odd fraction of a step, so no grid point lands exactly on a
    # flat face (a distance of exactly zero gives degenerate triangles).
    origin = shape.lo - 3.37 * h
    counts = np.ceil((shape.hi + 3 * h - origin) / h).astype(int) + 1
    axes = [origin[i] + h * np.arange(counts[i]) for i in range(3)]
    P = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    field = shape.sdf(P).reshape(counts)
    verts, faces, _, _ = marching_cubes(field, level=0.0, spacing=(h, h, h), allow_degenerate=False)
    V, F = verts + origin, faces.astype(np.int32)
    V, _, _, F = igl.remove_duplicate_vertices(V, F, 1e-9 * scale)
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    V, F, _, _ = igl.remove_unreferenced(V, F)
    if signed_volume(V, F) < 0:  # make faces point outward
        F = F[:, ::-1]
    return np.asarray(V), np.ascontiguousarray(F, dtype=np.int32)


def remesh(V, F):
    """Same surface, triangles with edges of about EDGE in the unit box."""
    import gpytoolbox
    lo = V.min(axis=0)
    scale = (V.max(axis=0) - lo).max()
    U, G = gpytoolbox.remesh_botsch((V - lo) / scale, F.astype(np.int32), i=10, h=EDGE, project=True)
    return np.asarray(U) * scale + lo, np.asarray(G, dtype=np.int32)


def checks(shape, V0, F0, V, F, seed):
    """Quality checks on the final mesh against the formula and the extracted surface."""
    import gpytoolbox
    import igl
    scale = (shape.hi - shape.lo).max()
    rng = np.random.default_rng(seed)
    probe = shape.lo + rng.random((20_000, 3)) * (shape.hi - shape.lo)  # points in the box
    inside_true = shape.sdf(probe) < 0
    inside_mesh = igl.winding_number(V, F, probe) > 0.5
    angles = np.degrees(gpytoolbox.tip_angles(V, F))
    c = {
        "open_edges": int(len(igl.boundary_facets(F))),
        "edge_manifold": bool(igl.is_edge_manifold(F)),
        "pieces": int(igl.facet_components(F).max() + 1),
        "holes": int(genus(V, F)),
        "volume_ratio": signed_volume(V, F) / signed_volume(V0, F0),
        "inside_mismatch": float((inside_true != inside_mesh).sum() / max(inside_true.sum(), 1)),
        "min_angle_deg": float(angles.min()),
        "edge_median": float(np.median(igl.edge_lengths(V, F)) / scale),
    }
    problems = []
    if c["open_edges"] or not c["edge_manifold"]:
        problems.append("not closed")
    if c["pieces"] != 1:
        problems.append(f"pieces: {c['pieces']}")
    if c["holes"] != shape.holes:
        problems.append(f"wrong number of holes: {c['holes']}, expected {shape.holes}")
    if not 0.975 <= c["volume_ratio"] <= 1.01:
        problems.append(f"volume changed in re-meshing: ratio {c['volume_ratio']:.3f}")
    if c["inside_mismatch"] > 0.04:  # edge rounding alone gives up to about 2% on thin shapes
        problems.append(f"inside test differs: {c['inside_mismatch']:.1%}")
    if c["min_angle_deg"] < 5:
        problems.append(f"sliver triangle: {c['min_angle_deg']:.1f} degrees")
    return c, problems


def shape_seed(shape_id, attempt):
    return int(np.random.SeedSequence([MASTER_SEED, shape_id, attempt]).generate_state(1)[0])


def draw_shape(shape_id, attempt):
    """One draw of one shape (runs in the helper process). Returns (V, F, info)."""
    t0 = time.time()
    seed = shape_seed(shape_id, attempt)
    try:
        shape = BUILDERS[CYCLE[shape_id % len(CYCLE)]](np.random.default_rng(seed))
        V0, F0 = surface(shape)
        V, F = remesh(V0, F0)
        c, problems = checks(shape, V0, F0, V, F, seed)
    except Exception as err:  # a failed surface or re-mesh is a failed draw, not a crash
        return None, None, {"seed": seed, "problems": [f"{type(err).__name__}: {err}"]}
    info = {"seed": seed, "problems": problems, "holes": shape.holes,
            "describe": shape.describe, "sizes": clean(shape.sizes), "checks": c,
            "seconds": time.time() - t0}
    return (None, None, info) if problems else (V, F, info)


class Helper:
    """A helper process for draws, replaced when a draw runs past its time limit."""

    def __init__(self):
        self.ctx = multiprocessing.get_context("spawn")  # what Windows does anyway
        self.pool = self.ctx.Pool(1)

    def draw(self, shape_id, attempt):
        job = self.pool.apply_async(draw_shape, (shape_id, attempt))
        try:
            return job.get(timeout=DRAW_TIMEOUT_S)
        except multiprocessing.TimeoutError:
            self.pool.terminate()
            self.pool = self.ctx.Pool(1)
            return None, None, {"seed": shape_seed(shape_id, attempt),
                                "problems": [f"timed out: over {DRAW_TIMEOUT_S} s"]}

    def close(self):
        self.pool.terminate()


def make_one(helper, shape_id):
    """Up to ATTEMPTS draws of one shape. Returns (V, F, record); V is None if all failed."""
    family = CYCLE[shape_id % len(CYCLE)]
    tried = []
    for attempt in range(ATTEMPTS):
        V, F, info = helper.draw(shape_id, attempt)
        if V is None:
            tried.append({"seed": info["seed"], "problems": info["problems"]})
            continue
        extent = V.max(axis=0) - V.min(axis=0)
        return V, F, {
            "id": shape_id, "family": family, "seed": info["seed"], "attempt": attempt,
            "status": "ok", "describe": info["describe"], "generator_version": GENERATOR_VERSION,
            "edge": EDGE, "holes": info["holes"], "triangles": int(len(F)), "vertices": int(len(V)),
            "size_m": r3(extent), "longest_m": r3(extent.max()),
            "box_ratio": r3(extent.min() / extent.max()),  # shortest / longest side of its box
            "volume_m3": r3(signed_volume(V, F)),
            "checks": {k: (r3(v) if isinstance(v, float) else v) for k, v in info["checks"].items()},
            "sizes": info["sizes"], "failed_draws": tried, "seconds": r3(info["seconds"])}
    return None, None, {"id": shape_id, "family": family, "status": "failed",
                        "generator_version": GENERATOR_VERSION, "edge": EDGE, "failed_draws": tried}


# ---------------------------------------------------------------- files


def parse_ids(text):
    """'0-19,40,50-59' -> sorted list of IDs."""
    ids = set()
    for part in text.split(","):
        a, _, b = part.strip().partition("-")
        ids.update(range(int(a), int(b or a) + 1))
    return sorted(ids)


def read_manifest(path):
    """Latest record per ID (the file is append-only; later lines win)."""
    records = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                records[r["id"]] = r
    return records


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def is_done(record, mesh_dir):
    return (record is not None and record.get("status") == "ok"
            and record.get("generator_version") == GENERATOR_VERSION
            and record.get("edge") == EDGE
            and (mesh_dir / f"{record['id']:05d}.obj").exists())


# ---------------------------------------------------------------- pictures


def draw(ax, V, F, title):
    """Shaded picture of a mesh. Drawn from the full mesh: a simplified copy made the
    flat faces look bumpy, which is not what the teacher gets."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    t = V[F]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])[:, [0, 2, 1]]  # outward, in drawing axes
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    P = V[:, [0, 2, 1]]  # matplotlib draws z up; our y is up
    tri = P[F]
    light = np.array([-0.4, -0.7, 0.6]) / np.linalg.norm([-0.4, -0.7, 0.6])
    shade = 0.35 + 0.6 * np.clip(n @ light, 0, 1)
    colours = np.stack([0.55 * shade + 0.1, 0.62 * shade + 0.1, 0.72 * shade + 0.1,
                        np.ones_like(shade)], axis=1)
    ax.add_collection3d(Poly3DCollection(tri, facecolors=colours, edgecolors="none"))
    lo, hi = P.min(axis=0), P.max(axis=0)
    mid, half = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.view_init(elev=22, azim=-58)
    ax.set_axis_off()
    ax.set_title(title, fontsize=8)


def sheet(plt, items, path, columns, suptitle):
    """items: list of (title, V, F), or None for an empty cell. Saves one picture."""
    rows = int(np.ceil(len(items) / columns))
    height = 3.0 * rows + 0.9
    fig = plt.figure(figsize=(3.0 * columns, height))
    for k, item in enumerate(items):
        if item is not None:
            draw(fig.add_subplot(rows, columns, k + 1, projection="3d"), *item[1:], item[0])
    fig.suptitle(suptitle, fontsize=11, y=1 - 0.25 / height)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - 0.9 / height,
                        wspace=0.02, hspace=0.15)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def label(r):
    return f"#{r['id']:05d} {r['family'].replace('_', ' ')}\n{r['describe']}, {r['triangles']:,} tri"


# ---------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser(description="Make the RIFT-Arch base shapes.")
    p.add_argument("--ids", default="0-19", help="IDs to make, e.g. 0-19 or 0-1999 or 3,7,40-59")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--redo", action="store_true", help="remake IDs that are already done")
    p.add_argument("--no-pictures", action="store_true")
    args = p.parse_args()
    ids = parse_ids(args.ids)
    mesh_dir = args.data_dir / "shapes"
    manifest = args.data_dir / "shapes.jsonl"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sheets").mkdir(parents=True, exist_ok=True)
    title = f"Shapes: RIFT-Arch generator v{GENERATOR_VERSION}, IDs {args.ids}"
    with ui.Run(title, args.out_dir, steps=3) as run:
        generate(run, args, ids, mesh_dir, manifest)


def save_one(run, mesh_dir, manifest, records, V, F, record):
    """Write one shape's mesh and its manifest line the moment it is done."""
    import igl
    shape_id = record["id"]
    if V is not None:
        path = mesh_dir / f"{shape_id:05d}.obj"
        tmp = path.with_suffix(".tmp.obj")
        igl.write_obj(str(tmp), V, F)
        tmp.replace(path)  # never leave half a file under the real name
        record["file"] = f"shapes/{path.name}"
        record["md5"] = md5(path)
    else:
        why = record["failed_draws"][-1]["problems"]
        run.say(f"[red]#{shape_id:05d} ({record['family']}) failed every draw[/]: {why}",
                f"#{shape_id:05d} ({record['family']}) failed every draw: {why}")
    with open(manifest, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    records[shape_id] = record


def generate(run, args, ids, mesh_dir, manifest):
    import igl
    records = read_manifest(manifest)
    todo = [i for i in ids if args.redo or not is_done(records.get(i), mesh_dir)]
    run.note(f"{len(ids)} IDs asked for, {len(ids) - len(todo)} already done, {len(todo)} to make. "
             f"Triangle edge {EDGE} of the longest side; thinnest part {MIN_PART}.")

    with run.step(f"Make and check {len(todo)} shapes"):
        helper = Helper()
        try:
            for shape_id in run.track(todo, "shapes"):
                save_one(run, mesh_dir, manifest, records, *make_one(helper, shape_id))
        finally:
            helper.close()

    asked = [records[i] for i in ids if i in records]
    ok = [r for r in asked if r["status"] == "ok"]
    with run.step("Draw preview sheets"):
        if not args.no_pictures:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            load = lambda r: igl.read_triangle_mesh(str(args.data_dir / r["file"]))
            for start in run.track(sorted({r["id"] // 20 * 20 for r in ok}), "sheets of 20"):
                block = [r for r in ok if start <= r["id"] < start + 20]
                path = args.out_dir / "sheets" / f"{start:05d}-{start + 19:05d}.png"
                if path.exists() and not any(r["id"] in todo for r in block):
                    continue  # nothing new in this block
                sheet(plt, [(label(r), *load(r)) for r in block], path, 5,
                      f"RIFT-Arch shapes {start}-{start + 19}")
            examples = []
            for fam in FAMILIES:  # one row per family: its first 4 shapes, for the dashboard
                mine = [(label(r), *load(r)) for r in ok if r["family"] == fam][:4]
                examples += mine + [None] * (4 - len(mine))
            if any(examples):
                sheet(plt, examples, args.out_dir / "families.png", 4,
                      "RIFT-Arch: the first four shapes of every family")

    with run.step("Summarise"):
        summarise(run, args, asked, ok)


def summarise(run, args, asked, ok):
    columns = ["Family", "Made", "Failed", "Redrawn", "Triangles (median, range)",
               "Box shortest / longest side (median)", "Seconds per shape"]
    rows, families = [], {}
    for fam in FAMILIES + ["all"]:
        mine = [r for r in asked if fam in ("all", r["family"])]
        good = [r for r in mine if r["status"] == "ok"]
        if not mine:
            continue
        tri = [r["triangles"] for r in good]
        families[fam] = {
            "made": len(good), "failed": len(mine) - len(good),
            "redrawn": sum(bool(r["failed_draws"]) for r in good),
            "triangles_median": float(np.median(tri)) if tri else None,
            "triangles_min": min(tri, default=None), "triangles_max": max(tri, default=None),
            "box_ratio_median": float(np.median([r["box_ratio"] for r in good])) if good else None,
            "seconds_mean": float(np.mean([r["seconds"] for r in good])) if good else None}
        f = families[fam]
        rows.append([fam.replace("_", " "), f["made"], f["failed"], f["redrawn"],
                     f"{f['triangles_median']:,.0f} ({f['triangles_min']:,}-{f['triangles_max']:,})" if tri else "-",
                     f"{f['box_ratio_median']:.3f}" if good else "-",
                     f"{f['seconds_mean']:.1f}" if good else "-"])
    problems = {}
    for r in asked:
        for d in r["failed_draws"]:
            for prob in d["problems"]:
                key = prob.split(":")[0]
                problems[key] = problems.get(key, 0) + 1
    worst = {k: max((r["checks"][k] for r in ok), default=None, key=abs) for k in
             ("inside_mismatch",)}
    worst["volume_ratio_min"] = min((r["checks"]["volume_ratio"] for r in ok), default=None)
    worst["min_angle_deg"] = min((r["checks"]["min_angle_deg"] for r in ok), default=None)
    (args.out_dir / "summary.json").write_text(json.dumps(
        {"generator_version": GENERATOR_VERSION, "master_seed": MASTER_SEED, "edge": EDGE,
         "min_part": MIN_PART, "ids": args.ids, "families": families,
         "failed_draw_reasons": problems, "worst_checks": worst}, indent=2))
    run.table("RIFT-Arch shapes", columns, rows)
    intro = (f"Generator v{GENERATOR_VERSION}, IDs {args.ids}. Triangle edge {EDGE} and thinnest "
             f"part {MIN_PART} of the longest side. Worst kept shape: volume ratio "
             f"{worst['volume_ratio_min']}, inside-test mismatch {worst['inside_mismatch']}, "
             f"smallest angle {worst['min_angle_deg']} degrees.")
    if problems:
        intro += " Redrawn because: " + "; ".join(f"{k} ({v})" for k, v in sorted(problems.items())) + "."
    run.note(intro)
    run.log_markdown("\n".join([intro, "", "| " + " | ".join(columns) + " |",
                                "|" + "---|" * len(columns)]
                               + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]))


if __name__ == "__main__":
    main()
