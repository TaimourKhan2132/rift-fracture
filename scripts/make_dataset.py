"""Make the RIFT-Arch training samples: the teacher's breaks for every shape, on 256 cells.

For each shape made by make_shapes.py, in its own helper process:
  1. Teacher homework (decision D1, option B, labels only) with the hit push of D5.
  2. Cells: fill the shape with small cubes (voxels) 0.0075 apart in the unit box,
     6 across the thinnest part any shape may have; cut them into 256 Voronoi cells
     (spread with 5 Lloyd rounds); make sure every cell is one connected piece; link
     cells whose voxels touch (link size = touching area).
  3. Strength (D3): 100 calibration hits set one break threshold per strength level,
     the level being the share of the shape the median hit breaks off: 2, 4, ... 12%.
  4. 50 labelled hits at random surface points. Each hit is labelled at all six
     strength levels (the expensive part, the hit's push, is computed once):
       cell_fragment  which fragment each cell ends up in (0 = the main body)
       link_broken    whether each link breaks
     Fragments are the groups of cells still joined by unbroken links, so the cell
     and link labels always agree; each cell first takes the teacher fragment that
     holds most of its volume.
  5. Quality and variety numbers per hit, and a picture of two hits.

Files are saved the moment a shape finishes, and shapes already done are skipped, so
a crash or a power cut costs one shape. A shape taking longer than --timeout is
stopped and recorded as failed.

Usage (Miniforge Prompt, rift-teacher active, repo root):
    python scripts/make_dataset.py --ids 0-19                      the pilot
    python scripts/make_dataset.py --ids 0-1999 --every 2 --offset 0   even IDs (laptop 1)
    python scripts/make_dataset.py --ids 0-1999 --every 2 --offset 1   odd IDs (laptop 2)

Outputs (gitignored):
    data/rift-arch/samples/<id>.npz    cells, links, hits and labels (listed in make_sample)
    data/rift-arch/samples/<id>.json   settings, timings, quality numbers
    outputs/dataset/                   summary.json, log_table.md, latest.png, pictures/
"""

import argparse
import hashlib
import json
import multiprocessing
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # find our other scripts
from teacher import (DEFAULT_FM_DIR, PUSH_RULES, build_teacher, decide, hit_distances,
                     load_shape, piece_volumes, sample_impacts, target_damage_threshold,
                     tet_pieces)
from cell_graph import locate_in_tets, make_cells, make_links, seeds_inside
import ui

DATASET_VERSION = 1   # bump whenever a change would alter any sample
DATA_SEED = 2027
TEACHER = {"cage": 5000, "modes": 20, "d": 1, "grid_size": 256, "minratio": 1.5}  # D1, option B
PUSH = "mass"         # D5 (proposed 7 Nov): weight the hit by material, not by mesh corner
STRENGTHS = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12)  # D3: share broken off
STRENGTH_STAT = "median"  # D3: the median calibration hit breaks off that share
HITS = 50
CALIBRATION_HITS = 100
CELLS = 256
LLOYD_ROUNDS = 5
VOXEL = 0.0075        # voxel spacing in the unit box: 6 across the thinnest part (0.045)
MAX_VOXELS = 800_000  # a larger shape gets coarser voxels (recorded per shape)
SHOW_LEVELS = (0, 2, 5)  # strength levels drawn in the pictures: 2%, 6%, 12%

DATA_DIR = Path("data/rift-arch")
OUT_DIR = Path("outputs/dataset")


# ---------------------------------------------------------------- voxels and cells


def normalisation(v_m):
    """(offset, scale) with v_unit = (v_m - offset) / scale, as fracture-modes scales shapes."""
    lo, hi = v_m.min(axis=0), v_m.max(axis=0)
    scale = float((hi - lo).max())
    return (lo + hi) / 2, scale


def voxel_grid(v, f, spacing=VOXEL, max_voxels=MAX_VOXELS):
    """Centres of the cubes (spacing apart) that lie inside the shape.

    Unlike Day 5's voxelize, the spacing is fixed in the unit box instead of taken from
    the bounding box: an L-corner's box is thick even though its walls are thin.
    Returns (centres, grid index, spacing used, grid corner).
    """
    import igl
    lo, hi = v.min(axis=0), v.max(axis=0)
    counts = np.maximum(np.ceil((hi - lo) / spacing).astype(int), 1)
    while True:
        axes = [lo[i] + (np.arange(counts[i]) + 0.5) * spacing for i in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        inside = igl.fast_winding_number_for_meshes(v, f.astype(np.int64), grid) > 0.5
        if inside.sum() <= max_voxels:
            break
        spacing *= (inside.sum() / max_voxels) ** (1 / 3) * 1.01  # too many: coarser cubes
        counts = np.maximum(np.ceil((hi - lo) / spacing).astype(int), 1)
    index = np.stack(np.meshgrid(*[np.arange(c) for c in counts], indexing="ij"),
                     axis=-1).reshape(-1, 3)
    return grid[inside], index[inside], float(spacing), lo


def face_pairs(index):
    """Pairs of voxels that share a face (each pair once)."""
    dims = index.max(axis=0) + 2
    ids = -np.ones(dims, dtype=np.int64)
    ids[tuple(index.T)] = np.arange(len(index))
    a_all, b_all = [], []
    for axis in range(3):
        nb = index.copy()
        nb[:, axis] += 1
        b = ids[tuple(nb.T)]
        ok = b >= 0
        a_all.append(np.flatnonzero(ok))
        b_all.append(b[ok])
    return np.concatenate(a_all), np.concatenate(b_all)


def join_split_cells(cell, pairs, rounds=5):
    """Make every cell one connected piece of voxels.

    A Voronoi cell takes the voxels nearest its seed, which in a non-convex shape (an
    arch, an L-corner) can include voxels across a gap that do not touch the rest of
    the cell. Each such stray piece joins the neighbouring cell it touches most.
    Returns (cell, voxels moved).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    a, b = pairs
    n, moved = len(cell), 0
    for _ in range(rounds):
        same = cell[a] == cell[b]
        graph = coo_matrix((np.ones(same.sum()), (a[same], b[same])), shape=(n, n))
        _, comp = connected_components(graph, directed=False)
        size = np.bincount(comp)
        comp_cell = np.zeros(len(size), dtype=int)
        comp_cell[comp] = cell
        main = np.zeros(cell.max() + 1, dtype=int)  # largest piece of every cell
        best = np.zeros(cell.max() + 1)
        for c, s in enumerate(size):
            if s > best[comp_cell[c]]:
                best[comp_cell[c]], main[comp_cell[c]] = s, c
        stray = np.ones(len(size), dtype=bool)
        stray[main] = False
        if not stray.any():
            break
        # For every stray piece, count its faces touching each other cell.
        touch = ~same & stray[comp[a]]
        touch_b = ~same & stray[comp[b]]
        src = np.concatenate([comp[a][touch], comp[b][touch_b]])
        dst = np.concatenate([cell[b][touch], cell[a][touch_b]])
        if len(src) == 0:
            break
        key, count = np.unique(src * (cell.max() + 1) + dst, return_counts=True)
        target = {}
        for k, cnt in zip(key, count):
            s, d = divmod(int(k), cell.max() + 1)
            if cnt > target.get(s, (0, -1))[0]:
                target[s] = (cnt, d)
        for s, (_, d) in target.items():
            idx = comp == s
            cell[idx] = d
            moved += int(idx.sum())
    return cell, moved


# ---------------------------------------------------------------- labels


def by_size(labels, weights):
    """Relabel groups so 0 is the largest (by weight), 1 the next, and so on."""
    size = np.bincount(labels, weights=weights)
    order = np.argsort(-size, kind="stable")
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    return rank[labels]


def label_hit(fm, distances, threshold, voxel_piece, cell, links, cell_vol):
    """Cell and link labels for one hit at one strength, plus quality numbers."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    n_frag, piece_frag = decide(fm, distances, threshold)
    frag = piece_frag[voxel_piece]  # teacher fragment of every voxel
    n_cells = len(cell_vol)
    table = np.bincount(cell * n_frag + frag, minlength=n_cells * n_frag).reshape(n_cells, n_frag)
    majority = table.argmax(axis=1)  # fragment holding most of each cell
    broken = majority[links[:, 0]] != majority[links[:, 1]]
    hold = links[~broken]
    adj = csr_matrix((np.ones(len(hold)), (hold[:, 0], hold[:, 1])), shape=(n_cells, n_cells))
    n_groups, group = connected_components(adj, directed=False)
    group = by_size(group, cell_vol)
    frag_vox = np.bincount(frag, minlength=n_frag)
    owned = np.zeros(n_frag, dtype=bool)
    owned[majority] = True
    # A teacher fragment the cells cannot hold together comes out as several cell groups
    # ("rebuilt" is False). It matters most for the teacher's main body: its cut-off
    # groups look like pieces that break off when they do not ("false break").
    group_vol = np.bincount(group, weights=cell_vol)
    of_main = np.unique(group[majority == frag_vox.argmax()])
    false_break = (group_vol[of_main].sum() - group_vol[of_main].max()) / cell_vol.sum() if len(of_main) else 0.0
    return {
        "cell_fragment": group.astype(np.int16), "link_broken": broken,
        "teacher_fragments": int((frag_vox > 0).sum()),
        "cell_fragments": int(n_groups),
        "broken_off_teacher": float(1 - frag_vox.max() / len(frag)),
        "broken_off_cells": float(1 - cell_vol[group == 0].sum() / cell_vol.sum()),
        "purity": float(table.max(axis=1).sum() / len(frag)),
        "lost_volume": float(frag_vox[(frag_vox > 0) & ~owned].sum() / len(frag)),
        "rebuilt": bool(n_groups == len(np.unique(majority))),
        "false_break": float(false_break),
    }


# ---------------------------------------------------------------- one shape (helper process)


def settings(opts):
    """Everything that decides a sample's content; a change means remaking samples."""
    return {"dataset_version": DATASET_VERSION, "data_seed": DATA_SEED,
            "teacher": {**TEACHER, "cage": opts["cage"], "modes": opts["modes"]},
            "push": opts["push"], "strengths": list(STRENGTHS), "strength_stat": STRENGTH_STAT,
            "hits": HITS, "calibration_hits": CALIBRATION_HITS, "cells": CELLS, "lloyd_rounds": LLOYD_ROUNDS,
            "voxel": VOXEL, "max_voxels": MAX_VOXELS}


def sub_seed(shape_id, part):
    return int(np.random.SeedSequence([DATA_SEED, shape_id, part]).generate_state(1)[0])


def prepare(shape_id, opts):
    """Everything a sample needs before thresholds are chosen: teacher homework, cells,
    and the push of every calibration and labelled hit (pushes do not depend on the
    threshold). strength_audit.py uses this too, so it sees exactly the same shape."""
    data_dir = Path(opts["data_dir"])
    shape = opts["shape"]
    times = {}

    v_m, f = load_shape_metres(data_dir / shape["file"])
    offset, scale = normalisation(v_m)
    v, f = load_shape(data_dir / shape["file"])  # the unit box fracture-modes works in

    T = build_teacher(v, f, opts["cage"], TEACHER["d"], opts["modes"], TEACHER["grid_size"],
                      TEACHER["minratio"], Path(opts["fm_dir"]), display_mesh=False)
    fm = T["fm"]
    times.update({k: round(x, 2) for k, x in T["timings"].items()})

    t = time.time()
    vox, vox_index, h, grid_lo = voxel_grid(v, f)
    pairs = face_pairs(vox_index)
    seeds, cell = make_cells(vox, seeds_inside(vox, min(CELLS, len(vox)),
                                               np.random.default_rng(sub_seed(shape_id, 0))),
                             LLOYD_ROUNDS)
    cell, moved = join_split_cells(cell, pairs)
    _, cell = np.unique(cell, return_inverse=True)
    n_cells = int(cell.max() + 1)
    links, link_area = make_links(vox_index, cell, np.full(3, h))
    cell_vox = np.bincount(cell, minlength=n_cells)
    cell_vol = cell_vox * h ** 3
    centroid = np.zeros((n_cells, 3))
    np.add.at(centroid, cell, vox)
    centroid /= cell_vox[:, None]
    voxel_tet, missed = locate_in_tets(vox, T["nodes"], T["elements"])
    voxel_piece = tet_pieces(fm)[voxel_tet]
    times["cells_s"] = round(time.time() - t, 2)

    t = time.time()
    push = opts["push"]
    cal_pts, cal_dirs = sample_impacts(v, f, CALIBRATION_HITS, sub_seed(shape_id, 2))
    cal_d = [hit_distances(fm, p, d, TEACHER["d"], push) for p, d in zip(cal_pts, cal_dirs)]
    hit_pts, hit_dirs = sample_impacts(v, f, HITS, sub_seed(shape_id, 1))
    hit_d = [hit_distances(fm, p, d, TEACHER["d"], push) for p, d in zip(hit_pts, hit_dirs)]
    times["pushes_s"] = round(time.time() - t, 2)

    return SimpleNamespace(
        shape=shape, T=T, fm=fm, f=f, offset=offset, scale=scale, times=times,
        vox=vox, vox_index=vox_index, h=h, grid_lo=grid_lo, seeds=seeds, cell=cell,
        moved=moved, n_cells=n_cells, links=links, link_area=link_area, cell_vol=cell_vol,
        centroid=centroid, missed=missed, voxel_piece=voxel_piece, piece_vol=piece_volumes(T),
        cal_d=cal_d, hit_pts=hit_pts, hit_dirs=hit_dirs, hit_d=hit_d)


def calibrate(fm, cal_d, piece_vol, stat=STRENGTH_STAT):
    """One threshold per strength level (D3), from the calibration hits.
    Returns (thresholds, the median or mean damage each one gives)."""
    found = [target_damage_threshold(fm, cal_d, piece_vol, s, stat) for s in STRENGTHS]
    return [thr for thr, _ in found], [got for _, got in found]


def label_all(s, thresholds):
    """label_hit for every labelled hit at every threshold: a list of rows, one per hit."""
    return [[label_hit(s.fm, d, thr, s.voxel_piece, s.cell, s.links, s.cell_vol) for thr in thresholds]
            for d in s.hit_d]


def make_sample(shape_id, opts):
    """Teacher, cells, calibration and labels for one shape. Writes its files; returns its record."""
    t_start = time.time()
    data_dir = Path(opts["data_dir"])
    s = prepare(shape_id, opts)
    shape, fm, times = s.shape, s.fm, s.times
    seeds, cell, links, cell_vol, hit_pts, hit_dirs = s.seeds, s.cell, s.links, s.cell_vol, s.hit_pts, s.hit_dirs

    t = time.time()
    thresholds, achieved = calibrate(fm, s.cal_d, s.piece_vol)
    times["calibration_s"] = round(time.time() - t, 2)

    t = time.time()
    res = label_all(s, thresholds)
    times["labels_s"] = round(time.time() - t, 2)

    grid = lambda key, dtype: np.array([[r[key] for r in row] for row in res], dtype=dtype)
    stats = {k: grid(k, float) for k in ("teacher_fragments", "cell_fragments", "broken_off_teacher",
                                         "broken_off_cells", "purity", "lost_volume", "rebuilt",
                                         "false_break")}
    cell_fragment = np.stack([np.stack([r["cell_fragment"] for r in row]) for row in res])
    link_broken = np.stack([np.stack([r["link_broken"] for r in row]) for row in res])

    samples = data_dir / "samples"
    path = samples / f"{shape_id:05d}.npz"
    tmp = samples / f"{shape_id:05d}.tmp.npz"
    np.savez_compressed(
        tmp,
        cell_seed=seeds.astype(np.float32), cell_centroid=s.centroid.astype(np.float32),
        cell_volume=cell_vol.astype(np.float32), links=links.astype(np.int16),
        link_area=s.link_area.astype(np.float32),
        hit_point=hit_pts.astype(np.float32), hit_normal=(-hit_dirs).astype(np.float32),
        strength=np.array(STRENGTHS, dtype=np.float32), threshold=np.array(thresholds),
        cell_fragment=cell_fragment, link_broken=link_broken,
        teacher_fragments=stats["teacher_fragments"].astype(np.int16),
        cell_fragments=stats["cell_fragments"].astype(np.int16),
        broken_off=stats["broken_off_cells"].astype(np.float32),
        purity=stats["purity"].astype(np.float32),
        voxel_spacing=np.float64(s.h), voxel_origin=s.grid_lo, cell_grid=cell_grid(s.vox_index, cell),
        to_metres_offset=s.offset, to_metres_scale=np.float64(s.scale),
        meta=json.dumps({"id": shape_id, "family": shape["family"], **settings(opts)}))
    tmp.replace(path)

    per_level = []
    for j, strength in enumerate(STRENGTHS):
        outcomes = {np.packbits(link_broken[i, j]).tobytes() for i in range(HITS)}
        per_level.append({
            "strength": strength, "threshold": thresholds[j], "calibration_damage": achieved[j],
            "broken_off_median": float(np.median(stats["broken_off_cells"][:, j])),
            "broken_off_mean": float(stats["broken_off_cells"][:, j].mean()),
            "cell_fragments_mean": float(stats["cell_fragments"][:, j].mean()),
            "teacher_fragments_mean": float(stats["teacher_fragments"][:, j].mean()),
            "distinct_outcomes": len(outcomes),
            "hits_with_a_break": int((stats["cell_fragments"][:, j] > 1).sum())})
    record = {
        "id": shape_id, "family": shape["family"], "describe": shape["describe"],
        "status": "ok", **settings(opts), "shape_md5": shape.get("md5"),
        "shape_generator_version": shape.get("generator_version"),
        "triangles": int(len(s.f)), "tets": int(len(s.T["elements"])),
        "pieces": int(fm.precomputed_num_pieces), "voxels": int(len(s.vox)), "voxel_used": s.h,
        "cells": s.n_cells, "links": int(len(links)), "voxels_moved_to_join_cells": s.moved,
        "voxels_not_in_a_tet": s.missed, "levels": per_level,
        "purity_mean": float(stats["purity"].mean()), "purity_min": float(stats["purity"].min()),
        "lost_volume_mean": float(stats["lost_volume"].mean()),
        "rebuilt_share": float(stats["rebuilt"].mean()),
        "false_break_mean": float(stats["false_break"].mean()),
        "false_break_max": float(stats["false_break"].max()),
        "timings": times, "seconds": round(time.time() - t_start, 1),
        "npz_md5": hashlib.md5(path.read_bytes()).hexdigest()}

    if opts.get("pictures", True):
        t = time.time()
        try:
            draw_sample(Path(opts["out_dir"]) / "pictures" / f"{shape_id:05d}.png", record,
                        s.vox, cell, cell_fragment, hit_pts)
        except Exception as err:  # a picture must never cost the sample
            record["picture_error"] = f"{type(err).__name__}: {err}"
        times["picture_s"] = round(time.time() - t, 2)
    return record


def cell_grid(index, cell):
    """The cells as a block of voxels: cell number per voxel, -1 outside the shape.
    Voxel (i, j, k) has its centre at voxel_origin + (i, j, k, + 0.5) * voxel_spacing."""
    grid = -np.ones(index.max(axis=0) + 1, dtype=np.int16)
    grid[tuple(index.T)] = cell
    return grid


def load_shape_metres(path):
    import igl
    v, f = igl.read_triangle_mesh(str(path))
    return v, f


def child_main(shape_id, opts):
    """Entry point of the helper process: make one sample, always leave a record behind."""
    out = Path(opts["data_dir"]) / "samples" / f"{shape_id:05d}.json"
    try:
        record = make_sample(shape_id, opts)
    except BaseException as err:  # recorded, never silently dropped
        record = {"id": shape_id, "family": opts["shape"]["family"], "status": "failed",
                  **settings(opts), "error": f"{type(err).__name__}: {err}",
                  "traceback": traceback.format_exc()[-2000:]}
    tmp = out.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(record, indent=1))
    tmp.replace(out)


# ---------------------------------------------------------------- pictures


def draw_sample(path, record, vox, cell, cell_fragment, hit_pts):
    """Cells, and two hits at 2%, 6% and 12% strength (voxels coloured by fragment)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    mid = SHOW_LEVELS[1]
    breaking = [i for i in range(HITS) if cell_fragment[i, mid].max() > 0]
    hits = (breaking + [i for i in range(HITS) if i not in breaking])[:2]
    show = np.random.default_rng(0).choice(len(vox), size=min(len(vox), 12_000), replace=False)
    P = vox[show][:, [0, 2, 1]]  # matplotlib draws z up; our y is up
    lo, hi = P.min(axis=0), P.max(axis=0)
    centre, half = (lo + hi) / 2, (hi - lo).max() / 2
    palette = colormaps["tab10"]

    def panel(ax, colours, title, star=None):
        ax.scatter(*P.T, c=colours, s=2, linewidths=0, depthshade=False)
        if star is not None:
            s = star[[0, 2, 1]]
            ax.scatter(*s, marker="*", s=220, c="red", edgecolors="black", zorder=5)
        for set_lim, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centre):
            set_lim(c - half, c + half)
        ax.view_init(elev=22, azim=-58)
        try:
            ax.set_box_aspect(None, zoom=1.3)
        except TypeError:  # older matplotlib: no zoom
            pass
        ax.set_axis_off()
        ax.set_title(title, fontsize=9)

    fig = plt.figure(figsize=(16, 8.4))
    shuffle = np.random.default_rng(1).permutation(cell.max() + 1)
    panel(fig.add_subplot(2, 4, 1, projection="3d"), colormaps["tab20"](shuffle[cell[show]] % 20),
          f"{record['cells']} cells, {record['links']} links")
    for r, i in enumerate(hits):
        for c, j in enumerate(SHOW_LEVELS):
            lab = cell_fragment[i, j][cell[show]]
            colours = np.array([(0.78, 0.78, 0.78, 0.35) if l == 0 else palette((l - 1) % 10)
                                for l in lab])
            lv = record["levels"][j]
            n_frag = int(cell_fragment[i, j].max()) + 1
            panel(fig.add_subplot(2, 4, r * 4 + c + 2, projection="3d"), colours,
                  f"hit {i}, strength {lv['strength']:.0%}: {n_frag} fragment{'s' * (n_frag != 1)}",
                  hit_pts[i])
    ax = fig.add_subplot(2, 4, 5)
    ax.set_axis_off()
    tm = record["timings"]
    homework = sum(tm.get(k, 0) for k in ("cage_s", "tetgen_s", "modes_s", "precompute_s"))
    lines = [f"#{record['id']:05d} {record['family'].replace('_', ' ')}", record["describe"], "",
             f"{record['triangles']:,} triangles, {record['tets']:,} tets",
             f"{record['pieces']} teacher pieces, homework {homework:.0f} s",
             f"{record['voxels']:,} voxels, {record['cells']} cells",
             f"purity {record['purity_mean']:.1%}, lost {record['lost_volume_mean']:.2%}",
             "", "strength: median broken off / distinct outcomes"]
    lines += [f"  {lv['strength']:.0%}: {lv['broken_off_median']:.1%} / {lv['distinct_outcomes']} of {HITS}"
              for lv in record["levels"]]
    ax.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")
    fig.suptitle(f"RIFT-Arch sample #{record['id']:05d}: grey = main body, colours = pieces "
                 f"that break off, red star = hit", fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.92, wspace=0.02, hspace=0.12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)


# ---------------------------------------------------------------- main (runs the helpers)


def parse_ids(text):
    ids = set()
    for part in text.split(","):
        a, _, b = part.strip().partition("-")
        ids.update(range(int(a), int(b or a) + 1))
    return sorted(ids)


def read_shapes(data_dir):
    """Latest record per shape ID from make_shapes.py's manifest."""
    shapes = {}
    path = data_dir / "shapes.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                shapes[r["id"]] = r
    return shapes


def read_record(data_dir, shape_id):
    path = data_dir / "samples" / f"{shape_id:05d}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def is_done(record, data_dir, opts, retry_failed):
    if record is None:
        return False
    if record.get("status") != "ok":
        return not retry_failed and all(record.get(k) == v for k, v in settings(opts).items())
    return (all(record.get(k) == v for k, v in settings(opts).items())
            and (data_dir / "samples" / f"{record['id']:05d}.npz").exists())


def main():
    p = argparse.ArgumentParser(description="Make RIFT-Arch training samples with the teacher.")
    p.add_argument("--ids", default="0-19", help="shape IDs, e.g. 0-19 or 0-1999 or 3,7,40-59")
    p.add_argument("--every", type=int, default=1, help="only IDs where id %% every == offset")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--workers", type=int, default=1,
                   help="shapes at once (each in its own process); 0 = in this process, for debugging")
    p.add_argument("--timeout", type=float, default=1800, help="seconds before a shape is stopped")
    p.add_argument("--push", choices=PUSH_RULES, default=PUSH)
    p.add_argument("--cage", type=int, default=TEACHER["cage"], help="testing only; D1 says 5000")
    p.add_argument("--modes", type=int, default=TEACHER["modes"], help="testing only; D1 says 20")
    p.add_argument("--retry-failed", action="store_true", help="try shapes that failed before again")
    p.add_argument("--no-pictures", action="store_true")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--fm-dir", type=Path, default=DEFAULT_FM_DIR)
    args = p.parse_args()
    ids = [i for i in parse_ids(args.ids) if i % args.every == args.offset]
    (args.data_dir / "samples").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "pictures").mkdir(parents=True, exist_ok=True)
    opts = {"data_dir": str(args.data_dir), "out_dir": str(args.out_dir), "fm_dir": str(args.fm_dir),
            "push": args.push, "cage": args.cage, "modes": args.modes,
            "pictures": not args.no_pictures}
    part = f", every {args.every} from {args.offset}" if args.every > 1 else ""
    title = f"Dataset: RIFT-Arch samples v{DATASET_VERSION}, IDs {args.ids}{part}"
    with ui.Run(title, args.out_dir, steps=2) as run:
        generate(run, args, ids, opts)


def generate(run, args, ids, opts):
    shapes = read_shapes(args.data_dir)
    missing = [i for i in ids if shapes.get(i, {}).get("status") != "ok"]
    ids = [i for i in ids if i not in missing]
    todo = [i for i in ids
            if not is_done(read_record(args.data_dir, i), args.data_dir, opts, args.retry_failed)]
    push_note = "mass push (D5)" if opts["push"] == "mass" else "corners push"
    run.note(f"{len(ids)} shapes asked for, {len(ids) - len(todo)} already done, {len(todo)} to make; "
             f"{push_note}, cage {opts['cage']}, {opts['modes']} modes, {args.workers or 'no'} helper "
             f"process{'es' * (args.workers != 1)}.")
    if missing:
        run.note(f"No finished shape for {len(missing)} IDs (make them with make_shapes.py first): "
                 f"{missing[:10]}{' ...' if len(missing) > 10 else ''}")
    if opts["cage"] != TEACHER["cage"] or opts["modes"] != TEACHER["modes"]:
        run.say("[yellow]Teacher setting differs from D1 (option B): testing only.[/]",
                "Teacher setting differs from D1 (option B): testing only.")

    with run.step(f"Make {len(todo)} samples"):
        for shape_id, why in run.track(run_helpers(run, args, opts, shapes, todo), "shapes",
                                       total=len(todo)):
            record = read_record(args.data_dir, shape_id)
            if record is None or why:  # the helper died or was stopped before writing a record
                record = {"id": shape_id, "family": shapes[shape_id]["family"], "status": "failed",
                          **settings(opts), "error": why or "the helper stopped without a record"}
                (args.data_dir / "samples" / f"{shape_id:05d}.json").write_text(json.dumps(record, indent=1))
            if record["status"] == "ok":
                pic = args.out_dir / "pictures" / f"{shape_id:05d}.png"
                if pic.exists():
                    (args.out_dir / "latest.png").write_bytes(pic.read_bytes())
                run.note(f"#{shape_id:05d} {record['family']}: {record['seconds']:.0f} s, "
                         f"{record['pieces']} pieces, 6% strength gives "
                         f"{record['levels'][2]['distinct_outcomes']} different outcomes from {HITS} hits")
            else:
                run.say(f"[red]#{shape_id:05d} failed[/]: {record.get('error')}",
                        f"#{shape_id:05d} failed: {record.get('error')}")

    with run.step("Summarise"):
        records = [r for r in (read_record(args.data_dir, i) for i in ids) if r is not None]
        summarise(run, args, records, len(ids))


def run_helpers(run, args, opts, shapes, todo):
    """Run each shape in its own process, up to --workers at once. Yields (id, problem)."""
    if args.workers <= 0:  # debugging: everything in this process
        for shape_id in todo:
            child_main(shape_id, {**opts, "shape": shapes[shape_id]})
            yield shape_id, None
        return
    ctx = multiprocessing.get_context("spawn")  # what Windows does anyway
    queue, running = list(todo), {}
    try:
        while queue or running:
            while queue and len(running) < args.workers:
                shape_id = queue.pop(0)
                proc = ctx.Process(target=child_main, args=(shape_id, {**opts, "shape": shapes[shape_id]}))
                proc.start()
                running[shape_id] = (proc, time.time())
                run.set_substep("working on " + ", ".join(f"#{i:05d}" for i in running))
            time.sleep(0.5)
            for shape_id, (proc, started) in list(running.items()):
                why = None
                if proc.exitcode is None:
                    if time.time() - started < args.timeout:
                        continue
                    proc.terminate()
                    why = f"stopped after {args.timeout:.0f} s"
                proc.join()
                if why is None and proc.exitcode != 0:
                    why = f"helper process crashed (exit code {proc.exitcode})"
                del running[shape_id]
                if running:
                    run.set_substep("working on " + ", ".join(f"#{i:05d}" for i in running))
                yield shape_id, why
    finally:
        for proc, _ in running.values():  # Ctrl+C or an error: stop the helpers too
            proc.terminate()


def summarise(run, args, records, asked):
    ok = [r for r in records if r["status"] == "ok"]
    failed = [r for r in records if r["status"] != "ok"]
    families = sorted({r["family"] for r in records})
    med = lambda xs: float(np.median(xs)) if len(xs) else None
    homework = lambda r: sum(r["timings"].get(k, 0) for k in ("cage_s", "tetgen_s", "modes_s", "precompute_s"))
    columns = ["Family", "Made", "Failed", "Triangles", "Homework s", "Total s",
               "Broken off at 2/6/12%", "Fragments at 2/6/12%", "Outcomes at 6%", "Purity", "Rebuilt"]
    rows, table = [], {}
    for fam in families + ["all"]:
        mine = [r for r in ok if fam in ("all", r["family"])]
        bad = [r for r in failed if fam in ("all", r["family"])]
        if not mine and not bad:
            continue
        lv = lambda j, key: med([r["levels"][j][key] for r in mine])
        t = {"samples": len(mine), "failed": len(bad),
             "triangles": med([r["triangles"] for r in mine]),
             "homework_s": med([homework(r) for r in mine]), "total_s": med([r["seconds"] for r in mine]),
             "broken_off": [lv(j, "broken_off_median") for j in range(len(STRENGTHS))],
             "fragments": [lv(j, "cell_fragments_mean") for j in range(len(STRENGTHS))],
             "distinct_6": lv(2, "distinct_outcomes"),
             "purity": med([r["purity_mean"] for r in mine]),
             "rebuilt": med([r["rebuilt_share"] for r in mine])}
        table[fam] = t
        if not mine:
            rows.append([fam.replace("_", " "), 0, len(bad)] + ["-"] * (len(columns) - 3))
            continue
        pick = lambda xs, fmt: " / ".join(fmt.format(xs[j]) for j in SHOW_LEVELS)
        rows.append([fam.replace("_", " "), len(mine), len(bad), f"{t['triangles']:,.0f}",
                     f"{t['homework_s']:.0f}", f"{t['total_s']:.0f}",
                     pick(t["broken_off"], "{:.1%}"), pick(t["fragments"], "{:.1f}"),
                     f"{t['distinct_6']:.0f}", f"{t['purity']:.1%}", f"{t['rebuilt']:.0%}"])
    pace = med([r["seconds"] for r in ok])
    estimate = (f" At the median pace ({pace:.0f} s per shape, {args.workers or 1} at once), 2,000 shapes "
                f"take about {2000 * pace / max(args.workers, 1) / 3600:.0f} hours on one laptop."
                if pace else "")
    errors = sorted({" ".join(str(r.get("error", "?")).split())[:160] for r in failed})
    (args.out_dir / "summary.json").write_text(json.dumps(
        {"dataset_version": DATASET_VERSION, "ids": args.ids, "asked": asked, "made": len(ok),
         "failed": len(failed), "settings": settings({"push": args.push, "cage": args.cage,
                                                      "modes": args.modes}),
         "families": table, "errors": errors}, indent=2))
    intro = (f"Dataset v{DATASET_VERSION}, IDs {args.ids}: {len(ok)} samples, {len(failed)} failed "
             f"(medians per shape; fragments and outcomes are per hit, of {HITS} hits). "
             f"Teacher option B, {args.push} push, strengths 2–12%, {HITS} hits per shape, "
             f"{CELLS} cells.{estimate}")
    if errors:
        intro += " Failures: " + "; ".join(e.rstrip(".") for e in errors[:5]) + ("." if len(errors) <= 5 else "; ...")
    run.table("RIFT-Arch samples", columns, rows)
    run.note(intro)
    run.log_markdown("\n".join([intro, "", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
                               + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]))


if __name__ == "__main__":
    main()
