"""Day 5: turn the teacher's breaks into RIFT's training format, and check what is lost.

RIFT does not predict tets. It works on a shape pre-cut into Voronoi cells and
predicts which links between touching cells break. This script:

  1. builds the teacher with the setting chosen on Day 4 (outputs/day4/chosen.json),
  2. fills the shape with a fine grid of small cubes (voxels), so the cells do not
     depend on how coarse the teacher's tets are,
  3. cuts the shape into 128 Voronoi cells: random seed points inside the shape,
     spread evenly with a few rounds of Lloyd relaxation (move each seed to the
     middle of its cell, repeat); every voxel joins its nearest seed,
  4. links two cells when their voxels touch (link size = touching area),
  5. finds which teacher tet each voxel sits in, so every hit's fragments can be
     read off per voxel,
  6. for each hit, gives each cell the fragment holding most of its volume, and
     marks a link "break" if its two cells ended up in different fragments,
  7. measures how faithfully the cells capture the teacher's break:
       purity       share of each cell's volume inside its own fragment (1.0 = perfect)
       rebuilt      do the cells, joined by unbroken links, give back exactly the
                    fragments the cells were labelled with?
       lost volume  volume of fragments too small to own a single cell
  8. draws 4 hits: teacher fragments | cells | cell graph with broken links in red.

This is a discrete Voronoi cut, good for labels. Cutting real cell meshes for the
game comes later, in the Unity demo.

Usage (Miniforge Prompt, rift-teacher active, repo root), after Day 4:
    python scripts/cell_graph.py

Outputs in outputs/day5/ (gitignored):
    per_hit.csv         one row per hit
    summary.json        averages over all hits
    cell_graph_<i>.png  pictures of 4 hits
    cell_graph.npz      cells, links and labels: a first look at one training sample
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # find teacher.py next to this file
from teacher import DEFAULT_FM_DIR, build_teacher, load_shape, project_impact, sample_impacts


# ---------------------------------------------------------------- voxels, cells and links


def voxelize(v_fine, f_fine, min_across=8, max_voxels=600_000):
    """Centres of small cubes filling the shape, on a regular grid.

    Spacing h: at least `min_across` cubes across the thinnest side, but no more
    than `max_voxels` cubes in the bounding box. Returns (centres, grid index, h).
    """
    import igl

    lo, hi = v_fine.min(axis=0), v_fine.max(axis=0)
    extent = hi - lo
    h = max(extent.min() / min_across, (np.prod(extent) / max_voxels) ** (1 / 3))
    counts = np.maximum(np.round(extent / h).astype(int), 1)
    axes = [lo[i] + (np.arange(counts[i]) + 0.5) * extent[i] / counts[i] for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    index = np.stack(np.meshgrid(*[np.arange(c) for c in counts], indexing="ij"),
                     axis=-1).reshape(-1, 3)
    inside = igl.winding_number(v_fine, f_fine, grid) > 0.5
    spacing = extent / counts
    return grid[inside], index[inside], spacing


def seeds_inside(points, n, rng):
    """n distinct random voxel centres as starting seeds."""
    return points[rng.choice(len(points), size=n, replace=False)].copy()


def make_cells(points, seeds, lloyd_rounds):
    """Voronoi cells: every voxel joins its nearest seed. Each Lloyd round moves
    every seed to the middle of its cell, which evens out cell sizes."""
    from scipy.spatial import cKDTree

    seeds = seeds.copy()
    for _ in range(lloyd_rounds):
        cell = cKDTree(seeds).query(points)[1]
        count = np.bincount(cell, minlength=len(seeds))
        sums = np.zeros_like(seeds)
        np.add.at(sums, cell, points)
        filled = count > 0
        seeds[filled] = sums[filled] / count[filled, None]
    cell = cKDTree(seeds).query(points)[1]
    used, cell = np.unique(cell, return_inverse=True)  # drop seeds that got no voxels
    return seeds[used], cell


def make_links(index, cell, spacing):
    """Pairs of cells whose voxels touch face to face, with the touching area."""
    dims = index.max(axis=0) + 2
    key = lambda ijk: (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    keys = key(index)
    order = np.argsort(keys)
    sorted_keys = keys[order]
    a_all, b_all, area_all = [], [], []
    for axis in range(3):
        step = np.zeros(3, dtype=int)
        step[axis] = 1
        nb_keys = key(index + step)
        pos = np.searchsorted(sorted_keys, nb_keys)
        pos[pos == len(sorted_keys)] = 0
        found = sorted_keys[pos] == nb_keys
        a = cell[found]
        b = cell[order[pos[found]]]
        differ = a != b
        face_area = np.prod(np.delete(spacing, axis))
        a_all.append(a[differ])
        b_all.append(b[differ])
        area_all.append(np.full(differ.sum(), face_area))
    a, b, area = map(np.concatenate, (a_all, b_all, area_all))
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    n = cell.max() + 1
    pair, inverse = np.unique(lo * n + hi, return_inverse=True)
    return np.stack([pair // n, pair % n], axis=1), np.bincount(inverse, weights=area)


def locate_in_tets(points, nodes, elements, k=24):
    """Index of the teacher tet containing each point.

    Checks the k tets with the nearest centres using barycentric coordinates
    (a point is inside a tet when all four weights are >= 0). Points found in
    none of them fall back to the nearest tet centre; their share is returned
    so we know how exact the lookup was.
    """
    from scipy.spatial import cKDTree

    a = nodes[elements[:, 0]]
    edges = np.stack([nodes[elements[:, j]] - a for j in (1, 2, 3)], axis=2)  # (T, 3, 3)
    det = np.linalg.det(edges)
    edges[np.abs(det) < 1e-18] = np.eye(3)  # flat tets: never "contain" anything useful
    inv = np.linalg.inv(edges)
    _, cand = cKDTree(nodes[elements].mean(axis=1)).query(points, k=k)
    found = np.full(len(points), -1)
    for j in range(k):
        todo = np.flatnonzero(found < 0)
        if todo.size == 0:
            break
        t = cand[todo, j]
        lam = np.einsum("nij,nj->ni", inv[t], points[todo] - a[t])
        ok = (lam >= -1e-9).all(axis=1) & (lam.sum(axis=1) <= 1 + 1e-9)
        found[todo[ok]] = t[ok]
    missed = found < 0
    found[missed] = cand[missed, 0]
    return found, float(missed.mean())


# ---------------------------------------------------------------- one hit


def label_hit(fm, voxel_tet, cell, links, n_cells):
    """Cell fragments, link breaks and quality numbers for the last impact."""
    from scipy.sparse import coo_matrix, csr_matrix
    from scipy.sparse.csgraph import connected_components

    frag = np.asarray(fm.tet_labels_after_impact).astype(int)[voxel_tet]  # fragment per voxel
    n_frag = int(fm.n_pieces_after_impact)
    table = coo_matrix((np.ones(len(frag)), (cell, frag)), shape=(n_cells, n_frag)).toarray()
    cell_frag = table.argmax(axis=1)  # fragment holding most of each cell
    purity = table.max(axis=1).sum() / table.sum()  # volume-weighted over the shape
    cell_purity = table.max(axis=1) / table.sum(axis=1)

    broken = cell_frag[links[:, 0]] != cell_frag[links[:, 1]]
    hold = links[~broken]
    adj = csr_matrix((np.ones(len(hold)), (hold[:, 0], hold[:, 1])), shape=(n_cells, n_cells))
    n_groups, _ = connected_components(adj, directed=False)
    frags_in_cells = np.unique(cell_frag)

    frag_vol = np.bincount(frag, minlength=n_frag)
    present = np.flatnonzero(frag_vol > 0)
    lost = np.setdiff1d(present, frags_in_cells)
    return {
        "cell_frag": cell_frag, "broken": broken, "frag": frag,
        "teacher_fragments": len(present),
        "cell_fragments": len(frags_in_cells),
        "cell_groups": int(n_groups),
        "rebuilt": bool(n_groups == len(frags_in_cells)),
        "purity": float(purity),
        "min_cell_purity": float(cell_purity.min()),
        "lost_volume_share": float(frag_vol[lost].sum() / frag_vol.sum()),
        "links_broken": int(broken.sum()),
    }


# ---------------------------------------------------------------- pictures


def draw_hit(path, title, xy_vox, xy_cells, links, cell, res, point_xy):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.collections import LineCollection

    def frag_colours(labels, weights=None):
        size = np.bincount(labels, weights=weights)
        order = np.argsort(-size)  # biggest fragment first -> grey
        palette = colormaps["tab10"]
        lut = {f: (0.75, 0.75, 0.75, 1.0) if rank == 0 else palette((rank - 1) % 10)
               for rank, f in enumerate(order)}
        return np.array([lut[l] for l in labels])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    axes[0].scatter(*xy_vox.T, c=frag_colours(res["frag"]), s=1.5, linewidths=0)
    axes[0].set_title(f"Teacher: {res['teacher_fragments']} fragments")

    shuffle = np.random.default_rng(1).permutation(len(xy_cells))  # neighbours get different colours
    axes[1].scatter(*xy_vox.T, c=shuffle[cell] % 20, cmap="tab20", s=1.5, linewidths=0)
    axes[1].scatter(*xy_cells.T, c="black", s=5)
    axes[1].set_title(f"{len(xy_cells)} Voronoi cells")

    segs = xy_cells[links]
    axes[2].add_collection(LineCollection(segs[~res["broken"]], colors=(0.55, 0.55, 0.55, 0.6),
                                          linewidths=0.6))
    axes[2].add_collection(LineCollection(segs[res["broken"]], colors="red", linewidths=1.6))
    cell_size = np.bincount(cell, minlength=len(xy_cells)).astype(float)
    axes[2].scatter(*xy_cells.T, c=frag_colours(res["cell_frag"], cell_size), s=22,
                    edgecolors="black", linewidths=0.3, zorder=3)
    axes[2].set_title(f"Cell graph: {res['links_broken']} of {len(links)} links break (red)")

    for ax in axes:
        ax.plot(*point_xy, marker="*", color="red", markersize=14, markeredgecolor="black",
                zorder=4)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser(description="Build RIFT's cell graph from teacher breaks.")
    p.add_argument("--chosen", type=Path, default=Path("outputs/day4/chosen.json"),
                   help="teacher setting from Day 4; the options below override it")
    p.add_argument("--mesh", type=Path)
    p.add_argument("--cage", type=int)
    p.add_argument("--modes", type=int)
    p.add_argument("--d", type=int, choices=[1, 3])
    p.add_argument("--threshold", type=float)
    p.add_argument("--grid-size", type=int)
    p.add_argument("--minratio", type=float)
    p.add_argument("--cells", type=int, default=128)
    p.add_argument("--lloyd", type=int, default=5, help="rounds of seed spreading")
    p.add_argument("--voxels-across", type=int, default=8,
                   help="voxels across the thinnest side of the shape")
    p.add_argument("--impacts", type=int, default=50)
    p.add_argument("--render", type=int, default=4, help="how many hits to draw")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/day5"))
    p.add_argument("--fm-dir", type=Path, default=DEFAULT_FM_DIR)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(args.chosen.read_text()) if args.chosen.is_file() else {}
    defaults = {"mesh": "outputs/day3/wall.obj", "cage": 5000, "modes": 20, "d": 3,
                "threshold": None, "grid_size": 256, "minratio": 1.5}
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, cfg.get(key, value))
    if args.threshold is None:
        raise SystemExit("No threshold: run scripts/sweep_threshold.py first, or pass --threshold")
    setting = f"cage{args.cage}_m{args.modes}_d{args.d}, threshold {args.threshold:.3g}"
    print(f"Teacher setting: {setting}" + (" (from chosen.json)" if cfg else ""), flush=True)

    v_fine, f_fine = load_shape(args.mesh)
    T = build_teacher(v_fine, f_fine, args.cage, args.d, args.modes, args.grid_size,
                      args.minratio, args.fm_dir)
    fm, nodes, elements = T["fm"], T["nodes"], T["elements"]
    print(f"Teacher built in {sum(T['timings'].values()):.0f} s: {len(elements)} tets, "
          f"up to {fm.precomputed_num_pieces} pieces", flush=True)

    vox, vox_index, spacing = voxelize(v_fine, f_fine, args.voxels_across)
    rng = np.random.default_rng(args.seed)
    seeds, cell = make_cells(vox, seeds_inside(vox, args.cells, rng), args.lloyd)
    n_cells = len(seeds)
    links, link_area = make_links(vox_index, cell, spacing)
    voxel_tet, missed = locate_in_tets(vox, nodes, elements)
    cell_size = np.bincount(cell, minlength=n_cells)
    cell_cent = np.zeros((n_cells, 3))
    np.add.at(cell_cent, cell, vox)
    cell_cent /= cell_size[:, None]
    tets_per_cell = np.array([np.unique(voxel_tet[cell == c]).size for c in range(n_cells)])
    print(f"{len(vox)} voxels -> {n_cells} cells ({cell_size.min()}-{cell_size.max()} voxels each), "
          f"{len(links)} links; teacher tets per cell: median {np.median(tets_per_cell):.0f}; "
          f"voxels not found in a tet: {missed:.2%}", flush=True)

    points, dirs = sample_impacts(v_fine, f_fine, args.impacts, args.seed + 1)
    rows, kept = [], []
    for i, (pt, dvec) in enumerate(zip(points, dirs)):
        project_impact(fm, pt, dvec, args.d, args.threshold)
        res = label_hit(fm, voxel_tet, cell, links, n_cells)
        rows.append({"hit": i, **{k: res[k] for k in (
            "teacher_fragments", "cell_fragments", "cell_groups", "rebuilt", "purity",
            "min_cell_purity", "lost_volume_share", "links_broken")},
            "links": len(links)})
        kept.append(res)

    with open(args.out_dir / "per_hit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Draw hits that break into 2+ fragments first, then any others.
    order = ([i for i, r in enumerate(rows) if r["cell_fragments"] > 1]
             + [i for i, r in enumerate(rows) if r["cell_fragments"] <= 1])[: args.render]
    extent = v_fine.max(axis=0) - v_fine.min(axis=0)
    axes2d = np.sort(np.argsort(-extent)[:2])  # the two widest directions (x and y for the wall)
    show = np.random.default_rng(2).choice(len(vox), size=min(len(vox), 40_000), replace=False)
    for i in order:
        r = rows[i]
        res_shown = dict(kept[i], frag=kept[i]["frag"][show])
        title = (f"{setting} | hit {i}: teacher {r['teacher_fragments']} fragments, cells "
                 f"{r['cell_fragments']} | purity {r['purity']:.1%} | rebuilt: "
                 f"{'yes' if r['rebuilt'] else 'NO'} | lost volume {r['lost_volume_share']:.1%}")
        draw_hit(args.out_dir / f"cell_graph_{i}.png", title, vox[show][:, axes2d],
                 cell_cent[:, axes2d], links, cell[show], res_shown, points[i][axes2d])

    broke = [r for r in rows if r["teacher_fragments"] > 1]
    mean = lambda key, rs: float(np.mean([r[key] for r in rs])) if rs else float("nan")
    summary = {
        "setting": setting, "cells": n_cells, "voxels": len(vox),
        "voxel_size": spacing.tolist(), "voxels_not_in_a_tet": missed,
        "teacher_tets_per_cell_median": float(np.median(tets_per_cell)),
        "links": int(len(links)), "mean_links_per_cell": 2 * len(links) / n_cells,
        "hits": len(rows), "hits_with_a_break": len(broke),
        "mean_purity": mean("purity", rows), "mean_purity_breaking_hits": mean("purity", broke),
        "worst_purity": float(min(r["purity"] for r in rows)),
        "rebuilt_share": mean("rebuilt", rows),
        "mean_lost_volume_share": mean("lost_volume_share", rows),
        "mean_teacher_fragments": mean("teacher_fragments", broke),
        "mean_cell_fragments": mean("cell_fragments", broke),
        "mean_links_broken_share": float(np.mean([r["links_broken"] / len(links) for r in broke]))
        if broke else float("nan"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        args.out_dir / "cell_graph.npz", seeds=seeds, cell_centroid=cell_cent,
        cell_volume=cell_size * np.prod(spacing), links=links, link_area=link_area,
        hit_points=points, hit_directions=dirs,
        cell_fragment=np.stack([k["cell_frag"] for k in kept]),
        link_broken=np.stack([k["broken"] for k in kept]), threshold=args.threshold)

    s = summary
    print(f"\nSaved results in {args.out_dir}\n\nPaste into docs/kill-test-log.md (Day 5):\n")
    print("| Setting | Cells | Links (per cell) | Teacher tets per cell | Hits with a break "
          "| Purity (all / breaking hits / worst) | Rebuilt | Lost volume "
          "| Fragments: teacher / cells | Links broken |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    print(f"| {setting} | {s['cells']} | {s['links']} ({s['mean_links_per_cell']:.1f}) | "
          f"{s['teacher_tets_per_cell_median']:.0f} | {s['hits_with_a_break']} of {s['hits']} | "
          f"{s['mean_purity']:.1%} / {s['mean_purity_breaking_hits']:.1%} / {s['worst_purity']:.1%} | "
          f"{s['rebuilt_share']:.0%} | {s['mean_lost_volume_share']:.2%} | "
          f"{s['mean_teacher_fragments']:.1f} / {s['mean_cell_fragments']:.1f} | "
          f"{s['mean_links_broken_share']:.1%} |")


if __name__ == "__main__":
    main()
