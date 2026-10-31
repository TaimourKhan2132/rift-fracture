"""Day 6c: can fracture-modes handle a SECOND hit without redoing its homework?

After a first hit breaks a crater out of the wall, the wall is a new shape. The
teacher's homework (its modes) was done for the intact wall. This script compares,
for second hits on the damaged wall:

  reuse   the intact wall's homework, used as if nothing had happened
          (fast: what a game would be tempted to do)
  redo    new homework computed for the damaged wall
          (correct: what the dataset will contain, but minutes per shape)

If reuse and redo give nearly the same breaks, fracture-modes could handle
repeated hits cheaply and RIFT's case for damaged shapes (RQ3) is weak. If they
differ, especially near the crater, RIFT has a real job.

Steps:
  1. build the teacher on the intact wall; calibrate its threshold (Day 4 rule),
  2. pick K first hits that each knock out a crater of 2-20% of the wall,
  3. for each: keep the largest remaining fragment, rebuild it as a closed mesh
     (voxels -> smoothed surface -> simplified to about 16,000 triangles),
     redo the homework on it and calibrate its own threshold,
  4. hit the damaged wall at M points (half near the crater, half far) with both
     reuse and redo, and compare the two breaks on the damaged wall's voxels:
       ARI           agreement between the two breaks (1 = identical, 0 = unrelated)
       detached IoU  overlap of the broken-off volume (1 = identical)
     Also compares redo at the intact wall's threshold, to separate the effect of
     the new shape from the effect of a new threshold.
  0. control: the INTACT wall rebuilt through the same pipeline, no crater. If reuse
     and redo already disagree here, the disagreement comes from rebuilding the
     shape, not from the damage.

Usage (Miniforge Prompt, rift-teacher active, repo root):
    python scripts/second_hit.py              (option B: about 12 minutes)
    python scripts/second_hit.py --preset C   (option C: about 45 minutes)

Outputs in outputs/day6c_<preset>/ (gitignored): summary.json, per_hit.csv,
case_<k>.png pictures and case_<k>_wall.obj meshes (k = control, 0, 1, 2).
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # find teacher.py next to this file
from teacher import (DEFAULT_FM_DIR, build_teacher, decide, load_shape, piece_distances,
                     project_impact, sample_impacts, tet_volumes_and_centroids)
from sweep_threshold import outcome_stats
from cell_graph import locate_in_tets, voxelize

PRESETS = {"B": {"cage": 5000, "modes": 20, "d": 1},
           "C": {"cage": 5000, "modes": 40, "d": 3}}
NEAR = 0.15  # second hits closer than this to the crater count as "near" (wall width = 1)
MIN_FRAGMENT = 30  # voxels; a Voronoi cell is about 540, so smaller bits are lookup specks


# ---------------------------------------------------------------- helpers


def calibrate(T, v, f, n_hits, n_thresholds, seed):
    """Day 4 rule: the threshold giving the most different outcomes over n_hits."""
    fm = T["fm"]
    tet_vol, tet_cent = tet_volumes_and_centroids(T["nodes"], T["elements"])
    p2t = fm.piece_to_tet_matrix
    piece_vol = np.asarray(p2t.T @ tet_vol).ravel()
    piece_cent = np.asarray(p2t.T @ (tet_vol[:, None] * tet_cent)) / piece_vol[:, None]
    points, dirs = sample_impacts(v, f, n_hits, seed)
    dists = []
    for p, dvec in zip(points, dirs):
        project_impact(fm, p, dvec, T["d"], 1.0)
        dists.append(piece_distances(fm))
    pooled = np.concatenate(dists)
    pooled = pooled[pooled > 0]
    grid = np.geomspace(pooled.min() * 0.5, pooled.max() * 2.0, n_thresholds)
    rows = [outcome_stats(fm, dists, t, points, piece_vol, piece_cent, tet_vol, tet_cent)
            for t in grid]
    best = max(rows, key=lambda s: (s["distinct_outcomes"], s["frac_partial"],
                                    -np.nan_to_num(s["locality_ratio"], nan=9.0)))
    return best, points, dirs, dists, piece_vol


def voxel_fragments(fm, voxel_tet):
    """Fragment of each voxel after the last impact."""
    return np.asarray(fm.tet_labels_after_impact).astype(int)[voxel_tet]


def ari(a, b):
    """Adjusted Rand index between two labelings of the same voxels (1 = identical)."""
    from scipy.sparse import coo_matrix

    n = len(a)
    table = coo_matrix((np.ones(n), (a, b))).tocsr()
    pairs = lambda x: x * (x - 1) / 2.0
    same_both = pairs(table.data).sum()
    same_a = pairs(np.asarray(table.sum(axis=1)).ravel()).sum()
    same_b = pairs(np.asarray(table.sum(axis=0)).ravel()).sum()
    expected = same_a * same_b / pairs(n)
    best = 0.5 * (same_a + same_b)
    return 1.0 if best == expected else float((same_both - expected) / (best - expected))


def split_disconnected(labels, index):
    """Split each fragment into the parts whose voxels still touch face to face.

    Without this, a fragment could stay "joined" through crater material that is
    already gone. Applied to reuse and redo alike, so reuse gets its best cheap version.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    dims = index.max(axis=0) + 2
    ids = -np.ones(dims, dtype=np.int64)
    ids[tuple(index.T)] = np.arange(len(index))
    a_all, b_all = [], []
    for axis in range(3):
        nb = index.copy()
        nb[:, axis] += 1
        b = ids[tuple(nb.T)]
        ok = (b >= 0) & (labels[np.maximum(b, 0)] == labels)  # neighbour exists, same fragment
        a_all.append(np.nonzero(ok)[0])
        b_all.append(b[ok])
    a, b = np.concatenate(a_all), np.concatenate(b_all)
    n = len(labels)
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def count_fragments(labels, min_voxels):
    """Fragments with at least min_voxels voxels (ignores specks from the voxel lookup)."""
    return int(np.sum(np.bincount(labels) >= min_voxels))


def detached(labels):
    """Voxels that broke off: everything outside the largest fragment."""
    return labels != np.argmax(np.bincount(labels))


def iou(a, b):
    union = np.sum(a | b)
    return 1.0 if union == 0 else float(np.sum(a & b) / union)


def signed_volume(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)


def damaged_mesh(vox_index, keep, lo, spacing, target_faces=16000, smooth=0.5):
    """Closed triangle mesh of the voxels in `keep` (the wall after the first hit).

    Voxels -> largest connected piece -> slightly blurred -> surface at the 50% level
    (marching cubes) -> simplified to about target_faces triangles.
    """
    import igl
    from scipy import ndimage
    from skimage.measure import marching_cubes

    dims = vox_index.max(axis=0) + 1
    occ = np.zeros(dims, dtype=bool)
    occ[tuple(vox_index[keep].T)] = True
    labels, n = ndimage.label(occ)
    if n > 1:  # keep only the largest connected piece
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        occ = labels == np.argmax(sizes)
    pad = 3
    field = ndimage.gaussian_filter(np.pad(occ.astype(float), pad), smooth)
    verts, faces, _, _ = marching_cubes(field, level=0.5)
    V = lo + (verts - pad + 0.5) * spacing  # grid index -> wall coordinates
    F = faces.astype(np.int64)
    if signed_volume(V, F) < 0:  # make faces point outward
        F = F[:, ::-1]
    if len(F) > target_faces:
        ok, U, G, _, _ = igl.decimate(V, F, target_faces)
        if ok and len(G) > 0:
            V, F = np.asarray(U), np.asarray(G)
    F = F.astype(np.int32)
    open_edges = igl.boundary_facets(F)
    if len(open_edges) != 0:
        raise RuntimeError(f"damaged mesh is not closed ({len(open_edges)} open edges)")
    return V, F


# ---------------------------------------------------------------- main


def main():
    import igl
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from scipy.spatial import cKDTree

    p = argparse.ArgumentParser(description="Compare reused vs redone homework for second hits.")
    p.add_argument("--preset", choices=sorted(PRESETS), default="B")
    p.add_argument("--mesh", type=Path, default=Path("outputs/day3/wall.obj"))
    p.add_argument("--cases", type=int, default=3, help="how many first hits (craters)")
    p.add_argument("--hits", type=int, default=40, help="second hits per crater")
    p.add_argument("--calibration-hits", type=int, default=60)
    p.add_argument("--thresholds", type=int, default=40)
    p.add_argument("--mesh-faces", type=int, default=16000,
                   help="triangles in each rebuilt wall (fewer rounds its corners)")
    p.add_argument("--grid-size", type=int, default=256)
    p.add_argument("--minratio", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--fm-dir", type=Path, default=DEFAULT_FM_DIR)
    args = p.parse_args()
    cfg = PRESETS[args.preset]
    out = args.out_dir or Path(f"outputs/day6c_{args.preset}")
    out.mkdir(parents=True, exist_ok=True)
    setting = f"option {args.preset}: cage{cfg['cage']}_m{cfg['modes']}_d{cfg['d']}"
    print(f"Teacher setting: {setting}", flush=True)

    # 1) Intact wall: homework, threshold, voxels.
    v0, f0 = load_shape(args.mesh)
    T1 = build_teacher(v0, f0, cfg["cage"], cfg["d"], cfg["modes"], args.grid_size,
                       args.minratio, args.fm_dir)
    t_intact = sum(T1["timings"].values())
    best1, cal_pts, cal_dirs, cal_dists, piece_vol1 = calibrate(
        T1, v0, f0, args.calibration_hits, args.thresholds, args.seed)
    thr1 = best1["threshold"]
    vox, vox_index, spacing = voxelize(v0, f0)
    lo = v0.min(axis=0)
    vt1, missed1 = locate_in_tets(vox, T1["nodes"], T1["elements"])
    print(f"Intact wall: homework {t_intact:.0f} s, threshold {thr1:.3g}, "
          f"{len(vox)} voxels ({missed1:.2%} not in a tet)", flush=True)

    # 2) First hits that knock out a crater of 2-20% of the wall, spread apart.
    total = piece_vol1.sum()
    candidates = []
    for i, d0 in enumerate(cal_dists):
        _, lab = decide(T1["fm"], d0, thr1)
        frag_vol = np.bincount(lab, weights=piece_vol1)
        share = 1 - frag_vol.max() / total
        if 0.02 <= share <= 0.20:
            candidates.append((i, share))
    chosen = []
    for i, share in candidates:
        if all(np.linalg.norm(cal_pts[i] - cal_pts[j]) > 0.3 for j, _ in chosen):
            chosen.append((i, share))
        if len(chosen) == args.cases:
            break
    if not chosen:
        raise SystemExit("No first hit made a 2-20% crater at this threshold.")

    rows, cases = [], []
    axes2d = np.sort(np.argsort(-(v0.max(axis=0) - v0.min(axis=0)))[:2])
    # The control rebuilds the INTACT wall through the same pipeline (no crater). Any
    # disagreement there comes from rebuilding the shape, not from damage.
    runs = [("control", None, 0.0)] + [(k, i1, share) for k, (i1, share) in enumerate(chosen)]
    for k, i1, crater_share in runs:
        if k == "control":
            print("Control: intact wall rebuilt through the same pipeline (no crater)", flush=True)
            keep = np.ones(len(vox), dtype=bool)
            crater_tree = None
        else:
            print(f"Case {k}: first hit {i1} knocks out {crater_share:.1%} of the wall",
                  flush=True)
            project_impact(T1["fm"], cal_pts[i1], cal_dirs[i1], cfg["d"], thr1)
            frag = voxel_fragments(T1["fm"], vt1)
            keep = frag == np.argmax(np.bincount(frag))
            crater_tree = cKDTree(vox[~keep])

        # 3) The (damaged) wall as a new shape: new homework, own threshold.
        V2, F2 = damaged_mesh(vox_index, keep, lo, spacing, target_faces=args.mesh_faces)
        igl.write_obj(str(out / f"case_{k}_wall.obj"), V2, F2)
        t = time.perf_counter()
        T2 = build_teacher(V2, F2, cfg["cage"], cfg["d"], cfg["modes"], args.grid_size,
                           args.minratio, args.fm_dir)
        t_redo = time.perf_counter() - t
        best2, _, _, _, _ = calibrate(T2, V2, F2, args.calibration_hits, args.thresholds,
                                      args.seed + 1)
        thr2 = best2["threshold"]
        vt2, missed2 = locate_in_tets(vox[keep], T2["nodes"], T2["elements"])
        vt1_keep = vt1[keep]
        index_keep = vox_index[keep]
        print(f"  rebuilt wall: {len(F2)} triangles, redo homework {t_redo:.0f} s, "
              f"threshold {thr2:.3g} ({missed2:.2%} voxels not in a tet)", flush=True)

        # 4) Second hits: reuse vs redo, compared on the wall's remaining voxels.
        # Random hits land near the crater only about 30% of the time, so draw a large
        # random pool and take half the hits near the crater and half far from it.
        seed = args.seed + 10 + (0 if k == "control" else k + 1)
        pool_pts, pool_dirs = sample_impacts(V2, F2, args.hits * 20, seed)
        if crater_tree is None:
            pick_hits = np.arange(args.hits)
        else:
            pool_dist = crater_tree.query(pool_pts)[0]
            near_ids = np.nonzero(pool_dist < NEAR)[0][:args.hits // 2]
            far_ids = np.nonzero(pool_dist >= NEAR)[0][:args.hits - len(near_ids)]
            pick_hits = np.r_[near_ids, far_ids]
        pts2, dirs2 = pool_pts[pick_hits], pool_dirs[pick_hits]
        examples, case_rows = [], []
        for j, (pt, dv) in enumerate(zip(pts2, dirs2)):
            project_impact(T1["fm"], pt, dv, cfg["d"], thr1)
            reuse = split_disconnected(voxel_fragments(T1["fm"], vt1_keep), index_keep)
            project_impact(T2["fm"], pt, dv, cfg["d"], thr2)
            redo = split_disconnected(voxel_fragments(T2["fm"], vt2), index_keep)
            project_impact(T2["fm"], pt, dv, cfg["d"], thr1)
            redo_same_thr = split_disconnected(voxel_fragments(T2["fm"], vt2), index_keep)
            dist = float("nan") if crater_tree is None else float(crater_tree.query(pt)[0])
            case_rows.append({
                "case": k, "hit": j, "distance_to_crater": dist,
                "near": bool(crater_tree is not None and dist < NEAR),
                "ari_reuse_vs_redo": ari(reuse, redo),
                "ari_reuse_vs_redo_same_threshold": ari(reuse, redo_same_thr),
                "detached_iou": iou(detached(reuse), detached(redo)),
                "fragments_reuse": count_fragments(reuse, MIN_FRAGMENT),
                "fragments_redo": count_fragments(redo, MIN_FRAGMENT),
                "detached_share_reuse": float(detached(reuse).mean()),
                "detached_share_redo": float(detached(redo).mean()),
            })
            examples.append((dist, j, pt, reuse, redo))
        rows += case_rows
        cases.append({"case": k, "first_hit": None if i1 is None else int(i1),
                      "crater_share": crater_share, "triangles": int(len(F2)),
                      "redo_homework_s": t_redo, "threshold_intact": thr1,
                      "threshold_rebuilt": thr2})

        # Picture: two second hits (control: two random ones; craters: nearest and farthest).
        if k == "control":
            pick = [examples[0], examples[len(examples) // 2]]
        else:
            examples.sort(key=lambda e: e[0])
            pick = [examples[0], examples[-1]]
        fig, axs = plt.subplots(2, 2, figsize=(13, 8))
        xy = vox[keep][:, axes2d]
        xy_crater = vox[~keep][:, axes2d]
        show = np.random.default_rng(3).choice(len(xy), size=min(len(xy), 40_000), replace=False)
        palette = colormaps["tab10"]
        for r, (dist, j, pt, reuse, redo) in enumerate(pick):
            for c, (lab, name) in enumerate([(reuse, "REUSE intact homework"),
                                             (redo, "REDO homework on rebuilt wall")]):
                ax = axs[r, c]
                big = np.argmax(np.bincount(lab))
                order = {f: n for n, f in enumerate(np.unique(lab[lab != big]))}
                colours = np.array([(0.78, 0.78, 0.78, 1.0) if l == big
                                    else palette(order[l] % 10) for l in lab[show]])
                if len(xy_crater):
                    ax.scatter(*xy_crater.T, s=1, c=[(1.0, 0.85, 0.85, 1.0)], linewidths=0)
                    ax.plot(*cal_pts[i1][axes2d], marker="X", color="black", markersize=11)
                ax.scatter(*xy[show].T, s=1.5, c=colours, linewidths=0)
                ax.plot(*pt[axes2d], marker="*", color="red", markersize=15,
                        markeredgecolor="black")
                where = "" if k == "control" else f", {dist:.2f} from crater"
                ax.set_title(f"{name}\nsecond hit {j}{where}: {count_fragments(lab, MIN_FRAGMENT)} fragments, "
                             f"ARI {case_rows[j]['ari_reuse_vs_redo']:.2f}", fontsize=9)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
        what = ("control: intact wall, rebuilt" if k == "control"
                else f"case {k}: pink = crater from first hit (black X)")
        fig.suptitle(f"{setting}, {what}; red star = second hit", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"case_{k}.png", dpi=110)
        plt.close(fig)

    with open(out / "per_hit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def avg(key, subset):
        vals = [r[key] for r in subset]
        return float(np.mean(vals)) if vals else float("nan")

    groups = {"control": [r for r in rows if r["case"] == "control"],
              "near": [r for r in rows if r["case"] != "control" and r["near"]],
              "far": [r for r in rows if r["case"] != "control" and not r["near"]]}
    summary = {"setting": setting, "intact_homework_s": t_intact, "threshold_intact": thr1,
               "cases": cases}
    for name, subset in groups.items():
        summary[f"{name}_hits"] = len(subset)
        for key in ("ari_reuse_vs_redo", "ari_reuse_vs_redo_same_threshold", "detached_iou",
                    "fragments_reuse", "fragments_redo"):
            summary[f"{name}_{key}"] = avg(key, subset)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    s = summary
    damaged = [c for c in cases if c["case"] != "control"]
    craters = ", ".join("%.0f%%" % (100 * c["crater_share"]) for c in damaged)
    redo_ctrl = "%.0f" % cases[0]["redo_homework_s"]
    redo_dmg = ", ".join("%.0f" % c["redo_homework_s"] for c in damaged)
    print(f"\nSaved results in {out}\n\nPaste into docs/kill-test-log.md (Day 6c):\n")
    print(f"Setting: {setting}. Intact-wall homework {t_intact:.0f} s. Craters: {craters}.\n")
    print("| Second hits on | Hits | Redo homework (s) | ARI reuse vs redo | Same threshold "
          "| Detached IoU | Fragments reuse / redo |")
    print("|---|---|---|---|---|---|---|")
    for name, label, redo in (("control", "Control: intact wall, rebuilt", redo_ctrl),
                              ("near", f"Damaged wall, near crater (< {NEAR})", redo_dmg),
                              ("far", "Damaged wall, far from crater", redo_dmg)):
        print(f"| {label} | {s[name + '_hits']} | {redo} | {s[name + '_ari_reuse_vs_redo']:.2f} | "
              f"{s[name + '_ari_reuse_vs_redo_same_threshold']:.2f} | "
              f"{s[name + '_detached_iou']:.2f} | "
              f"{s[name + '_fragments_reuse']:.1f} / {s[name + '_fragments_redo']:.1f} |")
    print("\nARI: 1.00 = reuse gives exactly the right break; near 0 = unrelated to the right break.")


if __name__ == "__main__":
    main()
