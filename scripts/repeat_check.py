"""Day 6d: does the teacher's break depend on the mesh, and which threshold rule
means the same strength on every mesh?

Day 6c found two problems. Rebuilding the wall (same shape, different triangles)
changed the teacher's break (ARI 0.63), and the Day 4 threshold rule picked very
different strengths on near-identical walls (thresholds 64 to 210). This script
separates the causes by building the teacher on four versions of the same wall:

  original         the Day 3 wall file (56 triangles)
  original again   the same file a second time: is the teacher repeatable?
  finer triangles  exactly the same shape, each triangle split into 256 smaller ones
                   (14,336 triangles): does the triangulation alone matter?
  rounded rebuild  the Day 6c pipeline (voxels -> smoothed surface, 16,000 triangles):
                   triangulation plus slightly rounded edges and about 1% less volume

Each version gets two thresholds:
  most outcomes    the Day 4 rule: the threshold giving the most different outcomes
  target damage    the threshold at which the median hit breaks off a fixed share of
                   the volume (default 5%): a candidate rule for "same strength"

The same 40 hit points then hit every version. Each version's breaks are compared
with the original's on one shared voxel grid (ARI and detached-volume IoU, as in 6c).

Usage (Miniforge Prompt, rift-teacher active, repo root):
    python scripts/repeat_check.py              (option B: about 15 minutes)
    python scripts/repeat_check.py --preset C   (option C: about an hour)

Outputs in outputs/day6d_<preset>/ (gitignored): summary.json, per_hit.csv,
hits.png, and the two new wall meshes.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # find teacher.py next to this file
from teacher import (DEFAULT_FM_DIR, build_teacher, decide, hit_distances, load_shape,
                     median_damage, piece_volumes, project_impact, sample_impacts,
                     target_damage_threshold)
from cell_graph import locate_in_tets, voxelize
from second_hit import (MIN_FRAGMENT, PRESETS, ari, calibrate, count_fragments, damaged_mesh,
                        detached, iou, split_disconnected)


# ---------------------------------------------------------------- helpers


def voxel_labels(fm, dists, threshold, tet_piece, voxel_tet, vox_index):
    """Fragment of each voxel, using the same break rule as fracture-modes (Day 4 check)."""
    _, piece_lab = decide(fm, dists, threshold)
    return split_disconnected(piece_lab[tet_piece[voxel_tet]], vox_index)


# ---------------------------------------------------------------- main


def main():
    import igl
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    p = argparse.ArgumentParser(description="Check how much the teacher depends on the mesh.")
    p.add_argument("--preset", choices=sorted(PRESETS), default="B")
    p.add_argument("--mesh", type=Path, default=Path("outputs/day3/wall.obj"))
    p.add_argument("--target-damage", type=float, default=0.05,
                   help="share of the volume the median hit breaks off (target-damage rule)")
    p.add_argument("--hits", type=int, default=40, help="shared hits compared across versions")
    p.add_argument("--calibration-hits", type=int, default=60)
    p.add_argument("--thresholds", type=int, default=40, help="grid size for the Day 4 rule")
    p.add_argument("--fine-subdivisions", type=int, default=4,
                   help="each subdivision splits every triangle into 4")
    p.add_argument("--mesh-faces", type=int, default=16000)
    p.add_argument("--grid-size", type=int, default=256)
    p.add_argument("--minratio", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--fm-dir", type=Path, default=DEFAULT_FM_DIR)
    args = p.parse_args()
    cfg = PRESETS[args.preset]
    out = args.out_dir or Path(f"outputs/day6d_{args.preset}")
    out.mkdir(parents=True, exist_ok=True)
    setting = f"option {args.preset}: cage{cfg['cage']}_m{cfg['modes']}_d{cfg['d']}"
    print(f"Teacher setting: {setting}", flush=True)

    # Four versions of the same wall.
    v0, f0 = load_shape(args.mesh)
    vox, vox_index, spacing = voxelize(v0, f0)
    lo = v0.min(axis=0)
    vf, ff = v0, f0
    for _ in range(args.fine_subdivisions):  # midpoint split: the shape does not change
        vf, ff = igl.upsample(vf, ff, number_of_subdivs=1)
    vr, fr = damaged_mesh(vox_index, np.ones(len(vox), dtype=bool), lo, spacing,
                          target_faces=args.mesh_faces)
    igl.write_obj(str(out / "finer_triangles.obj"), vf, ff)
    igl.write_obj(str(out / "rounded_rebuild.obj"), vr, fr)
    versions = [("original", v0, f0), ("original again", v0.copy(), f0.copy()),
                ("finer triangles", vf, ff), ("rounded rebuild", vr, fr)]

    # Shared hits, sampled once on the original surface and used on every version.
    cal_pts, cal_dirs = sample_impacts(v0, f0, args.calibration_hits, args.seed + 100)
    hit_pts, hit_dirs = sample_impacts(v0, f0, args.hits, args.seed + 200)
    show_hits = [0, 1]  # drawn in hits.png

    results, rows, pictures = [], [], {}
    ref = None  # the original's labels, kept to compare the other versions against
    for name, V, F in versions:
        print(f"{name}: {len(F)} triangles", flush=True)
        T = build_teacher(V, F, cfg["cage"], cfg["d"], cfg["modes"], args.grid_size,
                          args.minratio, args.fm_dir)
        fm = T["fm"]
        tet_piece = np.asarray(fm.piece_to_tet_matrix.argmax(axis=1)).ravel()
        piece_vol = piece_volumes(T)
        vt, missed = locate_in_tets(vox, T["nodes"], T["elements"])

        # Rule A: the Day 4 rule, exactly as Day 6c used it (points on this version's surface).
        best_a, _, _, _, _ = calibrate(T, V, F, args.calibration_hits, args.thresholds, args.seed)
        thr_a = float(best_a["threshold"])
        # Rule B: target damage, on the shared calibration hits.
        cal_d = [hit_distances(fm, pt, dv, cfg["d"]) for pt, dv in zip(cal_pts, cal_dirs)]
        thr_b, med_b = target_damage_threshold(fm, cal_d, piece_vol, args.target_damage)
        med_a = median_damage(fm, cal_d, piece_vol, thr_a)

        hit_d = [hit_distances(fm, pt, dv, cfg["d"]) for pt, dv in zip(hit_pts, hit_dirs)]
        lab_a = [voxel_labels(fm, d, thr_a, tet_piece, vt, vox_index).astype(np.int32)
                 for d in hit_d]
        lab_b = [voxel_labels(fm, d, thr_b, tet_piece, vt, vox_index).astype(np.int32)
                 for d in hit_d]

        # Our break rule must match fracture-modes' own result (Day 4 check, repeated here).
        project_impact(fm, hit_pts[0], hit_dirs[0], cfg["d"], thr_b)
        own = split_disconnected(np.asarray(fm.tet_labels_after_impact).astype(int)[vt], vox_index)
        rule_check = ari(own, lab_b[0])

        if ref is None:
            ref = {"a": lab_a, "b": lab_b}
        for j in range(args.hits):
            rows.append({
                "version": name, "hit": j,
                "ari_most_outcomes": ari(ref["a"][j], lab_a[j]),
                "ari_target_damage": ari(ref["b"][j], lab_b[j]),
                "detached_iou_target_damage": iou(detached(ref["b"][j]), detached(lab_b[j])),
                "detached_share_most_outcomes": float(detached(lab_a[j]).mean()),
                "detached_share_target_damage": float(detached(lab_b[j]).mean()),
                "fragments_target_damage": count_fragments(lab_b[j], MIN_FRAGMENT),
            })
        mine = [r for r in rows if r["version"] == name]
        res = {
            "version": name, "triangles": int(len(F)), "tets": int(len(T["elements"])),
            "pieces": int(fm.precomputed_num_pieces),
            "timings_s": {k: float(v) for k, v in T["timings"].items()},
            "homework_s": float(sum(T["timings"].values())),
            "voxels_not_in_a_tet": float(missed),
            "threshold_most_outcomes": thr_a, "median_damage_most_outcomes": med_a,
            "threshold_target_damage": thr_b, "median_damage_target_damage": med_b,
            "ari_vs_original_most_outcomes": float(np.mean([r["ari_most_outcomes"] for r in mine])),
            "ari_vs_original_target_damage": float(np.mean([r["ari_target_damage"] for r in mine])),
            "detached_iou_vs_original": float(np.mean([r["detached_iou_target_damage"] for r in mine])),
            "fragments_target_damage": float(np.mean([r["fragments_target_damage"] for r in mine])),
            "rule_check_ari": float(rule_check),
        }
        results.append(res)
        pictures[name] = [lab_b[j] for j in show_hits]
        t = res["timings_s"]
        print(f"  {res['tets']} tets, {res['pieces']} pieces; homework {res['homework_s']:.0f} s "
              f"(cage {t['cage_s']:.0f}, tets {t['tetgen_s']:.0f}, modes {t['modes_s']:.0f}, "
              f"precompute {t['precompute_s']:.0f}); most-outcomes threshold {thr_a:.3g} "
              f"(median damage {med_a:.1%}); target-damage threshold {thr_b:.3g} "
              f"(median {med_b:.1%}); ARI vs original {res['ari_vs_original_target_damage']:.2f}; "
              f"rule check {rule_check:.2f}", flush=True)
        del T, fm

    with open(out / "per_hit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {"setting": setting, "target_damage": args.target_damage, "hits": args.hits,
               "versions": results}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # Picture: two shared hits (rows) on each version (columns), at the target-damage threshold.
    axes2d = np.sort(np.argsort(-(v0.max(axis=0) - v0.min(axis=0)))[:2])
    xy = vox[:, axes2d]
    show = np.random.default_rng(3).choice(len(xy), size=min(len(xy), 40_000), replace=False)
    palette = colormaps["tab10"]
    fig, axs = plt.subplots(len(show_hits), len(versions), figsize=(18, 8.5))
    for r, j in enumerate(show_hits):
        for c, (name, _, _) in enumerate(versions):
            lab = pictures[name][r]
            big = np.argmax(np.bincount(lab))
            order = {f: n for n, f in enumerate(np.unique(lab[lab != big]))}
            colours = np.array([(0.78, 0.78, 0.78, 1.0) if l == big else palette(order[l] % 10)
                                for l in lab[show]])
            ax = axs[r, c]
            ax.scatter(*xy[show].T, s=1.5, c=colours, linewidths=0)
            ax.plot(*hit_pts[j][axes2d], marker="*", color="red", markersize=15,
                    markeredgecolor="black")
            row = next(x for x in rows if x["version"] == name and x["hit"] == j)
            ax.set_title(f"{name}, hit {j}\n{row['fragments_target_damage']} fragments, "
                         f"{row['detached_share_target_damage']:.1%} broken off, "
                         f"ARI vs original {row['ari_target_damage']:.2f}", fontsize=9)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"{setting}: same hit on four versions of the same wall, threshold set for "
                 f"{args.target_damage:.0%} median damage", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "hits.png", dpi=100)
    plt.close(fig)

    print(f"\nSaved results in {out}\n\nPaste into docs/kill-test-log.md (Day 6d):\n")
    print(f"Setting: {setting}. {args.hits} shared hits; target damage {args.target_damage:.0%} "
          "(median hit).\n")
    print("| Version | Triangles | Tets | Homework (s): cage / tets / modes / precompute "
          "| Most-outcomes rule: threshold (median damage) "
          "| Target-damage rule: threshold (median damage) "
          "| ARI vs original: most outcomes / target damage | Detached IoU vs original |")
    print("|---|---|---|---|---|---|---|---|")
    for res in results:
        t = res["timings_s"]
        first = res["version"] == "original"
        ari_cell = "reference" if first else (f"{res['ari_vs_original_most_outcomes']:.2f} / "
                                              f"{res['ari_vs_original_target_damage']:.2f}")
        iou_cell = "reference" if first else f"{res['detached_iou_vs_original']:.2f}"
        print(f"| {res['version']} | {res['triangles']} | {res['tets']} | "
              f"{res['homework_s']:.0f}: {t['cage_s']:.0f} / {t['tetgen_s']:.0f} / "
              f"{t['modes_s']:.0f} / {t['precompute_s']:.0f} | "
              f"{res['threshold_most_outcomes']:.3g} ({res['median_damage_most_outcomes']:.1%}) | "
              f"{res['threshold_target_damage']:.3g} ({res['median_damage_target_damage']:.1%}) | "
              f"{ari_cell} | {iou_cell} |")
    worst = min(r["rule_check_ari"] for r in results)
    print(f"\nRule check (our break rule vs fracture-modes' own, should be 1.00): {worst:.2f}")
    print("ARI: 1.00 = identical breaks. 'original again' below 1.00 means the teacher is "
          "not repeatable on the same file.")


if __name__ == "__main__":
    main()
