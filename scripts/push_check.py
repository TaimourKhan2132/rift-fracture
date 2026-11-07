"""Day 6f: does a fairer hit make the teacher less dependent on the mesh?

Day 6c found that fracture-modes pushes every tet corner within 0.05 of the hit by
the same amount, so a denser mesh near the hit means a stronger hit. Its own code
comments write the push weighted by the mass matrix M
(piece_impact = wave_piece_lsqr' M d), but that line is commented out. This script
compares the two push rules on the same teacher builds:

  corners  fracture-modes as its code runs: every corner within 0.05 pushed equally
  mass     each corner's push weighted by the material around it (teacher.PUSH_RULES)

on three versions of the Day 3 wall, as in Day 6d: the original (56 triangles), the
exact same shape cut into 14,336 triangles, and the rounded rebuild (16,000 triangles).
Each version is built once. Both push rules then hit it at the same 40 points, each
rule with its own threshold for 5% median damage (decision D3). If the mass rule
agrees across versions better than the corners rule, our labels carry less mesh noise.

It also measures how much push each rule delivers at the same spot on each version,
relative to the original: the number of corners within 0.05 (corners rule) and their
total mass (mass rule). A fair rule delivers the same push on every version.

Usage (Miniforge Prompt, rift-teacher active, repo root):
    python scripts/push_check.py              (option B: about 8 minutes)

Outputs in outputs/day6f_<preset>/ (gitignored): summary.json, per_hit.csv,
log_table.md and hits.png.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # find teacher.py next to this file
from teacher import (DEFAULT_FM_DIR, PUSH_RULES, build_teacher, hit_distances, load_shape,
                     median_damage, piece_volumes, sample_impacts, target_damage_threshold,
                     tet_pieces)
from cell_graph import locate_in_tets, voxelize
from second_hit import (MIN_FRAGMENT, PRESETS, ari, count_fragments, damaged_mesh, detached,
                        iou)
from repeat_check import voxel_labels
import ui

RADIUS = 0.05  # fracture-modes' hit radius (in the unit box the shape is scaled to)


def push_sizes(fm, point):
    """How much push each rule delivers at this point: (corners within RADIUS, their mass)."""
    near = np.linalg.norm(fm.vertices - point, axis=1) < RADIUS
    mass = np.asarray(fm.M.sum(axis=0)).ravel()  # each corner's share of the material
    return int(near.sum()), float(mass[near].sum())


def main():
    import igl
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    p = argparse.ArgumentParser(description="Compare fracture-modes' push with a mass-weighted push.")
    p.add_argument("--preset", choices=sorted(PRESETS), default="B")
    p.add_argument("--mesh", type=Path, default=Path("outputs/day3/wall.obj"))
    p.add_argument("--target-damage", type=float, default=0.05)
    p.add_argument("--hits", type=int, default=40)
    p.add_argument("--calibration-hits", type=int, default=60)
    p.add_argument("--fine-subdivisions", type=int, default=4)
    p.add_argument("--mesh-faces", type=int, default=16000)
    p.add_argument("--grid-size", type=int, default=256)
    p.add_argument("--minratio", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--fm-dir", type=Path, default=DEFAULT_FM_DIR)
    args = p.parse_args()
    cfg = PRESETS[args.preset]
    out = args.out_dir or Path(f"outputs/day6f_{args.preset}")
    out.mkdir(parents=True, exist_ok=True)
    setting = f"option {args.preset}: cage{cfg['cage']}_m{cfg['modes']}_d{cfg['d']}"
    with ui.Run(f"Day 6f: corners push vs mass push ({setting})", out, steps=1 + 3 * 2 + 1) as run:
        check(run, args, cfg, out, setting, igl, plt, colormaps)


def check(run, args, cfg, out, setting, igl, plt, colormaps):
    with run.step("Prepare three versions of the same wall"):
        v0, f0 = load_shape(args.mesh)
        vox, vox_index, spacing = voxelize(v0, f0)
        lo = v0.min(axis=0)
        vf, ff = v0, f0
        for _ in range(args.fine_subdivisions):  # midpoint split: the shape does not change
            vf, ff = igl.upsample(vf, ff, number_of_subdivs=1)
        vr, fr = damaged_mesh(vox_index, np.ones(len(vox), dtype=bool), lo, spacing,
                              target_faces=args.mesh_faces)
    versions = [("original", v0, f0), ("finer triangles", vf, ff), ("rounded rebuild", vr, fr)]

    # Shared hits, sampled once on the original surface and used on every version.
    cal_pts, cal_dirs = sample_impacts(v0, f0, args.calibration_hits, args.seed + 100)
    hit_pts, hit_dirs = sample_impacts(v0, f0, args.hits, args.seed + 200)
    show_hits = [0, 1]

    results, rows, pictures, ref = [], [], {}, {}
    for name, V, F in versions:
        with run.step(f"{name} ({len(F):,} triangles): teacher homework"):
            T = build_teacher(V, F, cfg["cage"], cfg["d"], cfg["modes"], args.grid_size,
                              args.minratio, args.fm_dir, display_mesh=False)
        fm = T["fm"]
        with run.step(f"{name}: both push rules, {args.hits} shared hits each"):
            pieces = tet_pieces(fm)
            piece_vol = piece_volumes(T)
            vt, missed = locate_in_tets(vox, T["nodes"], T["elements"])
            sizes = [push_sizes(fm, pt) for pt in hit_pts]
            res = {"version": name, "triangles": int(len(F)), "tets": int(len(T["elements"])),
                   "pieces": int(fm.precomputed_num_pieces),
                   "homework_s": float(sum(T["timings"].values())),
                   "voxels_not_in_a_tet": float(missed)}
            labels = {}
            for push in PUSH_RULES:
                cal_d = [hit_distances(fm, pt, dv, cfg["d"], push) for pt, dv in zip(cal_pts, cal_dirs)]
                thr, med = target_damage_threshold(fm, cal_d, piece_vol, args.target_damage)
                labels[push] = [voxel_labels(fm, hit_distances(fm, pt, dv, cfg["d"], push), thr,
                                             pieces, vt, vox_index).astype(np.int32)
                                for pt, dv in run.track(list(zip(hit_pts, hit_dirs)), f"{push} push")]
                res[f"threshold_{push}"] = thr
                res[f"median_damage_{push}"] = med
        if not ref:
            ref = {"labels": labels, "sizes": sizes}
        for j in range(args.hits):
            row = {"version": name, "hit": j,
                   "corners_in_radius": sizes[j][0], "mass_in_radius": sizes[j][1],
                   "corners_vs_original": sizes[j][0] / max(ref["sizes"][j][0], 1),
                   "mass_vs_original": sizes[j][1] / max(ref["sizes"][j][1], 1e-12)}
            for push in PUSH_RULES:
                a, b = ref["labels"][push][j], labels[push][j]
                row[f"ari_{push}"] = ari(a, b)
                row[f"detached_iou_{push}"] = iou(detached(a), detached(b))
                row[f"detached_share_{push}"] = float(detached(b).mean())
                row[f"fragments_{push}"] = count_fragments(b, MIN_FRAGMENT)
            rows.append(row)
        mine = [r for r in rows if r["version"] == name]
        for key in ("corners_vs_original", "mass_vs_original"):
            res[f"median_{key}"] = float(np.median([r[key] for r in mine]))
        for push in PUSH_RULES:
            res[f"ari_vs_original_{push}"] = float(np.mean([r[f"ari_{push}"] for r in mine]))
            res[f"detached_iou_vs_original_{push}"] = float(np.mean([r[f"detached_iou_{push}"] for r in mine]))
        results.append(res)
        pictures[name] = {push: [labels[push][j] for j in show_hits] for push in PUSH_RULES}
        run.note(f"{name}: push vs original {res['median_corners_vs_original']:.2f} (corners) / "
                 f"{res['median_mass_vs_original']:.2f} (mass); ARI vs original "
                 f"{res['ari_vs_original_corners']:.2f} / {res['ari_vs_original_mass']:.2f}")
        del T, fm

    with run.step("Save results and picture"):
        with open(out / "per_hit.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        (out / "summary.json").write_text(json.dumps(
            {"setting": setting, "target_damage": args.target_damage, "hits": args.hits,
             "radius": RADIUS, "versions": results}, indent=2))

        axes2d = np.sort(np.argsort(-(v0.max(axis=0) - v0.min(axis=0)))[:2])
        xy = vox[:, axes2d]
        show = np.random.default_rng(3).choice(len(xy), size=min(len(xy), 40_000), replace=False)
        palette = colormaps["tab10"]
        panels = [(j, push) for j in show_hits for push in PUSH_RULES]
        fig, axs = plt.subplots(len(panels), len(versions), figsize=(14, 3.6 * len(panels)))
        for r, (j, push) in enumerate(panels):
            for c, (name, _, _) in enumerate(versions):
                lab = pictures[name][push][show_hits.index(j)]
                big = np.argmax(np.bincount(lab))
                order = {f: n for n, f in enumerate(np.unique(lab[lab != big]))}
                colours = np.array([(0.78, 0.78, 0.78, 1.0) if l == big else palette(order[l] % 10)
                                    for l in lab[show]])
                ax = axs[r, c]
                ax.scatter(*xy[show].T, s=1.5, c=colours, linewidths=0)
                ax.plot(*hit_pts[j][axes2d], marker="*", color="red", markersize=14,
                        markeredgecolor="black")
                row = next(x for x in rows if x["version"] == name and x["hit"] == j)
                ax.set_title(f"{push} push, {name}, hit {j}\n"
                             f"{row[f'detached_share_{push}']:.1%} broken off, "
                             f"ARI vs original {row[f'ari_{push}']:.2f}", fontsize=9)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle(f"{setting}: same hit, same teacher builds, two push rules "
                     f"(threshold set for {args.target_damage:.0%} median damage)", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "hits.png", dpi=90)
        plt.close(fig)

    intro = (f"Setting: {setting}. {args.hits} shared hits; each push rule has its own threshold "
             f"for {args.target_damage:.0%} median damage. Push vs original: median ratio at the "
             f"same spot (corners rule: corners within {RADIUS}; mass rule: their mass).")
    columns = ["Version", "Triangles", "Tets", "Homework (s)", "Push vs original: corners / mass",
               "Median damage: corners / mass", "ARI vs original: corners / mass",
               "Detached IoU vs original: corners / mass"]
    table = []
    for res in results:
        first = res["version"] == "original"
        table.append([
            res["version"], f"{res['triangles']:,}", f"{res['tets']:,}", f"{res['homework_s']:.0f}",
            f"{res['median_corners_vs_original']:.2f} / {res['median_mass_vs_original']:.2f}",
            f"{res['median_damage_corners']:.1%} / {res['median_damage_mass']:.1%}",
            "reference" if first else f"{res['ari_vs_original_corners']:.2f} / {res['ari_vs_original_mass']:.2f}",
            "reference" if first else (f"{res['detached_iou_vs_original_corners']:.2f} / "
                                       f"{res['detached_iou_vs_original_mass']:.2f}")])
    run.note(intro)
    run.table("Day 6f: corners push vs mass push (ARI 1.00 = identical breaks)", columns, table)
    run.log_markdown("\n".join([intro, "", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
                               + ["| " + " | ".join(r) + " |" for r in table]))
    run.note("Read it: if the mass rule's ARI vs original is clearly higher than the corners "
             "rule's, a fairer hit removes part of the mesh noise, and the dataset should use it.")


if __name__ == "__main__":
    main()
