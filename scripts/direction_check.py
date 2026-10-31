"""Day 6b: does the impact DIRECTION change how the shape breaks?

We chose teacher option C (40 modes, d=3) partly because it should respond to
impact direction. Reading fracture-modes' code shows that its default impact
projection treats every axis identically, so a unit-length direction cannot change
which pieces break. The paper says the same (Sec. 6, Fig. 21). fracture-modes has
a second projection, project_on_modes, where direction can matter. This script
measures both:

  1. builds the teacher with the chosen setting (default: outputs/day4b/chosen.json),
  2. hits the shape at N points, each from 3 directions: straight in (0 degrees),
     tilted 45 degrees, and grazing (80 degrees),
  3. for each projection, picks its own best threshold from the straight-in hits
     (same rule as Day 4), then compares every tilted hit with the straight-in hit
     at the same point:
       changed       share of points where the break is different at all
       detached IoU  overlap of the broken-off volume (1.0 = identical, 0 = none shared)
  4. reports variety and locality for each projection, as on Day 4,
  5. checks our shortcut against fracture-modes itself on every tilted hit,
  6. draws 2 points x 3 directions for each projection.

Usage (Miniforge Prompt, rift-teacher active, repo root):
    python scripts/direction_check.py

Outputs in outputs/day6b/ (gitignored): summary.json and directions_<projection>.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # find teacher.py next to this file
from teacher import (DEFAULT_FM_DIR, build_teacher, decide, fine_triangle_labels, load_shape,
                     piece_distances, project_impact, render_break, sample_impacts,
                     tet_volumes_and_centroids)
from sweep_threshold import outcome_stats

TILTS = [45, 80]  # degrees away from straight in
PROJECTIONS = {"default": False, "project_on_modes": True}


def tilted(straight_in, tangent, degrees):
    """Rotate the straight-in direction by `degrees` toward a direction along the surface."""
    a = np.radians(degrees)
    v = np.cos(a) * straight_in + np.sin(a) * tangent
    return v / np.linalg.norm(v)


def random_tangents(normals, rng):
    """One random direction along the surface (perpendicular to the normal) per hit."""
    r = rng.standard_normal(normals.shape)
    t = r - np.sum(r * normals, axis=1, keepdims=True) * normals
    return t / np.linalg.norm(t, axis=1, keepdims=True)


def detached(labels, piece_vol):
    """Pieces that broke off: everything outside the largest fragment."""
    frag_vol = np.bincount(labels, weights=piece_vol)
    return labels != np.argmax(frag_vol)


def detached_iou(a, b, piece_vol):
    """Overlap of two broken-off regions, measured by volume (1.0 when both are empty)."""
    union = piece_vol[a | b].sum()
    return 1.0 if union == 0 else float(piece_vol[a & b].sum() / union)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser(description="Measure whether impact direction changes the break.")
    p.add_argument("--chosen", type=Path, default=Path("outputs/day4b/chosen.json"))
    p.add_argument("--impacts", type=int, default=50)
    p.add_argument("--thresholds", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/day6b"))
    p.add_argument("--fm-dir", type=Path, default=DEFAULT_FM_DIR)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(args.chosen.read_text())
    if cfg["d"] != 3:
        raise SystemExit("This setting uses d=1, which ignores direction by design.")
    setting = f"cage{cfg['cage']}_m{cfg['modes']}_d{cfg['d']}"
    print(f"Teacher setting: {setting} (from {args.chosen})", flush=True)

    v_fine, f_fine = load_shape(cfg["mesh"])
    T = build_teacher(v_fine, f_fine, cfg["cage"], cfg["d"], cfg["modes"],
                      cfg.get("grid_size", 256), cfg.get("minratio", 1.5), args.fm_dir)
    fm = T["fm"]
    print(f"Teacher built in {sum(T['timings'].values()):.0f} s: up to "
          f"{fm.precomputed_num_pieces} pieces", flush=True)

    tet_vol, tet_cent = tet_volumes_and_centroids(T["nodes"], T["elements"])
    p2t = fm.piece_to_tet_matrix
    piece_vol = np.asarray(p2t.T @ tet_vol).ravel()
    piece_cent = np.asarray(p2t.T @ (tet_vol[:, None] * tet_cent)) / piece_vol[:, None]

    points, straight = sample_impacts(v_fine, f_fine, args.impacts, args.seed)
    tangents = random_tangents(straight, np.random.default_rng(args.seed + 7))

    summary = {"setting": setting, "hits": args.impacts, "projections": {}}
    for name, on_modes in PROJECTIONS.items():
        print(f"Projection: {name} ...", flush=True)
        # 1) Straight-in hits: distances, then this projection's own best threshold.
        dist0 = []
        for pt, dvec in zip(points, straight):
            project_impact(fm, pt, dvec, 3, 1.0, on_modes)
            dist0.append(piece_distances(fm))
        pooled = np.concatenate(dist0)
        pooled = pooled[pooled > 0]
        grid = np.geomspace(pooled.min() * 0.5, pooled.max() * 2.0, args.thresholds)
        rows = [outcome_stats(fm, dist0, t, points, piece_vol, piece_cent, tet_vol, tet_cent)
                for t in grid]
        best = max(rows, key=lambda s: (s["distinct_outcomes"], s["frac_partial"],
                                        -np.nan_to_num(s["locality_ratio"], nan=9.0)))
        thr = best["threshold"]
        base = [decide(fm, d0, thr)[1] for d0 in dist0]

        # 2) Tilted hits at the same points, compared with straight in.
        result = {"threshold": thr, "different_outcomes": best["distinct_outcomes"],
                  "frac_partial": best["frac_partial"], "locality": best["locality_ratio"],
                  "mean_fragments": best["mean_fragments"], "mismatches": 0}
        for deg in TILTS:
            changed, ious = [], []
            for i, (pt, dvec) in enumerate(zip(points, straight)):
                direction = tilted(dvec, tangents[i], deg)
                project_impact(fm, pt, direction, 3, thr, on_modes)
                _, lab = decide(fm, piece_distances(fm), thr)
                if not np.array_equal(fm.piece_labels_after_impact, lab):
                    result["mismatches"] += 1
                changed.append(not np.array_equal(lab, base[i]))
                ious.append(detached_iou(detached(lab, piece_vol), detached(base[i], piece_vol),
                                         piece_vol))
            result[f"changed_{deg}"] = float(np.mean(changed))
            result[f"detached_iou_{deg}"] = float(np.mean(ious))
        summary["projections"][name] = result
        print(f"  threshold {thr:.3g}: {best['distinct_outcomes']} different outcomes, "
              f"tilt 45 changes {result['changed_45']:.0%} of breaks, "
              f"tilt 80 changes {result['changed_80']:.0%}", flush=True)

        # 3) Pictures: 2 points with a partial straight-in break, each from 3 directions.
        partial = [i for i, lab in enumerate(base)
                   if 1 < lab.max() + 1 < fm.precomputed_num_pieces][:2] or [0, 1]
        fig = plt.figure(figsize=(15, 9))
        slot = 0
        for i in partial:
            for deg in [0] + TILTS:
                direction = tilted(straight[i], tangents[i], deg)
                project_impact(fm, points[i], direction, 3, thr, on_modes)
                slot += 1
                ax = fig.add_subplot(len(partial), 3, slot, projection="3d")
                render_break(ax, fm.fine_vertices, fm.fine_triangles, fine_triangle_labels(fm),
                             points[i], direction,
                             title=f"hit {i}, {deg} deg: {fm.n_pieces_after_impact} fragments")
        fig.suptitle(f"{setting}, {name} projection, threshold {thr:.3g}: "
                     "same point, three directions", fontsize=11)
        fig.tight_layout()
        fig.savefig(args.out_dir / f"directions_{name}.png", dpi=110)
        plt.close(fig)

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved results in {args.out_dir}\n\nPaste into docs/kill-test-log.md (Day 6b):\n")
    print("| Projection | Threshold | Different outcomes (of %d hits) | Partial | Locality "
          "| 45 deg: changed / detached IoU | 80 deg: changed / detached IoU | Mismatches |"
          % args.impacts)
    print("|---|---|---|---|---|---|---|---|")
    for name, r in summary["projections"].items():
        print(f"| {name} | {r['threshold']:.3g} | {r['different_outcomes']} | "
              f"{r['frac_partial']:.0%} | {r['locality']:.2f} | "
              f"{r['changed_45']:.0%} / {r['detached_iou_45']:.2f} | "
              f"{r['changed_80']:.0%} / {r['detached_iou_80']:.2f} | {r['mismatches']} |")


if __name__ == "__main__":
    main()
