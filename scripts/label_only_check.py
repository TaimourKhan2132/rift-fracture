"""Day 6e: do labels stay the same without the display-mesh step, and how much time
does skipping it save?

fracture-modes' precompute step does two jobs. First it prepares the break decision.
Then it cuts the fine (display) mesh into one mesh per piece, so fragments can be
drawn. Reading fracture_modes.py (impact_precomputation, impact_projection), the
break decision never uses those cut meshes, and on a 14,336-triangle wall the
cutting took 166 of 251 s (Day 6d). This script checks the claim instead of trusting it:

  1. build the teacher on the same wall twice: with the display mesh, and without,
  2. set one threshold (target damage, 5%) and hit both at the same 40 points,
  3. every tet's fragment label must be identical, hit for hit.

The default wall is the Day 3 wall cut into 14,336 triangles (the same shape),
because that is where the time goes.

Usage (Miniforge Prompt, rift-teacher active, repo root):
    python scripts/label_only_check.py              (option B: about 6 minutes)

Outputs in outputs/day6e_<preset>/ (gitignored): summary.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # find teacher.py next to this file
from teacher import (DEFAULT_FM_DIR, build_teacher, hit_distances, load_shape, piece_volumes,
                     project_impact, sample_impacts, target_damage_threshold)
from second_hit import PRESETS


def main():
    import igl

    p = argparse.ArgumentParser(description="Check that labels do not need the display mesh.")
    p.add_argument("--preset", choices=sorted(PRESETS), default="B")
    p.add_argument("--mesh", type=Path, default=Path("outputs/day3/wall.obj"))
    p.add_argument("--subdivisions", type=int, default=4,
                   help="split every triangle into 4 this many times (0 = use the file as is)")
    p.add_argument("--target-damage", type=float, default=0.05)
    p.add_argument("--hits", type=int, default=40)
    p.add_argument("--calibration-hits", type=int, default=60)
    p.add_argument("--grid-size", type=int, default=256)
    p.add_argument("--minratio", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--fm-dir", type=Path, default=DEFAULT_FM_DIR)
    args = p.parse_args()
    cfg = PRESETS[args.preset]
    out = args.out_dir or Path(f"outputs/day6e_{args.preset}")
    out.mkdir(parents=True, exist_ok=True)
    setting = f"option {args.preset}: cage{cfg['cage']}_m{cfg['modes']}_d{cfg['d']}"

    v, f = load_shape(args.mesh)
    for _ in range(args.subdivisions):  # midpoint split: the shape does not change
        v, f = igl.upsample(v, f, number_of_subdivs=1)
    print(f"Teacher setting: {setting}; wall with {len(f)} triangles", flush=True)

    builds = {}
    for name, display in (("with display mesh", True), ("labels only", False)):
        print(f"Building: {name}", flush=True)
        builds[name] = build_teacher(v, f, cfg["cage"], cfg["d"], cfg["modes"], args.grid_size,
                                     args.minratio, args.fm_dir, display_mesh=display)
    full, lab = builds["with display mesh"], builds["labels only"]
    same_mesh = (np.array_equal(full["elements"], lab["elements"])
                 and full["fm"].precomputed_num_pieces == lab["fm"].precomputed_num_pieces)

    # One threshold for both, set on the full build; then the same hits on both.
    cal_pts, cal_dirs = sample_impacts(v, f, args.calibration_hits, args.seed)
    cal_d = [hit_distances(full["fm"], pt, dv, cfg["d"]) for pt, dv in zip(cal_pts, cal_dirs)]
    threshold, med = target_damage_threshold(full["fm"], cal_d, piece_volumes(full),
                                             args.target_damage)
    hit_pts, hit_dirs = sample_impacts(v, f, args.hits, args.seed + 1)
    identical, worst_tets = 0, 0
    for pt, dv in zip(hit_pts, hit_dirs):
        labels = []
        for T in (full, lab):
            project_impact(T["fm"], pt, dv, cfg["d"], threshold)
            labels.append(np.asarray(T["fm"].tet_labels_after_impact).astype(int).copy())
        differ = int(np.sum(labels[0] != labels[1]))
        identical += differ == 0
        worst_tets = max(worst_tets, differ)

    summary = {"setting": setting, "triangles": int(len(f)), "threshold": threshold,
               "median_damage": med, "hits": args.hits, "hits_identical": int(identical),
               "most_tets_differing": worst_tets, "same_tets_and_pieces": bool(same_mesh),
               "timings_s": {name: {k: float(x) for k, x in T["timings"].items()}
                             for name, T in builds.items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nSaved results in {out}\n\nPaste into docs/kill-test-log.md (Day 6e):\n")
    print(f"Setting: {setting}. Wall: {len(f)} triangles. Threshold {threshold:.3g} "
          f"(median damage {med:.1%}). Same tets and pieces in both builds: "
          f"{'yes' if same_mesh else 'NO'}.\n")
    print("| Build | Homework (s) | Cage | Tets | Modes | Precompute | Hits with identical labels |")
    print("|---|---|---|---|---|---|---|")
    for name, T in builds.items():
        t = T["timings"]
        result = "reference" if name == "with display mesh" else f"{identical} of {args.hits}"
        print(f"| {name} | {sum(t.values()):.0f} | {t['cage_s']:.0f} | {t['tetgen_s']:.1f} | "
              f"{t['modes_s']:.0f} | {t['precompute_s']:.0f} | {result} |")
    if identical == args.hits and same_mesh:
        print("\nPASS: labels are identical without the display mesh.")
    else:
        print(f"\nFAIL: {args.hits - identical} hits differ (up to {worst_tets} tets). "
              "Keep the display mesh until this is understood.")
        raise SystemExit(1)  # non-zero exit, so a chained command (&&) stops here


if __name__ == "__main__":
    main()
