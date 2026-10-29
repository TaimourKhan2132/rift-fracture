"""Day 4: find break thresholds where WHERE you hit changes HOW the shape breaks.

Day 3 found that with fracture-modes' default threshold (0.02), every impact
shattered the wall into every possible piece, so the labels did not depend on
the hit. For each teacher setting, this script:

  1. builds the teacher once (cage, tets, modes, precomputation),
  2. hits the shape at N random points on its surface, each aimed into the surface,
  3. tests every hit against ~40 thresholds and records the outcome,
  4. picks the threshold that gives the most different outcomes,
  5. checks our threshold shortcut against fracture-modes' own result for every hit,
  6. draws 6 of the breaks at the chosen threshold and saves them as OBJ files.

Settings tested (every combination): cage 2000 and 5000 faces, 10 and 20 modes,
d=3 (direction matters) and d=1 (direction ignored).

Usage (Miniforge Prompt, rift-teacher active, repo root):
    python scripts/make_wall.py          (if outputs/day3/wall.obj is missing)
    python scripts/sweep_threshold.py

Outputs in outputs/day4/ (gitignored):
    summary.csv               one row per setting
    sweep_<setting>.csv       outcome statistics at every threshold
    breaks_<setting>.png      6 example breaks at the chosen threshold
    <setting>_hit<i>.obj      the same breaks, to open in Blender
    threshold_curves.png      how outcomes change with threshold, all settings
    chosen.json               the setting and threshold Day 5 starts from

Words: see the top of scripts/teacher.py.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # find teacher.py next to this file
from teacher import (DEFAULT_FM_DIR, build_teacher, decide, fine_triangle_labels, load_shape,
                     piece_distances, project_impact, render_break, sample_impacts,
                     tet_volumes_and_centroids)

RESULT_TAG = "RIFT_RESULT "
DEFAULT_THRESHOLD = 0.02  # fracture-modes' default, kept in every sweep for comparison

SWEEP_FIELDS = ["threshold", "frac_intact", "frac_partial", "frac_shatter", "mean_fragments",
                "distinct_outcomes", "mean_broken_share", "mean_min_fragment_share",
                "locality_ratio"]
SUMMARY_FIELDS = [
    "setting", "status", "cage", "modes", "d", "cage_faces", "tets", "max_pieces",
    "per_shape_s", "best_threshold", "distinct_at_best", "frac_partial_at_best",
    "frac_intact_at_best", "frac_shatter_at_best", "locality_at_best",
    "mean_fragments_at_best", "band_lo", "band_hi", "distinct_at_default",
    "frac_shatter_at_default", "decision_ms_median", "full_call_ms_median",
    "mismatches", "peak_ram_mb", "error",
]


# ---------------------------------------------------------------- one setting (child process)


def outcome_stats(fm, distances, threshold, points, piece_vol, piece_cent, tet_vol, tet_cent):
    """Outcome statistics over all hits for one threshold."""
    n_max = fm.precomputed_num_pieces
    total = piece_vol.sum()
    counts, keys, broken, min_share, locality = [], set(), [], [], []
    for dist, p in zip(distances, points):
        k, lab = decide(fm, dist, threshold)
        counts.append(k)
        keys.add(lab.astype(np.int32).tobytes())  # same partition -> same labels
        if k > 1:
            frag_vol = np.bincount(lab, weights=piece_vol, minlength=k)
            main = np.argmax(frag_vol)
            broken.append(1 - frag_vol[main] / total)
            min_share.append(frag_vol.min() / total)
            # Locality: are the broken-off parts near the hit? Compare their distance
            # from the hit with the shape's average distance from the hit.
            off = lab != main
            off_centre = np.average(piece_cent[off], axis=0, weights=piece_vol[off])
            typical = np.average(np.linalg.norm(tet_cent - p, axis=1), weights=tet_vol)
            locality.append(np.linalg.norm(off_centre - p) / typical)
    counts = np.array(counts)
    nan_mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
    return {
        "threshold": threshold,
        "frac_intact": float(np.mean(counts == 1)),
        "frac_partial": float(np.mean((counts > 1) & (counts < n_max))),
        "frac_shatter": float(np.mean(counts == n_max)),
        "mean_fragments": float(counts.mean()),
        "distinct_outcomes": len(keys),
        "mean_broken_share": nan_mean(broken),
        "mean_min_fragment_share": nan_mean(min_share),
        "locality_ratio": nan_mean(locality),
    }


def run_setting(args):
    import matplotlib
    matplotlib.use("Agg")  # draw to files, no window
    import matplotlib.pyplot as plt
    from time_fracture import peak_memory_mb

    name = f"cage{args.cage}_m{args.modes}_d{args.d}"
    r = {"setting": name, "cage": args.cage, "modes": args.modes, "d": args.d}
    v_fine, f_fine = load_shape(args.mesh)
    T = build_teacher(v_fine, f_fine, args.cage, args.d, args.modes,
                      args.grid_size, args.minratio, args.fm_dir)
    fm = T["fm"]
    r["cage_faces"], r["tets"] = int(T["cage_f"].shape[0]), int(T["elements"].shape[0])
    r["max_pieces"] = int(fm.precomputed_num_pieces)
    r["per_shape_s"] = sum(T["timings"].values())

    tet_vol, tet_cent = tet_volumes_and_centroids(T["nodes"], T["elements"])
    p2t = fm.piece_to_tet_matrix  # tets x pieces, 1 where a tet belongs to a piece
    piece_vol = np.asarray(p2t.T @ tet_vol).ravel()
    piece_cent = np.asarray(p2t.T @ (tet_vol[:, None] * tet_cent)) / piece_vol[:, None]

    points, dirs = sample_impacts(v_fine, f_fine, args.impacts, args.seed)

    # One real projection per hit gives the piece distances; the threshold only
    # enters afterwards, so we can test many thresholds without re-projecting.
    distances, full_ms, decision_ms = [], [], []
    for p, dvec in zip(points, dirs):
        t = time.perf_counter()
        project_impact(fm, p, dvec, args.d, threshold=1.0)
        full_ms.append((time.perf_counter() - t) * 1000)
        dist = piece_distances(fm)
        distances.append(dist)
        # Time just the decision a game needs: distances, fragments, tet labels.
        t = time.perf_counter()
        _, lab = decide(fm, piece_distances(fm), np.median(dist) if dist.size else 1.0)
        _ = p2t @ lab
        decision_ms.append((time.perf_counter() - t) * 1000)
    r["full_call_ms_median"] = float(np.median(full_ms))
    r["decision_ms_median"] = float(np.median(decision_ms))

    pooled = np.concatenate(distances)
    pooled = pooled[pooled > 0]
    if pooled.size == 0:
        raise RuntimeError("no piece distances above zero: nothing can break")
    grid = np.geomspace(pooled.min() * 0.5, pooled.max() * 2.0, args.thresholds)
    thresholds = np.unique(np.append(grid, DEFAULT_THRESHOLD))

    rows = [outcome_stats(fm, distances, t, points, piece_vol, piece_cent, tet_vol, tet_cent)
            for t in thresholds]
    with open(args.out_dir / f"sweep_{name}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SWEEP_FIELDS)
        w.writeheader()
        w.writerows(rows)

    # Best threshold: most different outcomes; ties -> more partial breaks -> more local.
    best = max(rows, key=lambda s: (s["distinct_outcomes"], s["frac_partial"],
                                    -np.nan_to_num(s["locality_ratio"], nan=9.0)))
    band = [s["threshold"] for s in rows if s["frac_partial"] >= 0.5]
    default = next(s for s in rows if s["threshold"] == DEFAULT_THRESHOLD)
    r.update({
        "best_threshold": best["threshold"], "distinct_at_best": best["distinct_outcomes"],
        "frac_partial_at_best": best["frac_partial"], "frac_intact_at_best": best["frac_intact"],
        "frac_shatter_at_best": best["frac_shatter"], "locality_at_best": best["locality_ratio"],
        "mean_fragments_at_best": best["mean_fragments"],
        "band_lo": min(band) if band else float("nan"),
        "band_hi": max(band) if band else float("nan"),
        "distinct_at_default": default["distinct_outcomes"],
        "frac_shatter_at_default": default["frac_shatter"],
    })

    # Check our shortcut against fracture-modes itself at the chosen threshold, for every
    # hit, and pick up to 6 partial breaks (different outcomes first) to draw.
    mismatches, picked, seen = 0, [], set()
    for i, (p, dvec, dist) in enumerate(zip(points, dirs, distances)):
        k, lab = decide(fm, dist, best["threshold"])
        project_impact(fm, p, dvec, args.d, best["threshold"])
        if fm.n_pieces_after_impact != k or not np.array_equal(fm.piece_labels_after_impact, lab):
            mismatches += 1
        key = lab.astype(np.int32).tobytes()
        if 1 < k < fm.precomputed_num_pieces and key not in seen and len(picked) < 6:
            seen.add(key)
            picked.append(i)
    r["mismatches"] = mismatches
    if len(picked) < 6:  # not enough distinct partial breaks: fill with any hits
        picked += [i for i in range(len(points)) if i not in picked][: 6 - len(picked)]

    fig = plt.figure(figsize=(15, 9))
    for slot, i in enumerate(picked):
        project_impact(fm, points[i], dirs[i], args.d, best["threshold"])
        tri_labels = fine_triangle_labels(fm)
        fm.write_segmented_output(filename=str(args.out_dir / f"{name}_hit{i}.obj"))
        ax = fig.add_subplot(2, 3, slot + 1, projection="3d")
        render_break(ax, fm.fine_vertices, fm.fine_triangles, tri_labels, points[i], dirs[i],
                     title=f"hit {i}: {fm.n_pieces_after_impact} fragments")
    fig.suptitle(f"{name}: threshold {best['threshold']:.3g}, "
                 f"{best['distinct_outcomes']} different outcomes from {len(points)} hits",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out_dir / f"breaks_{name}.png", dpi=110)
    plt.close(fig)

    try:
        r["peak_ram_mb"] = peak_memory_mb()
    except Exception:
        r["peak_ram_mb"] = float("nan")
    r["status"] = "ok"
    return r


# ---------------------------------------------------------------- driver (parent process)


def plot_curves(out_dir, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for r in results:
        if r["status"] != "ok":
            continue
        with open(out_dir / f"sweep_{r['setting']}.csv") as fh:
            rows = list(csv.DictReader(fh))
        t = [float(x["threshold"]) for x in rows]
        style = "-" if r["d"] == 3 else "--"
        axes[0].plot(t, [float(x["frac_partial"]) for x in rows], style, label=r["setting"])
        axes[1].plot(t, [int(x["distinct_outcomes"]) for x in rows], style, label=r["setting"])
    axes[0].set_ylabel("share of hits giving a partial break")
    axes[1].set_ylabel("different outcomes")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("break threshold")
        ax.axvline(DEFAULT_THRESHOLD, color="grey", linewidth=0.8)
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle("Grey line: fracture-modes' default threshold (0.02). Solid: d=3, dashed: d=1.")
    fig.tight_layout()
    fig.savefig(out_dir / "threshold_curves.png", dpi=110)
    plt.close(fig)


def choose(results):
    """Starting point for Day 5. Rule, in order:
    keep settings where at least half the hits give a partial break;
    prefer d=3 (keeps impact direction, which RIFT promises);
    then most different outcomes, then finer cage, then more modes."""
    ok = [r for r in results if r["status"] == "ok"]
    good = [r for r in ok if r["frac_partial_at_best"] >= 0.5]
    pool = [r for r in good if r["d"] == 3] or good
    weak = not pool
    pool = pool or ok
    if not pool:
        return None
    best = max(pool, key=lambda r: (r["distinct_at_best"], r["cage"], r["modes"]))
    return {"setting": best["setting"], "cage": best["cage"], "modes": best["modes"],
            "d": best["d"], "threshold": best["best_threshold"], "weak": weak}


def main():
    p = argparse.ArgumentParser(description="Sweep the break threshold for several teacher settings.")
    p.add_argument("--mesh", type=Path, default=Path("outputs/day3/wall.obj"))
    p.add_argument("--cages", type=int, nargs="+", default=[2000, 5000])
    p.add_argument("--modes-list", type=int, nargs="+", default=[10, 20])
    p.add_argument("--dims", type=int, nargs="+", default=[3, 1], choices=[1, 3])
    p.add_argument("--impacts", type=int, default=100)
    p.add_argument("--thresholds", type=int, default=40, help="how many thresholds to test")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid-size", type=int, default=256)
    p.add_argument("--minratio", type=float, default=1.5)
    p.add_argument("--timeout", type=int, default=7200, help="seconds allowed per setting")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/day4"))
    p.add_argument("--fm-dir", type=Path, default=DEFAULT_FM_DIR)
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--cage", type=int, help=argparse.SUPPRESS)
    p.add_argument("--modes", type=int, help=argparse.SUPPRESS)
    p.add_argument("--d", type=int, help=argparse.SUPPRESS)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.child:
        print(RESULT_TAG + json.dumps(run_setting(args)), flush=True)
        return
    if not args.mesh.is_file():
        raise SystemExit(f"{args.mesh} not found. Run: python scripts/make_wall.py")

    results = []
    settings = [(c, m, d) for d in args.dims for c in args.cages for m in args.modes_list]
    for n, (cage, modes, d) in enumerate(settings, 1):
        name = f"cage{cage}_m{modes}_d{d}"
        print(f"[{n}/{len(settings)}] {name} ...", flush=True)
        cmd = [sys.executable, __file__, "--child", "--cage", str(cage), "--modes", str(modes),
               "--d", str(d), "--mesh", str(args.mesh), "--impacts", str(args.impacts),
               "--thresholds", str(args.thresholds), "--seed", str(args.seed),
               "--grid-size", str(args.grid_size), "--minratio", str(args.minratio),
               "--out-dir", str(args.out_dir), "--fm-dir", str(args.fm_dir)]
        t = time.perf_counter()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
            lines = [l for l in proc.stdout.splitlines() if l.startswith(RESULT_TAG)]
            if proc.returncode == 0 and lines:
                r = json.loads(lines[-1][len(RESULT_TAG):])
            else:
                tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["no output"]
                r = {"setting": name, "cage": cage, "modes": modes, "d": d,
                     "status": "failed", "error": tail[0]}
                print(proc.stderr[-3000:], file=sys.stderr)
        except subprocess.TimeoutExpired:
            r = {"setting": name, "cage": cage, "modes": modes, "d": d,
                 "status": "timeout", "error": f"over {args.timeout} s"}
        results.append(r)
        if r["status"] == "ok":
            print(f"  done in {time.perf_counter() - t:.0f} s: up to {r['max_pieces']} pieces; "
                  f"best threshold {r['best_threshold']:.3g} gives {r['distinct_at_best']} "
                  f"different outcomes, {r['frac_partial_at_best']:.0%} partial; "
                  f"shortcut mismatches: {r['mismatches']}", flush=True)
        else:
            print(f"  {r['status'].upper()}: {r['error']}", flush=True)

    with open(args.out_dir / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    plot_curves(args.out_dir, results)
    chosen = choose(results)
    if chosen:
        chosen.update({"grid_size": args.grid_size, "minratio": args.minratio,
                       "mesh": str(args.mesh)})
        (args.out_dir / "chosen.json").write_text(json.dumps(chosen, indent=2))

    n_hits = args.impacts
    print(f"\nSaved results in {args.out_dir}\n\nPaste into docs/kill-test-log.md (Day 4):\n")
    print("| Setting | Max pieces | Per shape (s) | Best threshold | Different outcomes "
          f"(of {n_hits} hits) | Partial | Intact | Shatter | Locality | At 0.02: outcomes / shatter "
          "| Decision / full call (ms) | Mismatches |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        if r["status"] != "ok":
            print(f"| {r['setting']} | {r['status']}: {r.get('error', '')[:80]} |||||||||||")
            continue
        print(f"| {r['setting']} | {r['max_pieces']} | {r['per_shape_s']:.0f} | "
              f"{r['best_threshold']:.3g} | {r['distinct_at_best']} | "
              f"{r['frac_partial_at_best']:.0%} | {r['frac_intact_at_best']:.0%} | "
              f"{r['frac_shatter_at_best']:.0%} | {r['locality_at_best']:.2f} | "
              f"{r['distinct_at_default']} / {r['frac_shatter_at_default']:.0%} | "
              f"{r['decision_ms_median']:.2f} / {r['full_call_ms_median']:.1f} | {r['mismatches']} |")
    if chosen:
        note = " (WEAK: no setting gave partial breaks for half the hits)" if chosen["weak"] else ""
        print(f"\nDay 5 will start from {chosen['setting']} at threshold "
              f"{chosen['threshold']:.3g}{note}. Saved in {args.out_dir / 'chosen.json'}")
    sys.exit(0 if all(r["status"] == "ok" for r in results) else 1)


if __name__ == "__main__":
    main()
