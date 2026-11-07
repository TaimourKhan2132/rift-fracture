"""Strength audit: why do some shapes' strength levels collapse? (dataset pilot follow-up, 7 Nov)

In the 20-shape pilot, walls, slabs, L-corners and beams broke more as the strength went
up. On the pillars and one arch, two or more levels got the same threshold, or a level
broke off far more than its target (a square pillar lost 35% at the 10% level). These
shapes break in big chunks, so the median calibration hit sits on a jump.

This script rebuilds the teacher for chosen shapes exactly as make_dataset.py does (same
seeds, same hits: make_dataset.prepare) and sets the six thresholds two ways on the same
100 calibration hits:

  median  D3 as in dataset v1: the median calibration hit breaks off the target share
  mean    the average calibration hit breaks off the target share

For each way it reports what the 50 labelled hits break off at every level and how many
of them change when the strength goes up one level (a "dead step" changes fewer than 3).
It also measures false breaks: volume the cell labels show as broken off while the
teacher keeps it in the main body (the "rebuilt" number in the pilot).

Usage (Miniforge Prompt, rift-teacher active, repo root, shapes made by make_shapes.py):
    python scripts/strength_audit.py               pilot problem shapes plus controls
    python scripts/strength_audit.py --ids 6,9,14

Outputs (gitignored): outputs/strength_audit/<id>.png and <id>.json, summary.json,
log_table.md
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # find our other scripts
import make_dataset as md
from teacher import DEFAULT_FM_DIR, PUSH_RULES, STRENGTH_STATS, damage_share, target_damage_threshold
import ui

# Controls: 0 wall, 3 slab. Problems: 6, 9, 18 pillars, 14 arch, 16 wall (2% level).
# Rebuilt below 80%: 1, 10 L-corners.
DEFAULT_IDS = "0,1,3,6,9,10,14,16,18"
SWEEP = 90       # thresholds on the damage curve
DEAD = 3         # a step that changes fewer than this many of the 50 hits does nothing
SHOW_VOXELS = 8000
OUT_DIR = Path("outputs/strength_audit")


def damage_matrix(fm, dists, piece_vol, thresholds):
    """Teacher share broken off for every hit (rows) at every threshold (columns)."""
    return np.array([[damage_share(fm, d, piece_vol, t) for t in thresholds] for d in dists])


def try_rule(s, stat):
    """Thresholds from the calibration hits by one rule; what the labelled hits then do."""
    thresholds, achieved = md.calibrate(s.fm, s.cal_d, s.piece_vol, stat)
    res = md.label_all(s, thresholds)
    grid = lambda key: np.array([[r[key] for r in row] for row in res], dtype=float)
    teacher, cells, fb = grid("broken_off_teacher"), grid("broken_off_cells"), grid("false_break")
    broken = np.stack([np.stack([r["link_broken"] for r in row]) for row in res])  # hit, level, link
    fragment = np.stack([np.stack([r["cell_fragment"] for r in row]) for row in res])
    before = np.concatenate([np.zeros_like(broken[:, :1]), broken[:, :-1]], axis=1)
    changed = (broken != before).any(axis=2).sum(axis=0)  # level 0: hits that break anything
    n_hits, n_levels = broken.shape[:2]
    distinct = [len({np.packbits(broken[i, j]).tobytes() for i in range(n_hits)})
                for j in range(n_levels)]
    own = np.median if stat == "median" else np.mean
    return {
        "stat": stat, "thresholds": thresholds, "calibration_damage": achieved,
        "same_threshold": int(sum(np.isclose(a, b, rtol=1e-3) for a, b in zip(thresholds, thresholds[1:]))),
        "teacher_median": np.median(teacher, axis=0).tolist(),
        "teacher_mean": teacher.mean(axis=0).tolist(),
        "cells_median": np.median(cells, axis=0).tolist(),
        "cells_mean": cells.mean(axis=0).tolist(),
        "off_target": float(np.max(np.abs(own(teacher, axis=0) - np.array(md.STRENGTHS)))),
        "hits_changed": changed.tolist(), "dead_steps": int((changed[1:] < DEAD).sum()),
        "distinct_outcomes": distinct,
        "rebuilt": float(grid("rebuilt").mean()),
        "false_break_mean": float(fb.mean()), "false_break_max": float(fb.max()),
        "false_break_over_1pct": float((fb > 0.01).mean()),
    }, fragment


def audit_shape(shape_id, opts):
    t0 = time.time()
    s = md.prepare(shape_id, opts)
    fm, pv = s.fm, s.piece_vol

    rules, fragments = {}, {}
    for stat in STRENGTH_STATS:
        rules[stat], fragments[stat] = try_rule(s, stat)

    # Damage against threshold, from 30% down to 0.5% broken off by either rule.
    lo = min(target_damage_threshold(fm, s.cal_d, pv, 0.30, st)[0] for st in STRENGTH_STATS)
    hi = max(target_damage_threshold(fm, s.cal_d, pv, 0.005, st)[0] for st in STRENGTH_STATS)
    sweep = np.exp(np.linspace(np.log(lo / 1.3), np.log(hi * 1.3), SWEEP))
    cal_curve = damage_matrix(fm, s.cal_d, pv, sweep)

    record = {"id": shape_id, "family": s.shape["family"], "describe": s.shape["describe"],
              "pieces": int(fm.precomputed_num_pieces), "triangles": int(len(s.f)),
              "settings": md.settings(opts), "rules": rules,
              "sweep": {"threshold": sweep.tolist(),
                        "calibration_median": np.median(cal_curve, axis=0).tolist(),
                        "calibration_mean": cal_curve.mean(axis=0).tolist(),
                        "calibration_q25": np.quantile(cal_curve, 0.25, axis=0).tolist(),
                        "calibration_q75": np.quantile(cal_curve, 0.75, axis=0).tolist()},
              "timings": s.times}
    record["seconds"] = round(time.time() - t0, 1)
    try:
        draw_audit(Path(opts["out_dir"]) / f"{shape_id:05d}.png", record, s, fragments)
    except Exception as err:  # a picture must never cost the numbers
        record["picture_error"] = f"{type(err).__name__}: {err}"
    return record


# ---------------------------------------------------------------- picture


def draw_audit(path, record, s, fragments):
    """Damage curves, what each rule gives per level, and one hit at all six levels by each rule."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    colour = {"median": "#d9771e", "mean": "#2b6cb0"}
    levels = np.array(md.STRENGTHS) * 100
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.15, 1, 1], hspace=0.28, wspace=0.05)

    ax = fig.add_subplot(gs[0, :3])
    sw = record["sweep"]
    x = np.array(sw["threshold"])
    ax.fill_between(x, np.array(sw["calibration_q25"]) * 100, np.array(sw["calibration_q75"]) * 100,
                    color="0.85", label="middle half of calibration hits")
    ax.plot(x, np.array(sw["calibration_median"]) * 100, color=colour["median"], lw=2, label="median hit")
    ax.plot(x, np.array(sw["calibration_mean"]) * 100, color=colour["mean"], lw=2, label="average hit")
    for lv in levels:
        ax.axhline(lv, color="0.6", lw=0.6, ls=":")
    for stat, marker, y in (("median", "v", 1.0), ("mean", "^", 0.0)):  # on the top and bottom edges
        ax.scatter(record["rules"][stat]["thresholds"], [y] * len(levels), marker=marker, s=60,
                   color=colour[stat], clip_on=False, zorder=5, transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    ax.set_ylim(0, 32)
    ax.set_xlabel("break threshold (higher = harder to break)")
    ax.set_ylabel("share broken off (%)")
    ax.set_title("Calibration hits: damage against threshold. The six thresholds: median rule "
                 "(top edge), mean rule (bottom edge)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")

    ax = fig.add_subplot(gs[0, 3:])
    ax.plot([0, 13], [0, 13], color="0.6", lw=0.8, ls=":")
    for stat, key, label in (("median", "teacher_median", "median rule: median labelled hit"),
                             ("mean", "teacher_mean", "mean rule: average labelled hit")):
        r = record["rules"][stat]
        y = np.array(r[key]) * 100
        ax.plot(levels, y, "o-", color=colour[stat], label=label)
        for xi, yi, n in zip(levels, y, r["hits_changed"]):
            ax.annotate(str(n), (xi, yi), textcoords="offset points", xytext=(0, 7 if stat == "mean" else -13),
                        ha="center", fontsize=7, color=colour[stat])
    ax.set_xlim(1, 13)
    ax.set_xticks(levels)
    ax.set_xlabel("strength level (%)")
    ax.set_ylabel("labelled hits: share broken off (%)")
    ax.set_title("What the 50 labelled hits break off (teacher). Small numbers: hits that change at that step",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left")

    # One hit at all six levels by each rule: the first hit that breaks something at 6%
    # by the median rule (the same choice as the pilot pictures).
    mid = 2
    med_frag = fragments["median"]
    hit = next((i for i in range(len(med_frag)) if med_frag[i, mid].max() > 0), 0)
    show = np.random.default_rng(0).choice(len(s.vox), size=min(len(s.vox), SHOW_VOXELS), replace=False)
    P = s.vox[show][:, [0, 2, 1]]
    lo, hi = P.min(axis=0), P.max(axis=0)
    centre, half = (lo + hi) / 2, (hi - lo).max() / 2
    palette = colormaps["tab10"]
    star = s.hit_pts[hit][[0, 2, 1]]
    for row, stat in enumerate(STRENGTH_STATS):
        for j in range(len(levels)):
            a = fig.add_subplot(gs[row + 1, j], projection="3d")
            lab = fragments[stat][hit, j][s.cell[show]]
            cols = np.array([(0.78, 0.78, 0.78, 0.35) if l == 0 else palette((l - 1) % 10) for l in lab])
            a.scatter(*P.T, c=cols, s=2, linewidths=0, depthshade=False)
            a.scatter(*star, marker="*", s=180, c="red", edgecolors="black", zorder=5)
            for set_lim, c in zip((a.set_xlim, a.set_ylim, a.set_zlim), centre):
                set_lim(c - half, c + half)
            a.view_init(elev=22, azim=-58)
            try:
                a.set_box_aspect(None, zoom=1.3)
            except TypeError:  # older matplotlib: no zoom
                pass
            a.set_axis_off()
            n_frag = int(fragments[stat][hit, j].max()) + 1
            a.set_title(f"{stat} rule, {levels[j]:.0f}%: {n_frag} fragment{'s' * (n_frag != 1)}",
                        fontsize=9, color=colour[stat])
    fig.suptitle(f"Strength audit #{record['id']:05d} {record['family'].replace('_', ' ')} "
                 f"({record['describe']}), {record['pieces']} teacher pieces. Bottom rows: labelled hit {hit}",
                 fontsize=12)
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.01, top=0.93)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=75)
    plt.close(fig)


# ---------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser(description="Compare the median and mean strength rules on chosen shapes.")
    p.add_argument("--ids", default=DEFAULT_IDS, help="shape IDs, e.g. 6,9,14 or 0-19")
    p.add_argument("--push", choices=PUSH_RULES, default=md.PUSH)
    p.add_argument("--cage", type=int, default=md.TEACHER["cage"], help="testing only; D1 says 5000")
    p.add_argument("--modes", type=int, default=md.TEACHER["modes"], help="testing only; D1 says 20")
    p.add_argument("--data-dir", type=Path, default=md.DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--fm-dir", type=Path, default=DEFAULT_FM_DIR)
    args = p.parse_args()
    ids = md.parse_ids(args.ids)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    opts = {"data_dir": str(args.data_dir), "out_dir": str(args.out_dir), "fm_dir": str(args.fm_dir),
            "push": args.push, "cage": args.cage, "modes": args.modes}
    with ui.Run(f"Strength audit: median vs mean rule, IDs {args.ids}", args.out_dir, steps=2) as run:
        shapes = md.read_shapes(args.data_dir)
        missing = [i for i in ids if shapes.get(i, {}).get("status") != "ok"]
        if missing:
            run.note(f"No finished shape for IDs {missing} (make them with make_shapes.py first).")
        ids = [i for i in ids if i not in missing]
        if opts["cage"] != md.TEACHER["cage"] or opts["modes"] != md.TEACHER["modes"]:
            run.say("[yellow]Teacher setting differs from D1 (option B): testing only.[/]",
                    "Teacher setting differs from D1 (option B): testing only.")
        records = []
        with run.step(f"Audit {len(ids)} shapes (about 2 minutes each)"):
            for shape_id in run.track(ids, "shapes"):
                try:
                    r = audit_shape(shape_id, {**opts, "shape": shapes[shape_id]})
                except Exception as err:
                    run.say(f"[red]#{shape_id:05d} failed[/]: {err}", f"#{shape_id:05d} failed: {err}")
                    r = {"id": shape_id, "family": shapes[shape_id]["family"], "status": "failed",
                         "error": f"{type(err).__name__}: {err}", "traceback": traceback.format_exc()[-2000:]}
                    (args.out_dir / f"{shape_id:05d}.json").write_text(json.dumps(r, indent=1))
                    continue
                (args.out_dir / f"{shape_id:05d}.json").write_text(json.dumps(r, indent=1))
                records.append(r)
                m, a = r["rules"]["median"], r["rules"]["mean"]
                run.note(f"#{shape_id:05d} {r['family']}: dead steps median rule {m['dead_steps']}, "
                         f"mean rule {a['dead_steps']}; false break up to {m['false_break_max']:.1%}")
        with run.step("Summarise"):
            summarise(run, args, records)


def summarise(run, args, records):
    pick = lambda xs: " / ".join(f"{100 * xs[j]:.1f}" for j in md.SHOW_LEVELS)
    columns = ["Shape", "Pieces", "Rule", "Same threshold", "Dead steps", "Hits changed per step",
               "Median hit at 2/6/12%", "Average hit at 2/6/12%", "Rebuilt", "False break mean / max"]
    rows = []
    for r in records:
        for stat in STRENGTH_STATS:
            q = r["rules"][stat]
            rows.append([f"#{r['id']} {r['family'].replace('_', ' ')}", r["pieces"], stat,
                         q["same_threshold"], q["dead_steps"], " ".join(str(n) for n in q["hits_changed"]),
                         pick(q["teacher_median"]), pick(q["teacher_mean"]), f"{q['rebuilt']:.0%}",
                         f"{q['false_break_mean']:.2%} / {q['false_break_max']:.1%}"])
    total = {stat: {"dead_steps": sum(r["rules"][stat]["dead_steps"] for r in records),
                    "same_threshold": sum(r["rules"][stat]["same_threshold"] for r in records),
                    "worst_off_target": max((r["rules"][stat]["off_target"] for r in records), default=None)}
             for stat in STRENGTH_STATS}
    (args.out_dir / "summary.json").write_text(json.dumps(
        {"ids": args.ids, "shapes": len(records), "totals": total,
         "per_shape": {r["id"]: {stat: {k: r["rules"][stat][k] for k in
                                        ("same_threshold", "dead_steps", "hits_changed", "off_target",
                                         "rebuilt", "false_break_mean", "false_break_max")}
                                 for stat in STRENGTH_STATS} for r in records}}, indent=2))
    intro = (f"Strength audit, IDs {args.ids}: {len(records)} shapes, {args.push} push. Shares are teacher "
             f"volume broken off by the 50 labelled hits, in %. A dead step changes fewer than {DEAD} hits. "
             + "; ".join(f"{stat} rule: {t['dead_steps']} dead steps, {t['same_threshold']} repeated "
                         f"thresholds, worst level {100 * t['worst_off_target']:.1f} points off target"
                         for stat, t in total.items() if t["worst_off_target"] is not None) + ".")
    run.table("Median vs mean strength rule", columns, rows)
    run.note(intro)
    run.log_markdown("\n".join([intro, "", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
                               + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]))


if __name__ == "__main__":
    main()
