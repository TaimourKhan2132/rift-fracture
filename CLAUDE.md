# CLAUDE.md — project rules for RIFT

Claude Code reads this file at the start of every session. Follow it.

## What this project is

RIFT: Real-Time Inference of Fracture Topology. Final Year Project at UMT, Lahore.

A graph neural network predicts how a game object breaks on impact. The object is pre-cut into Voronoi cells. Each cell is a node; touching cells are linked. Given the impact point and direction, the network classifies each link as hold or break. Connected cells form fragments.

Training labels come from the teacher, **fracture-modes** (Sellán et al., 2022), which is slow and runs offline only.

## Current phase

**Phase 1: data (from 7 Nov 2026).** The week-1 kill test ended with GO (`docs/kill-test-log.md`). Day 6f is done (D5 proposed). The shape generator (`scripts/make_shapes.py`) and the dataset generator (`scripts/make_dataset.py`) are built (v1). The 20-shape pilot and the strength audit are done (`docs/data-log.md`): the median strength rule fails on shapes that break in big chunks, and the mean rule fixes it (D3 amendment proposed 8 Nov). Two shapes at once give 1.46 times as many shapes per hour. Now: accept the D3 amendment and D5, make dataset v2 with the mean rule, then bulk generation (two shapes at once per laptop).

Do NOT yet build: training code, the Unity integration, or anything for the Breaking Bad dataset. Folders are created when their first real file is needed.

Plan B (decided): if fracture-modes does not run natively on Windows by the end of Day 3, use WSL for the teacher environment only.

## Environment

- Native Windows. No WSL unless Plan B is triggered.
- Repo on TKR's laptop: `C:\Users\taimo\Documents\work\RIFT\rift-fracture` (not synced by OneDrive).
- Shells:
  - VS Code terminal is **cmd**.
  - Conda is not on PATH. Conda commands work in **Miniforge Prompt** only.
  - Claude Code's shell (Git Bash) runs env commands without activating:
    `~/miniforge3/Scripts/conda.exe run -n rift-teacher --no-capture-output python <script>`
- Two environments, never mixed:
  - `rift-teacher`: conda, Python 3.9, fracture-modes and its dependencies. Data generation only.
  - `rift-train`: conda, Python 3.11. Today: torch 2.15.0 (CPU build), onnx 1.23.2, onnxscript 0.7.2, for the Sentis export test. Later: the CUDA build of torch plus PyTorch Geometric, for training.
  - The two stages communicate only through files on disk.

### rift-teacher install

- Follow `docs/setup-windows.md` exactly. Do NOT use fracture-modes' own `requirements.txt` (unpinned, and missing `tqdm`).
- Conda owns: python, numpy, scipy, scikit-sparse, mosek. Pip installs the rest and must never upgrade numpy or scipy.
- Pinned versions. Do not change any of these without re-running `scripts/check_env.py`:
  - python 3.9, numpy <2, scikit-sparse 0.4.16 (conda-forge), mosek 11.2 (mosek channel)
  - libigl 2.4.1, gpytoolbox 0.0.1, polyscope 2.5.0, tetgen 0.6.4, scikit-image 0.24.0 (pip)
  - rich 14.3.4 with markdown-it-py 3.0.0, mdurl 0.1.2, pygments 2.21.0 (pip; terminal output only, optional)
  - Reasons are in `docs/setup-windows.md`. Newer libigl and polyscope break fracture-modes.
- `python scripts/check_env.py` must report all checks passed before any data generation.

### fracture-modes

- Lives in `third_party/fracture-modes` (gitignored). Pinned commit: `bdf5051` (recorded in `README.md` and `scripts/check_env.py`).
- Run their scripts from inside `third_party/fracture-modes`: they load data by relative path, and their `scripts/context.py` sets the import path.
- Our own code imports it by adding `third_party/fracture-modes` to `sys.path`.
- **Never copy its code into this repo** (academic, non-commercial licence).
- The MOSEK licence file lives at `%USERPROFILE%\mosek\mosek.lic`. Never copy it into the repo. It expires after 365 days.

## Teacher settings (decisions D1–D4, accepted 7 Nov; see `docs/decisions.md`)

- Option B (D1): cage 5,000 faces, 20 modes, d=1, cage grid 256, tetgen minratio 1.5, `display_mesh=False`. About 111 s per shape on a realistic 14,000-triangle mesh (Day 6e).
- 256 Voronoi cells per shape. Labels and fragment metrics are defined at cell resolution.
- The break threshold is not a constant: calibrate a band per shape. It acts as impact strength.
- Strength rule (D3): set each shape's threshold so the median random hit breaks off a fixed share of its volume (`teacher.target_damage_threshold`). Hit strengths range over 2–12% median broken-off share. The Day 4 "most different outcomes" rule gave very different strengths on near-identical shapes (Day 6c).
- Build the teacher with `build_teacher(..., display_mesh=False)` when only labels are needed: identical labels, homework 278 → 111 s on a 14,000-triangle wall (Day 6e).
- Always build the teacher through `teacher.build_teacher`: it fixes the mode solver's starting vector, without which a shape's teacher depends on the shapes built before it in the same process (Day 6e).
- Known limits: teacher labels depend on the triangle layout (same shape, different triangles: ARI about 0.6 with the corners push, 0.65–0.68 with the mass push; Day 6d, 6f), and a hit's strength depends on how many tet corners lie within 0.05 of it (Day 6c). Never compare RIFT to a single teacher run without stating this ceiling.
- Hit push (D5, proposed 7 Nov): `push="mass"` weights the push by material, the formula in fracture-modes' own comments; `push="corners"` is fracture-modes as its code runs. Mass agrees better across triangulations (ARI 0.65–0.68 vs 0.58–0.60, Day 6f). Use mass for the dataset once D5 is accepted, and always record which push was used.
- Samples (dataset v1, `make_dataset.py`): 256 cells on voxels 0.0075 apart in the unit box, a fixed size (Day 5's `voxelize` sizes voxels from the bounding box, which gave an L-corner 183 voxels). Every cell is one connected piece. Each hit is labelled at strengths 2, 4, 6, 8, 10 and 12% (D3). A cell's fragment = the group of cells still joined by unbroken links, so cell and link labels always agree. The generator uses the mass push by default (D5); pass `--push corners` if D5 is rejected.
- Always use fracture-modes' default projection. `project_on_modes` shatters the wall far from the hit (Day 6b).
- The teacher's breaks do not depend on impact direction (Day 6b, paper Sec. 6). Never claim direction-dependent fracture; RIFT uses impact location and strength (D2).
- Network layers (D4): matrix multiplies on a dense adjacency matrix (at most 256 × 256), run on Sentis' CPU backend. Never scatter operations: they are silently wrong on Sentis' GPU backends.

## Rules

- Never commit `data/`, `outputs/`, checkpoints, model files or licence files.
- Every generated shape gets an integer ID and a random seed. Every output file is named by its ID.
- Generation scripts save each shape's output as soon as it finishes and skip IDs already done, so a crash costs one shape.
- Scripts that run work in parallel must use the `if __name__ == "__main__":` guard (Windows starts new processes from scratch).
- Record every timing and measurement: week 1 in `docs/kill-test-log.md`, phase 1 onwards in `docs/data-log.md`.
- Shapes come only from `scripts/make_shapes.py`. Any change that alters a shape must bump its `GENERATOR_VERSION`, so old and new shapes never mix.
- Ask before installing new packages or changing an environment file. Pin versions once something works.
- Keep changes small and comment anything non-obvious. The team must be able to explain every line in the viva.
- In writing, say "matches a physics-based reference", never "real fracture".
- **Dataset quality beats generation speed.** Never lower teacher settings (cage grid size, cage faces, number of modes) or skip quality checks to save time, unless the Day 4/5 checks show quality holds.
- Every label setting must be checked for variety: if every hit gives the same outcome, the labels are useless (Day 3 red flag).
- Every new long-running script wraps its work in `ui.Run` (steps, `run.track` for loops, `run.table` for results, `run.log_markdown` for the log table), so progress shows in the terminal and on the dashboard. Never print raw Markdown tables to the terminal.

## Scripts

Run from the repo root inside `rift-teacher`. Outputs go to `outputs/` (gitignored).

- `scripts/check_env.py`: checks the environment can run fracture-modes (must pass 20/20).
- `scripts/make_wall.py`: the Day 3 test shape (wall with a window).
- `scripts/make_shapes.py`: the RIFT-Arch base shapes (walls, L-corners, arches, slabs, beams, square and round pillars). One OBJ per ID in `data/rift-arch/shapes/`, one line per ID in `data/rift-arch/shapes.jsonl`, previews in `outputs/shapes/`. The ID alone decides the shape; IDs already made are skipped.
- `scripts/make_dataset.py`: the RIFT-Arch training samples. For each shape: teacher homework, strength calibration, 50 hits labelled at 6 strengths on 256 cells. One `.npz` and one `.json` per ID in `data/rift-arch/samples/`, pictures in `outputs/dataset/`. Each shape runs in its own process with a time limit; IDs already done are skipped; `--every 2 --offset 0|1` splits IDs between the laptops.
- `scripts/strength_audit.py`: dataset pilot follow-up. Rebuilds the teacher for chosen shapes exactly as `make_dataset.py` does and compares the median and mean strength rules on the same hits; also measures false breaks (cell labels showing part of the main body as broken off).
- `scripts/time_fracture.py`: Day 3 timing across cage sizes and mode dimensions.
- `scripts/sweep_threshold.py`: Day 4 break-threshold sweep; writes `outputs/day4/chosen.json`.
- `scripts/cell_graph.py`: Day 5 Voronoi cells, links and break labels from teacher results.
- `scripts/direction_check.py`: Day 6b check of whether impact direction changes the break, for both fracture-modes projections.
- `scripts/second_hit.py`: Day 6c test of whether a second hit on a damaged wall can reuse the intact wall's homework or needs it redone (includes a no-crater control).
- `scripts/repeat_check.py`: Day 6d check of how much the teacher's break depends on the mesh (same file twice, finer triangles, rounded rebuild), and of a target-damage threshold rule.
- `scripts/push_check.py`: Day 6f comparison of fracture-modes' push (every corner equal) with a mass-weighted push, on three versions of the same wall.
- `scripts/label_only_check.py`: Day 6e check that labels are identical when the teacher skips the display-mesh step (`build_teacher(..., display_mesh=False)`), and how much time that saves.
- `scripts/teacher.py`: shared helpers for the scripts above (not run directly).
- `scripts/ui.py`: readable terminal output for every long script: steps with timers, progress bars with time left, aligned tables, and the log table saved as `<out_dir>/log_table.md` (not run directly).
- `scripts/dashboard.py`: builds `outputs/dashboard.html`, a local page with the run in progress and every results folder. Scripts using `ui.Run` keep it updated; `python scripts/dashboard.py --open` rebuilds and opens it.
- `scripts/export_sentis_test.py` (**rift-train**): exports the two Sentis test models (dense vs scatter graph layer) to `outputs/sentis_test/`.
- `unity/SentisTest/SentisSmokeTest.cs`: Unity script that runs those models on every Sentis backend. Copy it into a separate Unity test project; the real Unity project is not in the repo yet.

## Git and commits

- One branch per task, merged through a pull request. Never push to `main` directly.
- Merge pull requests with **Rebase and merge**, never Squash and merge (squashing collapses the commit history).
- Commit after each logical step: one change you can describe in one line. Never batch a whole session into one commit.
- Commit messages: verb + what changed, e.g. `Add cage-size timing script`.
- Push at the end of every session.
- Line endings are LF everywhere (enforced by `.gitattributes`).
