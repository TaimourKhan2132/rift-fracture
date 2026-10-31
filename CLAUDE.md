# CLAUDE.md — project rules for RIFT

Claude Code reads this file at the start of every session. Follow it.

## What this project is

RIFT: Real-Time Inference of Fracture Topology. Final Year Project at UMT, Lahore.

A graph neural network predicts how a game object breaks on impact. The object is pre-cut into Voronoi cells. Each cell is a node; touching cells are linked. Given the impact point and direction, the network classifies each link as hold or break. Connected cells form fragments.

Training labels come from the teacher, **fracture-modes** (Sellán et al., 2022), which is slow and runs offline only.

## Current phase

**Week-1 kill test, 27 Oct – 2 Nov 2026.** Work only on the items in `docs/kill-test-log.md`.

Do NOT yet build: the full package layout, training code, the Unity integration, or anything for the Breaking Bad dataset. Folders are created when their first real file is needed.

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
  - Reasons are in `docs/setup-windows.md`. Newer libigl and polyscope break fracture-modes.
- `python scripts/check_env.py` must report all checks passed before any data generation.

### fracture-modes

- Lives in `third_party/fracture-modes` (gitignored). Pinned commit: `bdf5051` (recorded in `README.md` and `scripts/check_env.py`).
- Run their scripts from inside `third_party/fracture-modes`: they load data by relative path, and their `scripts/context.py` sets the import path.
- Our own code imports it by adding `third_party/fracture-modes` to `sys.path`.
- **Never copy its code into this repo** (academic, non-commercial licence).
- The MOSEK licence file lives at `%USERPROFILE%\mosek\mosek.lic`. Never copy it into the repo. It expires after 365 days.

## Teacher settings (decision D1, under review; see `docs/decisions.md`)

- Current choice: cage 5,000 faces, 40 modes, d=3, cage grid 256, tetgen minratio 1.5. About 470 s and 1 GB per shape. The team may switch to option B (20 modes, d=1, about 87 s per shape). Do not start bulk data generation until D1 is settled.
- 256 Voronoi cells per shape. Labels and fragment metrics are defined at cell resolution.
- The break threshold is not a constant: calibrate a band per shape. It acts as impact strength.
- Always use fracture-modes' default projection. `project_on_modes` shatters the wall far from the hit (Day 6b).
- The teacher's breaks do not depend on impact direction (Day 6b, paper Sec. 6). Never claim direction-dependent fracture; RIFT uses impact location and strength (proposed decision D2).

## Rules

- Never commit `data/`, `outputs/`, checkpoints, model files or licence files.
- Every generated shape gets an integer ID and a random seed. Every output file is named by its ID.
- Generation scripts save each shape's output as soon as it finishes and skip IDs already done, so a crash costs one shape.
- Scripts that run work in parallel must use the `if __name__ == "__main__":` guard (Windows starts new processes from scratch).
- Record every timing and measurement in `docs/kill-test-log.md`.
- Ask before installing new packages or changing an environment file. Pin versions once something works.
- Keep changes small and comment anything non-obvious. The team must be able to explain every line in the viva.
- In writing, say "matches a physics-based reference", never "real fracture".
- **Dataset quality beats generation speed.** Never lower teacher settings (cage grid size, cage faces, number of modes) or skip quality checks to save time, unless the Day 4/5 checks show quality holds.
- Every label setting must be checked for variety: if every hit gives the same outcome, the labels are useless (Day 3 red flag).

## Scripts

Run from the repo root inside `rift-teacher`. Outputs go to `outputs/` (gitignored).

- `scripts/check_env.py`: checks the environment can run fracture-modes (must pass 20/20).
- `scripts/make_wall.py`: the Day 3 test shape (wall with a window).
- `scripts/time_fracture.py`: Day 3 timing across cage sizes and mode dimensions.
- `scripts/sweep_threshold.py`: Day 4 break-threshold sweep; writes `outputs/day4/chosen.json`.
- `scripts/cell_graph.py`: Day 5 Voronoi cells, links and break labels from teacher results.
- `scripts/direction_check.py`: Day 6b check of whether impact direction changes the break, for both fracture-modes projections.
- `scripts/second_hit.py`: Day 6c test of whether a second hit on a damaged wall can reuse the intact wall's homework or needs it redone (includes a no-crater control).
- `scripts/repeat_check.py`: Day 6d check of how much the teacher's break depends on the mesh (same file twice, finer triangles, rounded rebuild), and of a target-damage threshold rule.
- `scripts/teacher.py`: shared helpers for the scripts above (not run directly).
- `scripts/export_sentis_test.py` (**rift-train**): exports the two Sentis test models (dense vs scatter graph layer) to `outputs/sentis_test/`.
- `unity/SentisTest/SentisSmokeTest.cs`: Unity script that runs those models on every Sentis backend. Copy it into a separate Unity test project; the real Unity project is not in the repo yet.

## Git and commits

- One branch per task, merged through a pull request. Never push to `main` directly.
- Merge pull requests with **Rebase and merge**, never Squash and merge (squashing collapses the commit history).
- Commit after each logical step: one change you can describe in one line. Never batch a whole session into one commit.
- Commit messages: verb + what changed, e.g. `Add cage-size timing script`.
- Push at the end of every session.
- Line endings are LF everywhere (enforced by `.gitattributes`).
