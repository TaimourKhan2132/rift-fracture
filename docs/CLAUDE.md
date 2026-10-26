# CLAUDE.md — project rules for RIFT

Claude Code reads this file at the start of every session. Follow it.

## What this project is

RIFT: Real-Time Inference of Fracture Topology. Final Year Project at UMT, Lahore.

A graph neural network predicts how a game object breaks on impact. The object is pre-cut into Voronoi cells. Each cell is a node; touching cells are linked. Given the impact point and direction, the network classifies each link as hold or break. Connected cells form fragments.

Training labels come from the teacher, **fracture-modes** (Sellán et al., 2022), which is slow and runs offline only.

## Current phase

**Week-1 kill test, 27 Oct – 2 Nov 2026.** Work only on the items in `docs/kill-test-log.md`.

Do NOT yet build: the full package layout, training code, the Unity integration, or anything for the Breaking Bad dataset. Folders are created when their first real file is needed.

## Environment

- Native Windows. **No WSL.** Use Windows paths. Shells: PowerShell or Miniforge Prompt.
- Repo lives at `C:\dev\rift-fracture`.
- Two environments, never mixed:
  - `rift-teacher`: conda, Python 3.9, fracture-modes and its dependencies. Data generation only.
  - `rift-train` (later): PyTorch + PyTorch Geometric. Training only.
  - The two stages communicate only through files on disk.
- In `rift-teacher`, install conda packages first and pip packages last.
- fracture-modes lives in `third_party/fracture-modes` (gitignored), cloned with `--recursive` and pinned to the commit recorded in `README.md`. Import it by adding that folder to the Python path. **Never copy its code into this repo** (academic, non-commercial licence).
- The MOSEK licence file lives at `%USERPROFILE%\mosek\mosek.lic`. Never copy it into the repo.

## Rules

- Never commit `data/`, `outputs/`, checkpoints, model files or licence files.
- Every generated shape gets an integer ID and a random seed. Every output file is named by its ID.
- Generation scripts save each shape's output as soon as it finishes and skip IDs already done, so a crash costs one shape.
- Scripts that run work in parallel must use the `if __name__ == "__main__":` guard (Windows starts new processes from scratch).
- Record every timing and measurement in `docs/kill-test-log.md`.
- Ask before installing new packages or changing an environment file. Pin versions once something works.
- One branch per task, merged through a pull request. Never push to `main` directly.
- Keep changes small and comment anything non-obvious. The team must be able to explain every line in the viva.
- In writing, say "matches a physics-based reference", never "real fracture".
