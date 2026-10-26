# Week-1 kill test log

**Window:** Tue 27 Oct – Mon 2 Nov 2026. Pass every item or drop the idea. No extensions.

## Plan B (decide before Day 1)

If fracture-modes still does not install natively on Windows by the end of Day 3, we will:

- [x] Use WSL for the teacher environment only
- [ ] Switch to the backup FYP (MalwareLens 2.0)
- [ ] Other: ______

Decided on: Use WSL

## Daily checklist

### Data path

| Day | Date | Task | Pass when | Done | Notes |
|---|---|---|---|---|---|
| 1 | Tue 27 Oct | Install fracture-modes in `rift-teacher` | `scripts/example.py` finishes in ~15 s and prints its success message | [x] | Passed 27 Oct on TKR's laptop. See "Day 1 results". |
| 2 | Wed 28 Oct | Buffer. Export the working environment to a file; rebuild it on the second laptop | Second laptop runs the example too | [ ] | |
| 3 | Thu 29 Oct | Wall with a window. Time the mode solve at coarse, medium and fine cage sizes | Table below filled in | [ ] | |
| 4 | Fri 30 Oct | 3 different impacts; export pieces; inspect in MeshLab or Blender | Fragments look plausible | [ ] | |
| 5 | Sat 31 Oct | Voronoi-cut the wall into 128 cells; label each cell by fragment; draw the cell graph with breaking links coloured | Graph image saved | [ ] | |
| 6 | Sun 1 Nov | Read the fracture-modes paper: timing and repeated-impact sections | Threat list written below | [ ] | |
| 7 | Mon 2 Nov | Go/no-go write-up (half a page) | Decision recorded below | [ ] | |

### Demo path (Days 1–5, in parallel)

| Task | Pass when | Done | Notes |
|---|---|---|---|
| Install Unity 6 LTS and the Sentis package | Empty scene runs | [ ] | |
| Export model A: message passing as plain matrix multiply (adjacency × features) | ONNX file loads in Sentis and prints output | [ ] | |
| Export model B: same layer using scatter (index-add of neighbour messages) | ONNX file loads in Sentis and prints output | [ ] | |

## Environment record (Day 1–2)

- fracture-modes pinned commit: `bdf5051` (20 Dec 2022, latest on their `main`)
- Python version: 3.9
- Package versions that worked:
  - Linux dry run (26 Oct): numpy 1.26.4, scipy 1.13.1, scikit-sparse 0.4.16, mosek 11.2.7, libigl 2.4.1, gpytoolbox 0.0.1, tetgen 0.6.4, polyscope 2.5.0, scikit-image 0.24.0. `check_env.py`: 18/20 passed; the 2 failures are the MOSEK checks (no licence on that machine).
  - Windows, TKR's laptop (27 Oct): Python 3.9.23, same package versions as the dry run. `check_env.py`: 20/20 passed. Their `example.py` ran successfully.
  - Windows, Waleed's laptop:
- Problems hit and how they were fixed (found in the Linux dry run, before Day 1):
  - `pip install scikit-sparse` fails: it needs `cholmod.h` and a C compiler. Fix: install it from conda-forge.
  - polyscope 2.6.1 crashes on import under Python 3.9 (uses 3.10+ syntax). fracture-modes imports it at load time. Fix: pin 2.5.0.
  - libigl 2.6.3: `boundary_facets` returns a tuple, which crashes fracture-modes. libigl 2.5.1: `random_points_on_mesh` returns 3 values, which breaks their dataset generator. Fix: pin 2.4.1.
  - Their `requirements.txt` omits `tqdm`, which their dataset generator imports. Fix: install it explicitly.
  - MOSEK 11 still has the old `appendcone` API their solver uses (deprecated, not removed). Confirmed on Windows with a licence, 27 Oct.
- Harmless output: scipy prints `SparseEfficiencyWarning` lines during `example.py`. These are speed hints about sparse matrix formats, not errors.

## Day 1 results: their bunny example (TKR's laptop, 27 Oct)

Bunny mesh, 2,000-face cage, giving 1,270 vertices and 4,048 tetrahedra.

| Step | Time | When it runs |
|---|---|---|
| Build matrices | 0.32 s | Once per shape (offline) |
| 10 fracture modes | ~9.0 s (0.90 s per mode) | Once per shape (offline) |
| Impact precomputation | 1.96 s (max 6 pieces) | Once per shape (offline) |
| Impact projection | ~0.001 s | **Every impact (runtime)** |

**What this means:**

- Once a shape is precomputed (~11 s here), fracture-modes already handles each impact in about 1 ms. RIFT cannot win on per-impact speed.
- RIFT's case has to be skipping the per-shape precomputation: shapes never seen before, procedural shapes, and pieces that are already broken.
- Good news for data generation: at this size, 300 shapes cost about 1 hour on one laptop, and each extra impact label costs about 1 ms.

## Day 3 timing — the core number

| Shape | Cage faces | Tetrahedra | Modes | Mode solve (s) | Impact precompute (s) | Impact projection (ms) | Peak RAM (MB) | Notes |
|---|---|---|---|---|---|---|---|---|
| Wall + window | ~100 | | | | | | | |
| Wall + window | ~1,000 | | | | | | | |
| Wall + window | ~2,000 (their example) | | | | | | | |
| Wall + window | ~5,000 (Breaking Bad setting) | | | | | | | |

Projected generation time for 300 shapes at the chosen cage size: ______ CPU-hours per laptop.

## Threats found (Day 6)

- (Found Day 1) After per-shape precomputation, fracture-modes handles each impact in ~1 ms. RIFT must justify itself on the precomputation cost (unseen, procedural and already-broken shapes), not on per-impact speed. Day 3 measures how that cost grows; Day 6 checks whether broken pieces can break again without recomputing their modes.

## Go / no-go (Day 7)

- Install works on both laptops from the environment file: yes / no
- Solve time per shape at usable quality: ______
- If the solve is around a second, the argument shifts to damaged and procedural shapes: yes / no
- Sentis model A loads: yes / no. Model B loads: yes / no
- Does fracture-modes already handle damaged shapes or repeated impacts?

**Decision:** GO / NO-GO

**Reason:**
