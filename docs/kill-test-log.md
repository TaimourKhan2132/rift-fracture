# Week-1 kill test log

**Window:** Tue 27 Oct – Mon 2 Nov 2026. Pass every item or drop the idea. No extensions.

## Plan B (decide before Day 1)

If fracture-modes still does not install natively on Windows by the end of Day 3, we will:

- [ ] Use WSL for the teacher environment only
- [ ] Switch to the backup FYP (MalwareLens 2.0)
- [ ] Other: ______

Decided on: Use WSL

## Daily checklist

### Data path

| Day | Date | Task | Pass when | Done | Notes |
|---|---|---|---|---|---|
| 1 | Tue 27 Oct | Install fracture-modes in `rift-teacher` | `scripts/example.py` finishes in ~15 s and prints its success message | [ ] | |
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

- fracture-modes pinned commit:
- Python version:
- Package versions that worked (igl, scikit-sparse, tetgen, mosek, gpytoolbox):
- Problems hit and how they were fixed:

## Day 3 timing — the core number

| Shape | Cage faces | Tetrahedra | Modes | Solve time (s) | Peak RAM (MB) | Notes |
|---|---|---|---|---|---|---|
| Wall + window | ~100 (their example) | | | | | |
| Wall + window | ~1,000 | | | | | |
| Wall + window | ~5,000 (Breaking Bad setting) | | | | | |

Projected generation time for 300 shapes at the chosen cage size: ______ CPU-hours per laptop.

## Threats found (Day 6)

-

## Go / no-go (Day 7)

- Install works on both laptops from the environment file: yes / no
- Solve time per shape at usable quality: ______
- If the solve is around a second, the argument shifts to damaged and procedural shapes: yes / no
- Sentis model A loads: yes / no. Model B loads: yes / no
- Does fracture-modes already handle damaged shapes or repeated impacts?

**Decision:** GO / NO-GO

**Reason:**
