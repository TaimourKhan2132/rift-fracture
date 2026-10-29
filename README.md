# RIFT: Real-Time Inference of Fracture Topology

Final Year Project, BS Data Science, University of Management and Technology (UMT), Lahore.
Supervisor: Daniyal Adeeb.

RIFT decides, at the moment of impact, how a game object breaks. A graph neural network reads a pre-cut object (Voronoi cells) plus the hit point and direction, then predicts for each link between neighbouring cells whether it holds or breaks. Cells still linked form one fragment.

The goal: breaks close to a physics-based reference, at roughly the cost of today's pre-broken tricks, on shapes the game has never seen.

## Status

Week-1 feasibility test, 27 Oct – 2 Nov 2026. Progress and measurements: [`docs/kill-test-log.md`](docs/kill-test-log.md).

## Setup (data generation)

The teacher (the slow, accurate method the network learns to copy) is **fracture-modes** (Sellán et al., ACM TOG 2022).

- Repository: https://github.com/sgsellan/fracture-modes
- Location: `third_party/fracture-modes` (not committed; cloned with `--recursive`)
- Pinned commit: [`bdf5051`](https://github.com/sgsellan/fracture-modes/commit/bdf5051fb0d78d787f49fabbf75b54bbc6698b17) (20 Dec 2022, latest on their `main`)

Environment: `rift-teacher` (conda, Python 3.9, native Windows). Install steps and the reasons behind each pinned version: [`docs/setup-windows.md`](docs/setup-windows.md). Check an install with `python scripts/check_env.py`.

## Licence and credits

fracture-modes is released for academic, non-commercial, non-military use. It is not copied or redistributed in this repository.

Licence for this repository: to be decided before public release.

## Team

Taimour Khan · Waleed Nadeem
