# Decision log

Decisions that shape the dataset or the model, with the evidence behind them. Newest first.

## D1: Teacher setting is option C (31 Oct 2026, provisional)

**Decision:** generate the dataset with cage 5,000 faces, 40 modes, d=3, cage grid 256, tetgen minratio 1.5. Cut shapes into 256 Voronoi cells. Calibrate a threshold band per shape instead of using one fixed value.

**Why:** crater-style breaks that look like game impact damage: 27 fragments per hit on average, close to the hit (locality 0.19), 100 different outcomes from 100 hits. Chosen by the team after comparing options A, B and C (kill-test log, Day 4b and 5b).

**Cost:** about 470 s per shape and 1.0 GB peak memory. 300 shapes is about 20 CPU-hours per laptop.

**Label quality at 256 cells:** purity 94.6% (worst hit 90.8%), lost volume 0.76%, 24.4 teacher fragments become 14.5 cell fragments, rebuilt 58%. Labels and fragment metrics are therefore defined at cell resolution.

**Open:** direction-awareness is not verified. With fracture-modes' default projection, impact direction has no effect on the break (code check and paper Sec. 6). Day 6b tests the `project_on_modes` projection.
- If it passes: keep C and use `project_on_modes`.
- If it fails: revisit. C still gives the crater look, but option B gives a similar look at about a fifth of the cost.
