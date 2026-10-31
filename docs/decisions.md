# Decision log

Decisions that shape the dataset or the model, with the evidence behind them. Newest first.

## D3: Strength means the same broken-off share on every shape (1 Nov 2026, proposed)

**Proposal:** set each shape's break threshold so that the median random hit breaks off a fixed share of the shape's volume (5% so far; a band of shares for weak to strong hits). This replaces Day 4's rule (the threshold giving the most different outcomes).

**Why:** Day 4's rule gave thresholds from 64 to 210 on near-identical walls (Day 6c): on one wall a hit broke off a quarter of it, on another 6%. Strength would mean different things on different shapes, and the network could not learn it as an input. The target-damage rule hit 4.9–5.0% on four versions of the same wall (Day 6d) and made the second-hit comparison fair (Day 6c re-run).

**Open:** which shares to use for weak, medium and strong hits.

**Status:** waiting for the team's agreement.

## D2: RIFT claims impact location and strength, not direction (31 Oct 2026, proposed)

**Proposal:** RIFT predicts how a shape breaks from where it is hit and how hard. It does not claim that the hit direction changes which pieces break. In the game demo, direction only pushes the debris.

**Why:** the teacher cannot provide direction-dependent breaks.
- The paper says its modes ignore the direction of the break, and the fault is the same regardless of impact direction (Sec. 6, Fig. 21).
- Day 6b: with the default projection, 0% of hits tilted by 45 or 80 degrees changed. With `project_on_modes`, direction changes the break slightly (93–95% of the broken-off volume stays the same), but a single hit shatters most of the wall far from the impact (locality 0.55).

**Still different from the closest work:** Neural Clustering ignores the impact entirely; RIFT uses its location and strength.

**Status:** waiting for the team's agreement. It changes the README pitch and the proposal wording.

## D1: Teacher setting is option C (31 Oct 2026, under review)

**Decision:** generate the dataset with cage 5,000 faces, 40 modes, d=3, cage grid 256, tetgen minratio 1.5. Cut shapes into 256 Voronoi cells. Calibrate a threshold band per shape instead of using one fixed value.

**Why:** crater-style breaks that look like game impact damage: 27 fragments per hit on average, close to the hit (locality 0.19), 100 different outcomes from 100 hits. Chosen by the team after comparing options A, B and C (kill-test log, Day 4b and 5b), partly because C was expected to respond to impact direction.

**Cost:** about 470 s per shape and 1.0 GB peak memory. 300 shapes is about 20 CPU-hours per laptop.

**Label quality at 256 cells:** purity 94.6% (worst hit 90.8%), lost volume 0.76%, 24.4 teacher fragments become 14.5 cell fragments, rebuilt 58%. Labels and fragment metrics are therefore defined at cell resolution.

**Day 6b result (31 Oct): the direction reason does not hold** (see D2). Option C's remaining advantage over option B is the look: about 22 vs 15 fragments per hit.

| | Option B | Option C |
|---|---|---|
| Setting | cage 5,000, 20 modes, d=1 | cage 5,000, 40 modes, d=3 |
| Fragments per hit | 15.3 | 21.7–27.3 |
| Locality (lower = closer to the hit) | 0.12 | 0.17–0.19 |
| Direction changes the break | No | No (default projection) |
| Time per shape | 87 s | 470 s |
| Shapes per laptop-night (about 10 hours) | about 400 | about 75 |
| 256 cells: purity / lost volume | 97.1% / 0.47% | 94.6% / 0.76% |

**Status:** under review. The team decides between keeping C and switching to B.
