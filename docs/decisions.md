# Decision log

Decisions that shape the dataset or the model, with the evidence behind them. Newest first.

## D5: A hit pushes by material, not by mesh corner (7 Nov 2026, proposed)

**Proposal:** generate the dataset with `push="mass"`. Each tet corner near the hit is pushed in proportion to the material around it (fracture-modes' mass matrix). This is the formula written in fracture-modes' own code comments (`piece_impact = wave_piece_lsqr' M d`). The line their code actually runs pushes every corner equally.

**Why:** Day 6f (kill-test log).
- Same wall, different triangles, ARI vs the original: 0.68 vs 0.58 for the finer-triangles version (paired difference +0.10, 95% interval 0.04 to 0.17) and 0.65 vs 0.60 for the rounded rebuild (+0.05, -0.00 to 0.11).
- Reusing vs redoing homework on damaged walls agrees more as well (ARI 0.58–0.68 vs 0.52–0.56).
- Same cost. The strength rule (D3) calibrates it equally well (4.8–5.0% median damage on every version).
- The threshold carries over between two meshes of the same shape within 3% (corners push: 7%), which may let damaged shapes keep their parent's threshold.

**Cost:** our teacher becomes fracture-modes with one documented change. The report and the dataset sheet must say so, and baseline B4 ("fracture modes") is this version. Their code is not edited: `teacher.hit_distances` computes the push.

**Not fixed by this:** most of the mesh noise. Agreement between triangulations stays at about 0.65 ARI, because the modes themselves change with the tets.

**Status:** proposed 7 Nov 2026, to be accepted before the dataset pilot.

## D4: Graph layers are matrix multiplies, run on Sentis' CPU backend (3 Nov 2026, accepted 7 Nov)

**Proposal:** write RIFT's message-passing layers as matrix multiplies over a dense adjacency matrix (at most 256 × 256), never as scatter operations, and run the network on Sentis' CPU backend in the demo.

**Why:** the Sentis smoke test (kill-test log, Demo path results), run on both laptops with Unity 6.3 LTS and 6.6. The matrix-multiply layer was correct on CPU, GPUCompute and GPUPixel. The scatter layer was correct only on the CPU and silently wrong on both GPU backends (errors 0.40–0.51, not even the same between the two laptops). For the 6-cell test graph the CPU was also fastest: 0.05–0.07 ms against 0.7–6.8 ms on the GPU.

**Open:** timing at the real size (256 cells, 3–4 layers) once a model exists.

**Status:** accepted by the team on 7 Nov 2026.

## D3: Strength means the same broken-off share on every shape (1 Nov 2026, accepted 7 Nov)

**Proposal:** set each shape's break threshold so that the median random hit breaks off a fixed share of the shape's volume (5% so far; a band of shares for weak to strong hits). This replaces Day 4's rule (the threshold giving the most different outcomes).

**Why:** Day 4's rule gave thresholds from 64 to 210 on near-identical walls (Day 6c): on one wall a hit broke off a quarter of it, on another 6%. Strength would mean different things on different shapes, and the network could not learn it as an input. The target-damage rule hit 4.9–5.0% on four versions of the same wall (Day 6d) and made the second-hit comparison fair (Day 6c re-run).

**Range (7 Nov):** each hit's strength is a median broken-off share between 2% and 12%. To be checked on pictures in the dataset pilot before bulk generation.

**Amendment (8 Nov, proposed): the average hit, not the median.** Set each threshold so that the average calibration hit breaks off the target share. In the dataset pilot, three of four pillars and one of three arches break in big chunks: a hit leaves the shape whole or snaps off a capital or an arch corner. The median hit sits on that jump, so two levels got the same threshold, some steps changed nothing, and a 10% hit broke off 35%. The average moves a little with each hit, so every level gets its own threshold. Evidence (strength audit, data log; 9 shapes, the same hits both ways):
- Median rule: 4 dead steps (one level up changes fewer than 3 of 50 hits), 2 repeated thresholds, worst level 24.7 points off target. Mean rule: 0, 0 and 3.2 points.
- Shapes that already worked hardly change: walls, slab and L-corners stay within 1.1 points of every target (median rule: 0.4–2.9).
- Trade-off: on chunky shapes most hits break nothing at low strength (the median labelled hit breaks off 0–2.2% at the 6% level), and the average comes from a few hits on weak spots. Strength there means how many places can break; the network learns which from the hit location.
- Changing it remakes every sample: dataset v2.

**Status:** accepted by the team on 7 Nov 2026; amendment proposed 8 Nov 2026, to be accepted before bulk generation.

## D2: RIFT claims impact location and strength, not direction (31 Oct 2026, accepted 7 Nov)

**Proposal:** RIFT predicts how a shape breaks from where it is hit and how hard. It does not claim that the hit direction changes which pieces break. In the game demo, direction only pushes the debris.

**Why:** the teacher cannot provide direction-dependent breaks.
- The paper says its modes ignore the direction of the break, and the fault is the same regardless of impact direction (Sec. 6, Fig. 21).
- Day 6b: with the default projection, 0% of hits tilted by 45 or 80 degrees changed. With `project_on_modes`, direction changes the break slightly (93–95% of the broken-off volume stays the same), but a single hit shatters most of the wall far from the impact (locality 0.55).

**Still different from the closest work:** Neural Clustering ignores the impact entirely; RIFT uses its location and strength.

**Status:** accepted by the team on 7 Nov 2026. README updated; the proposal wording still to update.

## D1: Teacher setting is option B (decided 7 Nov 2026; option C chosen first, on 31 Oct)

**Decision (7 Nov):** generate the dataset with option B: cage 5,000 faces, 20 modes, d=1, cage grid 256, tetgen minratio 1.5, teacher built with `display_mesh=False`. Cut shapes into 256 Voronoi cells. Set each shape's threshold by the strength rule (D3).

**Why B:** about 111 s per realistic shape against about 470 s for C (4–5 times cheaper), and cleaner labels at 256 cells (purity 97.1% vs 94.6%, lost volume 0.47% vs 0.76%). C's extra crater detail mostly disappears at cell resolution (24.4 teacher fragments become 14.5 cell fragments) and sits inside the teacher's own mesh noise (ARI about 0.6, Day 6d). C was first chosen partly because it was expected to respond to impact direction; Day 6b showed it does not.

**First choice, 31 Oct (option C), kept for the record:** crater-style breaks that look like game impact damage: 27 fragments per hit on average, close to the hit (locality 0.19), 100 different outcomes from 100 hits. Chosen by the team after comparing options A, B and C (kill-test log, Day 4b and 5b), partly because C was expected to respond to impact direction.

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

**Status:** decided by the team on 7 Nov 2026: option B.
