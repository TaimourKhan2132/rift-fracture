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
| 3 | Thu 29 Oct | Wall with a window. Time the mode solve at coarse, medium and fine cage sizes | Table below filled in | [x] | Done 29 Oct. Per shape 22–45 s; impact 5–37 ms (d=3). Red flag: every impact gives the same break. See "Day 3 timing". |
| 4 | Fri 30 Oct | Look at the Day 3 breaks in Blender. On the 2,000-face wall, sweep the break threshold (and try 20 modes) | Hitting different spots gives different fragments, and they look plausible | [x] | Done 29 Oct (a day early). Swept 8 settings at both cage sizes. 81 different outcomes from 100 hits at the chosen setting. See "Day 4 results". |
| 5 | Sat 31 Oct | Voronoi-cut the wall into 128 cells; label each cell by fragment; draw the cell graph with breaking links coloured | Graph image saved | [x] | Done 29 Oct. Purity 96.2%, lost volume 0.14%. See "Day 5 results". |
| 6 | Sun 1 Nov | Read the fracture-modes paper: timing and repeated-impact sections | Threat list written below | [ ] | Pre-read done 31 Oct (`docs/paper-notes-fracture-modes.md`). Team reading of the marked pages still to do. Direction check (6b) done 31 Oct. Second-hit test (6c) done for option B: inconclusive, but found mesh sensitivity and a calibration problem (see Day 6c). |
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
- **Correction (Day 3):** the ~1 ms is fracture-modes' internal timer, which stops before tets are assigned to pieces. Timing the whole call on Day 3 gave 5–37 ms for d=3. The example also skipped the dataset generator's slower cage settings. See "Day 3 timing".

## Day 3 timing — the core number (TKR's laptop, 29 Oct)

**Setup:**
- Shape: wall with a window (`scripts/make_wall.py`).
- Settings: same as their dataset generator (cage grid 256, tetgen minratio 1.5), 10 modes, 20 impacts on the front face, default break threshold 0.02.
- Script: `scripts/time_fracture.py`. Raw numbers: `outputs/day3/timing.csv` (not in Git).
- Laptop: Ryzen 5 8645HS, on charger.
- Impact times cover the whole `impact_projection` call, including assigning tets to pieces.

| Shape | Cage faces | d | Tets | Cage (s) | Modes (s) | Impact precompute (s) | Per shape total (s) | Impact (ms, median) | Peak RAM (MB) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Wall + window | 100 | 3 | 134 | 25.2 | 0.4 | 0.20 | 25.8 | 0.80 | 463 | window closed in cage; pieces median 8, max 8 of 8 |
| Wall + window | 1000 | 3 | 1619 | 19.2 | 2.4 | 0.64 | 22.2 | 4.57 | 463 | cage genus 1; pieces median 8, max 8 of 8 |
| Wall + window | 2000 | 3 | 3774 | 19.0 | 10.2 | 1.06 | 30.2 | 10.64 | 463 | cage genus 1; pieces median 5, max 5 of 5 |
| Wall + window | 5000 | 3 | 10261 | 20.9 | 22.1 | 2.52 | 45.5 | 36.84 | 736 | cage genus 1; pieces median 5, max 5 of 5 |
| Wall + window | 100 | 1 | 134 | 26.5 | 0.3 | 0.52 | 27.3 | 0.71 | 463 | window closed in cage; pieces median 39.5, max 41 of 42 |
| Wall + window | 1000 | 1 | 1619 | 19.4 | 3.8 | 1.42 | 24.6 | 2.25 | 463 | cage genus 1; pieces median 58, max 60 of 60 |
| Wall + window | 2000 | 1 | 3774 | 19.6 | 5.7 | 1.88 | 27.2 | 4.65 | 463 | cage genus 1; pieces median 58, max 58 of 58 |
| Wall + window | 5000 | 1 | 10261 | 21.7 | 18.1 | 3.06 | 43.0 | 16.04 | 463 | cage genus 1; pieces median 64, max 64 of 64 |

**What this means:**

- **Per shape: 22–45 s.** Building the cage alone takes 19–26 s at every size. A shape that appears during play (a fresh fragment, a procedural object) cannot be prepared live: that is 1,300–2,700 frames at 60 fps. RIFT's opening holds.
- **Per impact, timed in full: 5–37 ms for d=3**, growing with mesh size. At 10k tets that is more than two frames at 60 fps (16.7 ms per frame). This is Python; C++ would be faster, so we only claim that impacts are not free, not that fracture-modes is slow at runtime.
- **d=1 is barely cheaper than d=3** (43 s vs 46 s per shape at 5,000 faces). Cost does not force d=1, so direction-aware labels (d=3) stay affordable.
- **Memory peaks at 0.74 GB**, so 3–4 shapes can be generated in parallel.
- **A 100-face cage closes the window.** The minimum usable cage for this wall is about 1,000 faces.
- **Red flag: in every run, every impact broke the wall into the maximum number of pieces** (median = max). With the default threshold, where you hit doesn't matter. The labels would be identical for every impact and the network would learn nothing. Their dataset generator randomises the threshold for this reason. Day 4 must find thresholds where the hit location changes the result.
- With 10 modes, d=3 allows only 5–8 pieces in total. Richer breaks may need more modes (Day 4).

**Projected generation time:** ~30 s per shape (2,000 faces, d=3), so 300 shapes take about 2.5 CPU-hours on one laptop, or about 1 hour with 3–4 running in parallel (untested). Extra impacts on a shape cost ~11 ms each.

- **Correction (Day 4):** most of the 5–37 ms per impact is a visualisation-only linear solve inside `impact_projection`. The break decision itself takes 0.2–0.4 ms (Day 4). Once a shape is prepared, fracture-modes is fast per impact.

## Day 4 results: break threshold (TKR's laptop, 29 Oct)

**Setup:**
- Shape: the Day 3 wall.
- Hits: 100 random points on the whole surface, each aimed straight into it.
- Thresholds: about 40 per setting, spread over the range where anything changes.
- Teacher settings: the dataset generator's (cage grid 256, tetgen minratio 1.5).
- Script: `scripts/sweep_threshold.py`. Pictures and raw numbers: `outputs/day4/` (not in Git).
- Locality: distance from the hit to the broken-off parts, divided by the shape's average distance from the hit. Lower = breaks closer to the hit.

| Setting | Max pieces | Per shape (s) | Best threshold | Different outcomes (of 100 hits) | Partial | Intact | Shatter | Locality | At 0.02: outcomes / shatter | Decision / full call (ms) | Mismatches |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cage2000_m10_d3 | 5 | 36 | 25.9 | 18 | 96% | 0% | 4% | 0.37 | 1 / 100% | 0.33 / 12.4 | 0 |
| cage2000_m20_d3 | 17 | 41 | 40.8 | 63 | 100% | 0% | 0% | 0.26 | 4 / 97% | 0.22 / 10.8 | 0 |
| cage5000_m10_d3 | 5 | 44 | 223 | 21 | 97% | 3% | 0% | 0.34 | 1 / 100% | 0.40 / 36.0 | 0 |
| cage5000_m20_d3 | 30 | 83 | 157 | 81 | 100% | 0% | 0% | 0.25 | 4 / 97% | 0.40 / 35.9 | 0 |
| cage2000_m10_d1 | 58 | 26 | 6.11 | 98 | 100% | 0% | 0% | 0.37 | 37 / 52% | 0.21 / 4.5 | 0 |
| cage2000_m20_d1 | 317 | 40 | 35.8 | 100 | 100% | 0% | 0% | 0.12 | 98 / 1% | 0.25 / 4.7 | 0 |
| cage5000_m10_d1 | 64 | 42 | 21.2 | 96 | 100% | 0% | 0% | 0.35 | 10 / 90% | 0.30 / 13.1 | 0 |
| cage5000_m20_d1 | 416 | 87 | 120 | 100 | 100% | 0% | 0% | 0.12 | 83 / 17% | 0.39 / 16.1 | 0 |

**What this means:**

- **The Day 3 red flag is fixed.** At the default threshold (0.02), the d=3 settings give 1–4 different outcomes from 100 hits. With a tuned threshold they give 18–81, and nearly every hit is a partial break.
- **Our threshold shortcut is exact:** it matched fracture-modes' own result on all 800 checks (0 mismatches).
- **Per impact, fracture-modes is fast:** the break decision takes 0.2–0.4 ms. RIFT cannot beat it per impact. Its case rests only on the per-shape preparation: 26–87 s here.
- **Chosen for Day 5:** cage 5,000, 20 modes, d=3, threshold 157. That gives up to 30 pieces, 81 different outcomes, 5.7 fragments per hit on average, and locality 0.25. It takes 83 s per shape and 0.75 GB peak.
- **Modes matter more than cage size.** With 10 modes, d=3 allows only 5 pieces; 20 modes allow 17–30.
- **The threshold is not a constant.** The same wall needs about 41 at 2,000 faces but about 157 at 5,000. Thresholds must be calibrated per shape, which is cheap once the shape is prepared.
- **Use a band of thresholds, not one value.** The chosen 157 sits near the edge where breaks stop. A per-shape band could also become a "toughness" input to the network.
- **Open decision: d=3 and d=1 break differently.** d=3 splits the wall into 4–7 large slabs. d=1 punches a local crater of 6–20 small chunks (locality 0.12), which looks closer to game impact damage but ignores impact direction. Decide before data generation.

## Day 5 results: cells and link labels (TKR's laptop, 29 Oct)

**Setup:**
- Teacher: the Day 4 choice (cage 5,000, 20 modes, d=3, threshold 157).
- Voxels: 138,240 small cubes (0.00625 units, 8 across the wall's thickness).
- Cells: 128 Voronoi cells, spread evenly by 5 rounds of Lloyd relaxation.
- Hits: 50.
- Script: `scripts/cell_graph.py`. Outputs: `outputs/day5/` (not in Git).

| Setting | Cells | Links (per cell) | Teacher tets per cell | Hits with a break | Purity (all / breaking hits / worst) | Rebuilt | Lost volume | Fragments: teacher / cells | Links broken |
|---|---|---|---|---|---|---|---|---|---|
| cage5000_m20_d3, threshold 157 | 128 | 326 (5.1) | 100 | 50 of 50 | 96.2% / 96.2% / 94.7% | 94% | 0.14% | 5.3 / 4.8 | 11.2% |

**What this means:**

- **128 cells capture the teacher's breaks well.** 96.2% of the volume carries the right fragment label (worst hit 94.7%). In the pictures, the broken links follow the teacher's cracks.
- **Lost fragments are slivers.** 17 of 50 hits lose a fragment smaller than one cell, costing 0.14% of the volume on average and 1.4% at worst.
- **3 of 50 hits produce one extra island** of cells (rebuilt 94%).
- **The teacher is not the bottleneck:** there are about 100 teacher tets per cell, so labels are limited by cell size, not by the teacher's resolution.
- **About 11% of links break per hit.** The classes are unbalanced roughly 1 to 8, which the plan's weighted loss expects.
- **In each hit, the worst cell is split about half and half** between two fragments: a cell straddling a crack. Finer cells near cracks would help if needed.
- **Not yet tested:** the same check with d=1 craters, whose chunks may be smaller than a cell.

## Day 4b and 5b: slabs or craters (TKR's laptop, 29–30 Oct)

**Question:** which teacher setting makes the dataset? The settings break the same wall in visibly different ways. Same scripts and wall as Days 4 and 5; outputs in `outputs/day4b/`, `outputs/day5_d1/` and `outputs/day5_d1_256/`.

| Option | Setting | Looks like | Fragments per hit | Locality | Per shape (s) | Peak RAM (GB) | Cells: purity / lost volume / fragments teacher → cells |
|---|---|---|---|---|---|---|---|
| A: slabs | cage5000_m20_d3, threshold 157 | 4–7 large slabs | 5.7 | 0.25 | 83 | 0.75 | 128 cells: 96.2% / 0.14% / 5.3 → 4.8 |
| B: crater, direction ignored | cage5000_m20_d1, threshold 120 | Tight crater of small chunks | 15.3 | 0.12 | 87 | 0.55 | 128 cells: 96.2% / 1.57% / 14.1 → 6.0; 256 cells: 97.1% / 0.47% / 14.1 → 8.3 |
| C: crater (direction: see Day 6b) | cage5000_m40_d3, threshold 71.3 | Local crater of 11–32 chunks | 27.3 | 0.19 | 470 | 1.04 | 256 cells: 94.6% / 0.76% / 24.4 → 14.5 (rebuilt 58%, worst purity 90.8%) |

**What this means:**

- **More modes give the teacher more ways to break.** For d=3, maximum pieces rise from 30 (20 modes) to 291 (40 modes), and the cost per shape rises from 83 s to 470 s.
- **Option C strengthens RIFT's case:** about 8 minutes per shape, which no game can spend while running.
- **In crater-style breaks, most chunks are smaller than a cell.** Cells keep the volume right (purity 96–97%, lost volume 0.5–1.6%) but merge chunks: 14 teacher fragments become 6–8 cell fragments. In the game, debris can't be smaller than one cell.
- **So labels and scores must work at cell resolution.** Train on cell-level labels and judge with volume-based metrics (purity, detached-volume IoU, volume-weighted ARI). Measure fragment-count error against the cell-level labels, not the teacher's chunk count.
- **"Rebuilt" falls to 72–84% with craters:** majority labels split some fragments into separate islands of cells. In the game these become separate pieces of debris either way.
- **Team chose option C on 31 Oct** (`docs/decisions.md`, D1), partly because it was expected to respond to impact direction. Day 6b tested that.

## Day 6 and 6b: the paper, and impact direction (31 Oct)

**Paper pre-read** (`docs/paper-notes-fracture-modes.md`; pages from the arXiv version, to confirm):

- The authors did not simulate secondary fractures, and say computing new modes for each broken piece "would exceed realtime constraints" (Sec. 6). fracture-modes does not handle repeated impacts or damaged shapes. This is RIFT's opening.
- Their timings match ours: up to about 2 minutes per shape, and 1–2 ms per impact on their timer.
- The paper says its fracture modes ignore the direction of the break, and that the fault is the same regardless of impact direction (Sec. 6, Fig. 21).

**Direction check** (`scripts/direction_check.py`): option C, 50 hits, each from straight in, 45 degrees and 80 degrees. Outputs: `outputs/day6b/`.

| Projection | Threshold | Different outcomes (of 50 hits) | Partial | Locality | 45 deg: changed / detached IoU | 80 deg: changed / detached IoU | Mismatches |
|---|---|---|---|---|---|---|---|
| default | 84.9 | 50 | 100% | 0.17 | 0% / 1.00 | 0% / 1.00 | 0 |
| project_on_modes | 75.9 | 50 | 100% | 0.55 | 94% / 0.93 | 92% / 0.95 | 0 |

**What this means:**

- **With the default projection, direction has no effect at all.** No tilted hit changed the break, as the code predicted. Breaks stay local (locality 0.17, about 22 fragments per hit).
- **`project_on_modes` makes direction matter a little, but breaks the wall far from the hit.** Over 90% of tilted hits change, but 93–95% of the broken-off volume stays the same. In the pictures, one hit shatters most of the wall into 20–49 pieces (locality 0.55). That is not usable as a training target.
- **So this teacher cannot teach direction-dependent fracture.** RIFT should claim impact location and strength, not direction. Direction can still push the debris in the game's physics.
- **Option C's remaining advantage over B is the look:** about 22 vs 15 fragments per hit. It costs about 5 times more per shape, and many of its extra chunks are smaller than a cell anyway (24 → 14.5 at 256 cells). Decision D1 is under review (`docs/decisions.md`).

## Day 6c: second hit on a damaged wall (31 Oct)

**Question:** after a first hit knocks a crater out of the wall, can the teacher handle a second hit by reusing the intact wall's homework (seconds saved), or must it redo the homework for the damaged shape (minutes)? If reuse is nearly right, RQ3 (damaged shapes) is weak. If reuse is wrong, especially near the crater, RIFT has a real job.

**Method** (`scripts/second_hit.py`): build the teacher on the intact wall and calibrate its threshold. Pick 3 first hits that each knock out 2–20% of the wall. For each, rebuild the remaining wall as a closed mesh (about 16,000 triangles, about 1% volume lost), redo the homework on it and calibrate its own threshold. Hit it 40 times, half near the crater (< 0.15) and half far, with both reuse and redo. Compare the two breaks on the remaining wall's voxels.

- **Control:** the intact wall rebuilt through the same pipeline, with no crater. Reuse and redo can disagree here only because of the rebuild itself. Damage matters only as far as the crater rows fall below the control.
- **Fair to reuse:** after each break, both methods re-check which voxels still touch, so reuse cannot keep pieces joined through material that is already gone.
- **Same threshold column:** redo at the intact wall's threshold, to separate the effect of the new shape from the effect of a new threshold.

**Results, option B** (`outputs/day6c_B/`). Intact wall (56 triangles): homework 96 s, threshold 142.5. Craters: 4%, 4% and 6% of the wall.

| Second hits on | Hits | Redo homework (s) | Rebuilt threshold | ARI reuse vs redo | Same threshold | Detached IoU | Fragments reuse / redo |
|---|---|---|---|---|---|---|---|
| Control: intact wall, rebuilt | 40 | 307 | 129 | 0.63 | 0.62 | 0.52 | 9.0 / 14.1 |
| Damaged wall, near crater (< 0.15) | 60 | 219, 245, 234 | 64, 105, 210 | 0.41 | 0.49 | 0.34 | 7.4 / 16.2 |
| Damaged wall, far from crater | 60 | 219, 245, 234 | 64, 105, 210 | 0.44 | 0.57 | 0.37 | 9.8 / 18.0 |

Per crater (ARI near / far): crater 0 (threshold 64) 0.25 / 0.27, and redo breaks off 22–31% of the wall per hit against 4–7% for reuse; crater 1 (105) 0.38 / 0.46; crater 2 (210) 0.59 / 0.58.

**What this means:**

- **The test cannot answer its question yet.** Two effects are bigger than the damage itself.
- **The teacher is sensitive to the mesh.** Rebuilding the intact wall (same shape, 16,000 slightly rounded triangles instead of 56 flat ones) changes the break: ARI 0.63, and only half the broken-off volume overlaps. At the same threshold the result is the same (0.62), so the mesh causes it, not the threshold.
- **The Day 4 threshold rule gives different strength on near-identical shapes.** Rebuilt walls got thresholds from 64 to 210. At 64, redo breaks off a quarter of the wall per hit. Crater 0's low score comes from this, not from the crater.
- **No sign that the crater matters locally:** near and far hits score the same (0.41 vs 0.44). Where strength roughly matched (crater 2), reuse vs redo (0.58) is as good as the control (0.63). So far, small craters (4–6%) do not need new homework beyond the teacher's own mesh noise.
- **Realistic meshes cost more:** homework on the 16,000-triangle walls took 219–307 s against 96 s for the 56-triangle original. Game meshes have thousands of triangles, so option B may cost about 3 times the 87 s budgeted.
- **Option C run postponed:** it would repeat the same two problems. Next, Day 6d separates the causes of the mesh sensitivity and tests a strength rule that holds across shapes.

## Threats found (Day 6)

- (Found Day 1) After per-shape precomputation, fracture-modes handles each impact in ~1 ms. RIFT must justify itself on the precomputation cost (unseen, procedural and already-broken shapes), not on per-impact speed. Day 3 measures how that cost grows; Day 6 checks whether broken pieces can break again without recomputing their modes.
  - Update (Day 3): per-shape cost is 22–45 s; per-impact cost timed in full is 5–37 ms (d=3). The per-shape cost stays RIFT's main case.
  - Update (Day 4): the per-impact break decision itself is 0.2–0.4 ms, so this threat stands in full. RIFT's network will probably be slower per impact than fracture-modes. RIFT's only case is skipping the 26–87 s per-shape preparation.
- (Found Day 4) The right threshold depends on the shape's mesh (about 41 at 2,000 cage faces vs 157 at 5,000 for the same wall). Data generation must calibrate a threshold band per shape, or labels will be inconsistent across shapes.
- (Found Day 4) d=3 and d=1 give visibly different breaks: large slabs vs a local crater. This choice shapes the whole dataset and is not settled yet.
- (Found Day 4b) Debris size is capped by cell size: crater chunks smaller than a cell merge into cell-sized debris. Ground truth and fragment metrics must be defined at cell resolution, and the proposal must say so.
- (Found Day 3) With the default break threshold, every impact shatters the wall into the maximum number of pieces, so the labels don't depend on where you hit. Unless Day 4 finds thresholds that give varied outcomes, there is nothing for the network to learn.
- (Found Day 6) The teacher's breaks do not depend on impact direction (paper Sec. 6, Fig. 21; Day 6b: 0% of tilted hits changed). The only mode where direction matters shatters the wall far from the hit. RIFT cannot promise direction-dependent fracture; the README and proposal must say "impact location and strength".
- (Found Day 6c) Teacher labels depend on the mesh, not only the shape: a rebuilt copy of the same wall agrees at ARI 0.63. This is label noise. It caps how closely any student can match the teacher, so the evaluation must report teacher-vs-teacher agreement as the ceiling.
- (Found Day 6c) The Day 4 threshold rule (most different outcomes) gives different strength on near-identical shapes (thresholds 64–210). Strength must mean the same thing on every shape before bulk generation.
- (Found Day 6c) Homework on a 16,000-triangle wall takes 219–307 s for option B, about 3 times the 56-triangle wall. Dataset time estimates must use realistic meshes.
- (Found Day 6, in RIFT's favour) The paper's authors did not simulate secondary fractures and say per-piece modes would exceed realtime constraints. RQ3 (damaged shapes) is supported by the method's own limitations.

## Go / no-go (Day 7)

- Install works on both laptops from the environment file: yes / no
- Solve time per shape at usable quality: 41–83 s (cage 2,000–5,000, 20 modes, d=3; Day 4)
- If the solve is around a second, the argument shifts to damaged and procedural shapes: no, the solve is 26–87 s. But the per-impact decision is 0.2–0.4 ms, so RIFT's argument is the per-shape cost only
- Sentis model A loads: yes / no. Model B loads: yes / no
- Does fracture-modes already handle damaged shapes or repeated impacts? No. Secondary fractures were not simulated, and per-piece modes "would exceed realtime constraints" (paper Sec. 6; pages to confirm)

**Decision:** GO / NO-GO

**Reason:**
