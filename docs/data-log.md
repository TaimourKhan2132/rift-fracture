# Data log (phase 1)

Measurements and findings while building the RIFT-Arch dataset, oldest first. Week-1 results are in `kill-test-log.md`; decisions are in `decisions.md`.

## Shape generator v1 (7 Nov)

**What it makes:** `scripts/make_shapes.py`. Seven families, in a fixed cycle of 20 IDs: 5 walls, 3 L-corners, 3 arches, 3 slabs, 2 beams, 2 square pillars and 2 round pillars. Any 20 consecutive IDs hold the same mix, so the 20-shape pilot already covers every family. The ID alone decides the shape: the family comes from the cycle, the sizes from a seed made of the master seed (2026), the ID and the attempt number.

Variety in 2,000 IDs (first draw of each, from the sizes alone):

| Family | Shapes | Variants |
|---|---|---|
| wall | 500 | no openings 111, 1 window 122, 2 windows 99, 3 windows 46, a door plus 0–2 windows 122 |
| L-corner | 300 | 0–2 windows, one arm each |
| arch | 300 | semicircular 131, pointed 89, segmental 80; flat top 159, ring top 141 |
| slab | 300 | 0 holes 107, 1 hole 132, 2 holes 61 (square or round) |
| beam | 200 | rectangle 113, T 58, I 29 |
| square pillar | 200 | plain, base, capital, or both (about a quarter each, over both pillar families) |
| round pillar | 200 | as square pillars; base and capital square or round |

**How a shape is built:**

1. Sizes are drawn in metres with y up. fracture-modes scales every shape so its longest side is 1, so only proportions reach the teacher. Thin parts (wall thickness, piers, lintels, flanges) are therefore at least 0.045 of the longest side, which is 3 triangle edges.
2. The solid is a distance formula (negative inside): boxes, cylinders and arch outlines joined with min and cut with max.
3. Its surface is extracted on a grid 0.0075 apart (marching cubes), then re-meshed to triangle edges of 0.015 (Botsch–Kobbelt re-meshing, gpytoolbox). Every shape, and later every damaged shape, reaches the teacher with the same triangle density (Day 6d).
4. Checks: closed, one piece, the expected number of holes (windows and slab holes; a door is a notch, not a hole), faces pointing out, smallest triangle angle at least 5 degrees, volume within -2.5% to +1% of the extracted surface, and an inside/outside test at 20,000 random points: points where the mesh and the formula disagree must be at most 4% of the points inside. A shape that fails is drawn again with new sizes, up to 3 times. Every failed draw is recorded in `shapes.jsonl` with its reason.
5. Each draw runs in a helper process with a 180 s limit, so one stuck shape cannot stall a night run.

**Found while building it (cloud test, same package versions as `rift-teacher`):**

- Re-meshing the output of mesh booleans directly ate 33% of a thin wall's volume: the re-mesher handles long thin triangles badly. Splitting those triangles first kept 99.1%.
- The mesh booleans themselves left zero-area triangles where a round pillar meets its base (456 of 784 triangles had an angle under 1 degree), and the re-mesher then hung for over 10 minutes. The generator therefore uses distance formulas instead of booleans. This also gives exact circles instead of 48-sided ones.
- A simplified copy of each mesh, used to draw previews faster, made flat faces look bumpy. Previews now draw the full mesh. The real surface is within 0.2 mm of the formula on average for a 6 m wall; edges are rounded by up to 9 mm.

**Cloud test, IDs 0–59:** 60 of 60 made on the first draw, none failed. About 2 s per shape, so 2,000 shapes take about 70 minutes plus about 17 minutes of preview pictures. Triangles: median 14,300, range 3,700–44,800 (L-corners largest, round pillars smallest). Worst kept shape: volume 0.981 of the extracted surface, inside test off by 2.7%, smallest angle 21 degrees. The same ID made twice gives a byte-identical file (20 of 20). A draw stopped by the time limit is recorded as failed, and the next shape still works.

**On TKR's laptop (Windows), IDs 0–99:** 100 of 100 made on the first draw, none failed. 1.4 s per shape; 3.5 minutes including the pictures. Triangles: median 14,200, range 3,100–46,100.

- Closest to a limit: square pillar #66 (volume 0.977 of the extracted surface, limit 0.975; inside test off by 3.5%, limit 4%) and thin rectangular beams (volume 0.981–0.986). Thin straight shapes lose the most to edge rounding, so expect a few redraws among them in the full run.
- Same shapes as the cloud test: identical triangles for all 60 shared IDs, and vertex positions equal to within 0.00000000000004 m. The file checksums still differ, because Windows writes different line endings and the last digit can round differently. Compare checksums only between the two Windows laptops.

**Open before bulk generation:**

- Homework time against triangle count. Walls and L-corners reach 20,000–45,000 triangles, against 14,000–16,000 in the Day 6 timings. If homework grows a lot with triangles, raise the edge to 0.02: about half the triangles, but the thinnest part becomes 0.06 of the longest side. Measured in the dataset pilot.
- Which families are held out for testing (the proposal splits by family, for example all arches). To decide before training.
- Shapes are axis-aligned with y up. The teacher sees one orientation per shape, so rotations belong in training-time augmentation, not in the dataset.

## Dataset generator v1 (7 Nov)

**What it makes:** `scripts/make_dataset.py`. For each shape, one sample file (`data/rift-arch/samples/<id>.npz`, about 45 KB) and one record (`<id>.json`: settings, timings, quality numbers). Each shape runs in its own helper process with a 30-minute limit, and finished shapes are skipped on the next run.

| Step | Setting | From |
|---|---|---|
| Teacher | option B, labels only, mass push | D1, D5 (proposed) |
| Cells | 256 Voronoi cells on voxels 0.0075 apart (6 across the thinnest part), 5 Lloyd rounds | Day 5, with a fixed voxel size (below) |
| Strength | 100 calibration hits set one threshold per level: 2, 4, 6, 8, 10, 12% median broken-off share | D3 |
| Hits | 50 random surface points, each labelled at all 6 levels | the push is computed once per hit |
| Labels | fragment per cell (0 = main body), broken or not per link | groups of cells joined by unbroken links |

**Found while building it:**

- Day 5's `voxelize` took the voxel size from the thinnest side of the bounding box. That works for a wall, but an L-corner's box is thick while its walls are thin: it got 183 voxels in total, under one across a wall. Voxels are now a fixed 0.0075 in the unit box, 6 across the thinnest part any shape may have.
- In an arch or an L-corner, a Voronoi cell can take voxels across a gap that do not touch the rest of the cell. Such stray pieces now join the neighbouring cell they touch most, so every cell is one connected chunk. Checked on a made-up case; none of the first four test shapes needed it.
- Cell labels follow the links. Each cell first takes the teacher fragment holding most of its volume; fragments are then the groups of cells still joined by unbroken links. Cell and link labels therefore always agree, which is also how Unity will turn RIFT's link predictions into pieces.
- The cells are stored as a block of voxels (cell number per voxel, -1 outside): 35 KB instead of 400 KB as a list.

**Cloud test (Linux), IDs 0–3.** MOSEK has no licence in the cloud, so fracture-modes' conic solve was swapped for an equivalent linear program (exact for d=1) in a test-only script. The teacher also ran with a 1,000-face cage and 6 modes to save time. These numbers check the pipeline, not label quality.

- 4 of 4 samples made, 66–130 s per shape. Outside the teacher's homework: cells 0.7–2.5 s, calibration 5.4–6.2 s, labels 1.6–2.5 s, picture about 1 s.
- Purity 98.4–99.1%, rebuilt 98–100%, lost volume under 0.05%.
- A link broken at one strength stayed broken at every higher strength in 98.4–100% of cases. The rest is a cell switching to another fragment when more pieces break.
- With so few pieces (15–42), some strength levels shared a threshold. Option B's few hundred pieces should separate them; the pilot checks this.
- Also checked: a run restarts where it stopped, two shapes run at once, a shape over its time limit is stopped and recorded, and a solver error inside a helper is recorded with its traceback.

**Pilot on TKR's laptop (IDs 0–19, option B):** to fill in. To check: homework time against triangle count, whether the six strengths give different breaks (broken off at 2 / 6 / 12%), variety (distinct outcomes at 6%), purity, and the pictures.
