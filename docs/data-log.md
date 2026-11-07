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

**On TKR's laptop:** to fill in (IDs 0–99).

**Open before bulk generation:**

- Homework time against triangle count. Walls and L-corners reach 20,000–45,000 triangles, against 14,000–16,000 in the Day 6 timings. If homework grows a lot with triangles, raise the edge to 0.02: about half the triangles, but the thinnest part becomes 0.06 of the longest side. Measured in the dataset pilot.
- Which families are held out for testing (the proposal splits by family, for example all arches). To decide before training.
- Shapes are axis-aligned with y up. The teacher sees one orientation per shape, so rotations belong in training-time augmentation, not in the dataset.
