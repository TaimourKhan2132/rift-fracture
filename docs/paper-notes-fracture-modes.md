# Paper notes: fracture modes ("Breaking Good")

Sellán, Luong, Mattos Da Silva, Ramakrishnan, Yang, Jacobson. *Breaking Good: Fracture Modes for Realtime Destruction.* ACM Transactions on Graphics / SIGGRAPH Asia 2022.
[arXiv 2111.05249](https://arxiv.org/abs/2111.05249) · [project page](https://www.dgp.toronto.edu/projects/breaking-good/) · [code](https://github.com/sgsellan/fracture-modes)

**Status:** pre-read notes from the arXiv version, 31 Oct 2026. Page numbers may be off by one, and quotes came from a pre-read. Check every page and every quote against the PDF during the Day 6 reading before using them in the proposal.

## Read these first (about 45 minutes)

| Where | What to look for | Why it matters for RIFT |
|---|---|---|
| Sec. 3.3, p. 6, Fig. 10 | What the break threshold (sigma) means | Our per-shape threshold calibration (Day 4) |
| Sec. 4 and Table 1, pp. 7–8 | Precomputation and per-impact timings | RIFT's whole speed argument |
| Sec. 6, pp. 9–10, Figs. 20–21 | Limitations: secondary fractures, direction, global modes | RQ3 (damaged shapes) and the direction claim |

## Findings

1. **They do not re-break pieces.**
   - "Secondary fractures were not included in our simulations." (Sec. 6, p. 10)
   - "Computing a new set of fracture modes for each piece would exceed realtime constraints." (Sec. 6, p. 10)
   - Reusing the parent shape's modes on a piece: "there is no guarantee that these would match the individual piece's fracture modes." (Sec. 6, p. 10)
   - **For RIFT:** this is the authors' own limitation and exactly RIFT's opening. Go/no-go answer: fracture-modes does not handle repeated impacts or damaged shapes. RQ3 stands.

2. **Their timings match ours.**
   - Table 1 (p. 7): 0.63–11.6 s per mode, 10–30 modes, 3,545–12,162 tets. That is roughly 6 s to 2 minutes per shape.
   - Impact projection: 1–2 ms on their timer (Sec. 4, p. 8).
   - Ours (kill-test log): 41–470 s per shape with dataset-quality cages; break decision 0.2–0.4 ms.
   - **For RIFT:** no speed claim per impact. The claim is skipping the per-shape step.

3. **The threshold is impact strength.**
   - Neighbouring pieces stay glued while their displacement difference is below sigma (Sec. 3.3, p. 6).
   - Scaling sigma is equivalent to scaling the impact magnitude in their model (p. 6). A larger threshold gives fewer pieces (Fig. 10).
   - The paper used sigma = 0.001; the code's default is 0.02. Day 4 showed the right value depends on the shape's mesh.
   - **For RIFT:** feed hit strength to the network as an input, calibrated per shape.

4. **Impact direction barely matters in their method.**
   - "Our fracture mode computation considers only the magnitude of the vector-valued discontinuity and disregards its direction." (Sec. 6, p. 10)
   - Fig. 21: the fault is the same regardless of impact direction. "Our method belongs to the class of prefracture, not dynamic algorithms." (p. 10)
   - Code check (`fracture_modes.py`, `impact_precomputation` and `impact_projection`): with the default projection (`project_on_modes=False`), every axis is processed identically. A unit impact direction therefore has no effect on which pieces break.
   - **For RIFT:** do not claim direction-awareness until Day 6b (`scripts/direction_check.py`) shows the `project_on_modes` projection makes direction matter without ruining locality.

5. **Modes are global.**
   - The modes link distant regions of the object, so a hit can break something far away (Sec. 6, p. 9).
   - Their fallback is a least-squares constant-per-piece projection (Fig. 20), which is what the code does by default.
   - **For RIFT:** our locality metric tracks this.

6. **Other limits RIFT inherits from its teacher.**
   - No partial cracks: pieces either separate fully or not at all (Sec. 6, p. 10).
   - Visible artefacts on coarse simulation meshes (Sec. 6, p. 9). This supports our 5,000-face cage.
   - "For very large meshes, the projection step could exceed CPU usage allowances for realtime applications." (Sec. 6, p. 9)
