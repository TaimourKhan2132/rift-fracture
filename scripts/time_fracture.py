"""Day 3 timing test: how fracture-modes' cost grows with cage size.

For each (cage size, mode dimension) pair, a fresh child process runs the whole
per-shape pipeline on one mesh and reports timings and peak memory:

  per shape (offline, done once):
    1. lazy_cage              coarse cage around the mesh
    2. tetgen                 fill the cage with tetrahedra
    3. compute_modes          the fracture modes (the MOSEK solves)
    4. impact_precomputation  per-piece data for fast impacts
  per impact (runtime):
    5. impact_projection      decide which pieces break, for N impacts

Each run gets its own process so its peak memory is measured on its own.

Settings match fracture-modes' dataset generator (generate_fractures.py):
lazy_cage grid_size=256, tetgen minratio=1.5. Two mode dimensions are timed:
  d=3  modes respond to impact direction (what their example uses)
  d=1  one value per tet, direction ignored (what their Breaking Bad dataset used)

Usage (Miniforge Prompt, rift-teacher active, repo root):

    python scripts/make_wall.py
    python scripts/time_fracture.py

Outputs (outputs/ is gitignored):
    outputs/day3/timing.csv                 one row per run
    outputs/day3/cage<N>_d<D>_impact0.obj   broken wall from the first impact
and a Markdown table to paste into docs/kill-test-log.md.
"""

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FM_DIR = REPO_ROOT / "third_party" / "fracture-modes"
RESULT_TAG = "RIFT_RESULT "

CSV_FIELDS = [
    "cage_target", "d", "modes", "status", "cage_faces", "cage_genus", "nodes", "tets",
    "t_cage_s", "t_tetgen_s", "t_modes_s", "t_precompute_s", "t_per_shape_s",
    "impact_ms_median", "impact_ms_max", "pieces_median", "pieces_max", "max_pieces",
    "peak_ram_mb", "error",
]


# ---------------------------------------------------------------- helpers


def peak_memory_mb():
    """Peak memory this process has used so far, in MB (works without psutil)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = wintypes.HANDLE
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
        if not get_info(get_process(), ctypes.byref(counters), counters.cb):
            return float("nan")
        return counters.PeakWorkingSetSize / 2**20
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux, bytes on macOS
    return peak / 2**20 if sys.platform == "darwin" else peak / 1024


def genus(F):
    """Holes through a closed triangle mesh (Euler: V - E + F = 2 - 2g, E = 3F/2)."""
    import numpy as np
    euler = np.unique(F).size - F.shape[0] * 3 // 2 + F.shape[0]
    return (2 - euler) // 2


# ---------------------------------------------------------------- one run (child process)


def run_one(args):
    """Full pipeline for one cage size and one mode dimension. Returns a result dict."""
    import numpy as np
    sys.path.insert(0, str(args.fm_dir))
    import igl
    import gpytoolbox
    import tetgen
    from gpytoolbox.copyleft import lazy_cage
    import fracture_utility as fracture

    r = {"cage_target": args.cage, "d": args.d, "modes": args.modes}
    v_fine, f_fine = igl.read_triangle_mesh(str(args.mesh))
    v_fine = gpytoolbox.normalize_points(v_fine)  # same scaling as fracture-modes

    t = time.perf_counter()
    v, f = lazy_cage(v_fine, f_fine, num_faces=args.cage, grid_size=args.grid_size)
    r["t_cage_s"] = time.perf_counter() - t
    r["cage_faces"], r["cage_genus"] = int(f.shape[0]), int(genus(f))

    t = time.perf_counter()
    nodes, elements = tetgen.TetGen(v, f).tetrahedralize(minratio=args.minratio)
    r["t_tetgen_s"] = time.perf_counter() - t
    r["nodes"], r["tets"] = int(nodes.shape[0]), int(elements.shape[0])

    modes = fracture.fracture_modes(nodes, elements)
    params = fracture.fracture_modes_parameters(num_modes=args.modes, verbose=False, d=args.d)
    t = time.perf_counter()
    modes.compute_modes(parameters=params)
    r["t_modes_s"] = time.perf_counter() - t

    t = time.perf_counter()
    modes.impact_precomputation(v_fine=v_fine, f_fine=f_fine)
    r["t_precompute_s"] = time.perf_counter() - t
    r["max_pieces"] = int(modes.precomputed_num_pieces)
    r["t_per_shape_s"] = r["t_cage_s"] + r["t_tetgen_s"] + r["t_modes_s"] + r["t_precompute_s"]

    # Impacts: contact points are tet nodes on the side facing the hit, picked
    # with a fixed seed. fracture-modes applies the hit to nodes within 0.05 units.
    direction3 = np.asarray(args.direction, dtype=float)
    direction3 /= np.linalg.norm(direction3)
    facing = -(nodes @ direction3)  # larger = closer to the side being hit
    candidates = np.flatnonzero(facing >= facing.max() - 0.01)
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(candidates, size=args.impacts, replace=len(candidates) < args.impacts)
    direction = direction3 if args.d == 3 else np.array([1.0])

    times_ms, pieces = [], []
    for i, node in enumerate(picks):
        t = time.perf_counter()
        modes.impact_projection(contact_point=nodes[node], direction=direction)
        times_ms.append((time.perf_counter() - t) * 1000)
        pieces.append(int(modes.n_pieces_after_impact))
        if i == 0:  # save one broken wall per run to look at on Day 4
            out = args.out_dir / f"cage{args.cage}_d{args.d}_impact0.obj"
            modes.write_segmented_output(filename=str(out))

    r["impact_ms_median"] = statistics.median(times_ms)
    r["impact_ms_max"] = max(times_ms)
    r["pieces_median"] = statistics.median(pieces)
    r["pieces_max"] = max(pieces)
    try:
        r["peak_ram_mb"] = peak_memory_mb()
    except Exception:  # never lose a finished run over the memory reading
        r["peak_ram_mb"] = float("nan")
    r["status"] = "ok"
    return r


# ---------------------------------------------------------------- driver (parent process)


def markdown_row(r):
    if r["status"] != "ok":
        return (f"| Wall + window | {r['cage_target']} | {r['d']} | | | | | | | | "
                f"{r['status']}: {r.get('error', '')[:80]} |")
    note = "window closed in cage" if r["cage_genus"] == 0 else f"cage genus {r['cage_genus']}"
    return (f"| Wall + window | {r['cage_faces']} | {r['d']} | {r['tets']} | "
            f"{r['t_cage_s']:.1f} | {r['t_modes_s']:.1f} | {r['t_precompute_s']:.2f} | "
            f"{r['t_per_shape_s']:.1f} | {r['impact_ms_median']:.2f} | {r['peak_ram_mb']:.0f} | "
            f"{note}; pieces median {r['pieces_median']:g}, max {r['pieces_max']} of {r['max_pieces']} |")


def main():
    p = argparse.ArgumentParser(description="Time fracture-modes at several cage sizes.")
    p.add_argument("--mesh", type=Path, default=Path("outputs/day3/wall.obj"))
    p.add_argument("--cage-sizes", type=int, nargs="+", default=[100, 1000, 2000, 5000])
    p.add_argument("--dims", type=int, nargs="+", default=[3, 1], choices=[1, 3])
    p.add_argument("--modes", type=int, default=10)
    p.add_argument("--impacts", type=int, default=20)
    p.add_argument("--direction", type=float, nargs=3, default=[0.0, 0.0, -1.0],
                   help="impact direction; default hits the wall's front face")
    p.add_argument("--grid-size", type=int, default=256)
    p.add_argument("--minratio", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=int, default=3600, help="seconds allowed per run")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/day3"))
    p.add_argument("--fm-dir", type=Path, default=DEFAULT_FM_DIR)
    # Internal: used when this script launches itself for a single run.
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--cage", type=int, help=argparse.SUPPRESS)
    p.add_argument("--d", type=int, help=argparse.SUPPRESS)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.child:
        print(RESULT_TAG + json.dumps(run_one(args)), flush=True)
        return

    if not args.mesh.is_file():
        raise SystemExit(f"{args.mesh} not found. Run: python scripts/make_wall.py")

    results = []
    for d in args.dims:
        for cage in args.cage_sizes:
            label = f"cage {cage}, d={d}"
            print(f"Running {label} ...", flush=True)
            cmd = [sys.executable, __file__, "--child", "--cage", str(cage), "--d", str(d),
                   "--mesh", str(args.mesh), "--modes", str(args.modes),
                   "--impacts", str(args.impacts), "--direction", *map(str, args.direction),
                   "--grid-size", str(args.grid_size), "--minratio", str(args.minratio),
                   "--seed", str(args.seed), "--out-dir", str(args.out_dir),
                   "--fm-dir", str(args.fm_dir)]
            t = time.perf_counter()
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
                lines = [l for l in proc.stdout.splitlines() if l.startswith(RESULT_TAG)]
                if proc.returncode == 0 and lines:
                    r = json.loads(lines[-1][len(RESULT_TAG):])
                else:
                    tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["no output"]
                    r = {"cage_target": cage, "d": d, "modes": args.modes,
                         "status": "failed", "error": tail[0]}
                    print(proc.stderr[-3000:], file=sys.stderr)
            except subprocess.TimeoutExpired:
                r = {"cage_target": cage, "d": d, "modes": args.modes,
                     "status": "timeout", "error": f"over {args.timeout} s"}
            results.append(r)
            if r["status"] == "ok":
                print(f"  done in {time.perf_counter() - t:.0f} s: {r['tets']} tets, "
                      f"per-shape {r['t_per_shape_s']:.1f} s, impact {r['impact_ms_median']:.2f} ms, "
                      f"peak RAM {r['peak_ram_mb']:.0f} MB", flush=True)
            else:
                print(f"  {r['status'].upper()}: {r['error']}", flush=True)

    csv_path = args.out_dir / "timing.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {csv_path}\n\nPaste into docs/kill-test-log.md (Day 3 table):\n")
    print("| Shape | Cage faces | d | Tets | Cage (s) | Modes (s) | Impact precompute (s) "
          "| Per shape total (s) | Impact (ms, median) | Peak RAM (MB) | Notes |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        print(markdown_row(r))
    sys.exit(0 if all(r["status"] == "ok" for r in results) else 1)


if __name__ == "__main__":
    main()
