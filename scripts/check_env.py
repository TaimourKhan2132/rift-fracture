"""Check that the rift-teacher environment can run fracture-modes.

Run from the repo root, inside the rift-teacher environment:

    python scripts/check_env.py

This script does not run fracture-modes itself. It makes the same library calls
fracture-modes makes (same functions, same arguments, same number of return
values), so a version mismatch shows up here as a clear PASS/FAIL line instead
of a confusing crash halfway through a mode solve.

Exit code 0 means every check passed.
"""

import argparse
import importlib
import os
import platform
import sys
import tempfile
import traceback
from pathlib import Path

# The exact fracture-modes commit both laptops must use (also recorded in README.md).
PINNED_COMMIT = "bdf5051fb0d78d787f49fabbf75b54bbc6698b17"

REPO_ROOT = Path(__file__).resolve().parent.parent
FM_DIR = REPO_ROOT / "third_party" / "fracture-modes"
DEFAULT_MESH = FM_DIR / "data" / "bunny_oded.obj"

results = []  # (name, passed, detail)


def check(name):
    """Decorator: run one check, record PASS/FAIL, never stop the whole script."""

    def wrap(fn):
        def run(*args, **kwargs):
            try:
                detail = fn(*args, **kwargs)
                results.append((name, True, detail or ""))
                print(f"[PASS] {name}" + (f"  ({detail})" if detail else ""))
                return True
            except Exception as e:  # report and keep going
                results.append((name, False, f"{type(e).__name__}: {e}"))
                print(f"[FAIL] {name}\n       {type(e).__name__}: {e}")
                if os.environ.get("RIFT_DEBUG"):
                    traceback.print_exc()
                return False

        return run

    return wrap


# ---------------------------------------------------------------- basics


@check("Python version is 3.9")
def check_python():
    v = sys.version_info
    if (v.major, v.minor) != (3, 9):
        raise RuntimeError(f"found {platform.python_version()}, expected 3.9.x")
    return f"{platform.python_version()} on {platform.system()} {platform.machine()}"


# Installed package names, where they differ from the import name.
DIST_NAMES = {
    "igl": ["libigl", "igl"],
    "sksparse": ["scikit-sparse"],
    "skimage": ["scikit-image"],
}


def module_version(module_name):
    top = module_name.split(".")[0]
    val = getattr(sys.modules.get(top), "__version__", None)
    if isinstance(val, str):
        return val
    from importlib.metadata import version
    for dist in DIST_NAMES.get(top, [top]):
        try:
            return version(dist)
        except Exception:
            pass
    return "version unknown"


def check_import(module_name):
    @check(f"import {module_name}")
    def _run():
        mod = importlib.import_module(module_name)
        if module_name == "mosek":
            return ".".join(str(x) for x in mod.Env.getversion())
        return module_version(module_name)

    return _run()


def git_head(repo):
    """Commit hash a clone is on, read straight from its .git folder (no git needed)."""
    git_dir = repo / ".git"
    head = (git_dir / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head  # detached HEAD: the file holds the hash itself
    ref = head[len("ref: "):]
    if (git_dir / ref).is_file():
        return (git_dir / ref).read_text().strip()
    for line in (git_dir / "packed-refs").read_text().splitlines():
        if line.endswith(" " + ref):
            return line.split()[0]
    raise RuntimeError(f"could not resolve {ref}")


@check("fracture-modes is cloned at the pinned commit")
def check_fracture_modes_clone():
    if not (FM_DIR / "fracture_utility").is_dir():
        raise FileNotFoundError(
            f"{FM_DIR} not found. Clone it with:\n"
            "       git clone --recursive https://github.com/sgsellan/fracture-modes.git third_party/fracture-modes"
        )
    head = git_head(FM_DIR)
    if head != PINNED_COMMIT:
        raise RuntimeError(f"HEAD is {head[:7]}, expected {PINNED_COMMIT[:7]}")
    return head[:7]


# ---------------------------------------------------------------- MOSEK


@check("MOSEK licence file exists")
def check_mosek_licence_file():
    env_path = os.environ.get("MOSEKLM_LICENSE_FILE")
    path = Path(env_path) if env_path else Path.home() / "mosek" / "mosek.lic"
    if not path.is_file():
        raise FileNotFoundError(f"no licence at {path}")
    return str(path)


@check("MOSEK solves a cone problem with the old API fracture-modes uses")
def check_mosek_solve():
    # Same API style as fracture_utility/conic_solve.py:
    # appendcone(conetype.quad, ...) and getxx(soltype.itr, xx).
    # Problem: minimise t  subject to  t >= sqrt(x1^2 + x2^2),  x1 = 1,  x2 = 1.
    # The answer is t = sqrt(2) = 1.41421...
    import mosek

    with mosek.Env() as env:
        with env.Task(0, 0) as task:
            task.appendvars(3)  # variables: t, x1, x2
            task.putvarboundlistconst([0, 1, 2], mosek.boundkey.fr, 0.0, 0.0)
            task.appendcons(2)
            task.putaijlist([0, 1], [1, 2], [1.0, 1.0])
            task.putconboundlistconst([0, 1], mosek.boundkey.fx, 1.0, 1.0)
            task.putclist([0], [1.0])
            task.appendcone(mosek.conetype.quad, 0.0, [0, 1, 2])
            task.putobjsense(mosek.objsense.minimize)
            task.optimize()
            xx = [0.0] * 3
            task.getxx(mosek.soltype.itr, xx)
    if abs(xx[0] - 2 ** 0.5) > 1e-5:
        raise RuntimeError(f"t = {xx[0]}, expected 1.41421")
    return f"t = {xx[0]:.5f}"


# ---------------------------------------------------------------- sparse maths


@check("scikit-sparse Cholesky (as in sparse_sqrt.py)")
def check_cholesky():
    import numpy as np
    from scipy.sparse import csc_matrix, diags
    from sksparse.cholmod import cholesky

    A = diags([4.0, 9.0, 16.0]).tocsc()
    decomp = cholesky(csc_matrix(A), beta=1e-12, ordering_method="natural")
    L, D = decomp.L_D()
    R = D.sqrt() @ L.T
    if not np.allclose((R.T @ R).toarray(), A.toarray()):
        raise RuntimeError("R.T @ R != A")


# ---------------------------------------------------------------- mesh pipeline


@check("load mesh, build cage, tetrahedralize (as in example.py)")
def check_mesh_pipeline(mesh_path, state):
    import igl
    import gpytoolbox
    import tetgen
    from gpytoolbox.copyleft import lazy_cage

    v_fine, f_fine = igl.read_triangle_mesh(str(mesh_path))
    v_fine = gpytoolbox.normalize_points(v_fine)
    v, f = lazy_cage(v_fine, f_fine, num_faces=100)
    nodes, elements = tetgen.TetGen(v, f).tetrahedralize()
    state.update(v_fine=v_fine, f_fine=f_fine, cage_v=v, cage_f=f,
                 nodes=nodes, elements=elements)
    return (f"fine mesh {f_fine.shape[0]} faces -> cage {f.shape[0]} faces "
            f"-> {elements.shape[0]} tets")


@check("igl calls on the tet mesh (cotmatrix, volume, adjacency, boundary)")
def check_igl_tets(state):
    import igl
    import numpy as np

    V, T = state["nodes"], state["elements"]
    L = igl.cotmatrix(V, T)
    if L.shape != (V.shape[0], V.shape[0]):
        raise RuntimeError(f"cotmatrix shape {L.shape}")
    vol = igl.volume(V, T)
    if vol.shape[0] != T.shape[0]:
        raise RuntimeError(f"volume returned shape {vol.shape}")
    TT, TTi = igl.tet_tet_adjacency(T)  # must return exactly 2 arrays
    F = igl.boundary_facets(T)
    # Keep the faces for the next check even if the return type is wrong,
    # so one failure does not hide the others.
    state["boundary_F"] = F[0] if isinstance(F, tuple) else F
    if not isinstance(F, np.ndarray) or F.ndim != 2 or F.shape[1] != 3:
        raise RuntimeError(f"boundary_facets returned {type(F).__name__}, "
                           "fracture-modes expects one (#F, 3) array")
    return f"{F.shape[0]} boundary faces"


@check("igl calls on triangle meshes (cleanup, area, sampling, write)")
def check_igl_tris(state, tmpdir):
    import igl
    import numpy as np

    V, F = state["nodes"], state["boundary_F"]
    vi, fi = igl.remove_unreferenced(V, F)[:2]
    ui, I, J, _ = igl.remove_duplicate_vertices(vi, fi, 1e-10)  # must return 4
    gi = J[fi]
    areas = igl.doublearea(ui, gi)
    if not np.all(areas >= 0):
        raise RuntimeError("negative areas")
    # generate_fractures.py expects exactly 2 outputs, B with 3 columns
    B, FI = igl.random_points_on_mesh(10, ui, gi)
    if B.shape != (10, 3):
        raise RuntimeError(f"random_points_on_mesh B shape {B.shape}, expected (10, 3)")
    out = Path(tmpdir) / "piece.obj"
    igl.write_obj(str(out), ui, gi)
    if not out.is_file():
        raise RuntimeError("write_obj wrote nothing")


@check("gpytoolbox mesh_boolean (as in impact_precomputation)")
def check_mesh_boolean(state):
    import numpy as np
    from gpytoolbox.copyleft import mesh_boolean

    vf, ff = state["v_fine"], state["f_fine"]
    vc, fc = state["cage_v"], state["cage_f"]
    vo, fo = mesh_boolean(vf, ff.astype(np.int32), vc, fc.astype(np.int32),
                          boolean_type="intersection")
    if fo.shape[0] == 0:
        raise RuntimeError("intersection is empty")
    return f"{fo.shape[0]} faces"


# ---------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH,
                        help="triangle mesh to test with (default: their bunny)")
    args = parser.parse_args()

    print("RIFT teacher environment check\n")
    check_python()
    for name in ["numpy", "scipy", "igl", "tetgen", "sksparse.cholmod", "mosek",
                 "gpytoolbox", "gpytoolbox.copyleft", "polyscope", "skimage", "tqdm"]:
        check_import(name)
    check_fracture_modes_clone()
    check_mosek_licence_file()
    check_mosek_solve()
    check_cholesky()

    state = {}
    if check_mesh_pipeline(args.mesh, state):
        check_igl_tets(state)
        if "boundary_F" in state:
            with tempfile.TemporaryDirectory() as tmp:
                check_igl_tris(state, tmp)
        check_mesh_boolean(state)
    else:
        print("       (skipping mesh checks that need the tetrahedralized mesh)")

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed.")
    if failed:
        print("Failed:", ", ".join(r[0] for r in failed))
        print("Re-run with RIFT_DEBUG=1 set for full tracebacks.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
