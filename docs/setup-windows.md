# Windows setup: the `rift-teacher` environment

This environment runs fracture-modes, the teacher that generates training labels.

**Status:**
- Dry run on Linux, Python 3.9, 26 Oct 2026: every check passes except the two MOSEK checks, which need a licence.
- Windows confirmation: Day 1 of the kill test.

## 0. Before you start

- Git for Windows installed.
- Miniforge installed ("Just Me", not added to PATH).
- MOSEK personal academic licence saved at `%USERPROFILE%\mosek\mosek.lic`. Request it with your university email: https://www.mosek.com/products/academic-licenses/

## 1. Get fracture-modes

From the repo root, in any terminal:

```
git clone --recursive https://github.com/sgsellan/fracture-modes.git third_party\fracture-modes
```

The pinned commit is `bdf5051` (the latest on their `main`, 20 Dec 2022). `scripts/check_env.py` confirms it.

## 2. Create the environment

Open **Miniforge Prompt** and `cd` to the repo root. Then run these one at a time:

```
conda create -n rift-teacher -c conda-forge python=3.9 "numpy<2" scipy scikit-sparse=0.4.16
conda activate rift-teacher
conda install -c mosek mosek=11.2
python -m pip install "numpy<2" libigl==2.4.1 gpytoolbox==0.0.1 tetgen==0.6.4 polyscope==2.5.0 scikit-image==0.24.0 tqdm
```

- Conda goes first and pip goes last: conda can silently overwrite packages pip installed.
- `"numpy<2"` appears in the pip line too, so pip can never upgrade it.
- tetgen pulls in pyvista and VTK, so the pip step downloads a few hundred MB.

## 3. Check it

Still in Miniforge Prompt, with `rift-teacher` active, from the repo root:

```
python scripts\check_env.py
```

Every line must say `[PASS]`. Then run their example:

```
cd third_party\fracture-modes
python scripts\example.py
```

Pass: it ends with `Example ran successfully!`. Their README says about 15 seconds; note that the example uses a 2,000-face cage. Record the time in `docs/kill-test-log.md`.

## Why these versions

fracture-modes was written in 2022. Newer releases of some libraries changed their behaviour, so we pin versions from that era.

| Package | Version | Why |
|---|---|---|
| Python | 3.9 | The version the authors used. Several 2022-era pins have no builds for newer Pythons. |
| numpy | below 2 | gpytoolbox 0.0.1 is compiled for NumPy 1.x. |
| scikit-sparse | 0.4.16, from conda-forge | Wraps the CHOLMOD C library. `pip install` tries to compile it and fails without `cholmod.h` and a C compiler. Conda ships it pre-built. |
| mosek | 11.2, from the mosek channel | Their solver code uses the old `appendcone` API. MOSEK 11 marks it deprecated but still supports it. Check again before any upgrade. |
| libigl | 2.4.1 | 2.6.x: `boundary_facets` returns three values instead of one, which crashes fracture-modes. 2.5.x: `random_points_on_mesh` returns three values instead of two, which breaks their dataset generator. |
| gpytoolbox | 0.0.1 | The version fracture-modes imports (`lazy_cage`, `mesh_boolean`). |
| polyscope | 2.5.0 | 2.6.x uses Python 3.10+ syntax and crashes on import under 3.9. fracture-modes imports polyscope as soon as it loads. |
| tetgen | 0.6.4 | Tested version. `tetrahedralize()` returns two values, as their code expects. |
| scikit-image | 0.24.0 | Last release for Python 3.9. |
| tqdm | any | Used by their dataset generator but missing from their `requirements.txt`. |

## If something fails

- **`conda` is not recognised:** you are in a normal terminal. Use Miniforge Prompt.
- **`DLL load failed` when importing `sksparse` or `mosek`:** the environment is not active. Run `conda activate rift-teacher` first.
- **MOSEK error 1008 (licence cannot be located):** the file is not at `%USERPROFILE%\mosek\mosek.lic`, or it was saved as `mosek.lic.txt`. Check with `dir %USERPROFILE%\mosek`.
- **Anything else:** run `set RIFT_DEBUG=1` then `python scripts\check_env.py` again for full error details, and log it under "Problems hit" in the kill-test log.
