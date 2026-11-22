# Running the dataset generator (dataset v2)

How to make the RIFT-Arch training samples with `scripts/make_dataset.py`: start the run, pause it, back it up, and let it continue by itself after a crash or a restart. Read it once fully before the first long run.

## What a run makes

For every shape from `scripts/make_shapes.py`: the teacher's homework, 100 calibration hits that set six break thresholds, and 50 labelled hits on 256 cells, each labelled at all six strengths (2, 4, 6, 8, 10 and 12% broken off).

Dataset v2 settings (`settings()` in `make_dataset.py`, stored in every sample):

| Setting | Value | From |
|---|---|---|
| Teacher | option B: cage 5,000, 20 modes, d=1, labels only | D1 |
| Hit push | mass push (by material, not by mesh corner) | D5 |
| Strength rule | the **average** calibration hit breaks off the target share | D3 amendment (8 Nov) |
| Version | `DATASET_VERSION = 2` | v1 used the median rule |

Samples go to `data/rift-arch/samples_v2/` (one `.npz` and one `.json` per shape ID), pictures to `outputs/dataset/pictures/`. v1 samples in `data/rift-arch/samples/` are never read or overwritten; delete them or keep them for comparison. The generator refuses to start if `samples_v2/` holds samples made with any other setting, so two versions can never mix.

**Time:** about 83 s of real time per shape with two shapes at once on TKR's laptop (data log, "Two shapes at once"), so 2,000 shapes take about **46 hours**. Peak memory is about 1.5 GB per shape.

## 1. Before the first run

Do these once, in **Miniforge Prompt**, from the repo root.

1. **Get the new pipeline:**
   ```
   git fetch origin
   git switch data/dataset-v2
   git pull
   ```
   Once the pull request is merged, use `git switch main` and `git pull` instead.
2. **Check the environment** (must report all checks passed; it also checks the MOSEK licence):
   ```
   conda activate rift-teacher
   python scripts\check_env.py
   ```
3. **Make all 2,000 shapes** (about 70 minutes; shapes already made are skipped, so IDs 0–99 are not redone):
   ```
   python scripts\make_shapes.py --ids 0-1999
   ```
   The generator skips IDs that have no finished shape and lists them at the start.
4. **Set up the laptop for two days of work:**
   - Plugged in, OMEN performance mode, on a cooling pad.
   - **Settings → System → Power:** when plugged in, *Sleep: Never*.
   - **Control Panel → Power Options → Choose what closing the lid does:** *Plugged in: Do nothing*. The generator stops Windows from idle-sleeping, but closing the lid still sleeps the laptop.
   - **Settings → Windows Update → Pause updates** for 1–2 weeks, so an update restart cannot interrupt the run.
   - Close VS Code, browsers and other heavy programs.
   - Free disk space: at least 5 GB (samples are about 45 KB each, pictures more).
5. **Choose a backup folder** (recommended): a USB drive, or a Google Drive for desktop folder such as `G:\My Drive\RIFT\backup_v2`. Open `scripts\run_dataset.bat` in a text editor and set the line
   ```
   set "BACKUP=G:\My Drive\RIFT\backup_v2"
   ```
   Every finished sample is then copied there as soon as it is made.

## 2. Pilot first: shapes 0–19

Check that v2's labels look right before spending 46 hours. The 20 IDs hold every family, including the chunky pillars and arch (#6, #9, #14, #18) that the mean rule was made for.

```
python scripts\make_dataset.py --ids 0-19 --workers 2
```

About 30 minutes. Then check:

- The summary table: no failures; "Outcomes at 6%" well above 1 for every family.
- `outputs\dataset\pictures\00006.png`, `00009.png`, `00014.png`, `00018.png`: the strength levels should now differ (in v1 two levels were identical on #9 and #14).
- Add the summary (`outputs\dataset\log_table.md`) to `docs/data-log.md`.

These 20 samples are kept: the full run skips them.

## 3. Start the full run

Double-click **`scripts\run_dataset.bat`**, or run it from any window:

```
scripts\run_dataset.bat
```

With no options it runs `make_dataset.py --ids 0-1999 --workers 2` (plus `--backup-dir` if you set `BACKUP`). It activates `rift-teacher` itself, so no Miniforge Prompt is needed. Any `make_dataset.py` options can be passed instead, for example `scripts\run_dataset.bat --ids 0-1999 --every 2 --offset 0` for the even IDs only.

If Miniforge is not in `%USERPROFILE%\miniforge3` or `C:\dev\miniforge3`, first run `set CONDA_ROOT=<your Miniforge folder>` in the same window.

What the window shows: one line per finished shape (time, peak memory, pieces, distinct outcomes at 6%), a progress bar with time left, and a summary table at the end. `python scripts\dashboard.py --open` shows the same progress in a browser page.

### Continue by itself after a restart

`run_dataset.bat` already restarts the generator after a crash (up to 20 times, a minute apart). To also continue after the laptop restarts:

1. Press **Win+R**, type `shell:startup`, press Enter.
2. Right-click in that folder → **New → Shortcut** → browse to `scripts\run_dataset.bat` in the repo → Finish.

After any restart, once you sign in to Windows, the run starts again and skips everything already made. Delete the shortcut when the dataset is finished.

## 4. Pause, resume, stop, check

From **another** window (Miniforge Prompt, `rift-teacher` active, repo root):

| Command | What happens |
|---|---|
| `python scripts\dataset_control.py status` | Shapes finished and failed, what is running now, the last sign of life, time left |
| `python scripts\dataset_control.py pause` | No new shape starts; the running shapes finish; the run waits (and lets the laptop sleep once idle) |
| `python scripts\dataset_control.py resume` | New shapes start again within a second |
| `python scripts\dataset_control.py stop` | The running shapes finish, then the run ends; `run_dataset.bat` does not restart it |

Pause and stop never cut off a running shape, so no work is lost. A pause survives a restart: a run started while paused waits until you resume. Starting the run again always clears an old stop.

**Ctrl+C** in the run's window does the same as `stop`. Pressing **Ctrl+C a second time** stops at once; the shapes that were running are then simply made again next time (about 2 minutes each). If Windows asks "Terminate batch job (Y/N)?", either answer is fine.

**Closing the window** also stops at once and loses only the running shapes. Use `pause` if you need the laptop's full speed for a while.

## 5. If something goes wrong

| Problem | What the pipeline does | What you do |
|---|---|---|
| Power cut, crash, blue screen | Every sample is written as a temporary file, then swapped in, so a cut never leaves a half sample. Half-written temporary files are removed at the next start. Only the shapes that were running are lost | Start `run_dataset.bat` again (automatic with the Startup shortcut) |
| Generator crashes | `run_dataset.bat` starts it again after 60 s, at most 20 times | Read the error above the restart line if it keeps happening |
| Ctrl+C or closing the window | The running shapes leave no record, so they are made again next run. In v1 they were recorded as failed and then skipped for good | Nothing |
| A helper process crashes or runs out of memory | Recorded as failed with `retry: true`, and tried once more automatically on the next run | If it fails twice, read the error in its `.json` |
| A shape fails with a Python error, or passes the 30-minute limit | Recorded as failed and skipped from then on (the error and traceback are in its `.json`) | At the end, run once more with `--retry-failed --timeout 3600` |
| Five shapes fail in a row (MOSEK licence expired, broken environment) | The run stops instead of marking every remaining shape failed. `run_dataset.bat` does not restart it | Fix the cause (`python scripts\check_env.py`), then start again with `--retry-failed` |
| The laptop slept anyway (lid closed, battery low) | The run freezes and continues on wake. The time asleep does not count toward a shape's 30-minute limit | Fix the power settings (step 1.4) |
| A sample file is damaged later (disk error, accidental edit) | At every start each finished `.npz` is checked against the checksum saved when it was made; a mismatch is remade | Nothing |
| Samples from another version or setting in `samples_v2/` | The generator refuses to start and lists the IDs | Move the wrong files away; test runs go elsewhere with `--samples-dir` |
| The disk dies or the folder is deleted | With `BACKUP` set, every sample is also in the backup folder; anything missed is copied at the next start | Copy the backup folder's files back into `data\rift-arch\samples_v2\`, then start again; they are checked and skipped |
| Laptop too hot or slow | Nothing automatic | Check HWMonitor; pause during the hottest hours if needed; never put the laptop on a bed or sofa |
| The window seems frozen | Big shapes (L-corners, round pillars) can take 3–4 minutes each | `dataset_control.py status` shows the last sign of life; under 2 minutes ago means it is working |

## 6. When the run ends

1. Read the summary table and its note: samples made, failures, time per shape, peak memory.
2. If shapes failed: `python scripts\make_dataset.py --ids 0-1999 --workers 2 --retry-failed --timeout 3600`.
3. Copy the summary (`outputs\dataset\log_table.md`) and the totals into `docs/data-log.md`, on a branch, through a pull request.
4. Share `data\rift-arch\samples_v2\` and `data\rift-arch\shapes.jsonl` through Google Drive (never Git).
5. Delete the Startup shortcut.

## Splitting the run between two laptops (later)

Both laptops need the same shapes first: compare their `data\rift-arch\shapes.jsonl` (the `md5` of every ID must match). Then one laptop runs `scripts\run_dataset.bat --ids 0-1999 --workers 2 --every 2 --offset 0` and the other `... --offset 1`, and the two `samples_v2` folders are merged by copying (file names never clash, and every sample records its own settings).
