"""Pause, resume, stop or check a running make_dataset.py from another window.

The generator watches two files in its samples folder (see make_dataset.Control):
  PAUSE   no new shape starts; running shapes finish; the run waits until it is removed
  STOP    no new shape starts; running shapes finish; then the run ends for good
Neither cuts off a running shape, so no work is lost. The files survive a restart: a
run started (or restarted by run_dataset.bat) while PAUSE exists stays paused.

Usage (repo root, rift-teacher active):
    python scripts/dataset_control.py status
    python scripts/dataset_control.py pause
    python scripts/dataset_control.py resume
    python scripts/dataset_control.py stop
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_dataset import DATA_DIR, MAX_ATTEMPTS, SAMPLES  # noqa: E402


def ago(seconds):
    seconds = int(max(seconds, 0))
    if seconds < 120:
        return f"{seconds} s ago"
    if seconds < 7200:
        return f"{seconds // 60} min ago"
    return f"{seconds / 3600:.1f} hours ago"


def status(samples):
    status_file = samples / "_status.json"
    try:
        st = json.loads(status_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st = None
    ok = failed = retry = 0
    seconds = []
    for path in samples.glob("[0-9]*.json"):
        if ".tmp" in path.name:
            continue
        try:
            r = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if r.get("status") == "ok":
            ok += 1
            seconds.append(r.get("seconds", 0))
        elif r.get("retry") and r.get("attempts", 1) < MAX_ATTEMPTS:
            retry += 1
        else:
            failed += 1
    print(f"Samples folder: {samples}")
    print(f"  finished: {ok}   failed for good: {failed}   to be tried again: {retry}")
    if (samples / "STOP").exists():
        print("  STOP file present: the run ends once its running shapes finish.")
    if (samples / "PAUSE").exists():
        print("  PAUSE file present: no new shape starts (resume with: dataset_control.py resume).")
    if st is None:
        print("  No run has reported in yet.")
        return
    age = time.time() - st.get("updated", 0)
    state = st.get("state", "?")
    if state in ("running", "paused", "stopping") and age > 120:
        state += f" (but no sign of life for {ago(age)[:-4]}: the run has probably ended or the laptop slept)"
    print(f"  run: {state}, last report {ago(age)}, IDs {st.get('ids')}"
          + (f" every {st.get('every')} from {st.get('offset')}" if st.get("every", 1) > 1 else ""))
    if st.get("running"):
        print("  working on: " + ", ".join(f"#{i:05d}" for i in st["running"]))
    if st.get("stop_reason"):
        print(f"  stop reason: {st['stop_reason']}")
    made, to_make = st.get("made_this_run", 0), st.get("to_make", 0)
    print(f"  this run: {made} made, {st.get('failed_this_run', 0)} failed, of {to_make} to make")
    started = st.get("started")
    done = made + st.get("failed_this_run", 0)
    if started and done and to_make > done:
        pace = (st.get("updated", time.time()) - started) / done
        print(f"  real time per shape so far: {pace:.0f} s, so about {(to_make - done) * pace / 3600:.1f} "
              f"hours left (pauses included)")


def main():
    p = argparse.ArgumentParser(description="Pause, resume, stop or check make_dataset.py.")
    p.add_argument("command", choices=["status", "pause", "resume", "stop"])
    p.add_argument("--samples-dir", type=Path, default=DATA_DIR / SAMPLES,
                   help=f"the run's samples folder (default {DATA_DIR / SAMPLES})")
    args = p.parse_args()
    samples = args.samples_dir
    if not samples.exists():
        sys.exit(f"{samples} does not exist: run this from the repo root, or pass --samples-dir.")
    if args.command == "pause":
        (samples / "PAUSE").write_text("paused by dataset_control.py\n")
        print("Paused: running shapes finish, no new one starts. Resume with: "
              "python scripts/dataset_control.py resume")
    elif args.command == "resume":
        (samples / "PAUSE").unlink(missing_ok=True)
        print("Resumed: the run starts new shapes again within a second.")
    elif args.command == "stop":
        (samples / "STOP").write_text("stop requested by dataset_control.py\n")
        print("Stopping: running shapes finish, then the run ends (run_dataset.bat does not restart it). "
              "Start the same command again later to continue where it stopped.")
    else:
        status(samples)


if __name__ == "__main__":
    main()
