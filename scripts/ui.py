"""Readable terminal output and live progress for RIFT scripts.

A long script wraps its work in one Run:

    with ui.Run("Day 6c: second hit", out_dir, steps=9) as run:
        with run.step("Build the teacher on the intact wall"):
            ...
        for hit in run.track(hits, "Second hits"):
            ...
        run.table("Results", ["Second hits on", "Hits", "ARI"], rows)
        run.log_markdown(markdown)   # saved as <out_dir>/log_table.md, ready for the log

The terminal shows each step with a timer, and a progress bar with time left for
loops. The same progress goes to outputs/dashboard.html (scripts/dashboard.py), a
page that refreshes itself every few seconds while a run is going. Tables are drawn
with aligned columns instead of raw Markdown.

Uses the `rich` package when it is installed. Without it the same information is
printed as plain text, so a missing package never stops a run.
"""

import contextlib
import json
import sys
import time
import warnings
from pathlib import Path

try:
    from rich.console import Console
    from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                               TimeElapsedColumn)
    from rich.table import Table
    from rich.text import Text
    HAVE_RICH = True
except ImportError:  # plain-text fallback
    HAVE_RICH = False

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = REPO_ROOT / "outputs"
LIVE_FILE = OUTPUTS / "live.json"   # state of the current run, read by the dashboard
RUNS_FILE = OUTPUTS / "runs.jsonl"  # one line per finished run
WRITE_EVERY_S = 2.0                 # how often loops may refresh the dashboard

_current = None  # the active Run, so shared helpers (build_teacher) can report sub-steps


def substep(name):
    """Name the part of the current step that is running now (no-op outside a Run)."""
    if _current is not None:
        _current.set_substep(name)


def fmt_time(seconds):
    """3 s, 2 min 05 s, 1 h 04 min."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds // 60} min {seconds % 60:02d} s"
    return f"{seconds // 3600} h {seconds % 3600 // 60:02d} min"


def _looks_numeric(text):
    t = str(text).strip().replace(",", "").replace("%", "").replace("–", "-")
    for part in t.replace(" / ", "/").split("/"):
        try:
            float(part.split()[0] if part.split() else "x")
        except ValueError:
            return False
    return bool(t)


if HAVE_RICH:
    class _BarIfCounted(BarColumn):
        """A progress bar for loops; nothing for steps, whose length is unknown."""

        def render(self, task):
            return super().render(task) if task.total is not None else Text("")


class Run:
    """One script run: steps, loops, tables, and live progress for the dashboard."""

    def __init__(self, title, out_dir=None, steps=None):
        self.title = title
        self.out_dir = Path(out_dir) if out_dir else None
        self.total_steps = steps
        self.started = time.time()
        self.step_no = 0
        self.step_name = None
        self.step_started = None
        self.sub = None
        self.loop = None
        self.done_steps = []
        self.status = "running"
        self.error = None
        self.ended = None
        self._last_write = 0.0
        self._write_failed = False
        self._progress = None
        self._step_task = None
        self.console = Console(highlight=False) if HAVE_RICH else None

    # ------------------------------------------------------------ start and end

    def __enter__(self):
        global _current
        _current = self
        try:  # scipy's format warnings from fracture-modes are noise, not problems
            from scipy.sparse import SparseEfficiencyWarning
            warnings.simplefilter("ignore", SparseEfficiencyWarning)
        except ImportError:
            pass
        if HAVE_RICH:
            self.console.rule(f"[bold]{self.title}")
            self._progress = Progress(SpinnerColumn(), TextColumn("{task.description}"),
                                      _BarIfCounted(bar_width=28), TextColumn("{task.fields[count]}"),
                                      TimeElapsedColumn(), TextColumn("[dim]{task.fields[eta]}"),
                                      console=self.console, transient=True)
            self._progress.start()
        else:
            print(f"==== {self.title} ====", flush=True)
        self._write(force=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        global _current
        if exc_type is None:
            self.status = "finished"
        elif issubclass(exc_type, KeyboardInterrupt):
            self.status = "stopped"
        else:
            self.status = "failed"
            self.error = f"{exc_type.__name__}: {exc}"
        self.ended = time.time()
        self.step_name, self.sub, self.loop = None, None, None
        if self._progress is not None:
            self._progress.stop()
        took = fmt_time(self.ended - self.started)
        if HAVE_RICH:
            style = {"finished": "green", "stopped": "yellow", "failed": "red"}[self.status]
            self.console.rule(f"[{style}]{self.status.capitalize()} in {took}")
        else:
            print(f"==== {self.status.capitalize()} in {took} ====", flush=True)
        self._write(force=True)
        self._record()
        _current = None
        return False  # never hide an error

    # ------------------------------------------------------------ steps and loops

    @contextlib.contextmanager
    def step(self, name):
        """A named stage of the run, shown with a running timer."""
        self.step_no += 1
        self.step_name, self.sub, self.step_started = name, None, time.time()
        label = self._step_label()
        if HAVE_RICH:
            self._step_task = self._progress.add_task(label, total=None, count="", eta="")
        else:
            print(f"{label} ...", flush=True)
        self._write(force=True)
        try:
            yield
        finally:
            seconds = time.time() - self.step_started
            if HAVE_RICH and self._step_task is not None:
                self._progress.remove_task(self._step_task)
                self._step_task = None
            self.done_steps.append({"name": name, "seconds": seconds})
            self.say(f"[green]✓[/] {label}  [dim]{fmt_time(seconds)}[/]",
                     f"  done: {name} ({fmt_time(seconds)})")
            self.step_name, self.sub = None, None
            self._write(force=True)

    def set_substep(self, name):
        self.sub = name
        if HAVE_RICH and self._step_task is not None:
            self._progress.update(self._step_task, description=f"{self._step_label()}  [dim]{name}")
        elif not HAVE_RICH:
            print(f"    {name} ...", flush=True)
        self._write(force=True)

    def track(self, items, name, total=None):
        """Loop over items with a progress bar, counts and time left."""
        items = list(items) if total is None and not hasattr(items, "__len__") else items
        total = len(items) if total is None else total
        started = time.time()
        self.loop = {"name": name, "done": 0, "total": total, "started": started}
        task = None
        if HAVE_RICH:
            task = self._progress.add_task(f"  {name}", total=total, count=f"0/{total}", eta="")
        last_print = 0
        try:
            for i, item in enumerate(items):
                yield item
                done = i + 1
                elapsed = time.time() - started
                left = elapsed / done * (total - done)
                self.loop["done"] = done
                if HAVE_RICH:
                    self._progress.update(task, completed=done, count=f"{done}/{total}",
                                          eta=f"about {fmt_time(left)} left" if done < total else "")
                elif done == total or done - last_print >= max(1, total // 10):
                    last_print = done
                    print(f"    {name}: {done}/{total} ({done / total:.0%}), "
                          f"about {fmt_time(left)} left", flush=True)
                self._write()
        finally:
            if task is not None:
                self._progress.remove_task(task)
            done = self.loop["done"] if self.loop else 0
            self.say(f"  [green]✓[/] {name}: {done} of {total}  "
                     f"[dim]{fmt_time(time.time() - started)}[/]",
                     f"    {name}: {done} of {total} done ({fmt_time(time.time() - started)})")
            self.loop = None
            self._write(force=True)

    # ------------------------------------------------------------ output

    def say(self, rich_text, plain_text=None):
        """Print one line above the progress display."""
        if HAVE_RICH:
            self.console.print(rich_text)
        else:
            print(plain_text if plain_text is not None else rich_text, flush=True)

    def note(self, text):
        """A quiet detail line (no markup)."""
        if HAVE_RICH:
            self.console.print(Text("    " + text, style="dim"))
        else:
            print("    " + text, flush=True)

    def table(self, title, columns, rows):
        """Print a table with aligned columns; numbers are right-aligned."""
        rows = [[str(c) for c in r] for r in rows]
        numeric = [all(_looks_numeric(r[j]) for r in rows) and rows != [] for j in range(len(columns))]
        if HAVE_RICH:
            t = Table(title=title, title_style="bold", header_style="bold", show_lines=False)
            for name, num in zip(columns, numeric):
                t.add_column(name, justify="right" if num else "left", overflow="fold")
            for r in rows:
                t.add_row(*r)
            self.console.print(t)
            return
        widths = [max(len(str(c)), *(len(r[j]) for r in rows)) for j, c in enumerate(columns)]
        line = lambda cells: "  ".join(c.rjust(w) if num else c.ljust(w)
                                       for c, w, num in zip(cells, widths, numeric))
        print(f"\n{title}\n{line([str(c) for c in columns])}\n" + "  ".join("-" * w for w in widths))
        for r in rows:
            print(line(r))
        print(flush=True)

    def log_markdown(self, text, name="log_table.md"):
        """Save the Markdown meant for docs/kill-test-log.md instead of printing it raw."""
        if self.out_dir is None:
            return None
        path = self.out_dir / name
        path.write_text(text.strip() + "\n", encoding="utf-8")
        self.say(f"[dim]Table for docs/kill-test-log.md saved to[/] {path}",
                 f"Table for docs/kill-test-log.md saved to {path}")
        return path

    # ------------------------------------------------------------ dashboard

    def _step_label(self):
        of = f"/{self.total_steps}" if self.total_steps else ""
        return f"[{self.step_no}{of}] {self.step_name}"

    def state(self):
        return {"title": self.title, "script": Path(sys.argv[0]).name, "args": sys.argv[1:],
                "out_dir": self._rel(self.out_dir), "started": self.started, "ended": self.ended,
                "updated": time.time(), "status": self.status, "error": self.error,
                "step_no": self.step_no, "total_steps": self.total_steps,
                "step_name": self.step_name, "step_started": self.step_started, "sub": self.sub,
                "loop": self.loop, "done_steps": self.done_steps}

    def _rel(self, path):
        if path is None:
            return None
        try:
            return str(Path(path).resolve().relative_to(OUTPUTS.resolve())).replace("\\", "/")
        except ValueError:
            return str(path)

    def _write(self, force=False):
        """Save live state and rebuild the dashboard, at most every few seconds."""
        now = time.time()
        if not force and now - self._last_write < WRITE_EVERY_S:
            return
        self._last_write = now
        try:
            OUTPUTS.mkdir(parents=True, exist_ok=True)
            LIVE_FILE.write_text(json.dumps(self.state(), indent=1), encoding="utf-8")
            import dashboard  # sibling module in scripts/
            dashboard.build()
        except Exception as err:  # the dashboard must never stop a run; warn once, keep trying
            if not self._write_failed:
                self._write_failed = True
                self.say(f"[yellow]Dashboard not updated ({err}); the run continues.",
                         f"Dashboard not updated ({err}); the run continues.")

    def _record(self):
        try:
            with open(RUNS_FILE, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({k: self.state()[k] for k in (
                    "title", "script", "args", "out_dir", "started", "ended", "status", "error")}) + "\n")
            import dashboard
            dashboard.build()
        except Exception:
            pass
