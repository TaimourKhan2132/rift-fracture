"""Local results page: outputs/dashboard.html (open it in any browser).

Shows, in one page:
  - the run in progress: its current step, a timer, and a progress bar with time left;
  - recent runs: what ran, how long it took, whether it finished;
  - every results folder in outputs/: the table saved for the kill-test log, the
    summary as readable tables, small CSV files, and the pictures.

Scripts that use ui.Run rebuild it as they go, and the page refreshes itself every
3 seconds while a run is going. Rebuild it by hand any time:

    python scripts/dashboard.py          (add --open to open it in the browser)

Reads only files in outputs/; needs nothing outside the Python standard library.
"""

import argparse
import csv
import html
import json
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = REPO_ROOT / "outputs"
PAGE = OUTPUTS / "dashboard.html"
LIVE_FILE = OUTPUTS / "live.json"
RUNS_FILE = OUTPUTS / "runs.jsonl"
STALE_AFTER_S = 45 * 60  # a "running" state this old means the window was closed mid-run
MAX_CSV_ROWS = 30         # longer CSV files are linked, not drawn

# Plain-language titles for the folders we know. New folders show their own name.
KNOWN = {
    "day3": ("Day 3: timing the teacher", "Homework and per-hit time at four cage sizes, on the test wall."),
    "day4": ("Day 4: break-threshold sweep", "Which threshold gives varied, local breaks; options at cage 2,000 and 5,000."),
    "day4b": ("Day 4b: option C sweep", "Threshold sweep for the crater setting (40 modes, d=3)."),
    "day5": ("Day 5: cell labels, option A, 128 cells", "How well 128 Voronoi cells carry the teacher's breaks."),
    "day5_d1": ("Day 5b: cell labels, option B, 128 cells", "Same check for the crater setting with 20 modes, d=1."),
    "day5_d1_256": ("Day 5b: cell labels, option B, 256 cells", "Option B with 256 cells."),
    "day5_m40_256": ("Day 5b: cell labels, option C, 256 cells", "Option C with 256 cells."),
    "day6b": ("Day 6b: does hit direction change the break?", "Each hit straight in, tilted 45 and 80 degrees, for both projections."),
    "day6c_B": ("Day 6c: second hit (first run)", "Reuse vs redo homework on a damaged wall; Day 4 strength rule, which spoiled the comparison."),
    "day6c_B_target": ("Day 6c: second hit (matched strength)", "Reuse vs redo homework on a damaged wall, each wall set to 5% median damage."),
    "day6d_B": ("Day 6d: same wall, different triangles", "How much the teacher's answer depends on the triangle layout."),
    "day6c_B_target_mass": ("Day 6c: second hit (matched strength, mass push)", "Reuse vs redo homework on a damaged wall, with the mass-weighted push of Day 6f."),
    "day6f_B": ("Day 6f: corners push vs mass push", "Does pushing by material instead of by mesh corner make the teacher less dependent on the triangles?"),
    "day6e_B": ("Day 6e: labels without the display mesh", "Same labels without the slow display-mesh step? Plus the repeatability check."),
    "shapes": ("Shapes: the RIFT-Arch generator", "The dataset's base shapes: the first four of every family. Every shape is in outputs/shapes/sheets/, 20 per picture."),
    "sentis_test": ("Sentis test networks", "The two small networks for the Unity test (dense vs scatter graph layer)."),
}


# ---------------------------------------------------------------- formatting


def esc(x):
    return html.escape(str(x), quote=True)


def fmt_time(seconds):
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds // 60} min {seconds % 60:02d} s"
    return f"{seconds // 3600} h {seconds % 3600 // 60:02d} min"


def when(epoch):
    return datetime.fromtimestamp(epoch).strftime("%a %d %b, %H:%M")


PERCENT_WORDS = ("share", "damage", "frac", "purity", "not_in_a_tet", "changed", "lost")


def fmt_value(v, key=""):
    """Readable value; fractions whose name says they are shares show as percentages."""
    words = str(key).lower().split("_")
    if (isinstance(v, float) and 0 <= v <= 1 and any(w in str(key).lower() for w in PERCENT_WORDS)
            and "iou" not in words and "ari" not in words):  # scores, not shares
        return f"{v:.2%}" if 0 < v < 0.01 else f"{v:.1%}"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if v != v:
            return "n/a"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.3g}"
    if isinstance(v, list) and all(not isinstance(x, (dict, list)) for x in v):
        return ", ".join(fmt_value(x) for x in v)
    return str(v)


def humanize(key):
    words = str(key).replace("_", " ").split()
    words = [{"ari": "ARI", "iou": "IoU", "s": "(s)", "ms": "(ms)", "mb": "(MB)"}.get(w.lower(), w)
             for w in words]
    text = " ".join(words)
    return text[:1].upper() + text[1:]


def is_num(text):
    try:
        float(str(text).replace(",", "").replace("%", "").split()[0])
        return True
    except (ValueError, IndexError):
        return False


def table_html(columns, rows, caption=None):
    numeric = [bool(rows) and all(is_num(r[j]) for r in rows if j < len(r) and r[j] != "")
               for j in range(len(columns))]
    head = "".join(f"<th class='{'num' if n else ''}'>{esc(c)}</th>" for c, n in zip(columns, numeric))
    body = "".join("<tr>" + "".join(f"<td class='{'num' if n else ''}'>{esc(c)}</td>"
                                    for c, n in zip(r, numeric)) + "</tr>" for r in rows)
    cap = f"<caption>{esc(caption)}</caption>" if caption else ""
    return f"<div class='scroll'><table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


# ---------------------------------------------------------------- reading results


def json_tables(data, caption=None, depth=0):
    """A summary.json as tables: plain values as a two-column table, groups as their own tables."""
    if isinstance(data, list):
        if data and all(isinstance(x, dict) for x in data):
            cols = []
            for row in data:
                cols += [k for k in row if k not in cols and not isinstance(row[k], (dict, list))]
            rows = [[fmt_value(row.get(k, ""), k) for k in cols] for row in data]
            return table_html([humanize(c) for c in cols], rows, caption)
        return f"<p><b>{esc(caption)}:</b> {esc(fmt_value(data))}</p>" if caption else ""
    if not isinstance(data, dict):
        return ""
    plain = [(k, v) for k, v in data.items()
             if not isinstance(v, (dict, list)) or (isinstance(v, list) and all(not isinstance(x, (dict, list)) for x in v))]
    parts = []
    if plain:
        parts.append(table_html(["What", "Value"], [[humanize(k), fmt_value(v, k)] for k, v in plain], caption))
    for k, v in data.items():
        if isinstance(v, dict) and depth < 2:
            inner = list(v.values())
            if inner and all(isinstance(x, dict) for x in inner):  # a group of similar records
                cols = []
                for row in inner:
                    cols += [c for c in row if c not in cols and not isinstance(row[c], (dict, list))]
                rows = [[humanize(name)] + [fmt_value(row.get(c, ""), c) for c in cols] for name, row in v.items()]
                parts.append(table_html([""] + [humanize(c) for c in cols], rows, humanize(k)))
            else:
                parts.append(json_tables(v, humanize(k), depth + 1))
        elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            parts.append(json_tables(v, humanize(k), depth + 1))
    return "".join(parts)


def markdown_html(text):
    """Enough Markdown for our log tables: pipe tables, paragraphs, **bold** and `code`."""
    out, table = [], []

    def inline(s):
        s = esc(s)
        while "**" in s and s.count("**") >= 2:
            s = s.replace("**", "<b>", 1).replace("**", "</b>", 1)
        while s.count("`") >= 2:
            s = s.replace("`", "<code>", 1).replace("`", "</code>", 1)
        return s

    def flush():
        if len(table) >= 2:
            cells = [[c.strip() for c in row.strip().strip("|").split("|")] for row in table]
            head, rows = cells[0], [r for r in cells[2:]]
            numeric = [bool(rows) and all(is_num(r[j]) for r in rows if j < len(r)) for j in range(len(head))]
            h = "".join(f"<th class='{'num' if n else ''}'>{inline(c)}</th>" for c, n in zip(head, numeric))
            b = "".join("<tr>" + "".join(f"<td class='{'num' if n else ''}'>{inline(c)}</td>"
                                         for c, n in zip(r, numeric)) + "</tr>" for r in rows)
            out.append(f"<div class='scroll'><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>")
        table.clear()

    for line in text.splitlines():
        if line.strip().startswith("|"):
            table.append(line)
            continue
        flush()
        if line.strip():
            out.append(f"<p>{inline(line.strip())}</p>")
    flush()
    return "".join(out)


def csv_html(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return ""
    head, body = rows[0], rows[1:]
    link = f"<a href='{esc(quote(path.relative_to(OUTPUTS).as_posix()))}'>{esc(path.name)}</a>"
    if len(body) > MAX_CSV_ROWS:
        return f"<p class='quiet'>{link}: {len(body):,} rows, one per hit (open it in Excel).</p>"
    rows = [[fmt_value(float(c)) if is_num(c) and "." in c else c for c in r] for r in body]
    return f"<p class='quiet'>{link}</p>" + table_html([humanize(h) for h in head], rows)


def folder_card(folder):
    title, blurb = KNOWN.get(folder.name, (folder.name, ""))
    files = sorted(p for p in folder.iterdir() if p.is_file())
    changed = max((p.stat().st_mtime for p in files), default=folder.stat().st_mtime)
    parts = [f"<section class='card' id='{esc(folder.name)}'><h2>{esc(title)}</h2>"
             f"<p class='quiet'>{esc(blurb)} <span class='tag'>outputs/{esc(folder.name)}</span> "
             f"<span class='tag'>updated {esc(when(changed))}</span></p>"]
    log = folder / "log_table.md"
    if log.exists():
        parts.append("<h3>Results</h3>" + markdown_html(log.read_text(encoding="utf-8")))
    for name in ("summary.json", "chosen.json"):
        p = folder / name
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                parts.append(f"<details {'open' if not log.exists() else ''}><summary>{esc(name)}</summary>"
                             f"{json_tables(data)}</details>")
            except (ValueError, OSError) as err:
                parts.append(f"<p class='quiet'>{esc(name)} could not be read ({esc(err)}).</p>")
    for p in files:
        if p.suffix == ".csv":
            parts.append(csv_html(p))
        elif p.suffix == ".txt" and p.stat().st_size < 4000:
            parts.append(f"<p class='quiet'>{esc(p.name)}</p><pre>{esc(p.read_text(encoding='utf-8', errors='replace'))}</pre>")
    pngs = [p for p in files if p.suffix.lower() == ".png"]
    if pngs:
        figs = "".join(f"<a class='shot' href='{esc(quote(p.relative_to(OUTPUTS).as_posix()))}' target='_blank'>"
                       f"<img loading='lazy' src='{esc(quote(p.relative_to(OUTPUTS).as_posix()))}' alt='{esc(p.stem)}'>"
                       f"<span>{esc(p.stem)}</span></a>" for p in pngs)
        parts.append(f"<div class='shots'>{figs}</div>")
    others = {}
    for p in files:
        if p.suffix.lower() not in (".png", ".csv", ".json", ".md", ".txt"):
            others.setdefault(p.suffix.lower(), []).append(p.name)
    if others:
        kinds = {".obj": "3D meshes (open in Blender or MeshLab)", ".onnx": "network files", ".npz": "NumPy data"}
        parts.append("<p class='quiet'>Also here: " + "; ".join(
            f"{len(v)} {kinds.get(k, k + ' files')}" for k, v in others.items()) + ".</p>")
    parts.append("</section>")
    return changed, folder.name, title, "".join(parts)


# ---------------------------------------------------------------- live run and history


def live_panel(live, now):
    if not live:
        return "", False
    running = live.get("status") == "running" and now - live.get("updated", 0) < STALE_AFTER_S
    title = esc(live.get("title", "Run"))
    if live.get("status") == "running" and not running:
        return (f"<section class='live warn'><h2>{title}</h2><p>No update since "
                f"{esc(when(live.get('updated', now)))}. The window was probably closed or the "
                f"computer slept, so this run did not finish.</p></section>"), False
    if not running:
        took = fmt_time((live.get("ended") or now) - live.get("started", now))
        status = live.get("status", "finished")
        err = f"<p class='err'>{esc(live['error'])}</p>" if live.get("error") else ""
        folder = live.get("out_dir")
        link = f" Results: <a href='#{esc(folder)}'>outputs/{esc(folder)}</a>." if folder else ""
        return (f"<section class='live {esc(status)}'><h2>Last run: {title}</h2>"
                f"<p>{esc(status.capitalize())} at {esc(when(live.get('ended') or now))}, after {esc(took)}."
                f"{link}</p>{err}</section>"), False
    of = f" of {live['total_steps']}" if live.get("total_steps") else ""
    step = ""
    if live.get("step_name"):
        sub = f" <span class='quiet'>({esc(live['sub'])})</span>" if live.get("sub") else ""
        step = (f"<p class='step'>Step {live.get('step_no')}{of}: <b>{esc(live['step_name'])}</b>{sub} "
                f"<span class='timer' data-since='{live.get('step_started') or now}'></span></p>")
    bar = ""
    loop = live.get("loop")
    if loop and loop.get("total"):
        frac = loop["done"] / loop["total"]
        elapsed = now - loop["started"]
        left = elapsed / loop["done"] * (loop["total"] - loop["done"]) if loop["done"] else None
        left_txt = f", about {fmt_time(left)} left" if left is not None else ""
        bar = (f"<p>{esc(loop['name'])}: {loop['done']} of {loop['total']} ({frac:.0%}){esc(left_txt)}</p>"
               f"<div class='bar'><div style='width:{frac * 100:.1f}%'></div></div>")
    elif live.get("step_name"):
        bar = "<div class='bar busy'><div></div></div>"
    done = live.get("done_steps") or []
    done_html = ""
    if done:
        items = "".join(f"<li>{esc(s['name'])} <span class='quiet'>{esc(fmt_time(s['seconds']))}</span></li>"
                        for s in done[-6:])
        done_html = f"<details><summary>{len(done)} steps done</summary><ul>{items}</ul></details>"
    return (f"<section class='live running'><h2>Running now: {title}</h2>"
            f"<p class='quiet'>{esc(live.get('script', ''))} {esc(' '.join(live.get('args') or []))} "
            f"&middot; started {esc(when(live.get('started', now)))} "
            f"&middot; total <span class='timer' data-since='{live.get('started', now)}'></span></p>"
            f"{step}{bar}{done_html}</section>"), True


def history_html():
    if not RUNS_FILE.exists():
        return ""
    runs = []
    for line in RUNS_FILE.read_text(encoding="utf-8").splitlines():
        try:
            runs.append(json.loads(line))
        except ValueError:
            continue
    if not runs:
        return ""
    def result(r):
        err = r.get("error")
        return r.get("status", "") + (f": {err[:80]}" if err else "")

    rows = [[when(r["started"]), r.get("title", ""), fmt_time((r.get("ended") or r["started"]) - r["started"]),
             result(r), "outputs/" + r["out_dir"] if r.get("out_dir") else ""]
            for r in reversed(runs[-12:])]
    return "<section class='card'><h2>Recent runs</h2>" + table_html(
        ["Started", "Run", "Took", "Result", "Results folder"], rows) + "</section>"


# ---------------------------------------------------------------- page


CSS = """
:root{--bg:#f6f5f2;--card:#fff;--ink:#1d1d1b;--quiet:#6b6a66;--line:#e3e1dc;--accent:#2f6fdf;--good:#2e7d4f;--warn:#a86b00;--bad:#b3261e;--tint:#eef3fc}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--ink:#ecebe7;--quiet:#a3a19b;--line:#33322f;--accent:#7aa7ff;--good:#6cc18f;--warn:#e0b04a;--bad:#ff8a80;--tint:#1d2638}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 "Segoe UI",system-ui,-apple-system,sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 24px;display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
header h1{font-size:18px;margin:0}header nav{display:flex;gap:6px;flex-wrap:wrap}header nav a{font-size:12.5px;color:var(--quiet);text-decoration:none;border:1px solid var(--line);border-radius:99px;padding:2px 9px}
header nav a:hover{color:var(--ink);border-color:var(--quiet)}
main{max-width:1100px;margin:0 auto;padding:20px 24px 60px}
.card,.live{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin:0 0 18px}
.live.running{border-color:var(--accent);background:var(--tint)}.live.failed{border-color:var(--bad)}.live.warn,.live.stopped{border-color:var(--warn)}
h2{font-size:17px;margin:0 0 6px}h3{font-size:14px;margin:16px 0 6px;color:var(--quiet);text-transform:uppercase;letter-spacing:.04em}
.quiet{color:var(--quiet);font-size:13.5px}.tag{display:inline-block;border:1px solid var(--line);border-radius:6px;padding:0 6px;font-size:12px;margin-left:4px}
.err{color:var(--bad);font-family:Consolas,monospace;font-size:13px}.step{font-size:16px}
.scroll{overflow-x:auto;margin:8px 0}table{border-collapse:collapse;font-size:13.5px;min-width:50%}
caption{text-align:left;font-weight:600;padding:4px 0}th,td{padding:5px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-weight:600;background:var(--bg)}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
details{margin:8px 0}summary{cursor:pointer;color:var(--quiet);font-size:13.5px}
.shots{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px;margin-top:12px}
.shot{display:block;border:1px solid var(--line);border-radius:8px;overflow:hidden;text-decoration:none;color:var(--quiet);font-size:12px;background:#fff}
.shot img{width:100%;display:block}.shot span{display:block;padding:4px 8px;background:var(--card)}
.bar{height:12px;border-radius:99px;background:var(--line);overflow:hidden;margin:6px 0}.bar div{height:100%;background:var(--accent);border-radius:99px}
.bar.busy div{width:30%;animation:slide 1.6s ease-in-out infinite}@keyframes slide{0%{margin-left:-30%}100%{margin-left:100%}}
pre{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:12.5px;overflow:auto}
code{font-family:Consolas,monospace;font-size:.92em}a{color:var(--accent)}
@media (max-width:640px){main,header{padding-left:16px;padding-right:16px}}
"""

JS = """
function tick(){var now=Date.now()/1000;document.querySelectorAll('.timer').forEach(function(el){
var s=Math.max(0,Math.round(now-parseFloat(el.dataset.since)));var m=Math.floor(s/60),h=Math.floor(m/60);
el.textContent=h?h+' h '+String(m%60).padStart(2,'0')+' min':(m?m+' min '+String(s%60).padStart(2,'0')+' s':s+' s');});}
tick();setInterval(tick,1000);
try{var y=sessionStorage.getItem('rift-scroll');if(y)window.scrollTo(0,parseFloat(y));
window.addEventListener('beforeunload',function(){sessionStorage.setItem('rift-scroll',String(window.scrollY));});}catch(e){}
"""


def build(path=PAGE):
    """Write the dashboard page; returns its path."""
    now = time.time()
    live = None
    if LIVE_FILE.exists():
        try:
            live = json.loads(LIVE_FILE.read_text(encoding="utf-8"))
        except ValueError:
            live = None
    live_html, running = live_panel(live, now)
    cards = [folder_card(f) for f in OUTPUTS.iterdir() if f.is_dir()] if OUTPUTS.exists() else []
    cards.sort(key=lambda c: c[0], reverse=True)
    nav = "".join(f"<a href='#{esc(name)}'>{esc(name)}</a>" for _, name, _, _ in cards)
    refresh = "<meta http-equiv='refresh' content='3'>" if running else ""
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>{refresh}"
            f"<title>RIFT runs</title><style>{CSS}</style></head><body>"
            f"<header><h1>RIFT runs</h1><span class='quiet'>built {esc(when(now))}"
            f"{' &middot; refreshing every 3 s' if running else ''}</span><nav>{nav}</nav></header>"
            f"<main>{live_html}{history_html()}{''.join(c[3] for c in cards)}</main>"
            f"<script>{JS}</script></body></html>")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(page, encoding="utf-8")
    tmp.replace(path)  # swap in whole, so the browser never reads a half-written page
    return path


def main():
    p = argparse.ArgumentParser(description="Rebuild outputs/dashboard.html.")
    p.add_argument("--open", action="store_true", help="open the page in the default browser")
    args = p.parse_args()
    path = build()
    print(f"Dashboard: {path}")
    if args.open:
        webbrowser.open(path.resolve().as_uri())


if __name__ == "__main__":
    main()
