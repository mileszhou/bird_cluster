"""Shared plumbing for the small tools that post-process a run's CSVs.

There will be a family of these -- add a column, sort a different way, join
something in -- and each should be a short file that says what it does and
nothing else. What they have in common lives here:

**Where to look.** Runs get archived under new names (`./clean` moves `output/`
to `output_NNN_<description>/`), so a tool that hardcodes `./output` analyses
whatever happens to be sitting there. The default comes from config.toml's
`current_working_output`, which re-points every tool at once -- including at an
archive, to re-analyse an old run. An explicit `--output-dir` still wins.

**Rewriting safely.** These files cost seven minutes of fitting to regenerate,
so an in-place edit goes through a temporary file and an atomic replace. An
interrupted tool leaves the original intact rather than a truncated one.

**Knowing what already ran.** Each tool declares a bit in
`code/lib/csv_marks.py`, and the CSV carries the accumulated state in an empty
column named for it (`C_3`). A tool whose prerequisites are unmet refuses and
names the tool that provides them, or runs it first with `--resolve`. That
handles the case column-sniffing cannot: a tool that only reorders rows leaves
no column to detect it by.

    from code.lib.csv_post import add_arguments, resolve_runs, rewrite

    ap = argparse.ArgumentParser(...)
    add_arguments(ap)
    args = ap.parse_args()
    for run in resolve_runs(args):
        apply_tool(run / "assignments.csv", TOOL, transform, args, order=ORDER)
"""

import csv
import importlib
import json
import os
import tempfile
from pathlib import Path

from code.lib.config import PROJECT_ROOT, working_output
from code.lib import csv_marks


def embeddings_for(run_dir: Path, explicit: Path | None = None) -> Path:
    """The vectors a cluster run was actually built from.

    An analysis tool has to read the *same* vectors as the run it is describing,
    and no fixed default can promise that. Every one of these tools used to
    default to `data/embed`, which was right only while there was one embedding
    set in the world; with runs at several resolutions it silently describes a
    clustering using somebody else's vectors, and the numbers look perfectly
    reasonable.

    So the run says. `discover.py` records the absolute `source` it loaded, and
    that is preferred when it still resolves. When it does not -- `./clean`
    moves a whole run root, taking `output/embed` with it and stranding the
    recorded path -- the sibling `embed/` under the same root is the same file
    in its new home, which is why the structural fallback comes before any
    global default rather than after it.
    """
    if explicit is not None:
        return explicit

    run_json = run_dir / "run.json"
    if run_json.is_file():
        try:
            source = json.loads(run_json.read_text()).get("source")
        except (json.JSONDecodeError, OSError):
            source = None
        if source and Path(source).is_file():
            return Path(source)

    # <root>/cluster/mcs15 -> <root>/embed/embeddings.jsonl
    if len(run_dir.parents) >= 2:
        sibling = run_dir.parents[1] / "embed" / "embeddings.jsonl"
        if sibling.is_file():
            return sibling

    live = PROJECT_ROOT / "output" / "embed" / "embeddings.jsonl"
    if live.is_file():
        return live
    raise SystemExit(
        f"error: cannot tell which embeddings {run_dir} was built from.\n"
        f"       Its run.json names none that still exist, there is no "
        f"embed/ beside it, and {live} is absent. Pass --embeddings.")


def add_arguments(ap, subdir: str = "cluster"):
    """The two flags every post-processor takes."""
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="the run root to work on. Default: config.toml's "
                         f"current_working_output + /{subdir}")
    ap.add_argument("--run", type=Path, default=None,
                    help="a single run directory, instead of every one under the root")
    ap.add_argument("--resolve", action="store_true",
                    help="run any missing prerequisite tools first instead of refusing")
    ap.add_argument("--force", action="store_true",
                    help="apply again even if this tool's bit is already set")
    ap.set_defaults(_subdir=subdir)


def resolve_runs(args, marker: str = "assignments.csv"):
    """Run directories to act on, newest-name-last, each containing `marker`."""
    if args.run:
        return [args.run]
    root = args.output_dir or (working_output() / getattr(args, "_subdir", "cluster"))
    if not root.is_dir():
        raise SystemExit(f"error: {root} is not a directory. Set current_working_output "
                         f"in config.toml or pass --output-dir.")
    runs = sorted(p for p in root.iterdir() if (p / marker).is_file())
    if not runs:
        raise SystemExit(f"error: no {marker} found under {root}")
    return runs


def read_rows(path: Path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def rewrite(path: Path, transform, order=()):
    """Apply `transform(rows) -> rows` and write back atomically.

    `order` names columns to put first; anything else keeps its existing
    position after them, so a tool that adds a column need not restate the
    schema.
    """
    rows = read_rows(path)
    if not rows:
        return "empty"
    rows = transform(rows)
    fields = [c for c in order if c in rows[0]] + [c for c in rows[0] if c not in order]
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
    return f"{len(rows)} rows"


def apply_tool(path: Path, tool, transform, args, order=()):
    """Run one post-processor, honouring the recorded state in the file.

    Returns a short status string for printing. Does nothing if the tool has
    already been applied, unless --force: these are meant to be run repeatedly
    and in whatever order, so re-applying must be safe *and* cheap.
    """
    rows = read_rows(path)
    if not rows:
        return "empty"
    state, column = csv_marks.read_state(rows[0].keys())

    if state & tool.bit and not getattr(args, "force", False):
        return f"already applied (state {csv_marks.describe(state)})"

    gaps = csv_marks.missing(state, tool)
    if gaps:
        if not getattr(args, "resolve", False):
            names = ", ".join(g.name for g in gaps)
            raise SystemExit(
                f"error: {path} needs {names} first (state: "
                f"{csv_marks.describe(state)}).\n"
                f"       Run it, or pass --resolve to run it automatically.")
        for g in gaps:
            mod = importlib.import_module(g.module)
            apply_tool(path, g, mod.transform, args, order=order)
        rows = read_rows(path)
        state, column = csv_marks.read_state(rows[0].keys())

    rows = transform(rows)
    rows, column = csv_marks.apply_mark(rows, state, column, tool)
    fields = [c for c in order if c in rows[0]] + \
             [c for c in rows[0] if c not in order and c != column] + [column]
    write_rows(path, rows, fields)
    return f"{len(rows)} rows -> {column} ({csv_marks.describe(state | tool.bit)})"


def write_rows(path: Path, rows, fields):
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
