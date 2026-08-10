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

**Staying idempotent.** A post-processor is run repeatedly, in orders nobody
planned. Each should be safe to re-apply, and should not care whether an earlier
one has already run.

    from code.lib.csv_post import add_arguments, resolve_runs, rewrite

    ap = argparse.ArgumentParser(...)
    add_arguments(ap)
    args = ap.parse_args()
    for run in resolve_runs(args):
        rewrite(run / "assignments.csv", transform)
"""

import csv
import os
import tempfile
from pathlib import Path

from code.lib.config import working_output


def add_arguments(ap, subdir: str = "cluster"):
    """The two flags every post-processor takes."""
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="the run root to work on. Default: config.toml's "
                         f"current_working_output + /{subdir}")
    ap.add_argument("--run", type=Path, default=None,
                    help="a single run directory, instead of every one under the root")
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
