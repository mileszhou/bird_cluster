#!/usr/bin/env python3
"""Add a `cluster_size` column: how many images are in that row's cluster.

Lets an assignments.csv be sorted or filtered by cluster size directly -- the
big clusters first, or every cluster of exactly n members -- without grouping
the file first.

Noise gets an **empty** value, not 0. `cluster_id = -1` is not a cluster with no
members, it is the absence of one; a 0 would sort among real sizes and read as a
measurement.

    python3 -m tools.add_cluster_size                       # every run under the working output
    python3 -m tools.add_cluster_size --run output_003/cluster/mcs5

Idempotent, and safe to run before or after any other post-processor.
"""

import argparse
import collections

import os
import sys
from pathlib import Path

# Running this as `python3 tools/x.py` puts tools/ on sys.path, not the repo
# root -- and then `import code.lib` finds the *stdlib* `code` module, which is
# not a package. Put the root first so both invocation forms work.
sys.path.insert(0, os.environ.get("PROJECT_ROOT")
                or str(Path(__file__).resolve().parents[1]))

from code.lib.csv_marks import REGISTRY
from code.lib.csv_post import add_arguments, apply_tool, resolve_runs

TOOL = REGISTRY["add_cluster_size"]
NOISE = "-1"


def transform(rows):
    if "cluster_id" not in rows[0]:
        raise SystemExit("error: no cluster_id column")
    sizes = collections.Counter(r["cluster_id"] for r in rows if r["cluster_id"] != NOISE)
    for r in rows:
        r["cluster_size"] = "" if r["cluster_id"] == NOISE else sizes[r["cluster_id"]]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    args = ap.parse_args()
    for run in resolve_runs(args):
        print(f"  {run.name}: {apply_tool(run / 'assignments.csv', TOOL, transform, args)}")


if __name__ == "__main__":
    main()
