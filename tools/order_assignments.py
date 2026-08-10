#!/usr/bin/env python3
"""Number the rows, then sort for reading: biggest clusters first, noise last.

`seq` is assigned **before** the sort, so it is the row's position in the input.
Every run of a sweep loads the same vectors and drops duplicates
deterministically, which makes `seq` a stable identity: seq 8412 is the same
photo in `mcs3` and in `mcs40`. That is what lets two runs be joined row by row,
and sorting on it restores the input order after a spreadsheet has been sorted on
something else. Numbering *after* the sort would give a different identity per
run -- useless for the comparison it exists to serve.

The reading order is largest cluster first with its members contiguous, down to
the smallest, then noise. That is what someone opening the file wants, as
against the fitter's order, which is the order vectors happened to be loaded in.

Needs `cluster_size`; run `add_cluster_size.py` first if it is absent.

    tools/order_assignments.py
    tools/order_assignments.py --run output_003/cluster/mcs5
"""

import argparse

from code.lib.csv_post import add_arguments, resolve_runs, rewrite

ORDER = ["seq", "cluster_id", "cluster_size", "cluster_name", "key", "xmp",
         "probability", "is_noise", "species"]


def transform(rows):
    for col in ("cluster_id", "cluster_size"):
        if col not in rows[0]:
            raise SystemExit(f"error: no {col} column -- run add_cluster_size.py first")
    # Number first: seq is the input position, identical across runs. These files
    # are written in input order, so enumerate() recovers it; an existing seq is
    # kept so re-running cannot renumber against a sorted file.
    for i, r in enumerate(rows, 1):
        if not r.get("seq"):
            r["seq"] = i
    rows.sort(key=lambda r: (-int(r["cluster_size"] or 0), int(r["cluster_id"]),
                             int(r["seq"])))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    args = ap.parse_args()
    for run in resolve_runs(args):
        print(f"  {run.name}: {rewrite(run / 'assignments.csv', transform, ORDER)}")


if __name__ == "__main__":
    main()
