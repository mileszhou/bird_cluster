#!/usr/bin/env python3
"""Second-level clustering: group the leaf clusters by their medoids.

Level 1 -- `cluster.py` -- is HDBSCAN over the image vectors, and its output is
**leaves**: many small groups, each a thing to look at. This stage clusters the
*leaves* into **branches**, by taking each leaf's medoid as its representative
and running Ward over those. `findings/01` is the measurement behind every
choice here:

**Ward, not HDBSCAN again.** HDBSCAN at level 2 is not monotonic in its own
parameter -- `min_cluster_size` 3, 5 and 8 give 106, 2 and 11 groups -- so there
is no threshold to tune, and it calls two thirds of the leaves noise, which
costs the coverage the whole exercise is for. A taxonomy has no outliers.

**Medoids, not centroids.** The mean of unit vectors is shortened by internal
disagreement -- median `centroid_norm` 0.884, and worse for larger leaves -- and
medoids scored better. It is also the identity rule the rest of the project
already uses: a cluster's stable name is its medoid's image key, so a branch's
name is the medoid of its medoids.

**An adaptive cut, not a fixed k.** The tree is the result and a cut is a view
of it, so the cut should express the constraint that made us want one. Here that
constraint is physical: a branch has to fit in a calendar month. Descending from
the root until no node holds more than `--max-leaves` leaves turns the display
limit into the only parameter, and it is one with a meaning you can check by
looking. A fixed k cannot promise it -- at k=12 every group overflows, and the
overflow is not the tail but the bulk.

## The layout is this stage's output, and it carries no labels

`layout.csv` is what the second-level clustering *produces*: which leaf each
image is in, which branch each leaf is in, and the time that encodes both. The
JPEG export is a **representation** of it -- it copies pixels and writes
metadata, and decides nothing structural. The layout is the interesting
artifact, a few hundred KB, readable and diffable without touching 3.7 GB of
image.

**No species column, deliberately.** Nothing here reads a label, because
nothing here depends on one: the vectors are self-supervised and the grouping
is geometry. A labelling copied into a clustering artifact is a second copy of
a fact owned elsewhere, and it goes stale the moment a different labeller is
run -- which is not hypothetical, it is what `assignments.csv`'s frozen
`species` column did, putting one bird in a JPEG and another in the index
beside it across 70% of an export. The label belongs to the step that renders
for a human, and it is read fresh there from the label CSV.

It also un-merges `export_seriated`'s "one command, not two", and safely. That
merge happened because `index.csv` used to be written *during* the copy, so a
second process could read it half-finished and silently do partial work. Here it
is written to completion by an earlier stage and renamed into place atomically,
so a consumer either finds no file or finds all of it. The hazard was the
interleaving, not the seam.

## The encoding

    year+month   one calendar month per branch, in dendrogram order
    day          one date per leaf within its branch, 1..max-leaves
    minute       position within the leaf
    second       always 00, left free for inserting test images

Lightroom sorts by capture time and filters by date, so a fabricated time is the
only channel that carries an arbitrary order into it. Two levels of time give
two levels of structure for free: pick a month and you have a branch, pick a day
and you have a leaf.

**28 days, always.** `--max-leaves` defaults to 27, leaving day 28 as the pool,
so February behaves like every other month and there is no special case. A
branch with more leaves than will fit puts the remainder on the pool date, which
alternates colour label Red/Blue so the boundaries inside it stay visible -- the
same convention `export_seriated` uses for its tail. At the default cut the pool
never fills, which is the point of choosing the cut that way.

**A leaf larger than 1,440 images cannot be encoded** and is a hard error rather
than a silent overflow: at level 1 the next date belongs to the next leaf, so
spilling would corrupt the encoding rather than merely crowd it. The free
`second` field is where to go if that ever bites.

    python3 -m code.cluster.cluster2 --run output/cluster/mcs3
    python3 -m code.cluster.cluster2 --run output/cluster/mcs3 --max-leaves 20
"""

import argparse
import csv
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list

sys.path.insert(0, os.environ.get("PROJECT_ROOT")
                or str(Path(__file__).resolve().parents[2]))

from code.lib.config import working_output                       # noqa: E402
from code.lib.csv_post import embeddings_for, read_rows           # noqa: E402
from tools.plot_matrix import load_vectors                        # noqa: E402

logger = logging.getLogger("cluster2")
MINUTES_PER_DAY = 24 * 60
POOL_COLOURS = ("Red", "Blue")


def adaptive_cut(Z, n, max_leaves):
    """Nodes of the linkage tree, descending until none holds > max_leaves leaves.

    Returns a list of node ids, in no particular order; `leaf_members` expands one.
    """
    count = {i: 1 for i in range(n)}
    kids = {}
    for i, (a, b, _, _) in enumerate(Z):
        a, b, node = int(a), int(b), n + i
        count[node] = count[a] + count[b]
        kids[node] = (a, b)

    groups, stack = [], [2 * n - 2 if n > 1 else 0]
    while stack:
        node = stack.pop()
        if count[node] <= max_leaves or node < n:
            groups.append(node)
        else:
            stack.extend(kids[node])
    return groups, kids


def leaf_members(node, n, kids):
    """The original leaf indices under a linkage node."""
    out, stack = [], [node]
    while stack:
        v = stack.pop()
        if v < n:
            out.append(v)
        else:
            stack.extend(kids[v])
    return out


def medoid_of(vectors, idx):
    """The member of `idx` closest to the group's mean -- its stable representative."""
    sub = vectors[idx]
    centre = sub.mean(axis=0)
    centre /= np.linalg.norm(centre) or 1.0
    return idx[int(np.argmax(sub @ centre))]


def plan(Z, vectors, max_leaves):
    """Branches and their leaves, both in dendrogram order.

    Ordering is the whole reason to keep the tree: adjacent months are adjacent
    branches and adjacent days are adjacent leaves, so scrolling forward is
    always a move to the next most similar thing. Sorting by size instead would
    make date adjacency mean nothing.
    """
    n = len(vectors)
    nodes, kids = adaptive_cut(Z, n, max_leaves)
    order = {leaf: i for i, leaf in enumerate(leaves_list(Z))}
    branches = []
    for node in nodes:
        members = sorted(leaf_members(node, n, kids), key=lambda i: order[i])
        branches.append((min(order[i] for i in members), members))
    branches.sort()
    return [(medoid_of(vectors, np.array(m)), m) for _, m in branches]


def capture_times(branch_index, leaf_slot, count, base_year, max_leaves):
    """Times for one leaf: its branch's month, its own day, a minute per image."""
    year = base_year + branch_index // 12
    month = branch_index % 12 + 1
    day = min(leaf_slot, max_leaves) + 1
    start = datetime(year, month, day)
    return [start + timedelta(minutes=i) for i in range(count)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, required=True,
                    help="a level-1 run directory, holding centers.jsonl and assignments.csv")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <working output>/cluster2/<run name>")
    ap.add_argument("--embeddings", type=Path, default=None,
                    help="default: the vectors the level-1 run recorded")
    ap.add_argument("--max-leaves", type=int, default=27,
                    help="leaves per branch before the rest go to the pool date (default 27)")
    ap.add_argument("--base-year", type=int, default=2000)
    ap.add_argument("--dry-run", action="store_true",
                    help="report the layout and stop, writing nothing")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    centres = [json.loads(l) for l in open(args.run / "centers.jsonl", encoding="utf-8")]
    if not centres:
        raise SystemExit(f"error: no clusters in {args.run}/centers.jsonl")
    vec_path = embeddings_for(args.run, args.embeddings)
    logger.info("%d leaves from %s", len(centres), args.run)
    logger.info("vectors from %s", vec_path)
    medoids = load_vectors(vec_path, [c["medoid"] for c in centres])

    Z = linkage(medoids.astype(np.float64), method="ward")
    branches = plan(Z, medoids, args.max_leaves)
    sizes = [len(m) for _, m in branches]
    pooled = sum(max(0, s - args.max_leaves) for s in sizes)
    logger.info("%d branches over %d leaves | leaves per branch median %d max %d | "
                "%d leaves pooled | %.1f imaginary years",
                len(branches), len(centres), int(np.median(sizes)), max(sizes),
                pooled, len(branches) / 12)

    members = {}
    for r in read_rows(args.run / "assignments.csv"):
        if r["is_noise"] != "1":
            members.setdefault(int(r["cluster_id"]), []).append(r)
    for rows in members.values():
        rows.sort(key=lambda r: int(r["seq"]))

    rows, seq, stat = [], 0, Counter()
    for bi, (medoid_leaf, leaf_idx) in enumerate(branches):
        branch_name = centres[medoid_leaf]["medoid"]
        for slot, li in enumerate(leaf_idx):
            cid = centres[li]["cluster_id"]
            imgs = members.get(cid, [])
            if not imgs:
                stat["empty leaf"] += 1
                continue
            if len(imgs) > MINUTES_PER_DAY:
                raise SystemExit(
                    f"error: leaf {cid} holds {len(imgs)} images, more than the "
                    f"{MINUTES_PER_DAY} minutes in a day. The encoding cannot "
                    f"represent it; see this module's docstring.")
            overflow = slot >= args.max_leaves
            colour = POOL_COLOURS[slot % 2] if overflow else ""
            stat["pooled" if overflow else "own date"] += 1
            for when, r in zip(capture_times(bi, slot, len(imgs),
                                             args.base_year, args.max_leaves), imgs):
                seq += 1
                rows.append({
                    "seq": seq,
                    "capture_time": when.strftime("%Y-%m-%d %H:%M:%S"),
                    "branch": bi + 1,
                    "branch_name": branch_name,
                    "leaf": cid,
                    "leaf_size": len(imgs),
                    "leaf_name": centres[li]["medoid"],
                    "color": colour,
                    "key": r["key"],
                })
    logger.info("%d images | leaves with their own date %d, pooled %d, empty %d",
                len(rows), stat["own date"], stat["pooled"], stat["empty leaf"])
    if args.dry_run:
        logger.info("dry run: nothing written")
        return

    out = args.out or (working_output() / "cluster2" / args.run.name)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "layout.csv.tmp"
    with open(tmp, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(out / "layout.csv")         # atomic: never half-visible
    (out / "run.json").write_text(json.dumps({
        "level1_run": str(args.run.resolve()),
        "source": str(vec_path),
        "max_leaves": args.max_leaves,
        "base_year": args.base_year,
        "leaves": len(centres),
        "branches": len(branches),
        "images": len(rows),
        "leaves_pooled": stat["pooled"],
        "leaves_per_branch_median": int(np.median(sizes)),
        "leaves_per_branch_max": max(sizes),
    }, indent=2) + "\n", encoding="utf-8")
    logger.info("-> %s", out / "layout.csv")


if __name__ == "__main__":
    main()
