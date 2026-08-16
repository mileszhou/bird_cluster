#!/usr/bin/env python3
"""Export copies of a run's images with the seriated order encoded in EXIF time.

The point is to study clusters visually in Lightroom without touching the master
library. Lightroom sorts by capture time, filename, rating or colour label --
never by arbitrary metadata -- so the ordering that produced the adjacency curve
cannot travel in a keyword. Writing it into the capture time makes Lightroom's
*default* sort walk the seriation, with no sidecar writes at all.

**Copies, never links.** A hardlink is the master file, so anything Lightroom
writes to it writes through to the library -- which defeats the purpose. At
161 KB mean this is about 1.4 GB for the mcs15 set; disk is cheaper than a
damaged master.

**The encoding.**

    day     one calendar date per cluster, in seriated order
    minute  position within the cluster
    second  always 00, left free for inserting test images between two neighbours

The dates follow the *seriation*, not cluster size -- so two adjacent dates are
two clusters the seriation put next to each other, and the date sequence lines
up with `adjacency-cluster.csv`. (This line used to read "largest first", which
the code has never done and which would have made date adjacency meaningless.)

A date holds 24*60 = 1440 minutes. Most clusters fit easily -- the largest at
mcs15 is 427, about seven hours -- but a cluster larger than that continues onto
the next date, and so does the combined tail. So the date is a *bijection* with
cluster index rather than literally the day number: with 146 clusters the run
spans about five months of an imaginary year.

**The tail.** Clusters below `--min-cluster` are concatenated in seriated order
and share the dates after the big ones. The default is **15**, which at mcs15
pools nothing at all: no cluster is smaller than the `min_cluster_size` it was
clustered with, so all 146 get a date and every boundary is visible.

It used to default to 100, on the reasoning that "121 separate one-day clusters
of 20 photos each is not a thing anyone wants to page through". That was wrong
twice over. In Lightroom a date is not a page you turn -- it is a segment marker
in one continuous stream -- and at 100 the pool swallowed **51% of the export**
into a single undifferentiated run of three dates, which is precisely the half
you cannot study. A cluster is worth a date of its own; pooling earns its place
only when a cluster is too small to be worth looking at separately, somewhere
around 5. Raise it if a run has genuinely tiny clusters (mcs3 and mcs5 do); the
numbers are printed so the choice can be made by looking.

A pooled row keeps its **own** `cluster_id` in `index.csv` -- it used to be
written as the literal `tail`, which discarded the identity of every pooled
cluster, so nothing downstream could tell where one ended and the next began.

**The folder structure is preserved**, `Photos-YY/<trip>/<name>.jpg` exactly as
in `data/jpg`. Flattening would have collided: 429 of the 8,425 basenames repeat
across trips, because camera counters wrap. Keeping the tree also keeps the
original filenames untouched, and the capture time carries the ordering anyway
-- filtering by date is what the export is for.

    python3 -m tools.export_seriated --run output/cluster/mcs15
    python3 -m tools.export_seriated --run output/cluster/mcs15 --min-cluster 50
    python3 -m tools.export_seriated --run output/cluster/mcs15 --dry-run

Writes to output/lightroom/jpg/<run>/ plus an index CSV. Never touches data/.
The labels are written into the copies afterwards by tools/export_jpg_labels.py
-- Lightroom reads a sidecar only for raw formats, so a JPEG's keyword has to be
embedded in the file itself.
"""

import argparse
import csv
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("PROJECT_ROOT")
                or str(Path(__file__).resolve().parents[1]))

from code.lib.config import PROJECT_ROOT, data_dir  # noqa: E402
from code.lib.csv_post import add_arguments, read_rows, resolve_runs  # noqa: E402
from tools.plot_matrix import load_vectors, seriate  # noqa: E402

NOISE = "-1"
SLOTS_PER_DAY = 24 * 60


def seriated_groups(rows, X, min_cluster):
    """[(label, [row indices in order])], clusters seriated, tail concatenated.

    Cluster order comes from seriating the centroids; members are ordered by
    distance to their own centroid, so a cluster reads core-outwards. This is
    the ordering plot_adjacency uses, and the only one where cluster boundaries
    line up with the walls -- seriating images individually does not keep
    clusters contiguous (see project/ideas/03).
    """
    lab = np.array([r["cluster_id"] for r in rows])
    ids = list(dict.fromkeys(lab))
    C = np.stack([X[lab == c].mean(0) for c in ids])
    C /= np.linalg.norm(C, axis=1, keepdims=True)

    ordered, tail = [], []
    for i in seriate(C):
        cid = ids[i]
        idx = np.flatnonzero(lab == cid)
        blk = X[idx]
        members = idx[np.argsort(np.linalg.norm(blk - blk.mean(0), axis=1))]
        (ordered if len(idx) >= min_cluster else tail).append((cid, list(members)))
    if tail:
        ordered.append(("tail", [i for _, members in tail for i in members]))
    return ordered, len(tail)


def set_capture_time(src: Path, dst: Path, when: datetime):
    """Copy src to dst with the capture time replaced, pixels untouched.

    piexif rewrites the APP1 segment in place rather than re-encoding, so the
    scan data is byte-identical -- the same reason scrub_metadata edits the
    marker structure instead of re-saving through an image library.
    """
    import piexif
    shutil.copy2(src, dst)
    stamp = when.strftime("%Y:%m:%d %H:%M:%S").encode()
    try:
        exif = piexif.load(str(dst))
    except Exception:
        exif = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    exif.setdefault("0th", {})[piexif.ImageIFD.DateTime] = stamp
    exif.setdefault("Exif", {})[piexif.ExifIFD.DateTimeOriginal] = stamp
    exif["Exif"][piexif.ExifIFD.DateTimeDigitized] = stamp
    piexif.insert(piexif.dump(exif), str(dst))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    ap.add_argument("--embeddings", type=Path,
                    default=Path("./data/embed/embeddings.jsonl"))
    ap.add_argument("--out", type=Path, default=None,
                    help="default: output/lightroom/jpg/<run>")
    ap.add_argument("--min-cluster", type=int, default=15,
                    help="clusters smaller than this share the tail dates "
                         "(default 15: at mcs15 nothing is pooled)")
    ap.add_argument("--base-date", default="2000-01-01",
                    help="date of the first cluster; a year far from real photos")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the layout and stop, copying nothing")
    args = ap.parse_args()

    base = datetime.strptime(args.base_date, "%Y-%m-%d")
    for run_dir in resolve_runs(args):
        rows = [r for r in read_rows(run_dir / "assignments.csv")
                if r["cluster_id"] != NOISE]
        X = load_vectors(args.embeddings, [r["key"] for r in rows])
        groups, n_tail = seriated_groups(rows, X, args.min_cluster)

        out = args.out or (PROJECT_ROOT / "output" / "lightroom" / "jpg" / run_dir.name)
        jpg_root = data_dir() / "jpg"

        plan, day, seq = [], 0, 0
        for cid, members in groups:
            start_day = day
            for k, i in enumerate(members):
                when = base + timedelta(days=day + k // SLOTS_PER_DAY,
                                        minutes=k % SLOTS_PER_DAY)
                seq += 1
                plan.append((seq, cid, rows[i], when))
            day += max(1, -(-len(members) // SLOTS_PER_DAY))
            if len(members) > SLOTS_PER_DAY:
                print(f"    cluster {cid}: {len(members)} images spans "
                      f"{day - start_day} dates")

        print(f"  {run_dir.name}: {len(plan):,} images, "
              f"{len(groups) - (1 if n_tail else 0)} clusters kept separate, "
              f"{n_tail} in the tail, {day} dates "
              f"({base:%Y-%m-%d} .. {base + timedelta(days=day - 1):%Y-%m-%d})")
        if args.dry_run:
            continue

        out.mkdir(parents=True, exist_ok=True)
        with open(out / "index.csv", "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["seq", "capture_time", "cluster_id", "species", "key"])
            for seq, cid, r, when in plan:
                dst = out / r["key"]          # mirror data/jpg's tree
                dst.parent.mkdir(parents=True, exist_ok=True)
                set_capture_time(jpg_root / r["key"], dst, when)
                # r's own cluster_id, not the group label: a pooled row would
                # otherwise be recorded as `tail` and lose its identity.
                w.writerow([seq, when.strftime("%Y-%m-%d %H:%M:%S"),
                            r["cluster_id"], r.get("species", ""), r["key"]])
        print(f"    -> {out}  ({len(plan):,} files + index.csv)")


if __name__ == "__main__":
    main()
