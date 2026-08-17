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

**The tail.** Clusters below `--min-cluster` share the dates after the big ones.
The default is **15**, which at mcs15 pools nothing at all: no cluster is smaller
than the `min_cluster_size` it was clustered with, so all 146 get a date and
every boundary is visible.

A pooled date holds **100 images, not 1,440**. The two halves are browsed
differently: a kept cluster is *filtered to* -- pick its date and you have
exactly that cluster, so its size is its own business -- while the tail is
*scrolled through*, and 1,440 thumbnails on one date is a scroll nobody
finishes.

Pooled clusters also **alternate colour label, Red and Blue**, so a boundary
inside a shared date is visible without reading anything. Two colours rather
than five because the eye only has to answer "did it change?"; a longer cycle
invites the reading that a particular colour means something. Only the tail is
coloured, so a colour present at all says "this date holds several clusters".
Note this overwrites any colour label the photo already carried -- 516 of the
10,982 in one mcs15 export do, 450 of them a custom `Safari` label -- which is
acceptable on a derived export and is counted in the output.

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

**One command, not two.** Copying the images and writing their metadata used to
be `export_seriated` followed by `export_jpg_labels`, and the gap between them
was a trap: run the second while the first is still going and it reads a partial
`index.csv`, labels only those rows, and reports success. They are now one pass,
and `index.csv` is renamed into place at the end so it is never half-visible.
`--labels-only` rewrites metadata in an existing export without copying 1.4 GB
again, which is what the split was protecting.

The writing itself is `code/lib/jpg_meta.py` -- a library, because it does one
small thing to an argument. Lightroom reads a sidecar only for raw formats, so a
JPEG's keyword has to be embedded in the file itself, and it has to go into
*both* XMP and the legacy IPTC block or Lightroom shows neither.

    python3 -m tools.export_seriated --run output/cluster/mcs15
    python3 -m tools.export_seriated --run output/cluster/mcs3 --min-cluster 3
    python3 -m tools.export_seriated --run output/cluster/mcs15 --dry-run
    python3 -m tools.export_seriated --run output/cluster/mcs15 --labels-only

Writes to output/lightroom/jpg/<run>/ plus an index CSV. Never touches data/.
"""

import argparse
import csv
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("PROJECT_ROOT")
                or str(Path(__file__).resolve().parents[1]))

from code.lib.config import PROJECT_ROOT, data_dir  # noqa: E402
from code.lib.jpg_meta import (SegmentError, XmpEditError,  # noqa: E402
                               effective_label, write_keywords)
from code.lib.csv_post import (add_arguments, embeddings_for, read_rows,
                               resolve_runs)  # noqa: E402
from tools.plot_matrix import load_vectors, seriate  # noqa: E402

NOISE = "-1"
SLOTS_PER_DAY = 24 * 60
# A pooled date holds far fewer, because a tail date is browsed rather than
# filtered to: 1,440 thumbnails is a scroll nobody finishes, and the clusters
# inside it are small enough that a hundred still shows several of them.
TAIL_SLOTS_PER_DAY = 100
# Consecutive pooled clusters alternate between these, so a boundary inside a
# shared date is visible without reading anything. Two colours, not five: the
# eye only has to answer "did it change?", and a longer cycle invites the
# reading that a particular colour means something.
TAIL_COLORS = ("Red", "Blue")


def seriated_groups(rows, X, min_cluster):
    """(kept, tail) -- each a list of (cluster_id, [row indices in order]).

    Cluster order comes from seriating the centroids; members are ordered by
    distance to their own centroid, so a cluster reads core-outwards. This is
    the ordering plot_adjacency uses, and the only one where cluster boundaries
    line up with the walls -- seriating images individually does not keep
    clusters contiguous (see project/ideas/03).

    The tail keeps its per-cluster structure rather than being concatenated into
    one list. Pooled clusters share dates, but they are still separate clusters,
    and the export has to be able to say where one ends -- that is what the
    alternating colour marks.
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
    return ordered, tail


def plan_dates(kept, tail, base):
    """[(cluster_id, row index, when, colour)] for every image, in export order.

    Two layouts, because the two halves are browsed differently. A kept cluster
    owns its dates and holds up to a day's minutes; filtering to that date shows
    exactly that cluster, so its size is the cluster's business.

    The tail is walked rather than filtered to, so it is capped at
    TAIL_SLOTS_PER_DAY and its clusters alternate colour. Nothing else is
    coloured: a colour that appears only in the tail says "several clusters
    share this date", which is precisely when the reader needs telling.
    """
    plan, day = [], 0
    for cid, members in kept:
        for k, i in enumerate(members):
            plan.append((cid, i,
                         base + timedelta(days=day + k // SLOTS_PER_DAY,
                                          minutes=k % SLOTS_PER_DAY), ""))
        day += max(1, -(-len(members) // SLOTS_PER_DAY))

    k = 0                                  # position in the pooled stream
    for n, (cid, members) in enumerate(tail):
        colour = TAIL_COLORS[n % len(TAIL_COLORS)]
        for i in members:
            plan.append((cid, i,
                         base + timedelta(days=day + k // TAIL_SLOTS_PER_DAY,
                                          minutes=k % TAIL_SLOTS_PER_DAY), colour))
            k += 1
    if tail:
        day += max(1, -(-k // TAIL_SLOTS_PER_DAY))
    return plan, day


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
                    default=None,
                    help="the vectors to read. Default: whatever the run\n"
                         "being analysed recorded as its source")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: output/lightroom/jpg/<run>")
    ap.add_argument("--min-cluster", type=int, default=15,
                    help="clusters smaller than this share the tail dates "
                         "(default 15: at mcs15 nothing is pooled)")
    ap.add_argument("--base-date", default="2000-01-01",
                    help="date of the first cluster; a year far from real photos")
    ap.add_argument("--label-dir", type=Path,
                    default=PROJECT_ROOT / "data" / "label",
                    help="directory holding bird_identification_output.csv")
    ap.add_argument("--labels-only", action="store_true",
                    help="rewrite metadata in an existing export without copying "
                         "the images again -- for changing the label form or the "
                         "colours after the fact")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the layout and stop, copying nothing")
    args = ap.parse_args()

    with open(args.label_dir / "bird_identification_output.csv",
              encoding="utf-8-sig", newline="") as fh:
        labels = {r["jpg"]: r for r in csv.DictReader(fh)}

    base = datetime.strptime(args.base_date, "%Y-%m-%d")
    for run_dir in resolve_runs(args):
        rows = [r for r in read_rows(run_dir / "assignments.csv")
                if r["cluster_id"] != NOISE]
        X = load_vectors(embeddings_for(run_dir, args.embeddings), [r["key"] for r in rows])
        kept, tail = seriated_groups(rows, X, args.min_cluster)
        plan, day = plan_dates(kept, tail, base)

        out = args.out or (PROJECT_ROOT / "output" / "lightroom" / "jpg" / run_dir.name)
        jpg_root = data_dir() / "jpg"

        pooled = sum(len(m) for _, m in tail)
        print(f"  {run_dir.name}: {len(plan):,} images, {len(kept)} clusters kept "
              f"separate, {len(tail)} pooled ({pooled:,} images), {day} dates "
              f"({base:%Y-%m-%d} .. {base + timedelta(days=day - 1):%Y-%m-%d})")
        for cid, members in kept:
            if len(members) > SLOTS_PER_DAY:
                print(f"    cluster {cid}: {len(members):,} images spans "
                      f"{-(-len(members) // SLOTS_PER_DAY)} dates")
        if args.dry_run:
            continue

        out.mkdir(parents=True, exist_ok=True)
        stat, failures, recoloured = Counter(), [], 0
        index = out / "index.csv.tmp"
        with open(index, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["seq", "capture_time", "cluster_id", "species", "color", "key"])
            for seq, (cid, i, when, colour) in enumerate(plan, 1):
                r = rows[i]
                dst = out / r["key"]          # mirror data/jpg's tree
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not args.labels_only:
                    set_capture_time(jpg_root / r["key"], dst, when)
                source = labels.get(r["key"])
                label = effective_label(source) if source else None
                if not label:
                    stat["no label"] += 1
                    failures.append((r["key"], "no usable label in the label CSV"))
                else:
                    try:
                        stat[write_keywords(dst, label, when, colour)] += 1
                        recoloured += bool(colour)
                    except (SegmentError, XmpEditError, OSError) as exc:
                        stat["failed"] += 1
                        failures.append((r["key"], str(exc)))
                # r's own cluster_id, not the group label: a pooled row would
                # otherwise be recorded as `tail` and lose its identity.
                w.writerow([seq, when.strftime("%Y-%m-%d %H:%M:%S"),
                            r["cluster_id"], r.get("species", ""), colour, r["key"]])
                if seq % 2000 == 0:
                    print(f"    {seq:,}/{len(plan):,}", flush=True)
        # Renamed last, so a consumer never reads a half-written index: the file
        # either is not there or is complete. export_jpg_labels used to read it
        # while this was still writing and label only the rows that existed,
        # reporting success for a fraction of the export.
        index.replace(out / "index.csv")

        print(f"    -> {out}  ({len(plan):,} files + index.csv)")
        for what, count in sorted(stat.items()):
            print(f"       {what}: {count:,}")
        if recoloured:
            print(f"       colour label set on {recoloured:,} pooled images")
        for key, why in failures[:5]:
            print(f"       ! {key}: {why}")
        if len(failures) > 5:
            print(f"       ! ... and {len(failures) - 5} more")


if __name__ == "__main__":
    main()
