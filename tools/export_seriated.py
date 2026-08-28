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

**The export is one-way. Nothing here is ever written back to the library.**
Three of its fields are fabricated for browsing and are meaningless -- worse,
actively destructive -- anywhere else: the **capture time** says the year 2000,
the **colour label** encodes a tail boundary, and the **keyword** is this
pipeline's guess. Overwriting a photo's real colour label here is therefore
free, which is why the tail does it without asking; the same act against the
master would lose work. If anything ever does flow back it will be a *review
verdict*, extracted deliberately into its own file, and never a field copied
wholesale from these copies.

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

A pooled date holds **at most 100 images, not 1,440**. The two halves are
browsed differently: a kept cluster is *filtered to* -- pick its date and you
have exactly that cluster, so its size is its own business -- while the tail is
*scrolled through*, and 1,440 thumbnails on one date is a scroll nobody
finishes.

**100 is a ceiling, not a quota**: a date takes whole clusters only, and closes
early rather than splitting one. Packing to exactly 100 cuts a cluster across a
date boundary, which is the one thing the tail must not do -- the alternating
colour marks a boundary *within* a date, and half a cluster continuing onto the
next has no boundary to mark. A cluster larger than 100 must be split, and
starts on an empty date so the split falls inside it rather than between two.

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
    python3 -m tools.export_seriated --layout output/cluster2/mcs3/layout.csv

Writes to output/lightroom/jpg/<run>/ plus an index CSV. Never touches data/.
"""

import argparse
import csv
import json
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
                               effective_english, effective_label,
                               write_keywords)
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

    used = 0                               # images already on the current date
    for n, (cid, members) in enumerate(tail):
        colour = TAIL_COLORS[n % len(TAIL_COLORS)]
        # TAIL_SLOTS_PER_DAY is a ceiling, not a quota: start a fresh date rather
        # than split a cluster across two. Packing dates exactly full would cut
        # clusters in half, which is the one thing the tail must not do -- the
        # alternating colour marks a boundary *within* a date, and a cluster
        # continuing onto the next date has no boundary to mark.
        if used and used + len(members) > TAIL_SLOTS_PER_DAY:
            day, used = day + 1, 0
        for j, i in enumerate(members):
            plan.append((cid, i,
                         base + timedelta(days=day + (used + j) // TAIL_SLOTS_PER_DAY,
                                          minutes=(used + j) % TAIL_SLOTS_PER_DAY),
                         colour))
        used += len(members)
        # A cluster bigger than a whole date has to be split regardless; it began
        # on an empty date, so the split falls inside it rather than between two.
        day, used = day + used // TAIL_SLOTS_PER_DAY, used % TAIL_SLOTS_PER_DAY
    if tail and used:
        day += 1
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


def apply_row(key, when, colour, out, jpg_root, labels, labels_only,
              stat, failures, taxa=None, pred=None):
    """Copy one image and write its metadata. Returns its label CSV row, if any.

    Shared by the two planners: the flat seriation below, and a layout read from
    a `cluster2` index. Neither decides anything here -- this function is the
    representation layer, and it is deliberately ignorant of which level of
    clustering chose the time it is given.
    """
    dst = out / key
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not labels_only:
        set_capture_time(jpg_root / key, dst, when)
    source = labels.get(key)
    label = effective_label(source) if source else None
    if not label:
        stat["no label"] += 1
        failures.append((key, "no usable label in the label CSV"))
        return source
    keywords = ([label] + taxa_keywords((taxa or {}).get(key))
                + prediction_keywords((pred or {}).get(key)))
    try:
        stat[write_keywords(dst, keywords, when, colour)] += 1
    except (SegmentError, XmpEditError, OSError) as exc:
        stat["failed"] += 1
        failures.append((key, str(exc)))
    return source


# PROJECT_ROOT-anchored like every other default here, not cwd-relative: a
# relative default silently finds nothing when the tool is run from anywhere but
# the repo root, and the export then completes looking entirely normal with the
# rank keywords quietly absent.
TAXA_DEFAULT = PROJECT_ROOT / "output" / "taxa" / "label_taxonomy.csv"
TAXA_RANKS = (("ord", "order"), ("fam", "family"), ("gen", "genus"))
PRED_DEFAULT = PROJECT_ROOT / "output" / "taxa" / "taxa_predictions.csv"
# Buckets over `margin` (top-1 minus runner-up cosine), which rises monotonically
# across its deciles from 0.08 to 0.68 agreement with the existing labels. It is
# the only calibrated confidence this project has -- the VLM's own averages 0.968
# against a measured ~35% error -- so it is the right thing to sort a review by.
MARGIN_BUCKETS = ((0.15, "high"), (0.07, "mid"), (0.0, "low"))


def margin_bucket(margin: str) -> str:
    try:
        m = float(margin)
    except (TypeError, ValueError):
        return ""
    for floor, name in MARGIN_BUCKETS:
        if m >= floor:
            return name
    return "low"


def load_predictions(path: Path | None, required: bool):
    """BioCLIP's per-image call, which owes nothing to the labelling.

    The distinction from `load_taxa` is the whole point and is easy to lose: the
    label taxonomy is looked up *from the species string*, so where the labeller
    is wrong it is confidently wrong in the same direction. This is read off the
    pixels, so it is a genuine second opinion and can be compared against the
    species keyword rather than merely restating it.
    """
    if path is None and PRED_DEFAULT.is_file():
        path = PRED_DEFAULT
    if path is None:
        if required:
            raise SystemExit(f"error: no predictions at {PRED_DEFAULT}. Build them "
                             "with `python3 -m tools.predict_taxa`.")
        return {}
    if not Path(path).is_file():
        raise SystemExit(f"error: no predictions at {path}. Build them with "
                         "`python3 -m tools.predict_taxa`.")
    with open(path, encoding="utf-8-sig", newline="") as fh:
        pred = {r["jpg"]: r for r in csv.DictReader(fh)}
    print(f"  predictions: {len(pred):,} images from {path}")
    return pred


def prediction_keywords(row) -> list[str]:
    """`bc:` for what BioCLIP saw, so it cannot be mistaken for the label's own.

    Prefixed separately from `ord:`/`fam:`/`gen:` on purpose. A photo can carry
    a species keyword from one source and a genus from another -- that is true
    of 19% of this set today, and reading the pair as one opinion is exactly the
    confusion this is meant to end.
    """
    if not row:
        return []
    out = []
    for prefix, rank in (("bc-ord", "order"), ("bc-fam", "family"), ("bc-gen", "genus")):
        if row.get(rank):
            out.append(f"{prefix}:{row[rank]}")
    if row.get("common_name"):
        out.append(f"bc:{row['common_name']}")
    bucket = margin_bucket(row.get("margin", ""))
    if bucket:
        out.append(f"bc-conf:{bucket}")
    return out


def load_taxa(path: Path | None, required: bool):
    """Per-image ranks from `tools.map_label_taxa`, or nothing.

    Optional on purpose: the export predates the taxonomy and still works
    without one, and a checklist is a thing a study may not have. Named
    explicitly it must exist -- a silent fallback to no taxonomy would produce a
    complete-looking export missing the whole point of asking for it.

    The ranks become their own keywords rather than being folded into the
    species string. A cluster review asks "is this branch one family?", which a
    keyword list can answer by filtering and a composed caption cannot. Prefixed
    so they sort together in the keyword panel and are unmistakably ours.
    """
    if path is None and TAXA_DEFAULT.is_file():
        path = TAXA_DEFAULT
    if path is None:
        if required:
            raise SystemExit("error: --taxonomy named no file and none found at "
                             f"{TAXA_DEFAULT}. Build one with "
                             "`python3 -m tools.map_label_taxa`.")
        return {}
    if not Path(path).is_file():
        raise SystemExit(f"error: no taxonomy at {path}. Build one with "
                         "`python3 -m tools.map_label_taxa`.")
    with open(path, encoding="utf-8-sig", newline="") as fh:
        taxa = {r["jpg"]: r for r in csv.DictReader(fh)}
    print(f"  taxonomy: {len(taxa):,} images from {path}")
    return taxa


def taxa_keywords(row) -> list[str]:
    return [f"{prefix}:{row[rank]}" for prefix, rank in TAXA_RANKS
            if row and row.get(rank)]


def load_labels(label_dir: Path):
    with open(label_dir / "bird_identification_output.csv",
              encoding="utf-8-sig", newline="") as fh:
        return {r["jpg"]: r for r in csv.DictReader(fh)}


def export_index(args):
    """Render a `cluster2` layout: it already holds the times, so nothing is planned.

    The layout is the second-level clustering's output and this is a view of it,
    so the only decisions left are which bytes to copy, where, and which label to
    caption them with. **The label is decided here and nowhere earlier** -- the
    layout is pure structure, so the two cannot fall out of step, and re-running
    with a different `--label-dir` re-captions an export without touching the
    clustering that produced it.
    """
    labels = load_labels(args.label_dir or PROJECT_ROOT / "data" / "label")
    taxa = (load_taxa(args.taxonomy, required=args.taxonomy is not None)
            if args.taxonomy_source in ("label", "both") else {})
    pred = (load_predictions(args.predictions, required=args.predictions is not None)
            if args.taxonomy_source in ("image", "both") else {})
    # utf-8-sig, not utf-8: layout.csv carries a BOM so Excel reads its Chinese
    # trip names, and a plain read would leave it glued to the first field name
    # -- which then gets written back out behind a second BOM.
    with open(args.layout, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"error: {args.layout} is empty")
    out = args.out or (PROJECT_ROOT / "output" / "lightroom" / "jpg" /
                       f"{args.layout.parent.parent.name}-{args.layout.parent.name}")
    jpg_root = data_dir() / "jpg"
    span = (rows[0]["capture_time"][:10], rows[-1]["capture_time"][:10])
    branches = len({r["branch"] for r in rows})
    leaves = len({r["leaf"] for r in rows})
    print(f"  {args.layout}: {len(rows):,} images, {branches} branches, "
          f"{leaves:,} leaves ({span[0]} .. {span[1]})")
    if args.dry_run:
        return

    out.mkdir(parents=True, exist_ok=True)
    stat, failures, recoloured = Counter(), [], 0
    index = out / "index.csv.tmp"
    with open(index, "w", newline="", encoding="utf-8-sig") as fh:
        ranks = ([rank for _, rank in TAXA_RANKS] if taxa else []) \
            + (["bc_common", "bc_genus", "bc_margin"] if pred else [])
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) + ["species"] + ranks)
        w.writeheader()
        for seq, r in enumerate(rows, 1):
            when = datetime.strptime(r["capture_time"], "%Y-%m-%d %H:%M:%S")
            colour = r.get("color", "")
            source = apply_row(r["key"], when, colour, out, jpg_root, labels,
                               args.labels_only, stat, failures, taxa, pred)
            recoloured += bool(colour)
            # The export's own index: the layout it was given, plus the caption
            # it chose. The layout has no species column -- that is the point --
            # so this file is the only record of which labelling was rendered,
            # and it is written by the step that did the rendering.
            t, bc = taxa.get(r["key"]) or {}, pred.get(r["key"]) or {}
            extra = {rank: t.get(rank, "") for _, rank in TAXA_RANKS} if taxa else {}
            if pred:
                extra.update(bc_common=bc.get("common_name", ""),
                             bc_genus=bc.get("genus", ""),
                             bc_margin=bc.get("margin", ""))
            w.writerow({**r, "species": (effective_english(source) if source else "") or "",
                        **extra})
            if seq % 2000 == 0:
                print(f"    {seq:,}/{len(rows):,}", flush=True)
    index.replace(out / "index.csv")
    print(f"    -> {out}  ({len(rows):,} files + index.csv)")
    for what, count in sorted(stat.items()):
        print(f"       {what}: {count:,}")
    if recoloured:
        print(f"       colour label set on {recoloured:,} pooled images")
    for key, why in failures[:5]:
        print(f"       ! {key}: {why}")
    if len(failures) > 5:
        print(f"       ! ... and {len(failures) - 5} more")


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
    ap.add_argument("--label-dir", type=Path, default=None,
                    help="directory holding bird_identification_output.csv. Default: "
                         "data/label; with --index, whatever the cluster2 run recorded, "
                         "and passing a different one is an error rather than an override")
    ap.add_argument("--taxonomy", type=Path, default=None,
                    help="per-image ranks from tools.map_label_taxa, written as "
                         "extra keywords (ord:/fam:/gen:) so a photo manager can "
                         f"filter by rank. Default: {TAXA_DEFAULT} if it exists, "
                         "otherwise none; naming one that is absent is an error")
    ap.add_argument("--taxonomy-source", choices=("label", "image", "both"),
                    default="label",
                    help="where the rank keywords come from. `label` looks the "
                         "species string up in the checklist, so it restates the "
                         "labelling and inherits its errors. `image` is BioCLIP's "
                         "own per-image call (bc:), which owes the labelling "
                         "nothing and is therefore the one worth reviewing "
                         "against. `both` writes each under its own prefix "
                         "(default: label)")
    ap.add_argument("--predictions", type=Path, default=None,
                    help=f"per-image calls from tools.predict_taxa (default: "
                         f"{PRED_DEFAULT} when --taxonomy-source uses it)")
    ap.add_argument("--layout", type=Path, default=None,
                    help="render a cluster2 layout.csv instead of planning a flat "
                         "seriation; the layout already carries the times")
    ap.add_argument("--labels-only", action="store_true",
                    help="rewrite metadata in an existing export without copying "
                         "the images again -- for changing the label form or the "
                         "colours after the fact")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the layout and stop, copying nothing")
    args = ap.parse_args()

    if args.layout:
        export_index(args)
        return
    labels = load_labels(args.label_dir or PROJECT_ROOT / "data" / "label")
    taxa = (load_taxa(args.taxonomy, required=args.taxonomy is not None)
            if args.taxonomy_source in ("label", "both") else {})
    pred = (load_predictions(args.predictions, required=args.predictions is not None)
            if args.taxonomy_source in ("image", "both") else {})

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
                source = apply_row(r["key"], when, colour, out, jpg_root, labels,
                                   args.labels_only, stat, failures, taxa, pred)
                recoloured += bool(colour)
                # The species comes from --label-dir, the same CSV the keyword
                # above was written from, and *not* from `r["species"]` -- that
                # column is whatever labelling was current when the clustering
                # ran, frozen into assignments.csv. Exporting a second labeller
                # against an existing clustering therefore produced JPEGs saying
                # one species and an index.csv beside them saying another, on
                # 70% of rows. Fall back to the assignments value only when the
                # row has no usable label, the same case that leaves the JPEG
                # without a keyword.
                species = (effective_english(source) if source else None) \
                    or r.get("species", "")
                # r's own cluster_id, not the group label: a pooled row would
                # otherwise be recorded as `tail` and lose its identity.
                w.writerow([seq, when.strftime("%Y-%m-%d %H:%M:%S"),
                            r["cluster_id"], species, colour, r["key"]])
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
