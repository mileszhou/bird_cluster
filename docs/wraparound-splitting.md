# Wraparound splitting

A camera burns `(body code, frame counter)` into each raw filename and the jpg export
inherits it, so once a counter wraps — every ~10,000 frames, once or twice a year here —
two different photos in a year share a filename and an export folder keyed by filename
becomes ambiguous. The fix is to cut each year into date-contiguous **segments** in which
no filename repeats, and export one jpg folder per segment (`2019.1`, `2019.2`, ...).

Nothing is renamed. A Lightroom rename is not a local edit, and the filename is what ties
an exported jpg back to the raw file held separately — so the split absorbs the whole
problem, however many segments that takes.

**The current plan lives in [`project/reports/wraparound_split_report.md`](../project/reports/wraparound_split_report.md)**,
with the trip-to-segment assignment in `project/reports/split_plan.csv`. Regenerate both
with `./run-split` (read-only, <1s). The hand-maintained year-by-year lists that used to
live in this file were derived before deduplication and are superseded by that report.

## Method

`tools/analyze_wraparound.py` reads the unsplit library at `data/xmp0`
(`Photos-YY/<trip>/*.xmp`) and, per year:

1. Sorts trip folders by their `YYYY-MM[-DD]` name prefix into a timeline. Undated
   folders have no place on it, so they are appended in name order and named in the
   report rather than guessed at.
2. Collects each trip's filenames, recursively — two trips nest a sub-trip inside them,
   and since a split moves the whole top-level folder those photos go with the parent.
3. Sweeps the timeline greedily, keeping a segment open until a trip would repeat a
   filename already in it, then cutting. This gives the fewest possible segments: the
   no-repeat constraint only gets easier as trips are removed, so any valid partition
   must cut at or before this one does.
4. Re-checks every proposed segment for repeats. A repeat surviving here would mean one
   trip folder holds the same filename twice — no split can fix that, since a boundary
   never runs through a trip — so it is reported as unsplittable.

## Deduplicate first

Run `./run-dedup` and clear what it finds before splitting. The same photo imported into
two neighbouring trips reads exactly like a counter collision and forces a cut that was
never real; that is where the old, far denser splits came from. Clearing 476 redundant
sidecars dropped neighbouring-trip collisions from 227 to 7 and 2019 from 15 segments
to 6.

What remains between neighbouring trips is real: different photos sharing a frame number
a day apart, from the counter running backwards after a card change or a second body
using the same code. The report lists them, because a cut between trips days apart is
worth an eye before acting on it.

## What counts as the same filename

The plan keys on the **sidecar stem**, which is exactly the exported jpg's basename.
Where Lightroom hit a name clash on import it already renamed one raw file with a
`YYYYMMDD-` prefix, and virtual copies carry a `-N` suffix; both survive into the export
(`20181229-_D8S7785-2.jpg`, and 3,170 such jpgs are already in `data/jpg`), so neither
needs a cut.

`--collision-key frame` keys on `code/lib/trips.py:frame_id` instead, which strips those
decorations — the identity used for *deduplication*, where two names for one capture must
collapse. It is the wrong test for export collisions and cuts more often (41 segments
against 35); the report prints both counts so the difference stays visible.

A third count, **strict**, cuts on any per-code counter *range* overlap whether or not a
name actually repeats. It covers frames shot but never imported — deleted, or never
touched in Lightroom — which would still clash in a re-export while leaving no sidecar
behind. It is not the plan; the gap between it and the plan is what is being taken on
trust.
