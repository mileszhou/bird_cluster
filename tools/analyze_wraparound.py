#!/usr/bin/env python3
"""
Propose how to split each year of the photo library into segments whose photo
filenames are unique, so an exported jpg folder can be keyed by filename.

Reads the original unsplit library -- trips directly under a year folder, no
half-year level:

    <xmp-dir>/<Photos-YY>/<trip>/*.xmp

A camera burns `(body code, frame counter)` into the raw filename and the jpg
export inherits it, so two photos sharing a filename anywhere in one folder make
the export ambiguous. The counter wraps every ~10,000 frames, which in this
library is once or twice a year. Cutting the year into date-contiguous segments
at each wrap point removes the ambiguity without renaming anything -- which
matters, because renaming in Lightroom is not a local edit and the filename is
what ties an exported jpg back to the raw file held separately.

Run this after deduplication (`./run-dedup`). That order matters: before it, the
same photo filed into two neighbouring trips read as a counter collision and
forced a cut that was never real -- which is where the old, far denser splits
came from. Any collision left between *neighbouring* trips is reported
separately, since it means either a duplicate the dedup missed or genuinely fast
counter reuse.

Nothing is modified: this reads the library and writes a report.

    ./run-split
    python3 tools/analyze_wraparound.py --year 19 --stdout
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import NamedTuple, Optional

sys.path.insert(0, os.environ.get("PROJECT_ROOT")
                or str(Path(__file__).resolve().parents[1]))

from code.lib.config import display_path  # noqa: E402
from code.lib.trips import frame_id, trip_date  # noqa: E402

# What counts as "the same filename", and so as a collision:
#
# STEM -- the sidecar stem, which is the exported jpg's basename. This is the
#   real export-collision test. Where Lightroom hit a name clash on import it
#   already renamed one raw file with a `YYYYMMDD-` prefix, and a virtual copy
#   carries a `-N` suffix; both survive into the jpg (`20181229-_D8S7785-2.jpg`),
#   so neither needs a cut.
# FRAME -- the camera frame identity from `code/lib/trips.py`, which strips those
#   decorations. Stricter: it treats the import prefix as an accident rather than
#   part of the name. Useful as a second opinion, not as the plan.
STEM = "stem"
FRAME = "frame"


class TripInfo(NamedTuple):
    name: str
    path: Path
    when: Optional[date]
    keys: set                # collision keys held by this trip
    repeated: list           # keys held more than once *within* this trip
    ranges: dict             # camera code -> (min counter, max counter)
    sidecars: int
    unparsed: list           # filenames whose stem yielded no camera frame


class Segment(NamedTuple):
    name: str                # e.g. "2019.3"
    trips: list              # of TripInfo
    cut_by: list             # (key, name of the earlier trip holding it) pairs


def year_label(year_dir_name: str) -> str:
    """`Photos-19` -> `2019`, matching the folder names in `data/jpg`."""
    suffix = year_dir_name.split("-")[-1]
    return f"20{suffix}" if len(suffix) == 2 else suffix


def key_label(key) -> str:
    return key if isinstance(key, str) else f"_{key.code}{key.num}"


def read_trip(trip_dir: Path, collision_key: str) -> TripInfo:
    """One trip folder's filenames.

    Sidecars are collected recursively: two trips nest a sub-trip inside them
    (`2021-03-26 海湾 烟花节/2021-03-28 海湾 烟花节/`), and since a split moves
    the whole top-level folder, the nested photos belong to the parent's segment.
    """
    counts: dict = defaultdict(int)
    nums: dict = defaultdict(list)
    unparsed = []
    sidecars = 0
    for xmp in sorted(trip_dir.rglob("*.xmp")):
        sidecars += 1
        frame = frame_id(xmp.stem)
        if frame is None:
            # No camera frame in the name, so it cannot take part in a counter
            # collision by frame -- but it is still a filename that must be
            # unique within its segment.
            unparsed.append(str(xmp.relative_to(trip_dir.parent)))
            if collision_key == STEM:
                counts[xmp.stem] += 1
            continue
        counts[xmp.stem if collision_key == STEM else frame] += 1
        nums[frame.code].append(int(frame.num))
    ranges = {code: (min(v), max(v)) for code, v in nums.items()}
    repeated = [key for key, n in counts.items() if n > 1]
    return TripInfo(trip_dir.name, trip_dir, trip_date(trip_dir.name),
                    set(counts), repeated, ranges, sidecars, unparsed)


def order_trips(trips: list) -> list:
    """Chronological, with undated folders last.

    An undated folder has no place on the timeline, so it is appended in name
    order rather than guessed at; the report names them so the placement can be
    overridden by hand.
    """
    dated = sorted((t for t in trips if t.when), key=lambda t: (t.when, t.name))
    undated = sorted((t for t in trips if not t.when), key=lambda t: t.name)
    return dated + undated


def propose_segments(year: str, trips: list) -> list:
    """Greedily cut the ordered trips wherever a filename would repeat.

    Keeping each segment open until it actually collides gives the fewest
    possible segments: the no-repeat constraint only gets easier as trips are
    removed, so any valid partition must cut at or before this one does.
    """
    segments = []
    current: list = []
    owner: dict = {}          # key -> name of the trip in this segment holding it
    cut_by: list = []

    def close():
        segments.append(Segment(f"{year}.{len(segments) + 1}", current, cut_by))

    for trip in trips:
        hits = sorted(((k, owner[k]) for k in trip.keys if k in owner),
                      key=lambda h: key_label(h[0]))
        if current and hits:
            close()
            current, owner, cut_by = [], {}, hits
        current.append(trip)
        for key in trip.keys:
            owner.setdefault(key, trip.name)
    if current:
        close()
    return segments


def count_segments_strict(trips: list) -> int:
    """Segment count under the strictest test: cut on any per-code counter
    *range* overlap, not just a filename that actually repeats.

    A frame shot but never imported (deleted, never touched in Lightroom) still
    occupies its counter value and would clash in a re-export, but leaves no
    sidecar to prove it. The gap between this count and the plan's is the size
    of that blind spot.
    """
    count = 1
    seen: dict = {}
    started = False
    for trip in trips:
        collides = any(
            code in seen and seen[code][0] <= hi and lo <= seen[code][1]
            for code, (lo, hi) in trip.ranges.items()
        )
        if started and collides:
            count += 1
            seen = {}
        started = True
        for code, (lo, hi) in trip.ranges.items():
            if code in seen:
                seen[code] = (min(seen[code][0], lo), max(seen[code][1], hi))
            else:
                seen[code] = (lo, hi)
    return count


def collisions(trips: list) -> dict:
    """Key -> the ordered positions of the trips holding it, for repeats only."""
    where = defaultdict(list)
    for position, trip in enumerate(trips):
        for key in trip.keys:
            where[key].append(position)
    return {k: pos for k, pos in where.items() if len(pos) > 1}


def neighbouring(trips: list, repeats: dict, window: int) -> list:
    """Collisions between trips close together on the timeline.

    Post-dedup these should be empty or nearly so. Undated trips are excluded:
    their position is name order, not evidence of when they happened.
    """
    dated = {i for i, t in enumerate(trips) if t.when}
    found = []
    for key, positions in repeats.items():
        for a, b in zip(positions, positions[1:]):
            if b - a <= window and a in dated and b in dated:
                found.append((key, trips[a], trips[b], (trips[b].when - trips[a].when).days))
    return sorted(found, key=lambda f: (f[1].when, key_label(f[0])))


def verify(segments: list) -> list:
    """Any key repeating inside a proposed segment -- must come back empty.

    A repeat here means the plan does not actually deliver unique filenames. The
    greedy sweep rules out repeats *between* two trips of a segment, so what is
    left to catch is a trip folder holding the same name twice; no split fixes
    that, since a segment boundary never runs through a trip.
    """
    broken = []
    for segment in segments:
        seen: dict = {}
        for trip in segment.trips:
            for key in trip.repeated:
                broken.append((segment.name, key, trip.name, trip.name))
            for key in trip.keys:
                if key in seen:
                    broken.append((segment.name, key, seen[key], trip.name))
                seen[key] = trip.name
    return broken


def analyse(year_dir: Path, skip: list, window: int, collision_key: str) -> dict:
    year = year_label(year_dir.name)
    trip_dirs = [p for p in year_dir.iterdir() if p.is_dir() and p.name not in skip]
    trips = order_trips([read_trip(p, collision_key) for p in trip_dirs])
    segments = propose_segments(year, trips)
    other = FRAME if collision_key == STEM else STEM
    alt = order_trips([read_trip(p, other) for p in trip_dirs])
    repeats = collisions(trips)
    return {
        "year": year,
        "dir": year_dir,
        "trips": trips,
        "sidecars": sum(t.sidecars for t in trips),
        "collisions": repeats,
        "neighbouring": neighbouring(trips, repeats, window),
        "segments": segments,
        "alt": len(propose_segments(year, alt)),
        "strict": count_segments_strict(trips),
        "broken": verify(segments),
        "unparsed": [(t.name, f) for t in trips for f in t.unparsed],
        "undated": [t for t in trips if not t.when],
    }


def render(results: list, xmp_dir: Path, window: int, collision_key: str) -> str:
    L = []
    w = L.append
    w("# Wraparound split plan")
    w("")
    w(f"Generated by `tools/analyze_wraparound.py` over `{display_path(xmp_dir)}` -- the original")
    w("unsplit library. Read-only; nothing is moved.")
    w("")
    w("Each year is cut into date-contiguous segments in which no photo filename repeats,")
    w("so a jpg export of a segment can be keyed by filename. Segments are named to match")
    w("`data/jpg` (`2019.1`, `2019.2`, ...). Splitting is the whole fix: no photo is")
    w("renamed, since the filename is what ties an exported jpg back to its raw file.")
    w("")
    w("Run `./run-dedup` first and clear the duplicates it finds. A photo filed into two")
    w("neighbouring trips reads here as a counter collision and forces a cut that is not")
    w("real -- that is what made the previous splits so dense.")
    w("")

    w("## Summary")
    w("")
    alt_name = "by frame id" if collision_key == STEM else "by stem"
    w(f"| year | trips | sidecars | colliding names | segments | {alt_name} | strict |")
    w("|---|---|---|---|---|---|---|")
    for r in results:
        w(f"| {r['year']} | {len(r['trips'])} | {r['sidecars']} | {len(r['collisions'])} "
          f"| **{len(r['segments'])}** | {r['alt']} | {r['strict']} |")
    w(f"| **total** | {sum(len(r['trips']) for r in results)} "
      f"| {sum(r['sidecars'] for r in results)} | "
      f"| **{sum(len(r['segments']) for r in results)}** "
      f"| {sum(r['alt'] for r in results)} | {sum(r['strict'] for r in results)} |")
    w("")
    w("Two collisions are counted, and the plan uses the first:")
    w("")
    w("- **by stem** -- the sidecar stem, which is exactly the exported jpg's basename.")
    w("  Where Lightroom hit a name clash on import it already renamed one raw file with a")
    w("  `YYYYMMDD-` prefix, and virtual copies carry a `-N` suffix; both survive into the")
    w("  export (`20181229-_D8S7785-2.jpg`), so neither needs a cut.")
    w("- **by frame id** -- the camera frame with those decorations stripped. Stricter, and")
    w("  what the previous version of this tool used; the difference is the cuts that the")
    w("  import-time renaming had already made unnecessary.")
    w("- **strict** -- cut on any per-code counter *range* overlap, whether or not a name")
    w("  actually repeats. This covers frames shot but never imported, which would clash in")
    w("  a re-export while leaving no sidecar behind. The gap is what is taken on trust.")
    w("")

    broken = [(r, item) for r in results for item in r["broken"]]
    if broken:
        w(f"## ⚠️ Unsplittable collisions ({len(broken)})")
        w("")
        w("The same filename twice inside one trip folder. A segment boundary never runs")
        w("through a trip, so no split fixes these -- they need a rename or a deletion.")
        w("")
        w("| year | segment | name | trip | also in |")
        w("|---|---|---|---|---|")
        for r, (segment, key, first, second) in broken:
            w(f"| {r['year']} | {segment} | `{key_label(key)}` | `{first}` | `{second}` |")
        w("")
    else:
        w("## Verification")
        w("")
        w("Every proposed segment was re-checked: no filename repeats inside any of them.")
        w("")

    suspicious = [(r, item) for r in results for item in r["neighbouring"]]
    w(f"## Collisions between neighbouring trips ({len(suspicious)})")
    w("")
    w(f"The same name in two trips no more than {window} apart on the timeline. A counter")
    w("cannot wrap that fast, so before deduplication this meant one photo filed twice.")
    w("With the duplicates cleared, what is left is genuinely two different photos sharing")
    w("a frame number -- the counter running backwards after a card change, or a second")
    w("body using the same code. Each still forces a cut; they are listed because a cut")
    w("between trips days apart is worth an eye.")
    w("")
    if suspicious:
        w("| year | name | earlier trip | later trip | days apart |")
        w("|---|---|---|---|---|")
        for r, (key, first, second, gap) in suspicious:
            w(f"| {r['year']} | `{key_label(key)}` | `{first.name}` | "
              f"`{second.name}` | {gap} |")
    else:
        w("_None._")
    w("")

    for r in results:
        w(f"## {r['year']} -- {len(r['segments'])} segment(s)")
        w("")
        w("| segment | from | to | trips | sidecars | cut at |")
        w("|---|---|---|---|---|---|")
        for segment in r["segments"]:
            names = ", ".join(f"`{key_label(k)}`" for k, _ in segment.cut_by[:3])
            if len(segment.cut_by) > 3:
                names += f" +{len(segment.cut_by) - 3} more"
            w(f"| **{segment.name}** | {segment.trips[0].name} | {segment.trips[-1].name} "
              f"| {len(segment.trips)} | {sum(t.sidecars for t in segment.trips)} "
              f"| {names or '--'} |")
        w("")
        if r["undated"]:
            w("Undated folders, placed last in name order because the timeline says nothing")
            w("about where they belong -- move them by hand if they belong elsewhere: "
              + ", ".join(f"`{t.name}`" for t in r["undated"]))
            w("")
        if r["unparsed"]:
            w(f"{len(r['unparsed'])} filename(s) carry no camera frame number: "
              + ", ".join(f"`{t}/{f}`" for t, f in r["unparsed"][:10]))
            w("")

    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xmp-dir", type=Path, default=Path("./data/xmp0"),
                    help="unsplit library root holding Photos-YY year folders")
    ap.add_argument("--year", action="append", default=[],
                    help="limit to a year folder, by suffix or full name (repeatable): "
                         "--year 19 --year Photos-24")
    ap.add_argument("--skip", action="append", default=["0000-test"],
                    help="trip folder names to ignore (repeatable)")
    ap.add_argument("--collision-key", choices=[STEM, FRAME], default=STEM,
                    help=f"what counts as the same filename: '{STEM}' (default) is the "
                         f"exported jpg basename; '{FRAME}' strips the import prefix and "
                         "virtual-copy suffix first, which cuts more often")
    ap.add_argument("--neighbour-window", type=int, default=1,
                    help="how many trips apart still counts as neighbouring when reporting "
                         "collisions a counter wrap cannot explain (default 1)")
    ap.add_argument("--report", type=Path,
                    default=Path("./project/reports/wraparound_split_report.md"))
    ap.add_argument("--csv", type=Path,
                    default=Path("./project/reports/split_plan.csv"))
    ap.add_argument("--stdout", action="store_true",
                    help="print the report instead of writing any files")
    args = ap.parse_args()

    if not args.xmp_dir.is_dir():
        sys.exit(f"error: {args.xmp_dir} does not exist")

    year_dirs = sorted(p for p in args.xmp_dir.iterdir() if p.is_dir())
    if args.year:
        wanted = {y.lower() for y in args.year}
        year_dirs = [p for p in year_dirs
                     if p.name.lower() in wanted or p.name.split("-")[-1].lower() in wanted]
    if not year_dirs:
        sys.exit(f"error: no year folders to analyse under {args.xmp_dir}")

    results = [analyse(d, args.skip, args.neighbour_window, args.collision_key)
               for d in year_dirs]
    report = render(results, args.xmp_dir, args.neighbour_window, args.collision_key)

    if args.stdout:
        sys.stdout.write(report)
        return

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report)
    print(f"wrote {args.report}")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with open(args.csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["year", "segment", "trip", "trip_date", "sidecars", "source"])
        for r in results:
            for segment in r["segments"]:
                for trip in segment.trips:
                    writer.writerow([r["year"], segment.name, trip.name,
                                     trip.when or "", trip.sidecars, str(trip.path)])
                    rows += 1
    print(f"wrote {args.csv} ({rows} trips, "
          f"{sum(len(r['segments']) for r in results)} segments)")

    broken = sum(len(r["broken"]) for r in results)
    if broken:
        print(f"warning: {broken} filename(s) repeat inside a single trip; no split can "
              f"separate those -- see the report")


if __name__ == "__main__":
    main()
