#!/usr/bin/env python3
"""
Find sidecars that describe the same photo, across an unsplit XMP library.

Reads the original library layout (trips directly under a year folder, no
half-year level):

    <xmp-dir>/<Photos-YY>/<trip>/*.xmp

Two sidecars are the same capture when they share `xmpMM:OriginalDocumentID`
-- Lightroom's identifier for the original raw file, which survives renaming,
re-dating and timezone corrections. Where that is missing, normalised capture
time plus frame identity is used instead. See `code/lib/sidecar_meta.py` for
why the raw `exif:DateTimeOriginal` string cannot be compared directly.

Nothing is modified: this reads the library and writes a report.

    ./run-dedup
    python3 tools/dedup_sidecars.py --xmp-dir ./data/xmp0 --stdout
"""
import argparse
import collections
import csv
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, os.environ.get("PROJECT_ROOT")
                or str(Path(__file__).resolve().parents[1]))

from code.lib.config import data_dir, display_path  # noqa: E402
from code.lib.sidecar_meta import capture_key, read_meta  # noqa: E402
from code.lib.trips import frame_id, trip_date, trip_date_precision  # noqa: E402

# A capture appearing in more than one trip folder is the case that mattered for
# splitting: it made neighbouring trips look like frame-counter collisions.
CROSS_TRIP = "cross-trip"
WITHIN_TRIP = "within-trip"
VIRTUAL_COPY = "virtual-copy"


def scan(xmp_dir: Path):
    """One record per sidecar."""
    records = []
    unparsable = []
    for path in sorted(xmp_dir.rglob("*.xmp")):
        meta = read_meta(path)
        if meta is None:
            unparsable.append(path)
            continue
        rel = path.relative_to(xmp_dir)
        # A few trips nest a sub-trip inside another trip folder, so the trip is
        # whichever folder actually holds the sidecar -- not a fixed depth.
        records.append({
            "path": path,
            "rel": rel,
            "year": rel.parts[0] if rel.parts else "",
            "trip": rel.parts[-2] if len(rel.parts) >= 2 else "",
            "trip_path": str(rel.parent),
            "stem": path.stem,
            "frame": frame_id(path.stem),
            "meta": meta,
        })
    return records, unparsable


def group_duplicates(records):
    """Captures with more than one sidecar, classified."""
    by_key = collections.defaultdict(list)
    keyless = []
    for rec in records:
        key = capture_key(rec["meta"], rec["frame"])
        if key is None:
            keyless.append(rec)
            continue
        by_key[key].append(rec)

    groups = []
    for key, members in by_key.items():
        if len(members) < 2:
            continue
        trips = {m["trip_path"] for m in members}
        if all(m["meta"].is_virtual_copy for m in members[1:]) and len(trips) == 1:
            kind = VIRTUAL_COPY
        elif any(m["meta"].is_virtual_copy for m in members) and len(trips) == 1:
            kind = VIRTUAL_COPY
        elif len(trips) > 1:
            kind = CROSS_TRIP
        else:
            kind = WITHIN_TRIP
        groups.append({
            "key": key,
            "kind": kind,
            "members": order_members(members),
        })
    groups.sort(key=lambda g: (g["kind"], str(g["members"][0]["rel"])))
    return groups, keyless


def days_off(rec):
    """How far the capture date sits from the trip folder's own date.

    A month-precision folder (`2018-11 Yunnan`) names a whole month, so anything
    captured in that month is where it belongs and counts as 0 -- measuring
    against the 1st would flag the entire trip.
    """
    folder_date = trip_date(rec["trip"])
    captured = rec["meta"].capture_date
    if folder_date is None or not captured:
        return None
    try:
        year, month, day = (int(x) for x in captured.split("-"))
    except ValueError:
        return None
    if trip_date_precision(rec["trip"]) == "month":
        if (year, month) == (folder_date.year, folder_date.month):
            return 0
        first_of_capture_month = date(year, month, 1)
        return abs((first_of_capture_month - folder_date).days)
    return abs((date(year, month, day) - folder_date).days)


def order_members(members):
    """Put the copy to keep first.

    The keeper is the one filed in the trip whose date best matches when the
    photo was actually taken -- the other copies are the misfiled ones. A real
    edit (virtual copy) never wins the position, and path order settles ties so
    the report is stable between runs.
    """
    def rank(rec):
        off = days_off(rec)
        return (rec["meta"].is_virtual_copy,
                off if off is not None else 10**6,
                str(rec["rel"]))
    return sorted(members, key=rank)


def find_misplaced(records, threshold):
    """Sidecars whose capture date is far from their trip folder's date.

    Reported per trip, because the two causes need opposite fixes: a handful of
    stray photos in an otherwise consistent trip is a misfiling, while a whole
    trip sitting away from its folder date just means the folder is named for a
    different day than it holds.
    """
    by_trip = collections.defaultdict(list)
    for rec in records:
        if rec["trip"]:
            by_trip[rec["trip_path"]].append(rec)

    misfiled, mislabelled_trips = [], []
    for trip_path, members in sorted(by_trip.items()):
        year = members[0]["year"]
        offs = [(rec, days_off(rec)) for rec in members]
        usable = [(r, o) for r, o in offs if o is not None]
        if not usable:
            continue
        outliers = [(r, o) for r, o in usable if o > threshold]
        if not outliers:
            continue
        if len(outliers) > len(usable) / 2:
            spread = sorted({r["meta"].capture_date for r, _ in usable})
            mislabelled_trips.append({
                "year": year, "trip": trip_path, "photos": len(usable),
                "outliers": len(outliers),
                "dates": f"{spread[0]} .. {spread[-1]}" if spread else "",
            })
        else:
            for rec, off in sorted(outliers, key=lambda x: -x[1]):
                misfiled.append({"rec": rec, "days_off": off})
    misfiled.sort(key=lambda m: -m["days_off"])
    return misfiled, mislabelled_trips


def render(groups, records, unparsable, keyless, misfiled, mislabelled, xmp_dir, threshold):
    L = []
    w = L.append
    by_kind = collections.Counter(g["kind"] for g in groups)
    redundant = sum(len(g["members"]) - 1 for g in groups if g["kind"] != VIRTUAL_COPY)

    w("# Sidecar deduplication report")
    w("")
    w(f"Generated by `tools/dedup_sidecars.py` over `{display_path(xmp_dir)}`. Read-only -- nothing in the")
    w("library is modified.")
    w("")
    w("## How a duplicate is decided")
    w("")
    w("Two sidecars are the same capture when they share `xmpMM:OriginalDocumentID`,")
    w("Lightroom's identifier for the original raw file. That is preferred over comparing")
    w("date/time/filename because it survives three things that break a literal comparison:")
    w("")
    w("- **Sub-second precision is inconsistent.** Only ~10% of these sidecars record it, so")
    w("  the same photo appears as `2018-03-24T12:30:58.82` in one copy and")
    w("  `2018-03-24T12:30:58` in the other. Comparing the strings misses it; the times are")
    w("  normalised to whole seconds here.")
    w("- **Timezone corrections.** A copy re-dated to local time reads")
    w("  `2019-11-22T17:42:30+08:00` against `2019-11-23T00:42:30+08:00` for the same frame --")
    w("  a different calendar day, so a date-based key splits them apart.")
    w("- **Renaming on import.** A clash on import adds a `YYYYMMDD-` prefix to one copy only.")
    w("")
    w("Where `OriginalDocumentID` is absent the fallback key is normalised capture time plus")
    w("frame identity (`code/lib/trips.py:frame_id`, which strips the import prefix and the")
    w("virtual-copy suffix).")
    w("")

    w("## Summary")
    w("")
    w(f"- sidecars scanned: **{len(records)}**")
    w(f"- **cross-trip duplicates: {by_kind[CROSS_TRIP]} captures** -- the same photo filed in")
    w("  two or more trips. These are what made neighbouring trips look like frame-counter")
    w("  collisions during splitting.")
    w(f"- within-trip duplicates: {by_kind[WITHIN_TRIP]} captures -- two sidecars for one photo")
    w("  in a single folder")
    w(f"- virtual copies: {by_kind[VIRTUAL_COPY]} captures -- Lightroom alternate edits sharing")
    w("  one original. **Deliberate; do not delete.**")
    w(f"- redundant sidecars to remove (excluding virtual copies): **{redundant}**")
    if keyless:
        w(f"- sidecars with no usable identity field: {len(keyless)}")
    if unparsable:
        w(f"- sidecars that would not parse: {len(unparsable)}")
    w("")
    w("In each listing below the **first line is the copy to keep** -- the one whose trip")
    w("folder date best matches when the photo was actually taken -- and the indented lines")
    w("beneath it are the redundant copies.")
    w("")

    for kind, title, note in (
        (CROSS_TRIP, "Cross-trip duplicates",
         "The same capture filed under two or more trips. Removing the indented copies is what "
         "clears the false collisions before re-splitting."),
        (WITHIN_TRIP, "Within-trip duplicates",
         "Two sidecars for one capture inside a single trip folder."),
        (VIRTUAL_COPY, "Virtual copies -- review, do not bulk-delete",
         "These share an original but have their own `xmpMM:DocumentID`: they are alternate "
         "edits made on purpose. Listed so they are not mistaken for import duplicates."),
    ):
        selected = [g for g in groups if g["kind"] == kind]
        w(f"## {title} ({len(selected)})")
        w("")
        w(note)
        w("")
        if not selected:
            w("_None._")
            w("")
            continue
        w("```")
        for group in selected:
            keeper, *rest = group["members"]
            off = days_off(keeper)
            suffix = f"   [captured {keeper['meta'].capture_date}]" if off else ""
            w(f"{keeper['rel']}{suffix}")
            for member in rest:
                moff = days_off(member)
                tag = f"   [captured {member['meta'].capture_date}"
                tag += f", {moff}d from folder date]" if moff else "]"
                w(f"    {member['rel']}{tag}")
        w("```")
        w("")

    w("## Misplaced photos")
    w("")
    w(f"Sidecars whose capture date is more than {threshold} day(s) from their trip folder's")
    w("date. Two different problems live here and they need opposite fixes.")
    w("")
    w(f"### Strays in an otherwise consistent trip ({len(misfiled)})")
    w("")
    w("A few photos filed under the wrong trip. These are the ones to move.")
    w("")
    if misfiled:
        w("```")
        for item in misfiled[:400]:
            rec = item["rec"]
            w(f"{rec['rel']}   [captured {rec['meta'].capture_date}, "
              f"{item['days_off']}d from folder date]")
        if len(misfiled) > 400:
            w(f"... {len(misfiled) - 400} more, see the CSV")
        w("```")
    else:
        w("_None._")
    w("")
    w(f"### Trips whose folder date does not describe their contents ({len(mislabelled)})")
    w("")
    w("More than half the trip's photos sit away from the folder date, so the folder is named")
    w("for a different day than it holds -- rename the trip rather than move the photos.")
    w("")
    if mislabelled:
        w("| trip | photos | outliers | capture dates |")
        w("|---|---|---|---|")
        for item in mislabelled:
            w(f"| `{item['trip']}` | {item['photos']} | {item['outliers']} | "
              f"{item['dates']} |")
    else:
        w("_None._")
    w("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xmp-dir", type=Path, default=None,
                    help="Default: <data-dir>/xmp, resolved by config.data_dir()")
    ap.add_argument("--report", type=Path,
                    default=Path("./project/reports/sidecar_dedup_report.md"))
    ap.add_argument("--csv", type=Path,
                    default=Path("./project/reports/sidecar_duplicates.csv"))
    ap.add_argument("--misplaced-threshold", type=int, default=2,
                    help="days between capture date and trip folder date before a sidecar "
                         "counts as misplaced (default 2, to absorb multi-day trips)")
    ap.add_argument("--stdout", action="store_true",
                    help="print the report instead of writing any files")
    args = ap.parse_args()
    args.xmp_dir = args.xmp_dir or data_dir() / "xmp"

    if not args.xmp_dir.is_dir():
        sys.exit(f"error: {args.xmp_dir} does not exist")

    records, unparsable = scan(args.xmp_dir)
    if not records:
        sys.exit(f"error: no .xmp files under {args.xmp_dir}")
    groups, keyless = group_duplicates(records)
    misfiled, mislabelled = find_misplaced(records, args.misplaced_threshold)
    report = render(groups, records, unparsable, keyless, misfiled, mislabelled,
                    args.xmp_dir, args.misplaced_threshold)

    if args.stdout:
        sys.stdout.write(report)
        return

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report)
    print(f"wrote {args.report}")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["kind", "role", "group", "year", "trip", "stem", "captured",
              "days_off_folder_date", "is_virtual_copy", "original_document_id", "path"]
    rows = 0
    with open(args.csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for number, group in enumerate(groups, 1):
            for position, member in enumerate(group["members"]):
                off = days_off(member)
                writer.writerow({
                    "kind": group["kind"],
                    "role": "keep" if position == 0 else "duplicate",
                    "group": number,
                    "year": member["year"], "trip": member["trip"], "stem": member["stem"],
                    "captured": member["meta"].captured,
                    "days_off_folder_date": "" if off is None else off,
                    "is_virtual_copy": member["meta"].is_virtual_copy,
                    "original_document_id": member["meta"].original_document_id,
                    "path": str(member["path"]),
                })
                rows += 1
    print(f"wrote {args.csv} ({rows} rows, {len(groups)} captures)")

    misplaced_csv = args.csv.with_name("sidecar_misplaced.csv")
    with open(misplaced_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["year", "trip", "stem", "captured", "days_off_folder_date", "path"])
        for item in misfiled:
            rec = item["rec"]
            writer.writerow([rec["year"], rec["trip"], rec["stem"],
                             rec["meta"].captured, item["days_off"], str(rec["path"])])
    print(f"wrote {misplaced_csv} ({len(misfiled)} rows)")


if __name__ == "__main__":
    main()
