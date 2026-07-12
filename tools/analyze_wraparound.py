#!/usr/bin/env python3
"""
Analyze camera-filename-counter collisions across trip subfolders within a
year of raw XMP sidecars (e.g. data/Photos-18, data/Photos-19), and suggest
split points that partition the year into segments with no counter overlap
per camera code.

This does NOT touch any jpg/ export data (which doesn't exist yet for these
older years) -- it works purely off the xmp filenames + trip-folder dates,
since a jpg export of a raw file inherits the raw file's original camera
filename. A collision here is a strong predictor of a jpg collision once
these years get exported.

Usage:
    python3 scripts/analyze_wraparound.py data/Photos-18
    python3 scripts/analyze_wraparound.py data/Photos-19
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# Trip folders are named "YYYY-MM-DD ..." or "YYYY-MM ..." (a few older ones
# lack a day). "0000-test" is a scratch folder, not a real trip.
TRIP_DATE_RE = re.compile(r'^(\d{4})-(\d{2})(?:-(\d{2}))?')

# Filename shapes observed under data/Photos-18 and data/Photos-19:
#   _D8S7785.xmp                              plain camera file
#   _D8S7785-2.xmp                            Lightroom virtual copy
#   20181229-_D8S7785.xmp                     date-disambiguated on import
#   20181229-_D8S7785-2.xmp                   both of the above
#   20190113-_D8S0025_<host>_<ts>_Conflict.xmp  filesystem sync-conflict copy
# The camera "code" is a 2-4 char alnum body-id immediately before a run of
# 4+ digits (the counter); that (code, counter) pair is what a camera would
# also burn into the exported jpg filename.
STEM_RE = re.compile(
    r'^(?:\d{8}-)?'          # optional date-disambiguation prefix
    r'_?'                    # leading underscore
    r'(?P<code>[A-Za-z0-9_]*?)'
    r'(?P<num>\d{4,})'
    r'(?:-\d+)?'             # optional virtual-copy suffix
    r'(?:_.*)?$'             # optional sync-conflict suffix
)


def trip_sort_key(folder_name: str):
    m = TRIP_DATE_RE.match(folder_name)
    if not m:
        return (folder_name, 0)
    y, mo, d = m.groups()
    return (int(y), int(mo), int(d or 0))


def parse_stem(stem: str):
    m = STEM_RE.match(stem)
    if not m:
        return None
    return m.group('code'), m.group('num')


def propose_splits(trips, trip_pairs: dict[str, set]):
    """Greedily partition chronologically-ordered trips into the minimum
    number of contiguous segments such that no (code, num) stem repeats
    within a segment. A segment boundary is forced right before the first
    trip that would collide with something already in the open segment."""
    segments = []
    current_trips = []
    current_used: set = set()
    for trip in trips:
        pairs = trip_pairs.get(trip.name, set())
        if current_trips and (pairs & current_used):
            segments.append(current_trips)
            current_trips = []
            current_used = set()
        current_trips.append(trip)
        current_used |= pairs
    if current_trips:
        segments.append(current_trips)
    return segments


def propose_splits_strict(trips, trip_code_ranges: dict[str, dict[str, tuple[int, int]]]):
    """Same greedy idea, but cuts on any [min,max] counter *range* overlap
    per camera code, not just an exact surviving-frame match. This guards
    against frames that were shot but never made it into this XMP library
    (deleted, never touched in Lightroom, etc.) -- their counter value
    still occupies the range even though no XMP proves the collision.
    Produces >= as many segments as propose_splits(); the gap between the
    two segment counts is the size of that blind spot."""
    segments = []
    current_trips = []
    current_ranges: dict[str, tuple[int, int]] = {}

    def overlaps(a, b):
        return a[0] <= b[1] and b[0] <= a[1]

    for trip in trips:
        trip_ranges = trip_code_ranges.get(trip.name, {})
        collides = any(
            code in current_ranges and overlaps(current_ranges[code], rng)
            for code, rng in trip_ranges.items()
        )
        if current_trips and collides:
            segments.append(current_trips)
            current_trips = []
            current_ranges = {}
        current_trips.append(trip)
        for code, (lo, hi) in trip_ranges.items():
            if code in current_ranges:
                clo, chi = current_ranges[code]
                current_ranges[code] = (min(clo, lo), max(chi, hi))
            else:
                current_ranges[code] = (lo, hi)
    if current_trips:
        segments.append(current_trips)
    return segments


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('year_dir', help='e.g. data/Photos-18')
    ap.add_argument('--skip', action='append', default=['0000-test'], help='trip folder names to ignore (repeatable)')
    args = ap.parse_args()

    year_dir = Path(args.year_dir)
    trips = sorted(
        (p for p in year_dir.iterdir() if p.is_dir() and p.name not in args.skip),
        key=lambda p: trip_sort_key(p.name),
    )

    # (code, num) -> set of trip folder names it appears in
    occurrences: dict[tuple[str, str], set[str]] = defaultdict(set)
    # per code: trip -> (min_num, max_num) as ints, in chronological order
    per_code_trip_range: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    # per trip: set of (code, num) pairs it contains, for split proposal
    trip_pairs: dict[str, set] = defaultdict(set)
    unparsed = []

    for trip in trips:
        per_trip_nums: dict[str, list[int]] = defaultdict(list)
        for xmp in trip.glob('*.xmp'):
            parsed = parse_stem(xmp.stem)
            if parsed is None:
                unparsed.append((trip.name, xmp.name))
                continue
            code, num = parsed
            occurrences[(code, num)].add(trip.name)
            trip_pairs[trip.name].add((code, num))
            per_trip_nums[code].append(int(num))
        for code, nums in per_trip_nums.items():
            per_code_trip_range[code].append((trip.name, min(nums), max(nums)))

    if unparsed:
        print(f"⚠️  {len(unparsed)} filenames didn't match the expected pattern (showing up to 10):")
        for trip_name, fname in unparsed[:10]:
            print(f"    {trip_name}/{fname}")
        print()

    # --- Cross-trip collisions: exact same (code, counter) in >1 trip ---
    collisions = {k: v for k, v in occurrences.items() if len(v) > 1}
    print(f"=== {year_dir.name}: cross-trip filename collisions ===")
    if not collisions:
        print("None found (no exact (camera-code, counter) pair repeats across trips).\n")
    else:
        print(f"{len(collisions)} colliding (code, counter) pairs across trips:\n")
        for (code, num), trip_set in sorted(collisions.items()):
            print(f"  _{code}{num}  ->  {', '.join(sorted(trip_set, key=trip_sort_key))}")
        print()

    # --- Per-code chronological counter ranges, to spot wraparound resets ---
    print(f"=== {year_dir.name}: per-camera-code counter range by trip (chronological) ===")
    for code in sorted(per_code_trip_range):
        entries = sorted(per_code_trip_range[code], key=lambda e: trip_sort_key(e[0]))
        print(f"\n-- code {code} ({len(entries)} trips) --")
        prev_max = None
        for trip_name, lo, hi in entries:
            flag = ""
            if prev_max is not None and lo < prev_max:
                flag = "  <-- overlaps/resets vs previous trip's max"
            print(f"  {trip_name:40s} {lo:>6} - {hi:<6}{flag}")
            prev_max = hi

    # --- Proposed collision-free split points ---
    segments = propose_splits(trips, trip_pairs)
    print(f"\n=== {year_dir.name}: proposed split into {len(segments)} collision-free segment(s) ===")
    for i, seg in enumerate(segments, 1):
        start = seg[0].name
        end = seg[-1].name
        print(f"  Segment {i}: {start}  ->  {end}   ({len(seg)} trips)")

    # --- Same, but under the stricter range-overlap criterion ---
    trip_code_ranges: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
    for code, entries in per_code_trip_range.items():
        for trip_name, lo, hi in entries:
            trip_code_ranges[trip_name][code] = (lo, hi)
    strict_segments = propose_splits_strict(trips, trip_code_ranges)
    print(f"\n=== {year_dir.name}: strict (range-overlap) split into {len(strict_segments)} segment(s) ===")
    print("(guards against frames absent from this XMP library; compare count above to gauge the blind spot)")
    for i, seg in enumerate(strict_segments, 1):
        start = seg[0].name
        end = seg[-1].name
        print(f"  Segment {i}: {start}  ->  {end}   ({len(seg)} trips)")


if __name__ == '__main__':
    main()
