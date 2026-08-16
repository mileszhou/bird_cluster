#!/usr/bin/env python3
"""
Undo Lightroom's flat "folder-name-in-filename" export and recover the
mirrored `<Photos-YY>/<trip>/*.jpg` structure that the rest of this project
expects.

Lightroom's third-party structure-preserving exporter is unreliable, so the
export instead used Lightroom's custom filename template to flatten every
trip into one folder, encoding the source trip name in the filename:

    <jpg-dir>/_flat/<trip>~~<original filename>.jpg

This script reads each name, looks up which `Photos-YY` the trip belongs to,
and moves the file to `<jpg-dir>/<Photos-YY>/<trip>/<original filename>.jpg`.

The trip -> year mapping comes primarily from the existing `data/xmp` tree
(817 of 875 trips have sidecars there). A trip with no sidecars at all --
JPEG-only, the source was never a raw file -- has no entry to look up, so its
year is taken from the trip name's own leading `YYYY-` (64 of 875 trips; see
project/messages/2026-08-05.01). Every trip name in the current dataset
starts with `YYYY-`, so this fallback always applies; a trip that doesn't
match is left in `_flat` and reported rather than guessed at.

Read-only by default (--dry-run); pass --apply to actually move files.

    ./run-decode-jpg --dry-run
    ./run-decode-jpg --apply
"""
import argparse
import collections
import csv
import re
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("PROJECT_ROOT")
                or str(Path(__file__).resolve().parents[1]))

from code.lib.config import data_dir  # noqa: E402

DELIM = "~~"
YEAR_RE = re.compile(r"^(\d{4})-")


def trip_to_year_map(xmp_dir: Path) -> dict[str, str]:
    """trip folder name -> Photos-YY, from the actual xmp tree."""
    mapping: dict[str, str] = {}
    for year_dir in sorted(xmp_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        for trip_dir in sorted(year_dir.iterdir()):
            if trip_dir.is_dir():
                mapping[trip_dir.name] = year_dir.name
    return mapping


def year_from_trip_name(trip: str) -> str | None:
    m = YEAR_RE.match(trip)
    if not m:
        return None
    return "Photos-" + m.group(1)[2:]


def plan_moves(flat_dir: Path, trip_years: dict[str, str]):
    """Return (moves, unresolved) where moves is a list of (src, dest) and
    unresolved is a list of filenames whose trip has no year at all."""
    moves = []
    unresolved = []
    used_name_fallback = collections.Counter()
    for src in sorted(flat_dir.iterdir()):
        if not src.is_file():
            continue
        if DELIM not in src.name:
            unresolved.append(src.name)
            continue
        trip, filename = src.name.split(DELIM, 1)
        year = trip_years.get(trip)
        if year is None:
            year = year_from_trip_name(trip)
            if year is not None:
                used_name_fallback[trip] += 1
        if year is None:
            unresolved.append(src.name)
            continue
        dest = flat_dir.parent / year / trip / filename
        moves.append((src, dest))
    return moves, unresolved, used_name_fallback


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jpg-dir", default=None, help="JPEG export root (default: <data-dir>/jpg)")
    ap.add_argument("--xmp-dir", default=None, help="Sidecar root, for the trip->year mapping (default: <data-dir>/xmp)")
    ap.add_argument("--flat-subdir", default="_flat", help="Name of the flat folder under --jpg-dir (default: _flat)")
    ap.add_argument("--apply", action="store_true", help="Actually move files (default is a dry run)")
    ap.add_argument("--manifest", default="./project/reports/decode_flat_jpg_manifest.csv",
                     help="Where to record src/dest of every move actually made, for reversibility "
                          "(default: ./project/reports/decode_flat_jpg_manifest.csv)")
    args = ap.parse_args()
    root = data_dir()
    args.jpg_dir = args.jpg_dir or root / "jpg"
    args.xmp_dir = args.xmp_dir or root / "xmp"

    jpg_dir = Path(args.jpg_dir)
    xmp_dir = Path(args.xmp_dir)
    flat_dir = jpg_dir / args.flat_subdir

    if not flat_dir.is_dir():
        print(f"error: {flat_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    trip_years = trip_to_year_map(xmp_dir)
    moves, unresolved, used_name_fallback = plan_moves(flat_dir, trip_years)

    by_year = collections.Counter(dest.parts[-3] for _, dest in moves)
    print(f"{len(moves)} files resolved across {len(by_year)} years, {len(unresolved)} unresolved")
    for year in sorted(by_year):
        print(f"  {year}: {by_year[year]}")
    if used_name_fallback:
        print(f"{sum(used_name_fallback.values())} files ({len(used_name_fallback)} trips) resolved via name-derived year, no sidecars present:")
        for trip in sorted(used_name_fallback):
            print(f"  {trip}: {used_name_fallback[trip]}")
    if unresolved:
        print(f"UNRESOLVED ({len(unresolved)}), left in place:")
        for name in unresolved[:50]:
            print(f"  {name}")
        if len(unresolved) > 50:
            print(f"  ... and {len(unresolved) - 50} more")

    collisions = [(src, dest) for src, dest in moves if dest.exists()]
    if collisions:
        print(f"COLLISIONS ({len(collisions)}), destination already exists -- skipped:")
        for src, dest in collisions[:50]:
            print(f"  {src.name} -> {dest}")

    if not args.apply:
        print("\nDry run only -- pass --apply to move files.")
        return

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    moved = 0
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["src", "dest"])
        for src, dest in moves:
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dest)
            writer.writerow([str(src), str(dest)])
            moved += 1
    print(f"\nMoved {moved} files. Manifest: {manifest_path}")

    remaining = list(flat_dir.iterdir())
    if not remaining:
        flat_dir.rmdir()
        print(f"{flat_dir} is empty, removed.")
    else:
        print(f"{len(remaining)} files remain in {flat_dir} (unresolved/collisions).")


if __name__ == "__main__":
    main()
