#!/usr/bin/env python3
"""Fold a partial labelling run back into a full one.

Re-running a couple of years produces a complete run over those years and
nothing else -- a CSV with only their rows, and a `raw/` tree that is a *full*
copy of `data/xmp` in which only the selected years were touched. Merging is
therefore not a directory copy: taking `raw/` wholesale would replace every
other year's labelled sidecar with an unlabelled one.

The rule here is precise instead: **a sidecar is taken from the overlay only if
the overlay's CSV has a row for it.** Everything the overlay did not label stays
exactly as the base left it.

    python3 -m tools.merge_label_run --base data/label --overlay output_003_rerun-24-25 \\
                             --out data/label-v2 --dry-run

Neither input is modified. The output is a new label directory, so a bad merge
costs a re-run of this script.
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

CSV_NAME = "bird_identification_output.csv"


def read_rows(path: Path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        return list(reader), list(reader.fieldnames or [])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True,
                    help="the full run to merge into, e.g. data/label")
    ap.add_argument("--overlay", type=Path, required=True,
                    help="a partial run whose rows win, e.g. an output_NNN dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base_rows, base_fields = read_rows(args.base / CSV_NAME)
    over_rows, over_fields = read_rows(args.overlay / CSV_NAME)
    if base_fields != over_fields:
        sys.exit("error: the two runs have different columns; merging them would "
                 f"misalign rows.\n  base:    {base_fields}\n  overlay: {over_fields}")

    over_by_key = {r["jpg"]: r for r in over_rows}
    if len(over_by_key) != len(over_rows):
        sys.exit("error: the overlay CSV has duplicate `jpg` keys")

    merged, replaced, changed = [], 0, 0
    unseen = dict(over_by_key)          # drained as base rows match, leaving the extras
    for r in base_rows:
        hit = unseen.pop(r["jpg"], None)
        if hit is None:
            merged.append(r)
            continue
        merged.append(hit)
        replaced += 1
        if (hit["category"], hit["applied"]) != (r["category"], r["applied"]):
            changed += 1
    # Rows the overlay has and the base does not: a new export, or mismatched
    # datasets. Appended rather than dropped, and reported so it is not silent.
    added = list(unseen.values())
    merged.extend(added)

    print(f"base rows    : {len(base_rows)}")
    print(f"overlay rows : {len(over_rows)}")
    print(f"  replaced   : {replaced}")
    print(f"  appended   : {len(added)}" + ("  <- not in the base run" if added else ""))
    print(f"  of the replaced, category or applied differs: {changed}")
    print(f"merged rows  : {len(merged)}")

    if args.dry_run:
        print("dry run: nothing written")
        return
    if args.out.exists():
        sys.exit(f"error: {args.out} exists; remove it or pick another --out")

    print(f"copying {args.base} -> {args.out} ...")
    shutil.copytree(args.base, args.out)

    with open(args.out / CSV_NAME, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=base_fields)
        w.writeheader()
        w.writerows(merged)

    # Sidecars: only those the overlay actually has a row for.
    copied = missing = 0
    for r in over_rows:
        rel = (r.get("xmp") or "").strip()
        if not rel:
            continue                      # csv-only row; no sidecar either side
        src = args.overlay / "raw" / rel
        if not src.is_file():
            missing += 1
            continue
        dst = args.out / "raw" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    print(f"sidecars taken from the overlay: {copied}"
          + (f", {missing} named but absent" if missing else ""))

    for name in ("args.json", "bird_label.log", "processed.txt"):
        src = args.overlay / name
        if src.is_file():
            shutil.copy2(src, args.out / f"overlay.{name}")
    print(f"overlay's args.json / log / checkpoint kept alongside as overlay.*")
    print(f"merged run written to {args.out}")


if __name__ == "__main__":
    main()
