#!/usr/bin/env python3
"""Check whether a sidecar tree still matches what we wrote into it.

Written for the round trip through Lightroom. The labels go out as sidecars,
Lightroom reads them, and from that moment the catalog also holds an opinion
about those files -- with *Automatically write changes into XMP* enabled, or on
an explicit Save Metadata to File, Lightroom writes its own version back and its
copy wins. Nothing warns; the file simply stops saying what we wrote.

So compare, and be specific about the answer. A byte difference alone is not
informative: Lightroom re-serialising a file it agrees with looks identical to
Lightroom overwriting a label. The classes that matter are different:

    identical            nothing touched it
    keywords differ      our label is gone or altered -- the case to worry about
    keywords intact,     Lightroom rewrote the file but kept the keywords, e.g.
      other fields differ  develop settings or a metadata date. Usually benign,
                         but it means the catalog is writing, so the next edit
                         could take the keywords too
    missing              the file is not there at all

    tools/verify_applied.py --reference data/label-v2/raw --against /path/to/library

Read-only. Nothing is modified either side.
"""

import argparse
import collections
import csv
import hashlib
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, os.environ.get("PROJECT_ROOT")
                or str(Path(__file__).resolve().parents[1]))

from code.lib import xmp_write  # noqa: E402

IDENTICAL = "identical"
KEYWORDS_DIFFER = "keywords-differ"
OTHER_DIFFER = "keywords-intact-other-differs"
MISSING = "not-in-reference"
UNPARSEABLE = "unparseable"

FIELDS = ["status", "path", "reference_keywords", "library_keywords"]


def digest(path: Path):
    return hashlib.md5(path.read_bytes()).digest()


def keywords(path: Path):
    root = ET.parse(path).getroot()
    return (tuple(xmp_write.items_of(root, xmp_write.SUBJECT_TAG)),
            tuple(xmp_write.items_of(root, xmp_write.HIER_TAG)))


def compare(reference: Path, against: Path):
    """Walk the tree under scrutiny, not the reference.

    A partial copy-back is normal -- only the sidecars Lightroom actually
    touched are worth bringing home -- so anything in the reference that is
    absent here was simply not collected, which is not a finding. The reverse is:
    a file present here with no counterpart never came from us.
    """
    for lib in sorted(against.rglob("*.xmp")):
        rel = lib.relative_to(against)
        ref = reference / rel
        if not ref.is_file():
            yield MISSING, rel, "", ""
            continue
        if digest(ref) == digest(lib):
            yield IDENTICAL, rel, "", ""
            continue
        try:
            a, b = keywords(ref), keywords(lib)
        except ET.ParseError:
            yield UNPARSEABLE, rel, "", ""
            continue
        if a == b:
            yield OTHER_DIFFER, rel, "", ""
        else:
            yield KEYWORDS_DIFFER, rel, ";".join(a[0]), ";".join(b[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path, required=True,
                    help="what we wrote, e.g. data/label-v2/raw")
    ap.add_argument("--against", type=Path, required=True,
                    help="the tree to check, e.g. the Lightroom library's sidecars")
    ap.add_argument("--out", type=Path,
                    default=Path("project/reports/verify_applied.csv"))
    args = ap.parse_args()

    for d in (args.reference, args.against):
        if not d.is_dir():
            sys.exit(f"error: {d} is not a directory")

    rows, counts = [], collections.Counter()
    for status, rel, ref_kw, lib_kw in compare(args.reference, args.against):
        counts[status] += 1
        if status != IDENTICAL:
            rows.append({"status": status, "path": rel.as_posix(),
                         "reference_keywords": ref_kw, "library_keywords": lib_kw})

    total = sum(counts.values())
    in_ref = sum(1 for _ in args.reference.rglob("*.xmp"))
    print(f"compared {total} sidecars from {args.against}")
    print(f"  (the reference holds {in_ref}; the rest were not copied back)")
    for status in (IDENTICAL, OTHER_DIFFER, KEYWORDS_DIFFER, MISSING, UNPARSEABLE):
        if counts[status]:
            print(f"  {counts[status]:7d}  {status}")

    if counts[KEYWORDS_DIFFER]:
        print(f"\n  *** {counts[KEYWORDS_DIFFER]} sidecars no longer carry the label we wrote.")
        print("      Lightroom's copy won. The restic snapshot in project/bookeeping/ is")
        print("      the rollback; re-applying from the reference is the repair.")
    elif counts[OTHER_DIFFER]:
        print("\n  Keywords intact everywhere, but Lightroom has rewritten files -- so the")
        print("  catalog is writing sidecars. Check Catalog Settings -> Metadata ->")
        print("  'Automatically write changes into XMP' before the next edit.")
    elif not counts[MISSING]:
        print("\n  Every sidecar still says exactly what we wrote.")

    if rows:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
