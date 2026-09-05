#!/usr/bin/env python3
"""Write a scope manifest naming the images a previous labelling put in a category.

**Why a manifest and not a `--category` flag.** Scope in this project is a
manifest and nothing else: `--years` was removed for doing the same job through a
second mechanism, and one way to narrow a run means one place for it to be wrong.
A generated manifest keeps that property and gains two more -- the file is a
record of exactly which images a run saw, and `run.json` stores the path, so the
population behind a result stays recoverable.

**Why the category and not the species.** Two independent labellers agree on
`category` for 0.9626 of images and on `label` for 0.288. The category is the
part of a labelling worth trusting, and this is the one place that difference can
be spent: skipping the 22,030 images already called `people`, `scenery` or
`animal` removes 45% of a paid run's cost without touching the birds.

The effective category resolves the never-demote rule, the same way `embed.py`
does: where `applied` is `kept-existing` the library kept `prior_category`, so a
plain `category == 'bird'` test silently drops the rows that rule exists to
protect.

    python3 -m tools.manifest_from_labels --category bird
    python3 -m tools.manifest_from_labels --category people --category scenery \
        --out local/manifests/not-birds.txt
    python3 -m tools.manifest_from_labels --category bird --min-confidence 0.8
"""

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("PROJECT_ROOT")
                or str(Path(__file__).resolve().parents[1]))

from code.lib.config import PROJECT_ROOT, data_dir  # noqa: E402


def effective_category(row) -> str:
    """This run's verdict, unless the never-demote rule overruled it.

    Mirrors `embed.effective_category()`. Where `applied` is `kept-existing` the
    sidecar kept `prior_category` and the new verdict lost, so reading `category`
    alone describes a decision the library did not adopt.
    """
    if (row.get("applied") or "").strip() == "kept-existing":
        prior = (row.get("prior_category") or "").strip().lower()
        if prior:
            return prior
    return (row.get("category") or "").strip().lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label-dir", type=Path, default=None,
                    help="directory holding bird_identification_output.csv "
                         "(default: <data>/label)")
    ap.add_argument("--category", action="append", required=True,
                    help="category to include; repeat for several")
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="skip rows below this confidence. Note the labeller's "
                         "confidence is not calibrated -- it averages 0.968 "
                         "against a measured ~35%% species error -- so this is a "
                         "blunt instrument, useful mainly to exclude the tail")
    ap.add_argument("--out", type=Path, default=None,
                    help="manifest to write (default: "
                         "local/manifests/<category>.txt). Must resolve inside "
                         "manifests/ or local/manifests/, which is where a run "
                         "can find it again")
    args = ap.parse_args()

    label_dir = args.label_dir or (data_dir() / "label")
    csv_path = Path(label_dir) / "bird_identification_output.csv"
    if not csv_path.is_file():
        sys.exit(f"error: no labelling at {csv_path}. Name one with --label-dir.")

    wanted = {c.strip().lower() for c in args.category}
    out = args.out or (PROJECT_ROOT / "local" / "manifests"
                       / ("-".join(sorted(wanted)) + ".txt"))
    out = Path(out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    # The same containment rule path_filter enforces when reading one: a manifest
    # outside the repository makes a run.json a dangling reference.
    allowed = (PROJECT_ROOT / "manifests", PROJECT_ROOT / "local" / "manifests")
    if not any(str(out.resolve()).startswith(str(a.resolve())) for a in allowed):
        sys.exit(f"error: {out} is outside manifests/ and local/manifests/. A "
                 f"scope list read from elsewhere makes a run's recorded "
                 f"population unrecoverable.")

    kept, seen, skipped_conf = [], 0, 0
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if "jpg" not in (reader.fieldnames or []):
            sys.exit(f"error: {csv_path} has no 'jpg' column, so its rows name "
                     f"sidecars rather than the images a run walks.")
        for row in reader:
            seen += 1
            if effective_category(row) not in wanted:
                continue
            try:
                conf = float(row.get("confidence") or 0.0)
            except ValueError:
                conf = 0.0
            if conf < args.min_confidence:
                skipped_conf += 1
                continue
            kept.append(row["jpg"])

    if not kept:
        sys.exit(f"error: no rows in {csv_path} match {sorted(wanted)}. A manifest "
                 f"selecting nothing would make a run that does nothing look "
                 f"like a run with nothing to do.")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(f"# Generated by tools.manifest_from_labels from {csv_path}\n")
        fh.write(f"# category in {sorted(wanted)}"
                 + (f", confidence >= {args.min_confidence}"
                    if args.min_confidence else "") + "\n")
        fh.write(f"# {len(kept):,} of {seen:,} rows. Regenerate rather than edit:\n"
                 f"#   python3 -m tools.manifest_from_labels "
                 + " ".join(f"--category {c}" for c in sorted(wanted))
                 + f" --out {out.relative_to(PROJECT_ROOT)}\n#\n")
        for key in kept:
            fh.write(key + "\n")

    print(f"  {len(kept):,} of {seen:,} rows match {sorted(wanted)}"
          + (f" ({skipped_conf:,} dropped below {args.min_confidence})"
             if skipped_conf else ""))
    print(f"  -> {out}")
    print(f"     use with:  --include-from {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
