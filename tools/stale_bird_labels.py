#!/usr/bin/env python3
"""Find bird labels the never-demote rule is keeping alive for the wrong reason.

`set_keywords_in_xmp()` never demotes a `bird`: a non-bird result over an
existing `bird` category leaves the sidecar alone. That guards the clustering
set against a false negative, and it works -- but it cannot tell a *correct* old
label from a *misplaced* one, and the old runs were basename-keyed.

10,832 sidecars share a basename with another, so an old run could attach a
label to the wrong photo entirely. The proof is direct: a wombat
carrying `common myna`, and a Sydney photo with the same stem that really is a
common myna. Once the twin is found, the mechanism is not in doubt.

This tool buckets every never-demoted row by what the evidence supports:

  A  proven collision -- a same-stem twin carries exactly the kept label.
     The kept label describes the twin, not this photo. Take this run's verdict.

  B  this run mis-categorised a real bird -- its own label is a name it used
     for birds elsewhere (`great horned owl` filed as `animal`). The old label
     was right; never-demote did its job. Leave alone.

  C  no proof either way. This run says non-bird and nothing corroborates the
     old label. Needs a look, or a targeted re-ask.

Bird-ness is decided from the run's own vocabulary -- the set of labels it used
under `category == bird` -- so no external species list is needed and the test
stays consistent with whatever model produced the CSV.

    tools/stale_bird_labels.py                        # report + worklist CSV
    tools/stale_bird_labels.py --extract out/recheck  # copy bucket C images out

The worklist is the input to a fix; nothing here modifies the dataset.
"""

import argparse
import collections
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.lib.xmp_labels import parse_label  # noqa: E402

BUCKET_A = "A-collision"
BUCKET_B = "B-run-miscategorised"
BUCKET_C = "C-unproven"

FIELDS = ["bucket", "jpg", "xmp", "kept_category", "kept_species", "run_category",
          "run_label", "twin_jpg", "stem_shared_by", "confidence"]


def prior_species(row):
    """The species the sidecar is holding, from `prior_label`."""
    for part in (row.get("prior_label") or "").split(";"):
        label = parse_label(part.strip())
        if label:
            return label.english
    return None


def analyse(csv_path: Path):
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8-sig")))
    if "jpg" not in (rows[0] if rows else {}):
        sys.exit(f"error: {csv_path} has no 'jpg' column -- needs a current labelling run")
    for r in rows:
        r["_stem"] = Path(r["jpg"]).stem

    # The run's own bird vocabulary. Self-contained, so it tracks whatever model
    # wrote this CSV rather than a species list that would go stale.
    bird_names = {r["label"].strip().lower() for r in rows if r["category"] == "bird"}
    by_stem = collections.defaultdict(list)
    for r in rows:
        by_stem[r["_stem"]].append(r)

    out = []
    for r in rows:
        if r.get("applied") != "kept-existing":
            continue
        if (r.get("prior_category") or "").strip().lower() != "bird":
            continue
        if r.get("category") == "bird":
            continue          # agreed; nothing was overruled

        kept = prior_species(r)
        run_label = (r.get("label") or "").strip().lower()
        twin = next((t for t in by_stem[r["_stem"]]
                     if t is not r and kept
                     and (t["label"] or "").strip().lower() == kept), None)
        if twin is not None:
            bucket = BUCKET_A
        elif run_label in bird_names:
            bucket = BUCKET_B
        else:
            bucket = BUCKET_C
        out.append({
            "bucket": bucket,
            "jpg": r["jpg"],
            "xmp": r.get("xmp", ""),
            "kept_category": "bird",
            "kept_species": kept or "",
            "run_category": r.get("category", ""),
            "run_label": r.get("label", ""),
            "twin_jpg": twin["jpg"] if twin is not None else "",
            "stem_shared_by": len(by_stem[r["_stem"]]),
            "confidence": r.get("confidence", ""),
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label-dir", type=Path, default=Path("./data/label"))
    ap.add_argument("--data-dir", type=Path, default=Path("./data"))
    ap.add_argument("--out", type=Path, default=Path("project/reports/stale_bird_labels.csv"))
    ap.add_argument("--extract", type=Path, default=None,
                    help="copy bucket C images into this folder for a re-check, "
                         "preserving the library/trip structure")
    args = ap.parse_args()

    findings = analyse(args.label_dir / "bird_identification_output.csv")
    counts = collections.Counter(f["bucket"] for f in findings)

    print(f"never-demoted rows (library says bird, the run disagreed): {len(findings)}")
    for b, note in ((BUCKET_A, "a same-stem twin carries the kept label -> take the run's verdict"),
                    (BUCKET_B, "the run mis-categorised a real bird       -> leave alone"),
                    (BUCKET_C, "no proof either way                       -> re-check")):
        print(f"  {counts[b]:4d}  {b:22s} {note}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig: trip folders are Chinese and Excel reads a BOM-less UTF-8 CSV
    # as the system codepage. Read it back with encoding="utf-8-sig".
    with open(args.out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sorted(findings, key=lambda f: (f["bucket"], f["jpg"])))
    print(f"\nwrote {args.out} ({len(findings)} rows)")

    if args.extract:
        # Laid out as a miniature data-dir -- <extract>/jpg/<library>/<trip>/*.jpg
        # with an empty xmp/ -- so bird_label.py can run against it directly:
        #
        #     python3 -m code.bird_label --data-dir <extract> --output-dir <out> \
        #             --approach vllm --years all
        #
        # The empty sidecar tree is the point: with no prior label to defer to,
        # every row comes back `csv-only` carrying a clean verdict, instead of
        # hitting the same never-demote rule that caused this in the first place.
        # Paths stay relative to jpg/, so the result keys straight back.
        todo = [f for f in findings if f["bucket"] == BUCKET_C]
        (args.extract / "xmp").mkdir(parents=True, exist_ok=True)
        copied = missing = 0
        for f in todo:
            src = args.data_dir / "jpg" / f["jpg"]
            if not src.is_file():
                missing += 1
                continue
            dst = args.extract / "jpg" / f["jpg"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
        size = sum(p.stat().st_size for p in (args.extract / "jpg").rglob("*.jpg"))
        print(f"extracted {copied} bucket C images to {args.extract}/jpg "
              f"({size / 1e6:.0f} MB)" + (f", {missing} not on disk" if missing else ""))
        print("laid out as a data-dir; see the comment in --extract for how to re-run.")
        print("NOTE a plain re-run reproduces the same verdict -- the model already saw")
        print("     these and said non-bird. The extract earns its keep for human review")
        print("     or a sharper prompt, not for repeating the same question.")


if __name__ == "__main__":
    main()
