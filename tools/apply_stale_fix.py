#!/usr/bin/env python3
"""Apply decisions from the stale-bird worklist to a corrected labelling run.

Input is `stale_bird_labels.csv` from `tools/stale_bird_labels.py`, optionally
with a `decision` column filled in, plus the labelling run it describes. Output
is a **new** label directory -- the input run is never modified, so a wrong call
costs a re-run of this script and nothing else.

Decisions, per row:

    keep-bird   the sidecar is right; leave it. Default for bucket B, where the
                run mis-categorised a real bird and never-demote did its job.
    take-run    the kept label is stale; adopt the run's verdict. Default for
                bucket A, where a same-stem twin provably carries that label.
    (blank)     bucket C's default -- treated as `keep-bird`, i.e. no change,
                because demoting on no evidence would trade a known-good guard
                for a guess.

A `--recheck` CSV overrides bucket C: point it at the output of re-running
bird_label over the extracted images, and any row it calls non-bird becomes
`take-run`. Because the extract has no sidecars, that run carries no prior to
defer to, so its verdict is clean.

What changes when a row becomes `take-run`:

    the corrected CSV      category/label/label_cn/confidence <- the run's,
                           applied <- `written`, and the stale prior kept in
                           prior_category/prior_label so nothing is lost
    the corrected sidecar  the bird keyword replaced by the run's category
                           and label, via the same writer bird_label uses

Nothing here re-runs a model. Every value written is one the labelling run
already produced.

    tools/apply_stale_fix.py --dry-run
    tools/apply_stale_fix.py --out data/label-fixed
"""

import argparse
import csv
import shutil
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("PROJECT_ROOT")
                or str(Path(__file__).resolve().parents[1]))

import xml.etree.ElementTree as ET  # noqa: E402

from code.bird_label import mk_label  # noqa: E402
from code.lib import xmp_write  # noqa: E402
from code.lib.xmp_labels import split_keywords  # noqa: E402

KEEP, TAKE = "keep-bird", "take-run"


def force_keywords(xmp: Path, category: str, label: str) -> bool:
    """Write `category`/`label` into a sidecar, overriding the never-demote rule.

    `set_keywords_in_xmp()` cannot be reused here: it exists precisely to refuse
    this write, since a non-bird result over an existing `bird` normally means
    the run is wrong. Here we have established the opposite -- the kept label
    describes a different photo -- so the guard has to be stepped around, once,
    deliberately, for rows a human or the evidence has cleared.

    Everything else is the ordinary path: the user's own keywords are preserved,
    hierarchical paths keep their structure, and the edit is re-parsed and
    verified before it lands, so this cannot corrupt a file even though it is
    overriding a safety rule.
    """
    before = xmp.read_text(encoding="utf-8")
    root = ET.fromstring(before)

    existing = xmp_write.items_of(root, xmp_write.SUBJECT_TAG)
    ours, theirs = split_keywords(existing)
    keywords = [category, label]
    merged = list(theirs) + [k for k in keywords if k not in theirs]

    prior_hier = xmp_write.items_of(root, xmp_write.HIER_TAG)
    hier_ours, _ = split_keywords([(h or "").split("|")[-1] for h in prior_hier])
    hierarchical = xmp_write.merge_hierarchical(
        prior_hier, set(ours) | set(hier_ours), keywords)

    after = xmp_write.set_subject_keywords(before, merged, hierarchical)
    xmp_write.verify_only_keywords_changed(before, after, merged, hierarchical)
    xmp.write_text(after, encoding="utf-8")
    return True


def load_decisions(worklist: Path, recheck: Path | None):
    """{jpg: (decision, bucket)} -- explicit column first, then bucket defaults."""
    decided = {}
    for row in csv.DictReader(open(worklist, newline="", encoding="utf-8-sig")):
        bucket = row["bucket"]
        explicit = (row.get("decision") or "").strip().lower()
        if explicit in (KEEP, TAKE):
            decided[row["jpg"]] = (explicit, bucket)
        elif bucket.startswith("A"):
            decided[row["jpg"]] = (TAKE, bucket)
        else:
            decided[row["jpg"]] = (KEEP, bucket)

    if recheck:
        # A re-run over the extracted images: no sidecars there, so no prior to
        # defer to and the verdict stands on its own.
        n = 0
        for row in csv.DictReader(open(recheck, newline="", encoding="utf-8-sig")):
            jpg = row.get("jpg")
            if jpg in decided and decided[jpg][1].startswith("C"):
                verdict = (row.get("category") or "").strip().lower()
                decided[jpg] = (KEEP if verdict == "bird" else TAKE, decided[jpg][1])
                n += 1
        print(f"recheck applied to {n} bucket C rows")
    return decided


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label-dir", type=Path, default=Path("./data/label"))
    ap.add_argument("--worklist", type=Path,
                    default=Path("project/reports/stale_bird_labels.csv"))
    ap.add_argument("--recheck", type=Path, default=None,
                    help="a bird_label CSV from re-running the extracted images")
    ap.add_argument("--out", type=Path, default=Path("./data/label-fixed"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src_csv = args.label_dir / "bird_identification_output.csv"
    decided = load_decisions(args.worklist, args.recheck)
    take = {j for j, (d, _) in decided.items() if d == TAKE}
    print(f"worklist rows: {len(decided)}  ->  take-run: {len(take)}, "
          f"keep-bird: {len(decided) - len(take)}")

    rows = list(csv.DictReader(open(src_csv, newline="", encoding="utf-8-sig")))
    fields = list(rows[0].keys())
    changed = []
    for r in rows:
        if r["jpg"] not in take:
            continue
        # The run's own verdict is already in these columns; promoting it just
        # means saying it was applied. The stale label stays in prior_* so the
        # correction is auditable and reversible from the CSV alone.
        r["applied"] = "written"
        changed.append(r)

    print(f"rows corrected: {len(changed)}")
    if args.dry_run:
        for r in changed[:8]:
            print(f"   {r['jpg']}\n      bird '{r['prior_label'][:38]}' -> "
                  f"{r['category']} '{r['label'][:38]}'")
        print("dry run: nothing written")
        return

    if args.out.exists():
        sys.exit(f"error: {args.out} exists; remove it or pick another --out")
    print(f"copying {args.label_dir} -> {args.out} ...")
    shutil.copytree(args.label_dir, args.out)

    with open(args.out / "bird_identification_output.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    wrote = skipped = 0
    for r in changed:
        rel = (r.get("xmp") or "").strip()
        if not rel:
            skipped += 1           # csv-only row: nothing to correct on disk
            continue
        xmp = args.out / "raw" / rel
        if not xmp.is_file():
            skipped += 1
            continue
        spec = mk_label(r["category"], r["label"], r["label_cn"],
                        float(r["confidence"] or 0))
        if force_keywords(xmp, r["category"], spec):
            wrote += 1
        else:
            skipped += 1

    print(f"sidecars corrected: {wrote}, skipped (no sidecar): {skipped}")
    print(f"corrected run written to {args.out}")


if __name__ == "__main__":
    main()
