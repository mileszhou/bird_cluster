#!/usr/bin/env python3
"""Remove this pipeline's labels from a sidecar tree, leaving the user's own.

Written to prepare `sample_data/`: a published sample should ship the *input* to
the pipeline, not its output. Sidecars carrying `bird` and
`hn-黑鹳-black stork(95%)` already contain the answer, so a fresh clone running
the labeller cannot show it doing anything.

**What counts as ours is `xmp_labels.split_keywords()`**, not a pattern invented
here. Three generations exist -- bare categories, `py-cn-en(NN%)` labels, and
early GPT-4o free text with no confidence marker at all -- and the last is
recognisable only by two signals together (the sidecar carries a category, and
the text is a descriptive phrase). Single tokens (`Family`, `Rivertown`) and the
hand-written species shape (`xs-小隼-Kestrel`) are always the user's. Reusing
that function is the point: a regex written here would disagree with the
labeller about what it owns, and the disagreement would delete someone's work.

`lr:hierarchicalSubject` holds keyword *paths*, so an entry is dropped only when
its **leaf** is one of ours. `People|Family|Miles` survives; `Photo|bird` loses
just the entry. The block is never created where absent.

Edits are text surgery via `xmp_write`, and every file is re-parsed afterwards
with `verify_only_keywords_changed()` -- a file that would move anything outside
its keyword blocks is left untouched and reported, rather than written and
hoped for.

    python -m tools.strip_labels sample_data/xmp            # dry run, the default
    python -m tools.strip_labels sample_data/xmp --apply

`--all` removes everything instead, hand-written keywords included. That is the
right setting for a *sample* tree, where a keyword is somebody else's vocabulary
rather than work to protect, and the wrong setting anywhere near `data/xmp`.

Reports every keyword it would remove, grouped, so the list can be read before
anything is written.
"""

import argparse
import os
import sys
import collections
from pathlib import Path

# Running this as `python3 tools/x.py` puts tools/ on sys.path, not the repo
# root -- and then `import code.lib` finds the *stdlib* `code` module, which is
# not a package. Put the root first so both invocation forms work.
sys.path.insert(0, os.environ.get("PROJECT_ROOT")
                or str(Path(__file__).resolve().parents[1]))

from code.lib.xmp_labels import read_subjects, split_keywords
from code.lib import xmp_write


def hierarchical_kept(text: str, ours: set) -> list | None:
    """Existing hierarchical entries minus those whose leaf is ours.

    None when the sidecar has no hierarchicalSubject, which means leave it
    alone -- it is never created here.
    """
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(text)
    except ET.ParseError:
        return None
    existing = xmp_write.items_of(root, "hierarchicalSubject")
    if existing is None:
        return None
    return [e for e in existing if e.split("|")[-1].strip() not in ours]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="a sidecar tree, e.g. sample_data/xmp")
    ap.add_argument("--all", action="store_true",
                    help="remove every keyword, including hand-written ones, rather "
                         "than only this pipeline's. For preparing a sample tree, "
                         "where a stranger's keyword is noise rather than work to "
                         "protect -- never point this at data/xmp")
    ap.add_argument("--apply", action="store_true",
                    help="write the changes; without it nothing is modified")
    args = ap.parse_args()

    files = sorted(args.root.rglob("*.xmp"))
    if not files:
        raise SystemExit(f"error: no .xmp under {args.root}")

    removed = collections.Counter()
    changed = kept_all = failed = 0
    for path in files:
        subjects = read_subjects(path)
        if not subjects:
            continue
        ours, theirs = (tuple(subjects), ()) if args.all else split_keywords(subjects)
        if not ours:
            kept_all += 1
            continue
        removed.update(ours)

        text = path.read_text(encoding="utf-8")
        hier = hierarchical_kept(text, set(ours))
        after = xmp_write.set_subject_keywords(text, theirs, hier)
        try:
            xmp_write.verify_only_keywords_changed(text, after, theirs, hier)
        except Exception as exc:
            print(f"  SKIPPED {path}: {exc}")
            failed += 1
            continue
        if args.apply:
            path.write_text(after, encoding="utf-8")
        changed += 1

    print(f"\n{len(files)} sidecars: {changed} to change, {kept_all} already clean, "
          f"{failed} skipped")
    print(f"{sum(removed.values())} keywords removed, {len(removed)} distinct\n")
    for kw, n in removed.most_common(25):
        print(f"  {n:>4}  {kw}")
    if len(removed) > 25:
        print(f"  ... and {len(removed)-25} more")
    if not args.apply:
        print("\nDry run. Re-run with --apply to write.")


if __name__ == "__main__":
    main()
