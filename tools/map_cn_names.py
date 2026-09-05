#!/usr/bin/env python3
"""A canonical Chinese name per bird, so one species is one string.

A model asked for "the standard Chinese name" is consistent but not reliable:
over Qwen's bird rows, 36.6% of the names it used five or more times got more
than one Chinese form. Each extra form is a separate keyword in a photo manager
and a separate row in any grouping by label -- the species is simply listed
twice, with neither entry holding all its photos.

**Keyed on the binomial where one is known.** An English-keyed vote cannot see
that two of its keys are the same bird, and the labelling is unstable in that
direction too -- 935 Chinese names carry more than one English name. The
taxonomy from `tools.map_label_taxa` collapses those: two English names for one
species land on one taxon and get one Chinese name.

Only where the *checklist* named the label, though. Where the taxonomy came from
BioCLIP's vote over pixels the binomial is a guess, and merging two labels on a
guess would invent a synonymy the data does not support -- so those fall back to
being keyed on the English name, which is weaker but claims nothing extra.

**This reports; it does not rewrite.** The dictionary is an artifact to look at
and then decide about, and applying it is a separate act on a separate day.

    python3 -m tools.map_cn_names
    python3 -m tools.map_cn_names --label-dir local/output_gemma/label
"""

import argparse
import collections
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("PROJECT_ROOT")
                or str(Path(__file__).resolve().parents[1]))

from code.lib.config import PROJECT_ROOT, data_dir  # noqa: E402


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def effective(row) -> tuple[str, str]:
    """(category, label) the library holds, resolving the never-demote rule."""
    if (row.get("applied") or "").strip() == "kept-existing":
        prior = (row.get("prior_category") or "").strip().lower()
        if prior:
            return prior, (row.get("prior_label") or "").strip()
    return ((row.get("category") or "").strip().lower(),
            (row.get("label") or "").strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label-dir", type=Path, default=None,
                    help="the labelling to build from (default: <data>/label). "
                         "Qwen is the most self-consistent one measured: 91.8%% "
                         "modal agreement on birds against gemma's 85.3%%")
    ap.add_argument("--taxonomy", type=Path,
                    default=PROJECT_ROOT / "output" / "taxa" / "label_taxonomy.csv",
                    help="per-label taxonomy from tools.map_label_taxa")
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "output" / "taxa" / "cn_names.csv")
    ap.add_argument("--category", default="bird",
                    help="which category to build for (default: bird; for scenery "
                         "the Chinese text is a description, not a name)")
    args = ap.parse_args()

    label_dir = args.label_dir or (data_dir() / "label")
    rows = read_csv(Path(label_dir) / "bird_identification_output.csv")
    print(f"  labelling: {len(rows):,} rows from {label_dir}")

    taxonomy = {}
    if Path(args.taxonomy).is_file():
        for t in read_csv(args.taxonomy):
            # Only a checklist-named binomial is trusted to merge two labels.
            if (t.get("method") or "").startswith("string") and t.get("genus"):
                binom = " ".join(x for x in (t["genus"], t.get("species")) if x)
                taxonomy[(t.get("label") or "").strip().lower()] = binom
        print(f"  taxonomy : {len(taxonomy):,} labels carry a checklist binomial")
    else:
        print(f"  taxonomy : none at {args.taxonomy}; keying on the English name only")

    # key -> Chinese form -> count, plus the English names that fed each key
    forms: dict = collections.defaultdict(collections.Counter)
    english: dict = collections.defaultdict(collections.Counter)
    for r in rows:
        category, label = effective(r)
        if category != args.category:
            continue
        label = label.lower()
        cn = (r.get("label_cn") or "").strip()
        if not label or not cn:
            continue
        binom = taxonomy.get(label)
        key = ("binomial", binom) if binom else ("english", label)
        forms[key][cn] += 1
        english[key][label] += 1

    if not forms:
        sys.exit(f"error: no {args.category} rows with both names in {label_dir}.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    changed = same = 0
    multi = 0
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["key_type", "key", "canonical_cn", "images", "cn_forms",
                    "english_names", "alternatives"])
        for (kind, key), counter in sorted(forms.items(),
                                           key=lambda kv: -sum(kv[1].values())):
            # Modal, with a deterministic tie-break so two runs agree.
            canonical = max(sorted(counter), key=lambda c: (counter[c], -len(c)))
            n = sum(counter.values())
            changed += n - counter[canonical]
            same += counter[canonical]
            if len(counter) > 1:
                multi += 1
            w.writerow([kind, key, canonical, n, len(counter),
                        "; ".join(sorted(english[(kind, key)])),
                        "; ".join(f"{c}({counter[c]})"
                                  for c in sorted(counter, key=lambda x: -counter[x])
                                  if c != canonical)])

    by_kind = collections.Counter(kind for kind, _ in forms)
    print(f"\n  {len(forms):,} keys: "
          + ", ".join(f"{n:,} {k}" for k, n in by_kind.most_common()))
    print(f"  {multi:,} of them carry more than one Chinese form")
    print(f"  applying it would change {changed:,} of {changed + same:,} rows "
          f"({changed / (changed + same):.1%})")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
