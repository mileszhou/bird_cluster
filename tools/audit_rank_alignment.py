#!/usr/bin/env python3
"""Do the clusters correspond to Linnaean ranks, and at which one?

`project/findings/01` asked whether a level-2 branch is a *rank above species* or
just a bag of things that look alike, and could not answer it: every target
available to the metrics was appearance-tinged folk taxonomy, so "these look
alike" and "these are kin" predicted the same numbers. A real checklist
separates them, which is what `tools.map_label_taxa` produces.

The measure is **size-weighted modal purity**: for each cluster, the share of its
images sitting in its single most common value at that rank, averaged over images
rather than over clusters so a 300-image cluster is not outvoted by a 3-image one.

**Read the shape, not the level.** Purity rises with coarseness for free -- there
are 43 orders and 11,131 species -- so a bare number says little. Two things
make it readable: a shuffled null at the same rank and the same cluster sizes,
and the *gain* from species to family. A cluster that is already species-pure
gains almost nothing by coarsening; one that is family-pure but species-mixed
gains a lot, and that gain is the signature of a rank above species.

**`--source checklist` is the honest run.** Taxonomy reached ~20% of images
through BioCLIP's vote over their pixels, and the clustering is over vectors, so
including those lets one model mark its own homework. Restricting to labels the
checklist named by *string* costs a fifth of the population and buys a number
that no embedding had a hand in.

    python3 -m tools.audit_rank_alignment                      # the live run
    python3 -m tools.audit_rank_alignment --cluster-dir output/cluster-bioclip
    python3 -m tools.audit_rank_alignment --source any --shuffles 20
"""

import argparse
import collections
import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("PROJECT_ROOT")
                or str(Path(__file__).resolve().parents[1]))

RANKS = ("species", "genus", "family", "order")


def read_csv(path: Path) -> list[dict]:
    # utf-8-sig: every CSV this project writes carries a BOM, and a plain read
    # glues it to the first field name.
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def purity(groups: list[str], values: list[str]) -> float:
    g = collections.defaultdict(list)
    for k, v in zip(groups, values):
        g[k].append(v)
    total = sum(len(v) for v in g.values())
    if not total:
        return float("nan")
    return sum(collections.Counter(v).most_common(1)[0][1]
               for v in g.values()) / total


def report(title: str, groups: list[str], keys: list[str], tax: dict,
           shuffles: int, rng) -> None:
    print(f"\n{title}  --  {len(keys):,} images, "
          f"{len(set(groups)):,} clusters")
    print(f"  {'rank':<10}{'purity':>10}{'null':>9}{'lift':>8}{'gain vs species':>18}")
    base = None
    for rank in RANKS:
        vals = [tax[k][rank] for k in keys]
        keep = [i for i, v in enumerate(vals) if v]
        if not keep:
            continue
        g = [groups[i] for i in keep]
        v = [vals[i] for i in keep]
        got = purity(g, v)
        arr = np.array(v)
        null = float(np.mean([purity(g, list(rng.permutation(arr)))
                              for _ in range(shuffles)]))
        if base is None:
            base = got
        print(f"  {rank:<10}{got:>10.4f}{null:>9.4f}{got / null:>8.1f}x"
              f"{got - base:>+18.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--taxonomy", type=Path,
                    default=Path("./output/taxa/label_taxonomy.csv"),
                    help="per-image ranks from tools.map_label_taxa")
    ap.add_argument("--vocabulary", type=Path,
                    default=Path("./output/taxa/vocabulary_taxa.csv"),
                    help="how each label was resolved; needed by --source checklist")
    ap.add_argument("--cluster-dir", type=Path, default=Path("./output/cluster"),
                    help="a clustering run; every mcs*/ beneath it is reported, "
                         "so a whole sweep is one invocation")
    ap.add_argument("--layout", type=Path, action="append", default=None,
                    help="a cluster2 layout.csv, adding its branches. Repeatable")
    ap.add_argument("--source", choices=("checklist", "any"), default="checklist",
                    help="checklist: only labels the checklist named by string, so "
                         "no embedding had a hand in the taxonomy (default). "
                         "any: include the ~20%% resolved by BioCLIP's vote")
    ap.add_argument("--shuffles", type=int, default=5,
                    help="permutations averaged for the null (default 5)")
    args = ap.parse_args()

    tax = {r["jpg"]: r for r in read_csv(args.taxonomy)}
    print(f"  taxonomy: {len(tax):,} images from {args.taxonomy}")

    if args.source == "checklist":
        method = {r["label"]: r["method"] for r in read_csv(args.vocabulary)}
        tax = {k: r for k, r in tax.items()
               if method.get(r["label"], "").startswith("string")}
        print(f"  restricted to checklist-named labels: {len(tax):,} images")

    rng = np.random.default_rng(0)
    seen = False

    for assign in sorted(args.cluster_dir.glob("mcs*/assignments.csv")):
        rows = [r for r in read_csv(assign)
                if r["key"] in tax and r.get("is_noise") not in ("1", "True")]
        if not rows:
            continue
        seen = True
        report(f"level 1  {assign.parent.name}", [r["cluster_id"] for r in rows],
               [r["key"] for r in rows], tax, args.shuffles, rng)

    for layout in (args.layout or []):
        rows = [r for r in read_csv(Path(layout)) if r["key"] in tax]
        if not rows:
            continue
        seen = True
        keys = [r["key"] for r in rows]
        report(f"level 1  {Path(layout).parent.name} (leaves)",
               [r["leaf"] for r in rows], keys, tax, args.shuffles, rng)
        report(f"level 2  {Path(layout).parent.name} (branches)",
               [r["branch"] for r in rows], keys, tax, args.shuffles, rng)

    if not seen:
        sys.exit(f"error: nothing to audit. No mcs*/assignments.csv under "
                 f"{args.cluster_dir}, and no --layout given whose keys are in "
                 f"the taxonomy. A clustering of a different population than the "
                 f"one that was mapped will land here.")


if __name__ == "__main__":
    main()
