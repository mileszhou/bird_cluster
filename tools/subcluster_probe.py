#!/usr/bin/env python3
"""Can the species be recovered inside a cluster without any label guidance?

Two questions look alike and are not. *Label-guided analysis* groups images by
their label and asks whether the embedding agrees -- it can confirm a labelling
but never discover anything, because the answer was the input. *Discovery* holds
the labels back entirely, subdivides on the vectors alone, and consults the
labels only afterwards to score what came out. This probe does the second.

Nothing here chooses a parameter by looking at a label. For each parent cluster
the sweep over `min_cluster_size` is resolved by HDBSCAN's own
`relative_validity_` -- a DBCV approximation that sees only the vectors -- and
the AMI against species is computed after the fact. An `oracle` column reports
the best AMI anywhere in the grid, so the price of choosing label-free is
visible rather than assumed. On mcs40 that price is zero at the median: the
label-free criterion picks the same resolution the oracle would.

**Leaf selection, not excess-of-mass.** EOM is unstable here -- inside cluster 12
it scores AMI 0.017, 0.017, 0.747, 0.066 across min_cluster_size 3/5/10/20,
finding the structure at one setting and missing it either side, because it
prefers the single massive blob unless the parameter happens to land right. Leaf
climbs monotonically over the same grid (0.563, 0.720, 0.753, 0.754) and never
collapses.

**What it does not do is decide whether to subdivide at all**, and that is the
gap this probe exposes rather than fills. `relative_validity_` will happily split
a cluster that is already one species, where the best available AMI is zero by
construction. The effective dimension from `plot_radial.py` looks like the
missing gate: it correlates +0.72 with the AMI gained by splitting (+0.64
partialling out cluster size), against -0.53 for purity, which needs labels.

    python -m tools.subcluster_probe                     # every cluster in the run
    python -m tools.subcluster_probe --focus 12          # one, with the full grid shown
    python -m tools.subcluster_probe --selection eom     # the unstable comparison

Writes `subclusters.csv` beside the assignments: the recovered partition, one
row per image, so it can be joined back or fed to plot_matrix. Prints the score
table. Never modifies assignments.csv.
"""

import argparse
import os
import sys
import collections
import csv
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
import hdbscan
from sklearn.metrics import (adjusted_mutual_info_score as ami,
                             homogeneity_score, completeness_score)

# Running this as `python3 tools/x.py` puts tools/ on sys.path, not the repo
# root -- and then `import code.lib` finds the *stdlib* `code` module, which is
# not a package. Put the root first so both invocation forms work.
sys.path.insert(0, os.environ.get("PROJECT_ROOT")
                or str(Path(__file__).resolve().parents[1]))

from code.lib.csv_post import (add_arguments, embeddings_for, read_rows,
                               resolve_runs)

NOISE = "-1"


def distances(V):
    """Euclidean distance matrix. Vectors are L2-normalised, so this is a
    monotone function of cosine and HDBSCAN's ordering is the same either way."""
    G = np.clip(V @ V.T, -1, 1).astype(np.float64)
    D = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * G))
    np.fill_diagonal(D, 0.0)
    return D


def subdivide(V, labels, grid, selection):
    """Sweep min_cluster_size, choose by relative_validity_, score afterwards.

    Returns (chosen assignment, chosen mcs, its AMI, best AMI in the grid), or
    None when no setting produced two or more sub-clusters.
    """
    D = distances(V)
    trials = []
    for mcs in grid:
        if mcs * 2 > len(V):
            continue
        h = hdbscan.HDBSCAN(min_cluster_size=mcs, metric="precomputed",
                            cluster_selection_method=selection,
                            gen_min_span_tree=True).fit(D)
        L = h.labels_
        if L.max() < 1:
            continue
        m = L != -1
        trials.append((float(h.relative_validity_), mcs, L,
                       ami(labels[m], L[m]) if m.sum() > 5 else 0.0))
    if not trials:
        return None
    _, mcs, L, a = max(trials, key=lambda t: t[0])
    return L, mcs, a, max(t[3] for t in trials)


def run(run_dir: Path, args) -> str:
    rows = [r for r in read_rows(run_dir / "assignments.csv")
            if r["cluster_id"] != NOISE]
    if args.focus:
        rows = [r for r in rows if r["cluster_id"] == args.focus]
        if not rows:
            raise SystemExit(f"error: no cluster {args.focus!r} in {run_dir}")

    keys = {r["key"] for r in rows}
    vec = {}
    with open(embeddings_for(run_dir, args.embeddings), encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d["key"] in keys:
                vec[d["key"]] = d["embedding"]
    X = np.asarray([vec[r["key"]] for r in rows], dtype=np.float32)
    y = np.array([r["species"] for r in rows])
    parent = np.array([r["cluster_id"] for r in rows])

    by = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by[r["cluster_id"]].append(i)

    print(f"  {run_dir.name}: {len(rows):,} images, {len(by)} clusters, "
          f"{len(set(y))} species labels")
    print(f"  {'cluster':>8}{'n':>7}{'mcs':>5}{'k':>5}{'noise':>8}{'AMI':>7}"
          f"{'oracle':>8}  dominant species")

    fine = np.empty(len(rows), dtype=object)      # sub-noise falls back to parent
    strict = np.empty(len(rows), dtype=object)    # sub-noise dropped
    scored = []
    for cid, idx in sorted(by.items(), key=lambda kv: -len(kv[1])):
        idx = np.asarray(idx)
        got = subdivide(X[idx], y[idx], args.grid, args.selection)
        top = collections.Counter(y[idx]).most_common(1)[0][0]
        if got is None:
            fine[idx] = cid
            strict[idx] = cid
            print(f"  {cid:>8}{len(idx):>7}{'-':>5}{'-':>5}{'-':>8}{'-':>7}"
                  f"{'-':>8}  {top}  (not split)")
            continue
        L, mcs, a, oracle = got
        m = L != -1
        fine[idx] = [f"{cid}.{v}" if v != -1 else cid for v in L]
        strict[idx] = [f"{cid}.{v}" if v != -1 else None for v in L]
        print(f"  {cid:>8}{len(idx):>7}{mcs:>5}{L.max()+1:>5}{1-m.mean():>8.1%}"
              f"{a:>7.3f}{oracle:>8.3f}  {top}")
        scored.append((a, oracle))

    with open(run_dir / "subclusters.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "cluster_id", "subcluster", "subcluster_strict", "species"])
        for r, f, s in zip(rows, fine, strict):
            w.writerow([r["key"], r["cluster_id"], f, s or "", r["species"]])

    ok = np.array([s is not None for s in strict])
    print(f"\n  GLOBAL (labels used only to score)")
    print(f"    as it stands, {len(by):>4} clusters       AMI {ami(y, parent):.3f}")
    print(f"    recursive,    {len(set(fine)):>4} groups         AMI {ami(y, fine):.3f}"
          f"   (all images kept)")
    print(f"    recursive,    {len(set(strict[ok])):>4} groups         AMI "
          f"{ami(y[ok], strict[ok]):.3f}   ({ok.mean():.0%} of images kept)")
    print(f"    homogeneity {homogeneity_score(y[ok], strict[ok]):.3f}  "
          f"completeness {completeness_score(y[ok], strict[ok]):.3f}")
    if scored:
        a = np.array([s[0] for s in scored])
        o = np.array([s[1] for s in scored])
        print(f"\n    {len(scored)}/{len(by)} clusters subdivided; per-cluster AMI "
              f"median {np.median(a):.3f}, oracle {np.median(o):.3f}, "
              f"cost of label-free choice {np.median(o - a):+.3f}")
    return f"{len(by)} clusters -> subclusters.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    ap.add_argument("--embeddings", type=Path,
                    default=None,
                    help="the vectors to read. Default: whatever the run\n"
                         "being analysed recorded as its source")
    ap.add_argument("--focus", default=None, help="one cluster_id instead of all")
    ap.add_argument("--grid", type=int, nargs="+", default=[5, 10, 20],
                    help="min_cluster_size values to sweep")
    ap.add_argument("--selection", choices=("leaf", "eom"), default="leaf",
                    help="HDBSCAN cluster_selection_method; eom is unstable here")
    args = ap.parse_args()

    for r in resolve_runs(args):
        run(r, args)


if __name__ == "__main__":
    main()
