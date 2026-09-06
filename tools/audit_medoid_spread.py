#!/usr/bin/env python3
"""Medoid spread against branch coherence, across two or more clustering arms.

`project/findings/04` observed that DINOv3's level-1 medoids are spread far more
unevenly than BioCLIP's -- a coefficient of variation of 2.02 against 0.38 over
the pairwise medoid cosines -- and that DINOv3's *branches* are the more
family-coherent. From one comparison at one `min_cluster_size` it proposed a
claim: a two-level clustering recovers taxonomic structure to the extent its
level-1 medoids are unevenly spaced relative to their own scale. If that held it
would be a cheap screen, computable from vectors alone with no taxonomy and no
second clustering.

This is the tool that tests it, and the reason it takes **arms** rather than one
run is that a single backbone cannot answer the question. Sweeping
`min_cluster_size` within one embedding moves granularity and medoid geometry
together, so the correlation that falls out of it measures `mcs` and says nothing
about the spaces. What separates the two is the same granularity reached in
different spaces, which needs at least two arms swept over the same values.

Three things are reported, in the order the argument needs them:

1. **Per run** -- leaves, branches, mean pairwise medoid cosine, its CV, and
   size-weighted modal family purity of the branches.
2. **Within each arm** -- whether CV moves where purity moves. A predictor that
   is flat across a sweep whose outcome halves is not predicting it.
3. **Across arms at matched branch count** -- the comparison that is not
   confounded with granularity, reported both on each run's own population and
   on the images the two runs share. Noise fractions differ between backbones, so
   the shared-image figure is the fair one and the own-population figure says how
   much the population choice was worth.

The taxonomy defaults to the checklist route only, for the reason
`audit_rank_alignment` does: an embedding must not mark its own homework.
"""
import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from code.lib.csv_post import embeddings_for  # noqa: E402


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def purity(groups: list[str], values: list[str]) -> float:
    """Size-weighted modal purity -- the share of each group taking its own mode."""
    g = collections.defaultdict(list)
    for k, v in zip(groups, values):
        g[k].append(v)
    total = sum(len(v) for v in g.values())
    if not total:
        return float("nan")
    return sum(collections.Counter(v).most_common(1)[0][1]
               for v in g.values()) / total


def load_vectors(path: Path, keys: list[str], dims: int | None) -> np.ndarray:
    """The medoid vectors, in the order given.

    `dims` is checked rather than trusted. The level-1 run records the width of
    the vectors it clustered, and a run's recorded `source` can go stale -- an
    absolute path into `output/embed/` survives `./clean` moving the run but not
    a *different* embedding taking that directory over. Two backbones' vectors
    share key space and nothing else, so the mixture is undetectable downstream;
    the width is the one cheap thing that catches it.
    """
    want, got = set(keys), {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["key"] in want:
                got[row["key"]] = row["embedding"]
    missing = [k for k in keys if k not in got]
    if missing:
        raise SystemExit(f"error: {len(missing)} medoids are not in {path}, "
                         f"first {missing[0]!r}. Wrong embeddings for this run?")
    v = np.asarray([got[k] for k in keys], dtype=np.float64)
    if dims is not None and v.shape[1] != dims:
        raise SystemExit(f"error: {path} is {v.shape[1]}-dimensional, but "
                         f"the run clustered {dims}-dimensional vectors.")
    return v


def medoid_spread(vectors: np.ndarray) -> dict[str, float]:
    """How the medoids are spread, in both metrics -- and they disagree.

    `findings/04` first measured this as the coefficient of variation of the
    pairwise **cosine**, and read a 5x gap between two backbones as one space
    having far more evenly spaced medoids than the other. That reading was wrong,
    and the two extra columns here are why.

    The cosine standard deviations are nearly equal between the two backbones
    (~0.11 against ~0.12). The whole CV gap comes from the *mean*: a contrastive
    text-image space is anisotropic, so everything in it sits on a high
    similarity floor, and dividing a normal spread by 0.30 rather than 0.055
    manufactures a fivefold difference out of an offset.

    Worse, the cosine is not what the algorithm consumes. Ward is given
    **euclidean** distances between L2-normalised medoids, and by their CV the
    ordering reverses -- the space that looked five times more evenly spaced is
    about 50% *less* so. Both are reported because a scale-free number is only
    meaningful in the metric whose structure is being described, and the one that
    matters here is the one `cluster2.py` passes to `linkage`.

    (A rank measure such as the family-separability AUC is unaffected, since
    euclidean distance on the unit sphere is a strictly decreasing function of
    cosine and the two induce the same ordering.)
    """
    v = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    cos = (v @ v.T)[np.triu_indices(len(v), k=1)]
    dist = pdist(v)
    return dict(cos_mean=float(cos.mean()), cos_sd=float(cos.std()),
                cos_cv=float(cos.std() / cos.mean()),
                dist_mean=float(dist.mean()), dist_sd=float(dist.std()),
                dist_cv=float(dist.std() / dist.mean()))


def collect(label: str, cluster_dir: Path, cluster2_dir: Path,
            embeddings: Path | None, family: dict[str, str]) -> list[dict]:
    runs = []
    def mcs_order(p):
        m = re.search(r"\d+", p.name)
        return (int(m.group()) if m else 0, p.name)

    for level2 in sorted(cluster2_dir.glob("mcs*"), key=mcs_order):
        level1 = cluster_dir / level2.name
        layout = level2 / "layout.csv"
        if not (layout.is_file() and (level1 / "centers.jsonl").is_file()):
            continue
        meta1 = json.loads((level1 / "run.json").read_text())
        meta2 = json.loads((level2 / "run.json").read_text())
        centres = [json.loads(l) for l in
                   open(level1 / "centers.jsonl", encoding="utf-8")]
        vec_path = embeddings_for(level1, embeddings)
        spread = medoid_spread(load_vectors(
            vec_path, [c["medoid"] for c in centres], meta1.get("dims")))
        rows = {r["key"]: r["branch"] for r in read_csv(layout) if r["key"] in family}
        runs.append(dict(
            label=label, name=f"{label} {level2.name}", mcs=level2.name,
            leaves=meta1["clusters"], branches=meta2["branches"], rows=rows,
            purity=purity(list(rows.values()), [family[k] for k in rows]),
            **spread))
    if not runs:
        raise SystemExit(f"error: no mcs*/layout.csv with a matching "
                         f"centers.jsonl under {cluster2_dir}")
    return runs


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", nargs=3, action="append", required=True,
                    metavar=("LABEL", "CLUSTER_DIR", "CLUSTER2_DIR"),
                    help="one embedding's sweep: a name, its level-1 run "
                         "directory and its level-2 one. Repeatable, and at "
                         "least two are needed for the comparison to mean "
                         "anything")
    ap.add_argument("--embeddings", type=Path, action="append", default=None,
                    help="vectors for the matching --arm, when the run's own "
                         "recorded source has gone stale. Repeatable, positional "
                         "against --arm")
    ap.add_argument("--taxonomy", type=Path,
                    default=Path("./output/taxa/label_taxonomy.csv"),
                    help="per-image ranks from tools.map_label_taxa")
    ap.add_argument("--vocabulary", type=Path,
                    default=Path("./output/taxa/vocabulary_taxa.csv"),
                    help="how each label was resolved; needed by --source checklist")
    ap.add_argument("--rank", default="family",
                    help="the rank branches are scored against (default family)")
    ap.add_argument("--source", choices=("checklist", "any"), default="checklist",
                    help="checklist: only labels the checklist named by string, "
                         "so no embedding had a hand in the taxonomy (default)")
    ap.add_argument("--tolerance", type=int, default=3,
                    help="how many branches apart two runs may be and still "
                         "count as matched granularity (default 3)")
    args = ap.parse_args()

    tax = {r["jpg"]: r for r in read_csv(args.taxonomy)}
    if args.source == "checklist":
        method = {r["label"]: r["method"] for r in read_csv(args.vocabulary)}
        tax = {k: r for k, r in tax.items()
               if method.get(r["label"], "").startswith("string")}
    family = {k: r[args.rank] for k, r in tax.items() if r[args.rank]}
    print(f"  taxonomy: {len(family):,} images with a {args.rank}, "
          f"source={args.source}")

    vecs = list(args.embeddings or [])
    vecs += [None] * (len(args.arm) - len(vecs))
    runs = []
    for (label, c1, c2), vec in zip(args.arm, vecs):
        runs += collect(label, Path(c1), Path(c2), vec, family)

    print(f"\n  {'clustering':<18}{'leaves':>7}{'branches':>9}"
          f"{'cos mean':>10}{'cos sd':>8}{'cos CV':>8}"
          f"{'eucl mean':>11}{'eucl sd':>9}{'eucl CV':>9}"
          f"{args.rank[:6] + ' pur':>12}{'images':>9}")
    for r in runs:
        print(f"  {r['name']:<18}{r['leaves']:>7}{r['branches']:>9}"
              f"{r['cos_mean']:>10.4f}{r['cos_sd']:>8.4f}{r['cos_cv']:>8.3f}"
              f"{r['dist_mean']:>11.4f}{r['dist_sd']:>9.4f}{r['dist_cv']:>9.4f}"
              f"{r['purity']:>12.4f}{len(r['rows']):>9,}")
    print("\n  eucl is what Ward is given. The two CVs disagree in direction "
          "between arms;\n  see medoid_spread.__doc__ for why the cosine one is "
          "the misleading of the two.")

    print("\n  Within each arm -- does the spread move where the purity moves?")
    for label in dict.fromkeys(r["label"] for r in runs):
        arm = [r for r in runs if r["label"] == label]
        if len(arm) < 2:
            print(f"    {label:<10} one run only, nothing to say")
            continue
        pu = np.array([r["purity"] for r in arm])
        for key, shown in (("cos_cv", "cos CV "), ("dist_cv", "eucl CV")):
            cv = np.array([r[key] for r in arm])
            print(f"    {label:<10} {shown} {cv.min():.4f}-{cv.max():.4f} "
                  f"(spans {np.ptp(cv):.4f})", end="")
            print(f"   purity {pu.min():.4f}-{pu.max():.4f} "
                  f"(spans {np.ptp(pu):.4f})" if key == "cos_cv" else "")

    labels = list(dict.fromkeys(r["label"] for r in runs))
    if len(labels) < 2:
        print("\n  One arm only -- the matched-granularity comparison needs two.")
        return
    print(f"\n  Matched branch count (within {args.tolerance}), "
          f"own population then shared images:")
    base = labels[0]
    for a in [r for r in runs if r["label"] == base]:
        for other in labels[1:]:
            cand = [r for r in runs if r["label"] == other]
            b = min(cand, key=lambda x: abs(x["branches"] - a["branches"]))
            if abs(b["branches"] - a["branches"]) > args.tolerance:
                continue
            common = sorted(set(a["rows"]) & set(b["rows"]))
            fam = [family[k] for k in common]
            pa = purity([a["rows"][k] for k in common], fam)
            pb = purity([b["rows"][k] for k in common], fam)
            print(f"    {a['name']} ({a['branches']} br) vs "
                  f"{b['name']} ({b['branches']} br), {len(common):,} shared:  "
                  f"own {a['purity']:.4f}/{b['purity']:.4f} ({a['purity']-b['purity']:+.4f})"
                  f"   shared {pa:.4f}/{pb:.4f} ({pa-pb:+.4f})")


if __name__ == "__main__":
    main()
