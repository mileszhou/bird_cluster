"""Step 2 of the semantic-clustering pipeline: HDBSCAN over frozen embeddings.

    ./run-cluster                                  # the default sweep
    python3 -m code.cluster.discover --min-cluster-size 5,15,40

**A run is an experiment, not a build step.** Each `--min-cluster-size` gets its
own output directory and nothing overwrites anything, because how the structure
*changes* across the sweep is itself the finding: a group that survives from 5 to
40 is a different kind of object than one that dissolves at 10. Passing several
values loads the 452 MB of vectors once and fits each in turn, which is the only
reason the sweep is a list rather than three invocations.

**Clusters are named by their medoid's image key, not by an ordinal.** HDBSCAN's
integer labels are arbitrary and change between runs, so cluster 47 at
`min_cluster_size=5` has nothing to do with cluster 47 at 15. The medoid is a
real photo, stable under re-fitting, and the same string that joins back to the
labelling CSV -- which is what makes a cluster a thing you can point at across
two runs, or in a note six months from now.

**Duplicates are dropped before fitting.** 241 captures were exported more than
once, and their alternate edits are near-identical vectors. HDBSCAN is
density-based, so a duplicated point does not merely add a row -- it doubles the
local density at exactly the place a cluster is being decided.

Distances: the vectors are L2-normalised by the embed server, so euclidean
distance is a monotone function of cosine (`|a-b|^2 = 2 - 2cos`). Clustering with
`metric="euclidean"` is therefore semantically cosine clustering, and gets
HDBSCAN's fast path.

Labels are *not* ground truth here (see `project/ideas/01`): the premise is that
a vector carries more than a label does, so the agreement figures below describe
the labelling as much as the clustering.
"""

import argparse
import collections
import csv
import json
import logging
import pickle
import subprocess
import re
import shutil
import sys
from pathlib import Path

import numpy as np

from code.lib.config import data_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("discover")


def load(path: Path):
    """(vectors, rows) from the embeddings JSONL, in file order."""
    vecs, rows = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            vecs.append(r["embedding"])
            rows.append({k: v for k, v in r.items() if k != "embedding"})
    return np.asarray(vecs, dtype=np.float32), rows


def capture_of(row) -> str:
    """What counts as one photograph.

    The sidecar when there is one -- alternate edits of a capture share it -- and
    the image key otherwise, since an image with no sidecar is its own capture.
    """
    return (row.get("xmp") or "").strip() or row["key"]


def drop_duplicate_captures(X, rows):
    """One vector per capture, keeping the first in sorted key order.

    Deterministic so a re-run picks the same representative; which one is kept
    barely matters, since the whole reason they are dropped is that they are
    near-identical.
    """
    by_capture = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_capture[capture_of(r)].append(i)
    keep = sorted(min(idxs, key=lambda i: rows[i]["key"]) for idxs in by_capture.values())
    dropped = len(rows) - len(keep)
    return X[keep], [rows[i] for i in keep], dropped


def cluster_centres(X, labels, probabilities, rows):
    """Per cluster: centroid, medoid, density peak, and the medoid-derived name.

    The centroid of unit vectors is not itself a unit vector, and for a spread
    cluster its norm shrinks towards zero -- so the gap between centroid and
    medoid is a shape measurement, not an error. A large gap means the members do
    not surround their own mean, which the plan flags as a candidate for a
    cluster that is really two.
    """
    out = []
    for cid in sorted(set(labels) - {-1}):
        idx = np.flatnonzero(labels == cid)
        members = X[idx]
        centroid = members.mean(0)

        d = np.sqrt(np.maximum(0, 2 - 2 * (members @ members.T)))
        medoid_local = int(np.argmin(d.sum(1)))
        medoid = idx[medoid_local]
        peak = idx[int(np.argmax(probabilities[idx]))]

        cn = np.linalg.norm(centroid)
        gap = float(np.linalg.norm(members[medoid_local] - centroid))
        out.append({
            "name": rows[medoid]["key"],          # the stable identity
            "cluster_id": int(cid),               # the ordinal, for this run only
            "size": len(idx),
            "medoid": rows[medoid]["key"],
            "density_peak": rows[peak]["key"],
            "centroid_norm": float(cn),
            "centroid_medoid_gap": gap,
            "mean_probability": float(probabilities[idx].mean()),
            "centroid": [float(v) for v in centroid],
        })
    return out


def label_agreement(labels, rows):
    """How the clustering and the labelling disagree.

    Not a score. A cluster holding three species may be the embedding seeing
    through a naming artefact -- 1,246 species names in this set have a single
    photo, many of them a rare congener the model reached for -- and a species
    split across clusters may be real variation in pose, distance or plumage.
    """
    per_cluster, per_species = collections.defaultdict(collections.Counter), collections.defaultdict(set)
    for cid, r in zip(labels, rows):
        sp = (r.get("species") or "").strip()
        if not sp:
            continue
        if cid != -1:
            per_cluster[cid][sp] += 1
            per_species[sp].add(cid)
    pure = sum(1 for c in per_cluster.values() if len(c) == 1)
    dominant = [max(c.values()) / sum(c.values()) for c in per_cluster.values()]
    split = sum(1 for s, cs in per_species.items() if len(cs) > 1)
    return {
        "clusters_with_labels": len(per_cluster),
        "single_species_clusters": pure,
        "mean_dominant_species_share": float(np.mean(dominant)) if dominant else 0.0,
        "species_split_across_clusters": split,
        "species_seen": len(per_species),
    }



def write_assignments(path, labels, probs, rows, names):
    """assignments.csv, ordered for reading rather than for the fitter.

    Largest cluster first with its members together, then down to the smallest,
    then the noise. That is the order someone browsing actually wants -- open the
    file and the biggest groups are at the top, each contiguous.

    `seq` is assigned **before** sorting: it is the row's position in the input,
    which every run of the sweep shares because they all load the same file and
    drop duplicate captures deterministically. So `seq = 8412` is the same photo
    in `mcs3` and in `mcs40`, which is what makes two runs comparable row by row,
    and sorting on it restores the input order in any spreadsheet. Numbering
    after the sort would have produced a different identity per run -- useless
    for the comparison it exists to serve.

    `cluster_size` is blank for noise: `-1` is not a cluster with no members, it
    is the absence of one, and a 0 would sort among real sizes.
    """
    sizes = collections.Counter(int(c) for c in labels if c != -1)
    records = []
    for seq, (cid, p, r) in enumerate(zip(labels, probs, rows), 1):
        cid = int(cid)
        records.append({
            "seq": seq,                       # input position, shared by every run
            "cluster_id": cid,
            "cluster_size": sizes.get(cid, ""),
            "cluster_name": names.get(cid, ""),
            "key": r["key"], "xmp": r.get("xmp", ""),
            "probability": f"{p:.4f}", "is_noise": int(cid == -1),
            "species": r.get("species", ""),
        })
    # noise last; clusters contiguous, largest first; stable within a cluster.
    records.sort(key=lambda d: (-(d["cluster_size"] or 0), d["cluster_id"], d["seq"]))

    fields = ["seq", "cluster_id", "cluster_size", "cluster_name", "key", "xmp",
              "probability", "is_noise", "species"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(records)


def run_one(X, rows, min_cluster_size, min_samples, out_root: Path, source, git_hash):
    import hdbscan

    out = out_root / f"mcs{min_cluster_size}"
    # A size owns its own directory and clears it before writing. It used to
    # refuse instead, on the grounds that runs are kept side by side -- but that
    # is what ./clean is for, at the level of a whole run. Within one run root,
    # refusing meant an interrupted fit left an empty mcs<N>/ that blocked the
    # retry, which is worst exactly when it matters: a parallel sweep where one
    # job dies and the others do not.
    #
    # Two instances on the same size at once is a user error, not a case to
    # defend against. The name check is the one cheap guard kept, because this
    # deletes a directory tree and --output-dir is a caller-supplied path.
    if out.exists():
        if not re.fullmatch(r"mcs\d+", out.name):
            sys.exit(f"error: refusing to clear {out}: not an mcs<N> directory.")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    logger.info(f"fitting min_cluster_size={min_cluster_size} on {X.shape[0]} vectors ...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                metric="euclidean",
                                # Measured 2026-08-14: a no-op on this data, kept
                                # because it costs nothing and helps if the
                                # dimensionality ever drops. At 768 dims hdbscan
                                # takes the generic path -- KD/ball trees are
                                # useless up here -- and the generic MST is
                                # single-threaded, so n_jobs=1 and -1 both took
                                # 21.1 s on 6,000 vectors. BLAS does the whole
                                # pairwise product in 0.2 s on the same box, so
                                # the cores are there; hdbscan cannot use them.
                                # Running several min_cluster_size values as
                                # separate processes is the only way to fill the
                                # machine -- see local/cluster-parallel.
                                core_dist_n_jobs=-1)
    labels = clusterer.fit_predict(X)
    probs = clusterer.probabilities_

    n_clusters = len(set(labels) - {-1})
    noise = int((labels == -1).sum())
    sizes = sorted(collections.Counter(labels[labels != -1]).values(), reverse=True)
    centres = cluster_centres(X, labels, probs, rows)
    agree = label_agreement(labels, rows)

    names = {c["cluster_id"]: c["name"] for c in centres}
    write_assignments(out / "assignments.csv", labels, probs, rows, names)

    with open(out / "centers.jsonl", "w", encoding="utf-8") as fh:
        for c in centres:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    with open(out / "clusterer.pkl", "wb") as fh:
        pickle.dump(clusterer, fh)

    summary = {
        "min_cluster_size": min_cluster_size, "min_samples": min_samples,
        "vectors": int(X.shape[0]), "dims": int(X.shape[1]),
        "clusters": n_clusters, "noise": noise,
        "noise_fraction": round(noise / len(labels), 4),
        "largest_clusters": sizes[:10],
        "median_cluster_size": int(np.median(sizes)) if sizes else 0,
        "label_agreement": agree,
        "source": source, "git_commit": git_hash,
    }
    (out / "run.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    logger.info(f"  clusters={n_clusters}  noise={noise} ({100*noise/len(labels):.1f}%)  "
                f"median size={summary['median_cluster_size']}  largest={sizes[:3]}")
    logger.info(f"  single-species clusters {agree['single_species_clusters']}/"
                f"{agree['clusters_with_labels']}, mean dominant share "
                f"{agree['mean_dominant_species_share']:.2f}, "
                f"{agree['species_split_across_clusters']}/{agree['species_seen']} species split")
    logger.info(f"  -> {out}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings", type=Path, default=None,
                    help="the curated vector set (default: data/embed/)")
    ap.add_argument("--output-dir", type=Path, default=Path("./output/cluster"))
    ap.add_argument("--min-cluster-size", default="5,15,40",
                    help="comma-separated sweep; each value gets its own output directory")
    ap.add_argument("--min-samples", type=int, default=None,
                    help="HDBSCAN's conservativeness; defaults to min_cluster_size")
    ap.add_argument("--keep-duplicate-captures", action="store_true",
                    help="do not collapse captures exported more than once. Off by default: "
                         "near-identical vectors double the local density exactly where a "
                         "cluster is being decided")
    args = ap.parse_args()
    if args.embeddings is None:
        args.embeddings = data_dir() / "embed" / "embeddings.jsonl"

    if not args.embeddings.is_file():
        sys.exit(f"error: {args.embeddings} not found")

    logger.info(f"loading {args.embeddings} ...")
    X, rows = load(args.embeddings)
    logger.info(f"  {X.shape[0]} vectors, {X.shape[1]} dims")

    norms = np.linalg.norm(X, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        sys.exit(f"error: vectors are not unit norm (min {norms.min():.4f}, "
                 f"max {norms.max():.4f}). Euclidean would stop matching cosine.")

    if not args.keep_duplicate_captures:
        X, rows, dropped = drop_duplicate_captures(X, rows)
        logger.info(f"  dropped {dropped} duplicate-capture vectors -> {X.shape[0]}")

    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_hash = "unknown"

    sizes = [int(s) for s in args.min_cluster_size.split(",") if s.strip()]
    for mcs in sizes:
        run_one(X, rows, mcs, args.min_samples, args.output_dir,
                str(args.embeddings), git_hash)


if __name__ == "__main__":
    main()
