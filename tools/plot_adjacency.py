#!/usr/bin/env python3
"""Adjacent-row difference along a seriated order: valleys inside, walls between.

After spectral seriation, plot how much each row differs from the next. If the
ordering has found real structure, a run of similar images is a valley and the
step from one group to the next is a wall -- and the walls should line up with
cluster boundaries. That is a sharper test of a seriation than the matrix
picture, because it is one dimension instead of two and a boundary is a spike
rather than a corner you have to squint at.

Two curves, because "difference between successive rows" has two readings and
they answer different questions:

**adjacent distance** -- `1 - cos(x_i, x_i+1)`, how far apart the two *images*
are. Local and noisy: two photos of the same bird are close, the next frame in a
burst is closer still.

**row difference** -- `||S_i - S_i+1||`, how differently the two images relate to
*everything else*. This is the one seriation is actually about. Inside a
coherent group every member sees the world the same way, so the row barely
changes; at a boundary the whole pattern of affinities changes at once.

The second looks expensive -- each row is n long -- but is not. With
`S_i = X x_i`, the difference of two rows is `X(x_i - x_i+1)`, whose norm is
`sqrt(d' G d)` for `G = X^T X`, a 768x768 matrix computed once. So it is O(d^2)
per pair rather than O(n): the whole curve for 27,194 images costs about what
one dense row would.

    python3 -m tools.plot_adjacency --run output/cluster/mcs15
    python3 -m tools.plot_adjacency --run output/cluster/mcs15 --all-images

By default it seriates the clustered rows of one run, so the cluster boundaries
can be drawn on top and the claim checked. `--all-images` seriates the whole
embedding file instead, clusters unknown.
"""

import argparse
import os
import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("PROJECT_ROOT")
                or str(Path(__file__).resolve().parents[1]))

from code.lib.csv_post import add_arguments, read_rows, resolve_runs  # noqa: E402
from tools.plot_matrix import fiedler_order, load_vectors, seriate  # noqa: E402

NOISE = "-1"


def adjacent_curves(X):
    """(1 - cosine between neighbours, ||row_i - row_i+1||) along the given order."""
    Xd = np.asarray(X, dtype=np.float64)
    delta = Xd[1:] - Xd[:-1]
    cos_gap = 1.0 - np.einsum("ij,ij->i", Xd[:-1], Xd[1:])
    # ||X delta||^2 = delta^T (X^T X) delta, and X^T X is d x d.
    G = Xd.T @ Xd
    row_gap = np.sqrt(np.maximum(0.0, np.einsum("ij,jk,ik->i", delta, G, delta)))
    return cos_gap, row_gap


def draw(cos_gap, row_gap, boundaries, out_path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for ax, y, label, colour in (
            (axes[0], row_gap, r"$\|S_i - S_{i+1}\|$   (relation to everything else)", "tab:blue"),
            (axes[1], cos_gap, r"$1 - \cos(x_i, x_{i+1})$   (distance between the two images)", "tab:orange")):
        ax.plot(y, lw=0.4, color=colour)
        # A rolling median rides over burst-to-burst noise without inventing peaks.
        if len(y) > 200:
            k = max(5, len(y) // 200)
            pad = np.pad(y, (k // 2, k - k // 2 - 1), mode="edge")
            smooth = np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=1)
            ax.plot(smooth, lw=1.2, color="black", alpha=0.7, label=f"median, window {k}")
            ax.legend(loc="upper right", fontsize=8)
        for b in boundaries:
            ax.axvline(b, color="crimson", lw=0.3, alpha=0.35)
        ax.set_ylabel(label, fontsize=9)
        ax.margins(x=0)
    axes[1].set_xlabel("position in the seriated order"
                       + ("   (red = cluster boundary)" if boundaries else ""))
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    ap.add_argument("--embeddings", type=Path,
                    default=Path("./data/embed/embeddings.jsonl"))
    ap.add_argument("--order", choices=("cluster", "fiedler"), default="cluster",
                    help="'cluster' seriates the cluster centroids and sorts each "
                         "cluster's members by distance to it -- boundaries are few "
                         "and real. 'fiedler' seriates every image, which does NOT "
                         "keep clusters contiguous (91.7%% of adjacent pairs cross a "
                         "boundary against 98.6%% for a random order) and so cannot "
                         "show walls")
    ap.add_argument("--all-images", action="store_true",
                    help="seriate the whole embedding file, not one run's clustered rows")
    args = ap.parse_args()

    for run_dir in resolve_runs(args):
        if args.all_images:
            keys, X = [], []
            with open(args.embeddings, encoding="utf-8") as fh:
                for line in fh:
                    d = json.loads(line)
                    keys.append(d["key"]); X.append(d["embedding"])
            X = np.asarray(X, dtype=np.float32)
            labels = None
        else:
            rows = [r for r in read_rows(run_dir / "assignments.csv")
                    if r["cluster_id"] != NOISE]
            X = load_vectors(args.embeddings, [r["key"] for r in rows])
            labels = [r["cluster_id"] for r in rows]

        if args.all_images or args.order == "fiedler":
            order = fiedler_order(X)
        else:
            # Seriate the centroids, then order each cluster's members by distance
            # to their own centroid. Few boundaries, and each is a real one.
            lab = np.array(labels)
            ids = list(dict.fromkeys(lab))
            C = np.stack([X[lab == c].mean(0) for c in ids])
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            order = []
            for i in seriate(C):
                idx = np.flatnonzero(lab == ids[i])
                blk = X[idx]
                order.extend(idx[np.argsort(np.linalg.norm(blk - blk.mean(0), axis=1))])
            order = np.asarray(order)
        X = X[order]
        cos_gap, row_gap = adjacent_curves(X)

        boundaries = []
        if labels is not None:
            lab = [labels[i] for i in order]
            boundaries = [i for i in range(1, len(lab)) if lab[i] != lab[i - 1]]

        out = run_dir / ("adjacency-all.png" if args.all_images
                         else f"adjacency-{args.order}.png")
        draw(cos_gap, row_gap, boundaries, out,
             f"{run_dir.name}: {len(X):,} images, "
             f"{'seriated clusters, members by distance to centroid' if not args.all_images and args.order == 'cluster' else 'spectral seriation of every image'}"
             + (f", {len(boundaries)} cluster boundaries" if boundaries else ""))
        print(f"  {run_dir.name}: {len(X):,} rows -> {out.name}")

        if boundaries:
            b = np.array(boundaries)
            at = row_gap[np.clip(b - 1, 0, len(row_gap) - 1)]
            elsewhere = np.delete(row_gap, np.clip(b - 1, 0, len(row_gap) - 1))
            print(f"    row-difference at a boundary: {at.mean():.3f}   "
                  f"elsewhere: {elsewhere.mean():.3f}   ratio {at.mean()/elsewhere.mean():.2f}x")
            # Sharper than the mean: of the largest steps, how many are real
            # boundaries? A mean can be lifted by a few spikes; this asks whether
            # the spikes *are* the boundaries.
            k = len(b)
            top = set(np.argsort(row_gap)[-k:])
            hit = len(top & set(np.clip(b - 1, 0, len(row_gap) - 1)))
            print(f"    of the {k} largest steps, {hit} are boundaries "
                  f"({hit/k:.0%}; {k/len(row_gap):.1%} expected by chance)")


if __name__ == "__main__":
    main()
