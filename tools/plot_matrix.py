#!/usr/bin/env python3
"""Render the similarity matrix of an assignments.csv in the order the file is in.

The picture the sorted CSV already affords. Take every row in the order the file
lists them, compute the cosine similarity of every pair, and draw it. Because
`order_assignments.py` has made each cluster's members contiguous, a cluster
becomes a **bright block on the diagonal**, its side length its membership. So
one image shows how many clusters there are, how big each is relative to the
others, how internally tight each one is, and -- off the diagonal -- which
clusters resemble each other.

What to expect, and why:

**Blocks along the diagonal, noisy inside.** Within a cluster the rows are in
`seq` order, which is the order the vectors were loaded, unrelated to appearance.
So a block is bright but grainy. Ordering *within* a cluster (by distance to the
medoid, or a local seriation) is the obvious next refinement and would turn each
block from a bright square into a smooth mound.

**Off-diagonal scatter rather than off-diagonal blocks.** Clusters are laid out
largest-first, which is a reading order, not a similarity order. Two clusters of
related birds land wherever their sizes put them, so their mutual similarity
shows up as isolated bright dots far from the diagonal instead of a block beside
it. Seriating the *cluster centroids* and laying the blocks out in that order is
the second refinement; it is cheap, since there are only as many centroids as
clusters.

Both refinements are deliberately not done here. This tool draws the order that
exists so it can be looked at first.

**Noise is excluded by default.** At mcs15 noise is 69% of the rows; included, it
is a featureless field that the clustered structure occupies one corner of.
`--include-noise` puts it back, always sorted last.

The matrix is never materialised. 26,950 rows would be 2.9 GB as float32, and no
image is 26,950 px square anyway, so similarity is computed in row strips and
averaged straight into the output tiles. Peak memory is one strip.

    python -m tools.plot_matrix                              # every run under the working output
    python -m tools.plot_matrix --run output/cluster/mcs15
    python -m tools.plot_matrix --style contour --size 900
    python -m tools.plot_matrix --style surface              # the 3-d version of the same tiles

Reads the `C_` mark to check the file has been sorted, and writes none of its
own: this produces a PNG beside the CSV and does not touch the CSV.
"""

import argparse
import collections
from pathlib import Path

import numpy as np

from code.lib import csv_marks
from code.lib.csv_marks import REGISTRY
from code.lib.csv_post import add_arguments, read_rows, resolve_runs

NEEDS = REGISTRY["order_assignments"]
NOISE = "-1"

# A flat image resolves one tile per pixel, so it can take as many as it likes.
# A surface cannot: every tile becomes a facet, and past a couple of hundred the
# per-cluster spikes occlude the diagonal ridge they are supposed to sit on --
# the plot gets busier and says less. Coarser is the honest default in 3-d.
DEFAULT_SIZE = {"heat": 1200, "contour": 1200, "surface": 160}


def load_vectors(path: Path, keys):
    """Vectors for `keys`, in that order. Raises if any key is absent."""
    import json

    want = set(keys)
    found = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["key"] in want:
                found[r["key"]] = r["embedding"]
    missing = want - found.keys()
    if missing:
        raise SystemExit(f"error: {len(missing)} keys are not in {path}, "
                         f"e.g. {sorted(missing)[0]}")
    return np.asarray([found[k] for k in keys], dtype=np.float32)


def tile_starts(n: int, size: int, warp: float):
    """First row of each tile: `start(k) = n * (k/T)**warp`.

    `warp=1` is uniform tiling -- every tile the same number of images. Below 1
    the tiles start coarse and finish fine, which spends screen area on the
    small clusters at the tail instead of on the few big ones at the head. 0.5
    is the square-root warp (screen position proportional to i**2).

    Boundaries are deduplicated, so a warp steep enough to ask for sub-image
    tiles yields fewer tiles rather than repeated ones.
    """
    if warp >= 1.0:
        return np.arange(0, n, max(1, -(-n // size)))
    k = np.arange(size + 1) / size
    s = np.unique(np.round(n * k ** warp).astype(int).clip(0, n))
    return s[s < n]


def seriate(C):
    """Order clusters by similarity: the Fiedler vector of the centroid graph.

    Laying clusters out largest-first is a reading order that carries no
    similarity information -- measured at 1.05x enrichment against 1.02x for a
    random permutation, i.e. nothing. Two clusters of related birds land
    wherever their sizes put them, so their mutual similarity appears as
    isolated dots far from the diagonal and reads as noise.

    Ordering by the second eigenvector of the normalised Laplacian is the
    classical spectral seriation (Atkins-Boman-Hendrickson 1998): it is the
    continuous relaxation of minimising sum_ij S_ij (i-j)^2, so similar clusters
    are pulled adjacent. On this dataset it lifts enrichment to 1.64x against a
    1.82x ceiling for a matrix that is one-dimensional by construction.
    """
    W = np.clip(C @ C.T, 0, None)
    np.fill_diagonal(W, 0)
    d = W.sum(1)
    Dm = np.diag(1 / np.sqrt(np.maximum(d, 1e-9)))
    _, v = np.linalg.eigh(Dm @ (np.diag(d) - W) @ Dm)
    return np.argsort(v[:, 1])


def order_within(X, bounds):
    """Row permutation putting each cluster's members in distance-to-centre order.

    Untouched, members sit in `seq` order -- the order the vectors were loaded,
    unrelated to appearance -- so a diagonal block is bright but grainy. Sorting
    by distance to the centroid makes each block a smooth mound: the core is the
    top-left corner of its own block, the periphery the bottom-right, and how
    fast the block fades outwards is the cluster's radial profile drawn in
    place. It also makes N(r) readable straight off the block edge, since row
    position within a cluster is now rank in radius.
    """
    idx = []
    for a, b in bounds:
        block = X[a:b]
        idx.append(a + np.argsort(np.linalg.norm(block - block.mean(0), axis=1)))
    return np.concatenate(idx)


def tile_starts_by_cluster(bounds, size: int, warp: float):
    """Tile boundaries that give cluster *i* screen width proportional to its
    size**warp, and never let a tile straddle two clusters.

    The index warp moves resolution along a smooth curve that knows nothing
    about where the clusters are, so at a steep exponent a head tile averages
    across several of them and the diagonal dims -- an artefact of the tiling
    rather than of the data. Allocating per cluster instead keeps every boundary
    on a tile edge, so each block is drawn at its own resolution and the head
    keeps its structure while the tail gains some.

    warp=1 is proportional (uniform tiles again); warp=0 gives every cluster the
    same width, which compares internal texture at the cost of showing size.
    """
    sizes = np.array([e - s for s, e in bounds], dtype=float)
    share = sizes ** warp
    per = np.round(share / share.sum() * size).astype(int)
    per = np.clip(per, 1, sizes.astype(int))          # >=1 tile, <=1 image/tile
    starts = []
    for (s, e), k in zip(bounds, per):
        starts.extend(s + (np.arange(k) * (e - s) // k))
    return np.unique(np.asarray(starts, dtype=int))


def tile_matrix(X, starts):
    """Mean cosine similarity per tile, computed in strips.

    Vectors are L2-normalised upstream, so the dot product *is* the cosine.
    Tiles may have unequal widths (see `tile_starts`), which is why the strip
    loop walks tile indices rather than stepping by a constant.
    """
    n, T = len(X), len(starts)
    widths = np.diff(np.append(starts, n)).astype(np.float32)
    out = np.empty((T, T), dtype=np.float32)

    i = 0
    while i < T:
        j = i + 1
        while j < T and starts[j] - starts[i] < 4096:   # whole tiles per strip
            j += 1
        a, b = int(starts[i]), int(starts[j]) if j < T else n
        s = X[a:b] @ X.T                                        # (b-a, n)
        s = np.add.reduceat(s, starts, axis=1) / widths         # (b-a, T)
        s = np.add.reduceat(s, starts[i:j] - a, axis=0) / widths[i:j, None]
        out[i:j] = s
        i = j
    return out


def draw(M, out_path: Path, title: str, style: str, boundaries, xlabel: str,
         dpi: int = 140):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if style == "surface":
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d)
        fig = plt.figure(figsize=(11, 9))
        ax = fig.add_subplot(111, projection="3d")
        g = np.arange(M.shape[0])
        gx, gy = np.meshgrid(g, g)
        ax.plot_surface(gx, gy, M, cmap="magma", linewidth=0, antialiased=False,
                        rcount=200, ccount=200)
        ax.set_zlabel("cosine similarity")
        ax.view_init(elev=55, azim=-60)
    else:
        fig, ax = plt.subplots(figsize=(10, 9))
        if style == "contour":
            # contourf ignores origin= and puts row 0 at the bottom, which runs
            # the diagonal the opposite way from the heat map. Flip explicitly.
            im = ax.contourf(M, levels=14, cmap="magma")
            ax.invert_yaxis()
        else:
            im = ax.imshow(M, cmap="magma", interpolation="nearest", origin="upper")
        fig.colorbar(im, ax=ax, shrink=0.85, label="mean cosine similarity")
        # Cluster edges, but only where a block is wide enough to read; at 1,090
        # clusters every boundary would be a grid, not a annotation.
        for e in boundaries:
            ax.axhline(e, color="white", lw=0.35, alpha=0.5)
            ax.axvline(e, color="white", lw=0.35, alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("image, in plotted order")

    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def run(run_dir: Path, args) -> str:
    rows = read_rows(run_dir / "assignments.csv")
    if not rows:
        return "empty"
    state, _ = csv_marks.read_state(rows[0].keys())
    if not state & NEEDS.bit:
        raise SystemExit(
            f"error: {run_dir/'assignments.csv'} is not sorted "
            f"(state: {csv_marks.describe(state)}).\n"
            f"       The diagonal is only meaningful once cluster members are "
            f"contiguous. Run tools/{NEEDS.name}.py first.")

    if not args.include_noise:
        rows = [r for r in rows if r["cluster_id"] != NOISE]
    if args.focus:
        # One cluster, by id or by a substring of its medoid name. Zooming in is
        # the only way to ask whether a label means anything: at library scale a
        # species is a few pixels, and the question is whether it is a block.
        # Exact id first, and only fall back to the name. A bare number is a
        # cluster_id, but it is also a substring of half the medoid paths in the
        # library -- "12" silently pulled in 134 rows from other clusters.
        want = [r for r in rows if r["cluster_id"] == args.focus]
        if not want:
            want = [r for r in rows if args.focus in r["cluster_name"]]
            ids = {r["cluster_id"] for r in want}
            if len(ids) > 1:
                raise SystemExit(f"error: {args.focus!r} matches {len(ids)} clusters "
                                 f"in {run_dir}; use a cluster_id or a longer name")
        if not want:
            raise SystemExit(f"error: no cluster matches {args.focus!r} in {run_dir}")
        rows = want
    if not rows:
        return "no clustered rows"

    key = (lambda r: r["species"]) if args.group == "label" else \
          (lambda r: r["cluster_id"])
    if args.group == "label":
        # The file is grouped by cluster, not by label, so re-group before the
        # spans below mean anything. Largest label first, matching the cluster
        # convention; --order similarity then seriates them.
        rank = collections.Counter(key(r) for r in rows)
        # A label carried by one image cannot form a block, and 196 of this
        # cluster's 460 labels are singletons -- kept, they are 196 one-tile
        # groups whose boundaries alone fill the picture.
        rows = [r for r in rows if rank[key(r)] >= args.min_group]
        if not rows:
            raise SystemExit(f"error: no label reaches --min-group {args.min_group}")
        rows.sort(key=lambda r: (-rank[key(r)], key(r)))

    # Cluster spans, in file order. Contiguous by construction -- that is what
    # order_assignments guarantees and what the C_ check above verified.
    bounds, start, seen = [], 0, key(rows[0])
    for i, r in enumerate(rows + [None]):
        if (key(r) if r else None) != seen:
            bounds.append((start, i))
            start, seen = i, (key(r) if r else None)

    X = load_vectors(args.embeddings, [r["key"] for r in rows])

    if args.order == "similarity":
        # Seriate the real clusters; noise, if kept, is not a cluster and stays
        # pinned at the end where order_assignments put it.
        real = [b for b, r in zip(bounds, (rows[a] for a, _ in bounds))
                if r["cluster_id"] != NOISE or args.group == "label"]
        tail = bounds[len(real):]
        C = np.stack([X[a:b].mean(0) for a, b in real])
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        order = [real[i] for i in seriate(C)] + tail
        idx = np.concatenate([np.arange(a, b) for a, b in order])
        X, rows = X[idx], [rows[i] for i in idx]
        cuts = np.cumsum([0] + [b - a for a, b in order])
        bounds = list(zip(cuts[:-1], cuts[1:]))

    if args.within == "centre":
        idx = order_within(X, bounds)
        X, rows = X[idx], [rows[i] for i in idx]

    starts = (tile_starts_by_cluster(bounds, args.size, args.warp) if args.by_cluster
              else tile_starts(len(rows), args.size, args.warp))
    M = tile_matrix(X, starts)

    # Row index -> tile coordinate. Under a warp the tiles are unequal, so this
    # is an interpolation against the boundaries rather than a division.
    grid = np.append(starts, len(rows))
    at = lambda i: float(np.interp(i, grid, np.arange(len(grid))))

    # Boundaries of clusters wide enough to read *on screen*, measured after the
    # warp -- which is the point of the warp: a cluster too small to outline at
    # p=1 can become outlinable at p=0.5. Keyed on the cluster's own span, not
    # on distance from the previous line: the tail is hundreds of small clusters
    # and a line at each turns that corner into a mesh.
    edges = [x for lo, hi in ((at(a), at(b)) for a, b in bounds)
             for x in (lo, hi) if hi - lo >= args.min_edge]

    widths = np.diff(grid)
    n_clusters = len({key(r) for r in rows})
    scale = (f"{widths.min()}..{widths.max()} images/tile, "
             f"warp {args.warp:g}{' per cluster' if args.by_cluster else ''}"
             if args.warp < 1 or args.by_cluster else f"{widths[0]} images each")
    what = "species labels" if args.group == "label" else "clusters"
    head = f"{run_dir.name}" + (f" cluster {args.focus}" if args.focus else "")
    title = (f"{head}: {len(rows):,} images, {n_clusters} {what}"
             f"{' + noise' if args.include_noise else ''} "
             f"({len(starts)}x{len(starts)} tiles, {scale}"
             f"{', seriated' if args.order == 'similarity' else ''})")
    suffix = ("" if args.warp >= 1 and not args.by_cluster else
              f"-{'c' if args.by_cluster else 'p'}{args.warp:g}")
    suffix += "-ser" if args.order == "similarity" else ""
    suffix += "-ctr" if args.within == "centre" else ""
    if args.focus:
        suffix = f"-focus{args.focus.replace('/', '_')[:24]}" + suffix
    if args.group == "label":
        suffix += "-bylabel"
    out = (run_dir /
           f"matrix-{args.style}{suffix}{'-noise' if args.include_noise else ''}.png")
    draw(M, out, title, args.style, edges,
         f"image, {what} ordered by "
         + ("centroid similarity" if args.order == "similarity" else "size"),
         args.dpi)
    return f"{len(rows):,} rows -> {out.name}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    ap.add_argument("--embeddings", type=Path,
                    default=Path("./data/embed/embeddings.jsonl"))
    ap.add_argument("--style", choices=("heat", "contour", "surface"), default="heat")
    ap.add_argument("--size", type=int, default=None,
                    help="target tiles per side; one tile averages ceil(n/size)^2 pairs. "
                         "Default 1200 for heat/contour, 160 for surface, which cannot "
                         "carry that detail (see DEFAULT_SIZE)")
    ap.add_argument("--warp", type=float, default=1.0,
                    help="tile boundary exponent: start(k) = n*(k/T)**warp. 1 = uniform; "
                         "below 1 spends screen area on the small clusters at the tail "
                         "(0.5 is the square-root warp)")
    ap.add_argument("--order", choices=("size", "similarity"), default="size",
                    help="cluster layout order. 'size' is the file order (largest "
                         "first, a reading order); 'similarity' seriates the centroids "
                         "so related clusters sit adjacent")
    ap.add_argument("--focus", default=None,
                    help="restrict to one cluster, by cluster_id or a substring of its "
                         "cluster_name. Raises the default resolution, since one "
                         "cluster can afford far more tiles than the whole run")
    ap.add_argument("--group", choices=("cluster", "label"), default="cluster",
                    help="what a diagonal block is. 'label' groups by species instead, "
                         "so a block appearing means the label names something the "
                         "embedding also sees")
    ap.add_argument("--min-group", type=int, default=1,
                    help="drop groups smaller than this (only meaningful with "
                         "--group label, where most labels are near-singletons)")
    ap.add_argument("--dpi", type=int, default=None)
    ap.add_argument("--within", choices=("seq", "centre"), default="seq",
                    help="order inside each cluster. 'seq' is load order; 'centre' "
                         "sorts by distance to the cluster centroid, turning each "
                         "diagonal block into a mound and its edge into N(r)")
    ap.add_argument("--by-cluster", action="store_true",
                    help="allocate screen width per cluster as size**warp, snapped to "
                         "cluster boundaries, instead of warping the index smoothly")
    ap.add_argument("--min-edge", type=float, default=8,
                    help="only outline clusters at least this many tiles wide (0 = none)")
    ap.add_argument("--include-noise", action="store_true",
                    help="keep cluster_id -1 rows (sorted last) instead of dropping them")
    args = ap.parse_args()

    if args.size is None:
        args.size = DEFAULT_SIZE[args.style] * (3 if args.focus and
                                                args.style != "surface" else 1)
    if args.dpi is None:
        args.dpi = 300 if args.focus else 140
    if not args.embeddings.is_file():
        raise SystemExit(f"error: {args.embeddings} not found")
    for r in resolve_runs(args):
        print(f"  {r.name}: {run(r, args)}")


if __name__ == "__main__":
    main()
