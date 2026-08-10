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


def tile_matrix(X, size: int):
    """Mean cosine similarity per tile, computed in strips.

    Returns (T x T array, tile starts). Vectors are L2-normalised upstream, so
    the dot product *is* the cosine.
    """
    n = len(X)
    step = max(1, -(-n // size))                 # rows per tile, ceil
    starts = np.arange(0, n, step)
    widths = np.diff(np.append(starts, n)).astype(np.float32)
    T = len(starts)

    out = np.empty((T, T), dtype=np.float32)
    block = max(1, 4096 // step) * step          # whole tiles per strip
    for a in range(0, n, block):
        b = min(a + block, n)
        s = X[a:b] @ X.T                                        # (b-a, n)
        s = np.add.reduceat(s, starts, axis=1) / widths         # (b-a, T)
        local = np.arange(0, b - a, step)
        s = np.add.reduceat(s, local, axis=0) / widths[a // step:a // step + len(local), None]
        out[a // step:a // step + len(local)] = s
    return out, starts


def draw(M, out_path: Path, title: str, style: str, boundaries):
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
        ax.set_xlabel("image, in file order (clusters largest first)")
        ax.set_ylabel("image, in file order")

    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
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
    if not rows:
        return "no clustered rows"

    X = load_vectors(args.embeddings, [r["key"] for r in rows])
    M, starts = tile_matrix(X, args.size)

    # Boundaries of clusters wide enough to read. Keyed on the cluster's own
    # span, not on distance from the previous line: with 146 clusters the tail
    # is hundreds of small ones, and a line at each turns the corner of the
    # image into a mesh that hides the thing it is annotating.
    step = max(1, -(-len(rows) // args.size))
    edges, start, seen = [], 0, rows[0]["cluster_id"]
    for i, r in enumerate(rows + [{"cluster_id": None}]):
        if r["cluster_id"] != seen:
            if (i - start) / step >= args.min_edge:
                edges += [start / step, i / step]
            start, seen = i, r["cluster_id"]

    n_clusters = len({r["cluster_id"] for r in rows if r["cluster_id"] != NOISE})
    title = (f"{run_dir.name}: {len(rows):,} images, {n_clusters} clusters"
             f"{' + noise' if args.include_noise else ''} "
             f"({M.shape[0]}x{M.shape[0]} tiles, {step} images each)")
    out = run_dir / f"matrix-{args.style}{'-noise' if args.include_noise else ''}.png"
    draw(M, out, title, args.style, edges)
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
    ap.add_argument("--min-edge", type=float, default=8,
                    help="only outline clusters at least this many tiles wide (0 = none)")
    ap.add_argument("--include-noise", action="store_true",
                    help="keep cluster_id -1 rows (sorted last) instead of dropping them")
    args = ap.parse_args()

    if args.size is None:
        args.size = DEFAULT_SIZE[args.style]
    if not args.embeddings.is_file():
        raise SystemExit(f"error: {args.embeddings} not found")
    for r in resolve_runs(args):
        print(f"  {r.name}: {run(r, args)}")


if __name__ == "__main__":
    main()
