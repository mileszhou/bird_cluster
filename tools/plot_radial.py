#!/usr/bin/env python3
"""Per-cluster radial profile: how many members lie within radius r of the centre.

Order a cluster's members by distance to its centroid and N(r) -- the count
inside radius r -- is just rank against radius. Drawn normalised (share of
members against share of the cluster's own radius) every cluster is comparable
regardless of size.

**The shape, and what it turned out to be.** For a uniform cloud N(r) goes as
r**d, so the log-log slope is the effective dimension and d=1 is the straight
diagonal dividing concave (a core, saturating outward) from convex (a shell,
centre empty). For a *Gaussian* cloud the radial density is a chi distribution,
r**(d-1)exp(-r^2/2), so N(r) is a sigmoid: convex up to one inflection at the
shell radius, concave after it. Measured on mcs40, that is exactly what every
cluster is -- one inflection each, at 0.49..0.94 of r_max, and 27 of 30 are
statistically indistinguishable from a fitted chi. The embedding's clusters are
Gaussian blobs.

**And that shape is nearly worthless as a diagnostic, which is worth knowing.**
Synthetic control, 768 dimensions, 600 points:

    one Gaussian mode                     1 peak, KS 0.038, p 0.36
    two modes, centres 30 sigma apart     1 peak, KS 0.028, p 0.74
    five well-separated modes             2 peaks, KS 0.105, p 0.00

Two modes flatly separated still give a textbook single-mode radial profile,
because in high dimensions everything in a mixture sits at about the same
distance from the joint centroid -- the shell is exactly where a mixture puts
its mass. So a clean chi fit is *not* evidence the cluster is one thing. The
5,469-member over-merge has the smallest KS statistic of any cluster here
(0.025) while being 6% pure. Radius alone cannot see the structure the matrix
plot shows plainly; do not read this plot as a unimodality test.

**What does survive is the exponent.** d ranks cluster quality without ever
consulting a label -- correlation -0.43 with species purity on mcs40, with the
over-merge at 19 against a median near 7. Since the labels are the object of
study rather than ground truth (project/ideas/01), a diagnostic that does not
consult them is worth more than one that does. d is biased by sample size, so
every cluster is resampled to a common n and both are reported; on mcs40 the
correction is small (19.1 -> 14.7) and the ranking survives.

    python -m tools.plot_radial
    python -m tools.plot_radial --run output/cluster/mcs40
    python -m tools.plot_radial --resample 0        # skip the equal-n control

Writes `radial.png` and `radial.csv` beside the assignments, and does not touch
the CSV. `tools/plot_matrix.py --within centre` draws the same information in
place, as the fade across each diagonal block.
"""

import argparse
import os
import sys
import collections
import csv
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.signal import find_peaks

# Running this as `python3 tools/x.py` puts tools/ on sys.path, not the repo
# root -- and then `import code.lib` finds the *stdlib* `code` module, which is
# not a package. Put the root first so both invocation forms work.
sys.path.insert(0, os.environ.get("PROJECT_ROOT")
                or str(Path(__file__).resolve().parents[1]))

from code.lib.csv_post import add_arguments, read_rows, resolve_runs
from tools.plot_matrix import load_vectors

NOISE = "-1"
FIT_LO, FIT_HI = 0.1, 0.9     # quantile range for the log-log fit, avoiding both tails
PROMINENCE = 0.15             # peak height, as a share of the density's maximum


def profile(X):
    """Radial statistics for one cluster."""
    n = len(X)
    r = np.sort(np.linalg.norm(X - X.mean(0), axis=1))
    F = np.arange(1, n + 1) / n
    lo, hi = max(1, int(FIT_LO * n)), max(2, int(FIT_HI * n))
    d = float(np.polyfit(np.log(r[lo:hi]), np.log(F[lo:hi]), 1)[0])

    # The inflection of N(r) is the mode of the density. Smooth first: the raw
    # spacing of order statistics is far too noisy to differentiate twice.
    grid = np.linspace(r[0], r[-1], 400)
    pdf = stats.gaussian_kde(r)(grid)
    peaks, _ = find_peaks(pdf, prominence=PROMINENCE * pdf.max())
    infl = float(grid[peaks[np.argmax(pdf[peaks])]] / r[-1]) if len(peaks) else float("nan")

    df, loc, scale = stats.chi.fit(r, floc=0)          # radius of an isotropic normal
    ks = stats.kstest(r, "chi", args=(df, loc, scale))
    return dict(r=r, F=F, d=d, infl=infl, peaks=len(peaks),
                ks=float(ks.statistic), ks_p=float(ks.pvalue), chi_df=float(df),
                chi_scale=float(scale), grid=grid, pdf=pdf,
                auc=float(np.trapezoid(F, r / r[-1])))


def draw(stats_, out_path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    ax1, ax2, ax3 = axes
    ds = np.array([s["d"] for s in stats_])
    norm = plt.Normalize(ds.min(), ds.max())
    cmap = plt.get_cmap("viridis")

    for s in stats_:
        c = cmap(norm(s["d"]))
        x = s["r"] / s["r"][-1]
        ax1.plot(x, s["F"], color=c, lw=0.9, alpha=0.8)
        ax2.loglog(x, s["F"], color=c, lw=0.9, alpha=0.8)
        ax3.plot(s["grid"] / s["r"][-1], s["pdf"] / s["pdf"].max(),
                 color=c, lw=0.9, alpha=0.8)
        if np.isfinite(s["infl"]):
            ax1.plot([s["infl"]], [np.interp(s["infl"], x, s["F"])], ".",
                     color=c, ms=5)

    for ax in (ax1, ax2):
        ax.plot([1e-3, 1], [1e-3, 1], "k--", lw=1, alpha=0.6)   # d = 1
        ax.set_xlabel("r / r_max")
    ax1.set_ylabel("N(r) / n"); ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.set_title("N(r): convex below the dashed d=1 line;\ndots mark the inflection")
    # Nothing lives near the centre -- the smallest r/r_max is around 0.4, which
    # is the shell effect itself -- so a decade of empty axis would hide the fits.
    lo = min(float(s["r"][0] / s["r"][-1]) for s in stats_)
    ax2.set_xlim(lo * 0.9, 1.05); ax2.set_ylim(1e-3, 1.3)
    ax2.set_title("log-log: the slope is the\neffective dimension d")
    ax3.set_xlabel("r / r_max"); ax3.set_ylabel("density (scaled)")
    ax3.set_xlim(0, 1)
    ax3.set_title("radial density: one peak each --\nchi-shaped, and blind to mixtures")

    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax3,
                 label="effective dimension d")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def run(run_dir: Path, args) -> str:
    rows = [r for r in read_rows(run_dir / "assignments.csv") if r["cluster_id"] != NOISE]
    if not rows:
        return "no clustered rows"

    X = load_vectors(args.embeddings, [r["key"] for r in rows])
    by = collections.defaultdict(list)
    species = collections.defaultdict(collections.Counter)
    for r, x in zip(rows, X):
        by[r["cluster_id"]].append(x)
        species[r["cluster_id"]][r.get("species", "")] += 1

    rng = np.random.default_rng(args.seed)
    m = min(len(v) for v in by.values()) if args.resample else 0
    out = []
    for cid, v in by.items():
        V = np.asarray(v, dtype=np.float32)
        s = profile(V)
        s["cluster_id"], s["n"] = cid, len(V)
        s["d_resampled"] = (float(np.mean([profile(V[rng.choice(len(V), m, False)])["d"]
                                           for _ in range(args.resample)]))
                            if args.resample else s["d"])
        top, k = species[cid].most_common(1)[0]
        s["purity"], s["species"] = k / len(V), top
        out.append(s)

    out.sort(key=lambda s: -s["n"])
    with open(run_dir / "radial.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["cluster_id", "n", "d", "d_resampled", "auc", "shape",
                    "inflection", "peaks", "chi_ks", "chi_p", "gaussian_like",
                    "purity", "dominant_species"])
        for s in out:
            w.writerow([s["cluster_id"], s["n"], f"{s['d']:.3f}",
                        f"{s['d_resampled']:.3f}", f"{s['auc']:.4f}",
                        "concave" if s["auc"] > 0.5 else "convex",
                        f"{s['infl']:.3f}", s["peaks"], f"{s['ks']:.4f}",
                        f"{s['ks_p']:.4g}", "yes" if s["ks_p"] > 0.05 else "no",
                        f"{s['purity']:.3f}", s["species"]])

    ds = np.array([s["d"] for s in out])
    ok = sum(s["ks_p"] > 0.05 for s in out)
    title = (f"{run_dir.name}: {len(out)} clusters, {len(rows):,} images "
             f"(d {ds.min():.1f}..{ds.max():.1f}, median {np.median(ds):.1f}; "
             f"{ok}/{len(out)} fit a chi distribution)")
    draw(out, run_dir / "radial.png", title)
    return (f"{len(out)} clusters -> radial.png, radial.csv "
            f"(d median {np.median(ds):.1f}, max {ds.max():.1f}; "
            f"{ok}/{len(out)} Gaussian-like)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    ap.add_argument("--embeddings", type=Path,
                    default=Path("./data/embed/embeddings.jsonl"))
    ap.add_argument("--resample", type=int, default=25,
                    help="draws per cluster for the equal-n control (0 to skip)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.embeddings.is_file():
        raise SystemExit(f"error: {args.embeddings} not found")
    for r in resolve_runs(args):
        print(f"  {r.name}: {run(r, args)}")


if __name__ == "__main__":
    main()
