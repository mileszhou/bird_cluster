# 02 — Clustering that adapts its resolution per species

*Raised 2026-08-09, after the first sweep. Partly answered by the literature and
by hdbscan itself; the rest is open.*

## The question

`min_cluster_size` is one number applied to the whole library, but the library
is not uniform. The first run showed both extremes at once, in the same fit:

    n= 396  common kingfisher 100%
    n= 242  yellow-legged gull 37%, black-headed gull 26%, great black-backed gull 12%
    n= 202  yellow-browed warbler 43%, arctic warbler 24%, sedge warbler 16%

A distinctive, well-photographed species resolves perfectly at the same setting
where three gulls merge into one blob. Gulls need finer resolution than
kingfishers, and no single global threshold can give it to them. Is there a
formulation that adapts?

## What is already adaptive, and what is not

Worth being precise, because half of this is solved and it is easy to reinvent.

**HDBSCAN's cluster *selection* is already adaptive.** That is the whole
difference from DBSCAN: the condensed tree holds clusters at every density
level, and excess-of-mass (`cluster_selection_method='eom'`, the default) picks
the most *stable* subtree in each branch independently. Different parts of the
data are already being cut at different density thresholds.

**`min_cluster_size` is not.** It is a size floor applied everywhere, and it does
double duty — it is simultaneously "the smallest thing I will call a cluster" and
a smoothing parameter on the density estimate. Those are two different
intentions wearing one number.

## Candidate answers, cheapest first

- **`cluster_selection_method='leaf'`** takes the *finest* clusters in the tree
  rather than the most stable ones. That is precisely the gull case: the gull
  blob is stable as a blob, so EOM keeps it whole, while leaf would descend to
  its children. One-line experiment, already installed. Expect many more, smaller,
  purer clusters and a different noise profile.

- **DBCV** (`hdbscan.validity`) scores a clustering *without labels*, from
  density alone. That is the direct answer to "an intrinsic way to decide": sweep
  the parameter, pick the maximum, no ground truth involved. Worth knowing that
  DBCV is reported to be unreliable in high dimensions, which is exactly where
  this data sits (228 components for 90% variance) — so it needs validating
  against the label agreement before being trusted, not instead of it.

- **`cluster_selection_epsilon`** sets a distance floor below which clusters are
  not split — a hybrid of DBSCAN and HDBSCAN. Useful in the opposite direction:
  it *prevents* over-splitting rather than enabling it.

- **Branch detection** (`hdbscan.branches`) subdivides a cluster by its internal
  structure rather than by density. A gull blob with three lobes is exactly the
  shape it is designed for.

- **Recursive clustering** — fit, then re-fit within each large or impure
  cluster with its own parameters. Crude, but it makes the per-species
  adaptivity explicit rather than hoping one number delivers it, and the
  hierarchy it produces is inspectable.

## Why it may not be a parameter problem at all

The gulls may be merged because *in DINOv3's space they are genuinely close* —
three grey-and-white birds on water, differing in details the backbone was never
trained to preserve. If so, no clustering parameter recovers them, and the
finding is about the embedding rather than the algorithm. A cheap way to tell:
check whether the gull species separate under supervision (do their centroids
differ significantly? is a linear probe able to tell them apart?). If a probe
can and clustering cannot, it is a resolution problem. If neither can, the
information is not in the vectors.

That distinction should probably come before any of the parameter work above.

## Related

- `project/reports/label_run_audit.md` — the 58% species self-agreement, which
  bounds how much of the "impurity" is real disagreement versus label noise.
- `project/ideas/01` — the same data used to study the labelling instead.
- The first sweep: `output/cluster/mcs{3,5,8,15,40}/run.json`.
