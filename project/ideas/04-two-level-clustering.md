# 04 — Two-level clustering: cluster the clusters

*Raised 2026-08-21 by Miles, while waiting for expert identification of the
species clusters and out of the original scope of the project. Measured the same
day. **Promising, unrefined, not committed.***

## The idea

Take a fine clustering — HDBSCAN at `min_cluster_size=3` — and cluster its
clusters. Each level-1 cluster is reduced to one representative vector, and
HDBSCAN runs again over those few thousand representatives to produce
superclusters. Every image inherits the supercluster of the cluster it belongs
to.

The comparison worth making is against the obvious alternative: just raise
`min_cluster_size`. Both produce coarser groups; the question is whether they
produce *the same kind* of coarser group.

## Why it looked promising

Raising `min_cluster_size` does not merge clusters. It re-runs the density
estimate at a coarser smoothing scale, which yields a different partition and
pushes more points into noise — on the whole library at 512px, mcs3 leaves 41.8%
noise and mcs8 leaves 55.1%. Coarseness is bought by discarding an eighth of the
library.

Clustering the representatives leaves level 1 intact and asks a separate
question above it, so nothing is re-estimated and nothing new is discarded at
the point where the coarse structure is decided.

## What was measured

Whole library, 49,224 images, DINOv3 ViT-B/16 at 512×512. Level 1 is mcs3:
2,477 clusters covering 57.8%. Level 2 is HDBSCAN `min_cluster_size=3` over the
2,477 representatives. Scores are agreement with the pipeline's four-way
`category` label, computed on the population every method clusters, since
comparing across different coverage is the confound that made the first attempt
meaningless.

| method | groups | coverage | AMI | homogeneity |
|---|---:|---:|---:|---:|
| flat mcs3 (level 1) | 2,477 | 57.8% | 0.2678 | 0.9468 |
| flat mcs8 (the obvious alternative) | 618 | 44.7% | 0.3066 | 0.8531 |
| **two-level on medoids** | **106** | 57.8% | **0.3516** | 0.8561 |
| condensed-tree ancestry cut | 35 | 57.8% | 0.0372 | 0.0204 |

Two-level agrees with the labels better than flat mcs8 at **one sixth** the
number of groups, and gives up no homogeneity doing it (0.8561 against 0.8531).
Coarseness normally costs purity; here it did not.

## The finding that was not expected

**HDBSCAN already computes a hierarchy, and it is useless for this.** The
condensed tree is stored in `clusterer.pkl`, so grouping level-1 clusters by
their ancestors costs nothing. It does not work, and the reason is structural
rather than a property of this dataset:

- median depth of a selected cluster below the root is **750**, max 992, for
  2,477 clusters — a caterpillar, not a balanced tree;
- at any cut coarse enough to be useful, one node swallows everything. At 35
  groups a single node holds **2,440 of 2,477 clusters (98.5%)**, the rest in
  ones and twos, hence homogeneity 0.02.

The condensed tree records *the order in which clusters shed from the main mass
as the density threshold rises*. In a high-dimensional embedding where nearly
everything is weakly connected, that is one long spine. Sibling relations in it
mean "we detached at about the same density", which carries almost no semantic
information.

So the two constructions ask different questions. The tree asks about **density
continuity**; clustering representatives asks about **proximity of cluster
meanings**. For browsing — which is what the export is for — the second is the
useful relation, and it is not recoverable from the first.

## Details that already have evidence behind them

**Use medoids, not centroids.** `centers.jsonl` records `centroid_norm`, and the
mean of unit vectors is shortened by internal disagreement: median 0.884, with
62% of clusters (79% of images) below 0.9, and it is *worse* for larger clusters
(correlation −0.33 with size). Medoids scored better (AMI 0.3507 against 0.3423)
— a small margin, but in the direction the diagnostic predicted. It also agrees
with the existing decision that a cluster's stable name is its medoid's key.

**Orphans must be attached, not dropped.** Level 2 left 1,610 of 2,477 clusters
as noise, and discarding them would cost two thirds of the library — coverage
falls from 57.8% to 19.6%. Assigning each orphan to the nearest supercluster
restores full level-1 coverage.

## What would have to be true

1. **The orphan attachment has to be worth something.** It is currently
   unmeasured: the shared-population comparison excludes orphans by
   construction, so nothing above scores the 1,610 clusters that were assigned
   by nearest-neighbour rather than found. If those assignments are arbitrary,
   two thirds of the coverage is decoration.
2. **The level-2 parameter has to be less brittle than it looks.**
   `min_cluster_size` 3 gives 106 superclusters; 5 gives **2**. A cliff that
   sharp means the setting is not robust, and nobody should trust a number
   chosen on one side of it without understanding why the other side collapses.
3. **The result has to survive a sharper target.** `category` has four classes
   against hundreds of groups, so every AMI here is low in absolute terms and
   only comparable to its neighbours in the table. Species on the bird subset
   would test whether superclusters are semantically real or merely separating
   birds from people.

## What it would cost

Almost nothing to try again: level 2 runs over 2,477 points and is instant, and
`centers.jsonl` and `clusterer.pkl` are already written by every clustering run.
The expense is entirely in evaluation, and — as everywhere else in this project
— the only evaluation that finally counts is someone looking at the groups.

## If it were adopted

It would want to become part of `discover.py` rather than a separate tool, since
it consumes exactly what a run already produces and produces the same kind of
artifact. Nothing has been designed and nothing should be until the three
questions above have answers.
