# 04 — Two-level clustering: find the taxonomy above the clusters

*Raised 2026-08-21 by Miles, while waiting for expert identification of the
species clusters and out of the original scope of the project. Measured and
refined the same day. **Promising, unrefined, not committed.***

## The idea

Take a fine clustering — HDBSCAN at `min_cluster_size=3` — reduce each cluster
to one representative vector, and cluster the representatives. Every image
inherits the group its cluster lands in.

The point is not merely coarser groups. It is that **a taxonomy is a hierarchy**,
and the structure worth finding is the one where birds become one large group,
people another, and scenery its own — unnamed, and discovered without being told
what any of those are.

## Why raising `min_cluster_size` cannot do this

Miles's argument, and the measurements agree with it.

A large `min_cluster_size` does not merge small clusters into larger ones. It
**blocks them from forming at all**, and their members become noise — discrete
points belonging nowhere. They cannot join a neighbouring cluster, because they
are not similar to one; that is why they were a separate small cluster in the
first place.

This dataset is **unevenly sampled**: some species have hundreds of photographs,
others a handful, some exactly one. Raising the threshold therefore rules out a
large part of the collection from being recognised at all, and the part it rules
out is systematically the rare part — which for a study of species is the part
that matters.

Lowering it does the opposite: small groups find their own clusters, purity
rises, and there are more clusters and fewer stranded singletons. The whole
library at 512px shows exactly that:

| | clusters | noise | homogeneity |
|---|---:|---:|---:|
| mcs3 | 2,477 | 41.8% | 0.9468 |
| mcs8 | 618 | 55.1% | 0.8531 |

So coarseness bought by raising the parameter costs an eighth of the library and
a tenth of the purity. Coarseness should instead be found *above* a fine
clustering, leaving level 1 intact.

## What was measured

Whole library, 49,224 images, DINOv3 ViT-B/16 at 512×512. Level 1 is mcs3:
2,477 clusters covering 57.8%. Scores are agreement with the pipeline's four-way
`category` label, computed on the population every method clusters — comparing
across different coverage is the confound that made the first attempt
meaningless.

### First attempt: HDBSCAN over the representatives

| method | groups | coverage | AMI | homogeneity |
|---|---:|---:|---:|---:|
| flat mcs3 (level 1) | 2,477 | 57.8% | 0.2678 | 0.9468 |
| flat mcs8 (the obvious alternative) | 618 | 44.7% | 0.3066 | 0.8531 |
| two-level, HDBSCAN mcs3 on medoids | 106 | 57.8% | 0.3516 | 0.8561 |
| condensed-tree ancestry cut | 35 | 57.8% | 0.0372 | 0.0204 |

Better than raising the parameter — one sixth the groups at higher agreement and
no loss of purity, where coarsening normally costs purity. But the level-2
parameter is **not even monotonic**: mcs3 gives 106 groups, mcs5 gives 2, mcs8
gives 11. That is not a threshold to tune; the selection is unstable.

### Better: Ward over the representatives, keeping the whole tree

Agglomerative clustering gives every level at once, has no parameter, and calls
nothing noise — which is what a taxonomy needs, since a taxonomy has no
outliers. Cutting the dendrogram over the same 2,477 representatives:

| k | AMI | homogeneity | the largest groups |
|---:|---:|---:|---|
| 2 | 0.2326 | 0.1828 | bird 42% (20,111), bird 100% (8,340) |
| **3** | **0.5439** | 0.5510 | scenery 53% (11,713), bird 97% (8,398), bird 100% (8,340) |
| 4 | 0.5229 | 0.5520 | scenery 53%, bird 97%, bird 100% |
| 8 | 0.4501 | 0.6076 | scenery 61%, bird 95%, bird 100% |
| 24 | 0.3522 | 0.6476 | scenery 58%, bird 100%, bird 100% |
| 106 | 0.2968 | 0.7672 | scenery 66%, bird 100%, animal 54% |

**The taxonomy appears at k=3**: 16,738 images in two groups that are 97% and
100% bird, and one mixed non-bird group. AMI 0.5439 against 0.3516 for two-level
HDBSCAN and 0.3066 for flat mcs8 — discovered with no labels, no names and no
parameter.

AMI falls monotonically after k=3 while homogeneity rises monotonically. The
taxonomy level and the browsing level are **different cuts of the same tree**,
which is the argument for keeping the tree rather than choosing one cut.

### The unexpected finding: HDBSCAN's own hierarchy is useless here

HDBSCAN already computes a hierarchy and stores it in `clusterer.pkl`, so
grouping level-1 clusters by their ancestors costs nothing. It does not work,
for a structural reason rather than a property of this dataset:

- median depth of a selected cluster below the root is **750**, max 992, for
  2,477 clusters — a caterpillar, not a balanced tree;
- at any cut coarse enough to be useful, one node swallows everything: at 35
  groups a single node holds **2,440 of 2,477 clusters (98.5%)**, hence
  homogeneity 0.02.

The condensed tree records *the order in which clusters shed from the main mass
as the density threshold rises*. In a high-dimensional embedding where nearly
everything is weakly connected, that is one long spine, and sibling relations in
it mean "we detached at about the same density" — almost no semantic content.
The tree asks about **density continuity**; clustering representatives asks
about **proximity of cluster meanings**, and only the second is a taxonomy.

## Details that already have evidence behind them

**Use medoids, not centroids.** `centers.jsonl` records `centroid_norm`, and the
mean of unit vectors is shortened by internal disagreement: median 0.884, 62% of
clusters (79% of images) below 0.9, and worse for larger clusters (correlation
−0.33 with size). Medoids scored better (0.3507 against 0.3423) — a small margin
in the direction the diagnostic predicted, and consistent with the existing
decision that a cluster's stable name is its medoid's key.

**Orphans must be attached, not dropped.** HDBSCAN at level 2 left 1,610 of
2,477 clusters as noise; discarding them costs two thirds of the library
(coverage 57.8% → 19.6%). Ward avoids the problem entirely by not having a noise
class.

## What would have to be true

1. **k=3 has to be more than an artefact of the scoring.** AMI penalises group
   count, so fewer groups score better on a four-class target by construction.
   The composition is the stronger evidence — two groups at 97% and 100% bird
   are not an artefact — but the peak's *location* is not trustworthy on this
   metric alone.
2. **The two bird groups need explaining.** They do not merge with each other at
   k=2; one joins the non-bird mass first. Whatever separates them is larger
   than what separates birds from scenery, which is either an artefact or
   something real about the collection, and nobody has looked.
3. **The result has to survive a sharper target.** `category` is four classes
   against hundreds of groups, so every AMI here is comparative only. Species on
   the bird subset would test whether the finer levels are semantically real or
   merely separating birds from people.
4. **Someone has to look at the groups.** As everywhere else in this project,
   agreement with the pipeline's own labels measures agreement with the noise as
   much as anything, and the only evaluation that finally counts is a person
   opening a group.

## What it would cost

Almost nothing to try again. Level 2 runs over 2,477 points and is instant, and
`centers.jsonl` and `clusterer.pkl` are already written by every clustering run.
The expense is entirely in evaluation.

## If it were adopted

The recommendation would be **Ward over the level-1 medoids, keeping the whole
linkage tree as an artifact** rather than any single cut — the tree is the
result, and a cut is a view of it. That belongs in `discover.py`, which already
consumes and produces exactly these things. Nothing has been designed, and
nothing should be until the questions above have answers.
