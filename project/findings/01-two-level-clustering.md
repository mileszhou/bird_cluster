# 01 — Two-level clustering: find the taxonomy above the clusters

*Raised 2026-08-21 by Miles, while waiting for expert identification of the
species clusters and out of the original scope of the project. Measured and
refined the same day. Filed as `ideas/04` until it outgrew that, which is what
prompted this directory. **Measured, then looked at — see the update at the
end, which qualifies the taxonomy claim. Not committed.***

Companion to `findings/02`, which sets out what the embedding is a model of and why a taxonomy is the right thing to look for above the clusters.

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
   metric alone. **Answered below, in the direction of caution.**
2. **The two bird groups need explaining.** They do not merge with each other at
   k=2; one joins the non-bird mass first. Whatever separates them is larger
   than what separates birds from scenery, which is either an artefact or
   something real about the collection, and nobody has looked.
3. **The result has to survive a sharper target.** `category` is four classes
   against hundreds of groups, so every AMI here is comparative only. Species on
   the bird subset would test whether the finer levels are semantically real or
   merely separating birds from people. **Done below: species and a genus-ish
   target both scored on the bird subset, and they separate the two levels.**
4. **Someone has to look at the groups.** As everywhere else in this project,
   agreement with the pipeline's own labels measures agreement with the noise as
   much as anything, and the only evaluation that finally counts is a person
   opening a group. **Done; see the update at the end.**

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

## Measured again, on the export — and looked at

*2026-08-24. This is what questions 1 and 4 above were waiting for: a person
opened the groups. The structure is real and it is coarser than species; what
it is coarser **by** is still open, and the honest answer is probably
appearance rather than descent.*

**A different population from the tables above**, so the numbers are not
comparable to them. This is the bird-set clustering at mcs3 — 1,194 leaves
over 16,889 images — grouped by `cluster2.py` into 70 branches, with
the adaptive cut at 27 leaves. The earlier tables are the whole-library
embedding, 2,477 clusters.

### The two levels are shaped like two different ranks

Agreement with four targets, scored for the leaves and again for the branches.
The *head noun* of a species name — the last word — is a rough stand-in for
genus or family, and is the row that matters.

| target | classes | AMI leaf | AMI branch |
|---|---:|---:|---:|
| species | 2,066 | 0.6850 | 0.5777 |
| head noun (genus-ish) | 479 | 0.6448 | 0.6285 |
| trip (place and date) | 526 | 0.5852 | 0.4682 |
| library (year) | 10 | 0.3162 | 0.1775 |

There is a clean crossover. Leaves agree better with **species** than with the
head noun (0.6850 against 0.6448); branches agree better with the **head
noun** than with species (0.6285 against 0.5777). Level 2 is not simply a
blurrier level 1 — it is aligned to a coarser rank than the one below it, which
is what a taxonomy would predict.

And the grouping is emphatically not arbitrary. Leaves in a branch share that
branch's dominant head noun **0.4208** of the time, against **0.1125** when the
leaves are shuffled between branches at the same branch sizes (200 draws, sd
0.0041) — **76 sigma**. Branches also track that structure
more closely than they track place: 0.6285 against 0.4682
for the trip, so they are not mainly sorting by background or by season.

### Why this does not settle it

**The head noun is folk taxonomy, not taxonomy.** English bird names group by
appearance at least as much as by descent, and two unrelated birds that look
alike often share a head noun. So "branches agree with head nouns" is what
*both* hypotheses predict, and the vocabulary being scored against has the
confound built into it. What the numbers establish is that the level-2 grouping
is real and coarser than species. What they cannot say is whether the thing it
is coarser by is descent or resemblance.

**Looking settled that, as far as looking can.** Reviewing the export: branches
do pull near-identical leaves together, and sometimes gather one species that
level 1 had split — which is the useful case. But they mostly do not read as a
rank above species. What brings leaves into a branch is that they *look alike*,
and that is not the same relation as sharing an ancestor.

### What that changes

1. **Question 1 is answered in the direction of caution.** The composition
   argument stands — the structure is 70 sigma from chance and rank-shaped — but
   the taxonomy reading does not survive being looked at, and no metric here
   would have caught that, because every available target is appearance-tinged.
2. **Question 4 is answered.** Someone looked, and the answer changed the claim.
3. **The value is as a labelling aid, and that does not depend on the answer.** A
   branch that gathers visually similar leaves is reviewable in one pass, which
   is what makes it useful; it is useful *because* it is resemblance. Reading the
   hierarchy as a statement about the birds is the part to drop.
4. **It is a case for `findings/02` rather than against it.** Appearance-not-
   descent is what that document argues the embedding is a model of. A level-2
   grouping that is coarse, real, and not taxonomic is the predicted result, not
   a disappointment.

### The test that would settle it

Map each species to its real family from a checklist, then look only where
appearance and descent **disagree** — the convergent look-alikes that sit in
different families. If branches follow appearance those land together; if they
follow descent they separate. Everything else is confounded. The same checklist
is what `findings/03` needs to turn its consistency bound into an accuracy one,
so it buys two answers.
