# 03 — Calibrating a labeller: what a second opinion can bound

*Raised 2026-08-24 by Miles, on seeing that a second labelling of the whole library was
consistently but only slightly behind the first. Measured the same day. The observation
that made it a finding: a single labelling carries no estimate of its own error, and two
labellings bound one.* **Measured, not committed to anything.**

Companion to `findings/02`, which argues what the embedding is a model of; route 2 below
depends on that being an independent measurement. Gives a number to `ideas/01`.

## The problem

A labelling run produces a `confidence` column. It looks like an error estimate and is
not one: averaged over the 26,520 images in this comparison it reads **0.968**,
implying 3.2% error. Nothing inside a single run can check that figure — the
only thing it can be compared against is itself.

Two things this project already had turn out to bound it, and neither is a label.

## Route 1 — two labellings bound each other

Two independent labellings of one population disagree on some fraction `D`. On every
disagreement at least one of them is wrong, so `a + b >= D` with no assumptions at all,
and therefore the worse of the two is wrong on at least `D/2`.

**Match the vocabularies before measuring `D`.** Two labellers name the same bird
differently — one writes the qualifier, the other drops it — and counting strings makes
that look like disagreement. Optimal one-to-one assignment (Hungarian on the contingency
table) strips it:

| | share of images |
|---|---:|
| exact string agreement | 0.2880 |
| agreement after optimal renaming | 0.3798 |
| the renaming's contribution — naming convention | 0.0919 |
| irreducible disagreement `D` | 0.6202 |

Skipping the matching would have overstated the error bound by 9.2% of the
population, which is the whole of what a vocabulary difference is worth here.

So, distribution-free: **the worse of the two labellings is wrong on at least 31.0%**
of species calls.

Add one assumption — that when both are wrong they are usually wrong in *different*
directions, so agreement implies correctness — and `(1-a)(1-b) = A` closes the system.
It needs a second equation; the measured skill gap supplies it.

| assumed accuracy gap | better labelling | worse labelling |
|---|---|---|
| 0.05 | 35.8% error | 40.8% error |
| 0.09 *(measured, from 1-NN)* | 33.6% error | 42.8% error |
| 0.15 | 30.4% error | 45.4% error |

**These are ceilings on accuracy, not estimates.** Correlated errors — two models trained
on overlapping data making the same mistake — can only inflate agreement, and inflated
agreement makes both look better. The true error is at least this.

A corollary about *which* second labeller to run: for calibration the useful one is the
most **decorrelated** one, not the best one. A stronger model from the same family would
share more errors and bound less.

## Route 2 — a clustering bounds it, once a human has looked

HDBSCAN over an embedding computed from pixels alone gives a partition no labeller
influenced. Purity — the dominant species' share of a cluster — is normally read as a
property of the clustering. **Invert it.** If a human confirms the clusters are
single-species, then everything that is not the dominant name is a label error, and
`1 - purity` measures the labelling instead.

The human step is what licenses the inversion, and it is not optional: without it the
same number is equally consistent with clean labels over mixed clusters.

| mcs | clusters | images | error, better labelling | error, worse labelling | Wilcoxon p |
|---:|---:|---:|---:|---:|---:|
| 3 | 1,186 | 16,485 | 36.6% | 45.2% | 9.5e-17 |
| 5 | 597 | 14,692 | 38.4% | 47.2% | 6.9e-12 |
| 8 | 328 | 13,455 | 41.9% | 49.0% | 2.7e-06 |
| 15 | 174 | 10,677 | 43.0% | 48.9% | 1.9e-03 |
| 40 | 49 | 7,691 | 51.0% | 56.0% | 8.1e-03 |

The monotone rise is the internal consistency check. The finest clustering has the
smallest, tightest groups and is the most likely to be genuinely single-species, so its
figure is the label-noise floor; the climb above it is real species merging as clusters
coarsen, not labels getting worse.

## The two routes converge

| route | assumption it rests on | error of the better labelling |
|---|---|---:|
| 1 — two labellings | errors are decorrelated | 33.6% |
| 2 — clustering + a human | clusters are single-species | 36.6% |
| the labeller's own confidence | none; it is self-report | 3.2% |

Two measurements sharing no assumption land within three points of each other, and both
are an order of magnitude away from what the model says about itself. **The confidence
column is not an error estimate**, and there was no way to establish that from one run.

## What this does not establish

1. **Neither route measures correctness against a taxonomy.** Route 1 bounds disagreement
   between two opinions; route 2 bounds inconsistency within a visual group. A labeller
   that is confidently and consistently wrong about a species passes both. Only a
   reference checklist, or an expert, closes that.
2. **Route 2's caveats run both ways.** The dominant name in a cluster can itself be the
   wrong species, which pushes true error up; a cluster that genuinely holds two species
   inflates the measured figure, which pushes it down. The second is the more likely here,
   so the floor is if anything an overestimate.
3. **The independence assumption in route 1 is the weak link**, and it is the reason the
   result is stated as a ceiling rather than a value.
4. **One pair of labellers, one collection.** The procedure is general; the numbers are not.

## What follows

**A second labelling's value is as an error bar, not as a better label.** That reframes
what the never-arbitrate design in `CLAUDE.md` produces: the paired verdict is worth
keeping even when — especially when — the second model loses, because the losing
comparison is what carries the bound. Promoting the winner was never the point.

**Purity at the finest clustering measures label noise, not clustering quality**, once
monospecificity has been checked by eye. Every purity figure recorded before that check
was answering a question nobody meant to ask.

Nothing here is a commitment. If it became one, the shape would be a tool that takes two
label CSVs and a clustering run and emits the three routes — the computation is a minute,
and the matching step is the only part that is easy to get wrong.
