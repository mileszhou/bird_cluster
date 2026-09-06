# 04 — The better leaves and the better branches come from different backbones

*Measured 2026-09-05, from artifacts in `local/output_bioclip/` and
`output_012_before_binomial/`. Prompted by Miles asking what the BioCLIP
clustering had shown — the answer at the time being "nothing, it was never
analysed". **The result is a measurement without an explanation.** What produces
it is not known, and the guess this document originally carried was removed for
being a guess.*

Companion to `findings/01`, which asks whether a level-2 branch is a rank above
species. This asks the same question of a second embedding and gets a different
answer, which is not what either of us expected.

## What was compared

Two clusterings of the same 27,194 bird images, differing only in the vectors:

| | backbone | supervision | leaves | branches | images clustered |
|---|---|---|---:|---:|---:|
| A | DINOv3 ViT-B/16 @512 | self-supervised | 1,194 | 70 | 16,889 (62%) |
| B | BioCLIP 2.5 ViT-H/14 @224 | Linnaean names | 871 | 49 | 22,780 (84%) |

Both are HDBSCAN at `min_cluster_size=3`, then Ward over the leaf medoids cut
adaptively at 27 leaves per branch. **B declines to cluster far less**: HDBSCAN
assigns 84% of the population against A's 62%.

Purity is measured against the taxonomy in `output_012_before_binomial/taxa`,
restricted to labels the checklist named by *string* — so no per-image model call
enters the taxonomy, and neither backbone marks its own homework.

## The measurement

Raw purity across the two is not comparable: B forms fewer, larger clusters, and
purity falls with coarseness for free. So the comparison is on the **12,637
images both clusterings assign**, with a shuffled null over the same cluster
sizes to give a lift.

| | level | rank | purity | null | lift |
|---|---|---|---:|---:|---:|
| **B** BioCLIP | leaf | species | 0.6786 | 0.0863 | **7.9×** |
| A DINOv3 | leaf | species | 0.6979 | 0.1123 | 6.2× |
| **B** BioCLIP | leaf | family | 0.8716 | 0.1391 | **6.3×** |
| A DINOv3 | leaf | family | 0.8793 | 0.1643 | 5.4× |
| B BioCLIP | branch | species | 0.2361 | 0.0304 | 7.8× |
| **A** DINOv3 | branch | species | 0.2995 | 0.0325 | **9.2×** |
| B BioCLIP | branch | family | 0.4220 | 0.0812 | 5.2× |
| **A** DINOv3 | branch | family | **0.6090** | 0.0840 | **7.3×** |

**BioCLIP makes the better leaves; DINOv3 makes the better branches.** The
crossover is the result, and it is not small at level 2: 0.609 family purity
against 0.422, which survives the size correction (7.3× against 5.2×).

Note the leaf-level reversal only appears after correcting for granularity. On
raw purity BioCLIP's leaves look *worse* — 0.6786 against 0.6979 — and they are
not; they are fewer and larger. **Raw purity must not be compared across
clusterings of different granularity**, which is the methodological point worth
carrying out of this even if the rest does not survive.

## The mechanism, as far as it has been measured

*Miles arrived independently at the hypothesis this document originally carried
as a guess and then dropped: that BioCLIP's medoids are more evenly distributed,
so Ward has less to grip. Two people guessing alike is not evidence, but the
claim is sharp enough to predict things, so it was tested rather than reinstated.
The section is ordered the way the causation runs — geometry first, taxonomy
after — which is the correction Miles made to the first version of it.*

Ward at level 2 sees **only the leaf medoids**, and knows nothing about what is
under them. So the question is what the medoid geometry looks like, and that can
be asked without any labels at all.

### The medoids are more evenly spaced in BioCLIP's space

| | medoids | branches | silhouette of Ward's branches | mean cos | sd | **CV** |
|---|---:|---:|---:|---:|---:|---:|
| DINOv3 | 1,194 | 70 | **0.0954** | 0.0550 | 0.1112 | **2.02** |
| BioCLIP | 871 | 49 | 0.0579 | 0.3029 | 0.1147 | **0.38** |

The *absolute* spread of pairwise similarities is nearly the same (sd 0.111
against 0.115). What differs is where it sits: DINOv3's medoids scatter around a
mean similarity of 0.055, BioCLIP's are packed into a narrow band at 0.303.
Relative to their own scale — the coefficient of variation — DINOv3's medoids are
**five times more variably spaced**. That is the "more even distribution", stated
without reference to any taxonomy.

**And it is a shortage of structure, not misaligned structure.** Those are
different and they are distinguishable: if BioCLIP's medoids were strongly
grouped along some non-taxonomic axis, Ward would find geometrically solid
branches that merely failed to match families — high silhouette, low family
purity. Instead the silhouette is *lower* (0.058 against 0.095). Ward did not
find the wrong structure; there was less structure to find.

Both silhouettes are low in absolute terms, which is worth saying plainly: 0.095
is a faint partition too. Ward is extracting a weak signal in one space and a
weaker one in the other.

### The taxonomy follows from that, rather than causing it

Given the above, family should be less recoverable from BioCLIP's medoid
geometry. Over every pair of leaf medoids, labelled same-family or not by the
leaf's modal family:

| space | same-family cos | cross-family cos | **AUC** |
|---|---:|---:|---:|
| DINOv3 | 0.2810 | 0.0492 | **0.9006** |
| BioCLIP | 0.4565 | 0.3002 | **0.8227** |

AUC is the comparable figure — how often a same-family pair outranks a
cross-family one — because it is scale-free where raw cosines are not comparable
between spaces of different dimension and concentration. 0.90 against 0.82, in
the direction the geometry predicts.

This was measured first and initially written up as *the* cause. It is not: it is
a consequence. Medoids spread thinly and unevenly carry more of every kind of
structure, taxonomy included. The label-free measurement is the one with the
causal arrow pointing the right way.

### The claim, stated so it can be refuted

**A two-level clustering recovers taxonomic structure to the extent that its
level-1 medoids are unevenly spaced relative to their own scale.** On this
evidence a coefficient of variation of ~2 supports it and ~0.4 does not.

That is a single comparison at one `min_cluster_size` with one clustering method
on one dataset, so it is a claim to attack rather than a result to rely on. It
would be refuted by a backbone with a low medoid CV that nonetheless produces
family-coherent branches, or a high-CV one that does not.

### The first test of it, and what it costs the claim

The claim was tested the same afternoon it was written, by generating level-2
groupings across the whole DINOv3 `min_cluster_size` sweep and correlating medoid
CV against branch family purity:

| clustering | leaves | branches | medoid CV | branch family purity |
|---|---:|---:|---:|---:|
| DINOv3 mcs3 | 1194 | 70 | 2.022 | 0.6066 |
| DINOv3 mcs5 | 597 | 35 | 1.930 | 0.5364 |
| DINOv3 mcs8 | 329 | 19 | 1.777 | 0.4689 |
| DINOv3 mcs15 | 174 | 9 | 1.619 | 0.3826 |
| DINOv3 mcs40 | 49 | 3 | 1.644 | 0.2461 |
| BioCLIP mcs3 | 871 | 49 | 0.379 | 0.4211 |

Over all six: Pearson r=+0.322 (p=0.534), Spearman rho=+0.771 (p=0.072). Neither
is significant, and at n=6 nothing here could have been.

**The within-backbone half of that correlation is worth nothing.** Across the
five DINOv3 points CV and purity move together almost perfectly (rho=+0.900) —
but so do branch count and purity (rho=**+1.000**), and both are simply following
`min_cluster_size`. Fewer branches over a fixed taxonomy is mechanically less
family-pure, and coarser leaves have more evenly spaced medoids for reasons that
have nothing to do with the geometry the claim is about. Five points that all
share a backbone cannot test a claim about backbones.

**The one cross-backbone point is the one that does not fit.** BioCLIP's medoid
CV is 0.379 — five times lower than any DINOv3 point, below the sweep's whole
range — yet its branches are 0.4211 family-pure, mid-range, with 49 branches.
DINOv3 scores 0.5364 at 35 branches and 0.6066 at 70, so interpolated to
BioCLIP's granularity it sits near 0.57. The direction the claim predicts is
therefore right: BioCLIP is worse at matched granularity, and that is the
crossover this finding is about. The magnitude is not. A CV five times lower buys
a purity deficit of roughly 0.15, not the collapse a proportional reading of
"~2 supports it and ~0.4 does not" implies.

**So the claim comes down a rank.** What survives is an ordering over two
backbones — the space with the more unevenly spread medoids gives the more
family-coherent branches. What does not survive is CV as a *quantity* predicting
purity: the only evidence for that was the within-backbone sweep, which is
confounded with granularity, and the single point that is not confounded lies off
the line. As a cheap screen computable from vectors alone it is not yet usable,
and it was proposed as one.

The measurement that would separate "CV predicts purity" from "`mcs` predicts
both" is BioCLIP across its own sweep, giving cross-backbone pairs at matched
branch counts. Until those exist this rests on n=2 backbones, which is what it
rested on before the correlation was run.

Reproducing it: `./run-cluster2 --run output_012_before_binomial/cluster/mcs$M`
for each `M` in the sweep, then modal family purity per branch as in
`tools/audit_rank_alignment.py`, with CV taken over the pairwise cosines between
level-1 medoids.

## What is still not known

Why BioCLIP's medoid space is compressed in that way. A raised similarity floor
is what you might expect from a space organised by a supervised objective over
200M images of every kind of organism, where all birds occupy one region — but
that is a story again, not a measurement, and it is the next thing down the chain
rather than part of what has been established. Two other candidates remain
untested:

- an interaction with the adaptive cut, which stops at 27 leaves per branch and
  therefore produces fewer branches when there are fewer leaves;
- the noise class doing different work in the two spaces — A declines 38% of the
  population, B only 16%, and what A declines is not a random 38%.

## What would test it

1. **A sweep, not one `min_cluster_size`.** *Half done.* DINOv3 now exists from
   mcs3 to mcs40 (above), which showed how strongly branch purity follows
   granularity alone. BioCLIP still exists only at mcs3, so the crossover itself
   is still measured at one setting — and it is BioCLIP's sweep, not a third
   backbone, that is now the cheapest useful run.
2. **The same leaf count.** Cutting Ward to equal branch counts, or clustering B
   at a smaller `mcs` until it yields ~1,194 leaves, removes the granularity
   difference that the lift only partly controls for.
3. **The noise class.** Compare on A's noise specifically: those 38% are images A
   declines and B accepts, and whether they are the hard ones or merely the
   sparse ones is directly measurable.
4. **A second clustering method.** This is one algorithm at one setting. The
   stability study designed in `status/08` is the right instrument, and this is a
   good reason to build it.
5. **Medoid CV over a third backbone.** *Attempted, and it did not go well.*
   The correlation above is confounded within a backbone and the one unconfounded
   point misses the line, so CV is not yet the cheap screen it was proposed as.
   A third backbone would still help, but only alongside BioCLIP's own sweep —
   points at matched branch counts are what the question needs, not more points.

## Why it is filed anyway

Because the clustering existed for a week and had never been looked at, and
because the crossover is the kind of result that is cheap to lose and expensive
to rediscover. `findings/01` concluded that branches are a rank above species
using **A**; that conclusion is unchanged, but it is now known to be a property
of that embedding rather than of two-level clustering as such — B's branches are
markedly less family-coherent doing the same thing to the same photographs.

## Caveats

- The taxonomy comes from a labelling measured at roughly 35% species error
  (`findings/03`). Restricting to checklist-named labels removes the model-call
  half of the error, not the labelling's own.
- 12,637 of 27,194 images survive the intersection with the checklist
  restriction. That is under half the population, and the images both clusterings
  agree to cluster are not a random half.
- One `min_cluster_size`, one clustering algorithm, one cut rule, one dataset.
- The BioCLIP artifacts were recovered after the fact from `local/output_bioclip/`
  and an export that predates `run.json` for renders. The clustering's own
  `run.json` records 871 leaves, 49 branches and 22,780 images, which matches the
  export, so the provenance is confirmed — but it was nearly not recoverable, and
  that is the reason exports now name their backbone.
