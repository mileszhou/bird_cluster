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

1. **A sweep, not one `min_cluster_size`.** Both clusterings exist only at mcs3.
   If the crossover holds from 3 to 40 it is a property of the spaces; if it
   moves, it is a property of that parameter.
2. **The same leaf count.** Cutting Ward to equal branch counts, or clustering B
   at a smaller `mcs` until it yields ~1,194 leaves, removes the granularity
   difference that the lift only partly controls for.
3. **The noise class.** Compare on A's noise specifically: those 38% are images A
   declines and B accepts, and whether they are the hard ones or merely the
   sparse ones is directly measurable.
4. **A second clustering method.** This is one algorithm at one setting. The
   stability study designed in `status/08` is the right instrument, and this is a
   good reason to build it.
5. **Medoid CV over a third backbone.** If the coefficient of variation of
   pairwise medoid similarity predicts branch family purity across embeddings
   generally, it is a cheap screen — computable from vectors alone, with no
   taxonomy and no second clustering. That is the version worth testing, because
   it needs nothing the label-free measurement does not already have.

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
