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

### The medoids sit differently in the two spaces

| | medoids | branches | silhouette of Ward's branches | mean cos | sd | cos CV | **eucl CV** |
|---|---:|---:|---:|---:|---:|---:|---:|
| DINOv3 | 1,194 | 70 | **0.0954** | 0.0550 | 0.1112 | 2.02 | **0.0618** |
| BioCLIP | 871 | 49 | 0.0579 | 0.3029 | 0.1147 | 0.38 | **0.0848** |

The *absolute* spread of pairwise similarities is nearly the same (sd 0.111
against 0.115). What differs is where it sits: DINOv3's medoids scatter around a
mean similarity of 0.055, BioCLIP's are packed into a narrow band at 0.303.

> **Correction.** This section originally read the coefficient of variation of
> those cosines — 2.02 against 0.38 — as DINOv3's medoids being *five times more
> variably spaced*, and built the claim below on it. That was wrong twice over,
> and the sd column above was the tell that went unread.
>
> The two standard deviations are equal to within 3%. The entire fivefold gap is
> the **denominator**: a contrastive image-text space is anisotropic, everything
> in it sits on a high similarity floor, and dividing an ordinary spread by 0.303
> instead of 0.055 manufactures a difference out of an offset.
>
> And the cosine is not the metric being described. `cluster2.py` hands Ward
> **euclidean** distances between L2-normalised medoids, and by *their* CV the
> ordering reverses: 0.0618 for DINOv3 against 0.0848 for BioCLIP. In the metric
> the algorithm actually consumes, BioCLIP's medoids are the more unevenly spaced
> — by about 50% — and they still give the less family-coherent branches. The
> last column is the honest one, and it points the other way.
>
> The rank-based results in this document are unaffected: euclidean distance on
> the unit sphere is a strictly decreasing function of cosine, so the AUC below is
> identical under either metric. It is only the scale-free *spread* that was being
> measured in the wrong units. `tools/audit_medoid_spread.py` now reports both,
> and `medoid_spread.__doc__` carries the reasoning.

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

This was measured first, then demoted to a *consequence* of the spacing story —
medoids spread thinly and unevenly carry more of every kind of structure,
taxonomy included, so the label-free measurement was said to have the causal
arrow pointing the right way.

**That demotion is withdrawn with the spacing story it rested on.** The AUC and
the silhouette are now the two surviving measurements of the mechanism, and
between them they say something narrower than a story: DINOv3's medoid geometry
separates families better (0.90 against 0.82) *and* Ward finds a geometrically
firmer partition in it (0.095 against 0.058). Less structure, and what there is
aligned better with taxonomy. Why that is true of a self-supervised space and not
of a taxonomy-supervised one is exactly the thing still unexplained — the spacing
account was an attempt at it, and it failed.

### The claim, stated so it can be refuted

**A two-level clustering recovers taxonomic structure to the extent that its
level-1 medoids are unevenly spaced relative to their own scale.** On this
evidence a coefficient of variation of ~2 supports it and ~0.4 does not.

That is a single comparison at one `min_cluster_size` with one clustering method
on one dataset, so it is a claim to attack rather than a result to rely on. It
would be refuted by a backbone with a low medoid CV that nonetheless produces
family-coherent branches, or a high-CV one that does not.

*It did not survive, and it was refuted from two directions at once — the
predictor does not track the outcome, and it was measured in the wrong metric.
The two sections below are the record; the claim is withdrawn, and the crossover
it tried to explain is not.*

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
branch counts. That was run the next day.

### The second test: both sweeps, and the claim does not survive it

BioCLIP now exists at the same five `min_cluster_size` values, so the comparison
can be made at matched granularity in both directions. Purity here is over each
run's own non-noise population, which is why BioCLIP mcs3 reads 0.4140 against
the 0.4211 above — that figure was over the two clusterings' intersection.

| clustering | leaves | branches | cos mean | cos sd | cos CV | eucl CV | family purity | images |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DINOv3 mcs3 | 1194 | 70 | 0.0550 | 0.1112 | 2.022 | 0.0618 | 0.6066 | 13,488 |
| DINOv3 mcs5 | 597 | 35 | 0.0593 | 0.1144 | 1.930 | 0.0637 | 0.5364 | 12,059 |
| DINOv3 mcs8 | 329 | 19 | 0.0652 | 0.1158 | 1.777 | 0.0646 | 0.4689 | 10,856 |
| DINOv3 mcs15 | 174 | 9 | 0.0737 | 0.1194 | 1.619 | 0.0671 | 0.3826 | 8,584 |
| DINOv3 mcs40 | 49 | 3 | 0.0646 | 0.1062 | 1.644 | 0.0586 | 0.2461 | 6,206 |
| BioCLIP mcs3 | 871 | 49 | 0.3029 | 0.1147 | 0.379 | 0.0848 | 0.4140 | 18,311 |
| BioCLIP mcs5 | 637 | 37 | 0.3091 | 0.1188 | 0.384 | 0.0884 | 0.3648 | 17,851 |
| BioCLIP mcs8 | 463 | 26 | 0.3135 | 0.1232 | 0.393 | 0.0921 | 0.3929 | 17,216 |
| BioCLIP mcs15 | 317 | 18 | 0.3200 | 0.1299 | 0.406 | 0.0977 | 0.3338 | 15,712 |
| BioCLIP mcs40 | 123 | 6 | 0.3481 | 0.1340 | 0.385 | 0.1042 | 0.2379 | 11,965 |

It answers the two questions in opposite directions.

**The crossover is confirmed, and at every granularity where the two can be
compared.** Noise fractions differ between the backbones — BioCLIP declines far
less — so the honest figure is over the images both runs place, with each run's
own population beside it:

| matched pair | shared images | DINOv3 | BioCLIP | gap |
|---|---:|---:|---:|---:|
| 35 vs 37 branches | 11,355 | 0.5361 | 0.3904 | **+0.1457** |
| 19 vs 18 branches | 9,672 | 0.4875 | 0.3720 | **+0.1155** |
| 9 vs 6 branches | 6,693 | 0.4145 | 0.2465 | **+0.1679** |

That is this finding's headline measured five ways instead of one, and it holds.
(The fourth pairing the tolerance admits, DINOv3's 3 branches against BioCLIP's
6, is not a matched pair — a factor of two in granularity — and near the floor
DINOv3 loses it, 0.2536 against 0.2933. At three branches over 283 families
neither number means much.)

**The mechanism is refuted, twice.**

*It does not track the outcome.* Across BioCLIP's whole sweep the cosine CV is
effectively constant — 0.379 to 0.406, a span of 0.027 — while branch family
purity falls from 0.4140 to 0.2379. The proposed predictor does not move where
the outcome moves. Worse for it, the two arms' CV ranges are disjoint by a factor
of four (1.62–2.02 against 0.38–0.41), so over the ten points CV is a relabelling
of *which backbone*: as a continuous predictor the design still has n=2, and eight
new points bought none of them. Over all ten, log branch count predicts purity
better than CV does — rho +0.758 (p=0.011) against +0.527 (p=0.117).

*And it was the wrong quantity.* Adding the sd and euclidean columns — which the
sweep forced, since ten rows make a pattern two rows hide — shows the cosine CV
was never measuring spread at all. The cosine standard deviations of the two arms
overlap completely (0.106–0.119 against 0.115–0.134; BioCLIP's are if anything
slightly *wider*), so the fivefold CV gap is the mean cosine and nothing else. In
the euclidean distances Ward is actually handed, the ordering reverses at every
single one of the five granularities: DINOv3 0.0586–0.0671 against BioCLIP
0.0848–0.1042. The claim said the more unevenly spaced medoids give the more
family-coherent branches; measured in the metric the algorithm consumes, the more
unevenly spaced medoids are BioCLIP's, and its branches are worse at every
matched granularity. That is not a weak correlation — it is the wrong sign.

The `test_the_cosine_cv_is_inflated_by_a_low_mean_not_a_wide_spread` case in
`test/tools/test_audit_medoid_spread.py` is this error in eight dimensions: one
cloud, shifted onto a similarity floor, reproduces both the inflated cosine CV
and the euclidean reversal.

**So the claim is withdrawn.** What is left is what was there before the CV was
ever computed: two backbones and an ordering, plus the AUC and silhouette that
say DINOv3's medoid geometry is both firmer and better aligned with family. The
geometry may still be the cause — those two measurements are consistent with it —
but *evenness of spacing* is not the form of it, and the coefficient of variation
is not how to detect it. A quantity with almost no variance inside a space cannot
explain what varies inside that space; its between-space variance is not separable
from everything else that differs between two backbones; and taken in the right
metric it points the wrong way.

The general lesson is cheaper than the finding: **a scale-free statistic is only
scale-free in the metric it is computed in.** Normalising by a mean makes an
anisotropy look like a spread, and this project reached for the CV precisely
because the two spaces were not comparable in raw units — which is the situation
where that substitution is most tempting and most wrong.

Reproducing it — one `--arm` per embedding, and the tool does the whole table
including the matched-granularity pairs:

```bash
python3 -m tools.audit_medoid_spread \
    --arm DINOv3  output_012_before_binomial/cluster output_012_before_binomial/cluster2 \
    --arm BioCLIP output/cluster output/cluster2 \
    --taxonomy   output_012_before_binomial/taxa/label_taxonomy.csv \
    --vocabulary output_012_before_binomial/taxa/vocabulary_taxa.csv
```

It takes arms rather than a run because a single backbone cannot answer this
question, which is what the first test found out the expensive way.

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

1. ~~**A sweep, not one `min_cluster_size`.**~~ **Done, and it holds.** Both
   backbones now run mcs3 to mcs40. The crossover survives at every matched
   branch count, by 0.12 to 0.17 of family purity — so it is a property of the
   two spaces and not of the parameter.
2. **The same leaf count.** *Mostly answered by the sweep* — matched branch
   counts within 3 give the same verdict as the lift did. What is still not
   matched is the **population**: B declines far less as `mcs` rises, so the two
   arms are scored over sets differing by thousands of images. The shared-image
   column above controls for that at each pair and moves the gap by under 0.03,
   which is the evidence it is not the explanation.
3. **The noise class.** Compare on A's noise specifically: those 38% are images A
   declines and B accepts, and whether they are the hard ones or merely the
   sparse ones is directly measurable.
4. **A second clustering method.** This is one algorithm at one setting. The
   stability study designed in `status/08` is the right instrument, and this is a
   good reason to build it.
5. ~~**Medoid CV over a third backbone.**~~ **Tested and refuted**, twice: the
   within-backbone correlation is confounded with granularity, and CV is flat
   across BioCLIP's whole sweep while its purity nearly halves. It is not the
   cheap screen it was proposed as. A third backbone would now be measuring the
   *ordering*, not the CV — and the useful third one is a supervised backbone
   that is not BioCLIP, since one self-supervised and one taxonomy-supervised
   arm cannot separate the training objective from everything else that differs.

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
- Five `min_cluster_size` values now, but still one clustering algorithm, one
  cut rule, one dataset, and two backbones.
- The spread statistics and the matched-granularity table are computed by
  `tools.audit_medoid_spread`; the branch purities agree to four decimals with
  `tools.audit_rank_alignment` run over the same layouts, which is the check that
  the two implementations of size-weighted modal purity have not drifted.
- The silhouette figures are the only numbers here not recomputed under the
  correction. They are euclidean already and their direction is unchanged, but
  they date from the mcs3-only pass and have not been swept.
- The BioCLIP artifacts were recovered after the fact from `local/output_bioclip/`
  and an export that predates `run.json` for renders. The clustering's own
  `run.json` records 871 leaves, 49 branches and 22,780 images, which matches the
  export, so the provenance is confirmed — but it was nearly not recoverable, and
  that is the reason exports now name their backbone.
