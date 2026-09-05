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

## The proximate cause, measured

*Added after Miles arrived independently at the same hypothesis this document
originally carried as a guess: that BioCLIP separates species so strongly that
the branch level has less to work with. Two people guessing the same thing is not
evidence, but it is a sharp enough claim to test — and it makes a prediction.*

Ward at level 2 sees only the leaf **medoids**. If the hypothesis holds, family
should be less recoverable from BioCLIP's medoid geometry than from DINOv3's.
Over every pair of leaf medoids, labelled same-family or not by the leaf's modal
family:

| space | same-family cos | cross-family cos | **AUC** |
|---|---:|---:|---:|
| DINOv3 | 0.2810 | 0.0492 | **0.9006** |
| BioCLIP | 0.4565 | 0.3002 | **0.8227** |

**AUC is the comparable figure** — how often a same-family pair outranks a
cross-family pair — because it is scale-free, where raw cosines are not
comparable between spaces of different dimension and concentration. Family is
markedly more recoverable from DINOv3's medoids (0.90) than from BioCLIP's
(0.82), which is precisely what predicts Ward recovering more family structure
there. **The crossover has a proximate cause.**

**The mechanism is not the one the hypothesis named, though.** BioCLIP's medoids
are not further apart — they are *closer together across the board*, with
cross-family pairs at 0.30 against DINOv3's 0.05. It is a compression, a raised
similarity floor that leaves less contrast for Ward to exploit, rather than
species being pushed apart. Both of us said "separated more"; the data says
"contrasted less".

## What is still not known

Why BioCLIP's medoid space is compressed in that way. A raised floor is what you
would expect from a space organised by a supervised objective over 200M images
of every kind of organism, where all birds occupy one region — but that is again
a story, not a measurement. Two other candidates remain untested:

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
5. **The same AUC over a third backbone.** If medoid-family AUC predicts branch
   family purity across embeddings generally, it is a cheap proxy: computable
   from vectors and a taxonomy alone, without clustering twice.

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
