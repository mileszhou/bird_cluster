# 02 — What the embedding is a model of

*Arrived 2026-08-21, in conversation while the species clusters waited for expert
identification. Miles's argument throughout; the measurements were run against
the whole-library embedding to check it. **Interpretive, with evidence. Not a
plan, and nothing here obliges anyone to build anything.***

This is the conceptual residue of the project so far — the things understood
about *what is being studied*, as distinct from the things built. Several arrived
as corrections to earlier assumptions, which is why they are worth writing down:
the earlier assumption is usually still lying around somewhere.

## 1. The object of study is the model, not semantics

The original objective included studying **the topology of semantics**. That was
too ambitious by one step, and the correction is Miles's:

> The distinction in the study of semantics still survives, but the study itself
> is not about the topology of semantics itself, but the model as a result of
> the training process.

Everything the clustering shows is a property of DINOv3 under its training
objective. That is not a limitation to be apologised for — it is the thing being
measured — but conclusions of the form "these two pictures are semantically
close" are not available. Only "this model, trained this way, places them close"
is.

The project already had a narrower version of this rule (labels are not ground
truth, so metrics computed against them measure agreement with their noise). The
general form is stronger: **the geometry is evidence about the training, and only
indirectly about meaning.**

## 2. Semantics wants order; the model hands us a metric

> Topology as to semantics is mainly order relation, thus we are talking about
> order topology. In this aspect, geometry contains too much information as a
> semantics description is needed.

Order, topology and metric are increasing amounts of structure. If semantic
description needs only an order, then a 768-dimensional metric supplies far more
than the phenomenon has, and the surplus is not neutral — it is the training
objective's notion of similarity wearing the costume of meaning. Distances in
this space are not semantic distances.

## 3. Language is countable, and a dense countable linear order is ℚ

> Language is represented by a string of alphabets of finite length without a
> prescribed upper bound. This is a typical countably many infinite set. One
> extremely interesting point is, between any two semantic elements, there is
> one (of course, thus infinitely) more elements.

The recollection behind this is exact, and stronger than an analogy.
**Cantor's isomorphism theorem (1895):** any two countable dense linear orders
without endpoints are order-isomorphic. Under those premises — countable because
language is, dense because between any two meanings there is a third, no
endpoints — ℚ is not *a* model of the semantic order. It is *the* model, up to
isomorphism.

## 4. But the order is not total, and that is the interesting part

Cantor's theorem needs a *linear* order, and semantics does not look linear: a
taxonomy branches, and two leaves of different branches are incomparable rather
than ordered.

Miles supplied the reason himself, in the course of a separate point about
whether a statement is quantitative:

> This is exactly what the statement that any two elements of semantics has
> something in between, just richer than in one dimensional space.

**"Richer than one dimensional" is precisely what forces the order to be
partial.** Several independent gradable directions induce a *product* order, and
product orders are partial by construction: `(heavy, vague)` and
`(light, precise)` are incomparable. The density claim and the
richer-than-1D claim are not two observations; the second answers the first.

The universal object then changes. For countable dense *partial* orders the
analogue of Cantor's theorem is the Fraïssé limit of the finite partial orders —
the generic countable partial order. That, not ℚ, is what a countable dense
partial semantic order would be isomorphic to.

## 5. Negation is a cut, and the embedding is already the completion

> A word 'not' digs a hole in the semantic space.

A Dedekind cut *is* a partition into a downward-closed set and its complement,
which is what "not P" names. The completion of ℚ then fills those holes with the
irrationals: positions fully determined by a partition and denoted by no word.

This lands unexpectedly well on the actual artifact. ℝ⁷⁶⁸ is already complete, so
density in it is automatic and says nothing; the content of "between any two
there is a third" is a claim about **nameable** meanings — a countable dense
subset — while the space supplies the continuum around them. The embedding hands
us the completion and 49,224 samples of the dense subset.

The Cantor set was also raised as a candidate and is held more loosely here: its
content would be that fully-determined meanings are nowhere dense yet
uncountable within the space of partial descriptions, and no one has proposed a
way to test that.

## 6. Whether a statement is quantitative is itself semantic

> Steel is more heavy than aluminium by alpha times, vs. steel is heavier than
> aluminum has similar semantics (close in many models) but different in the
> sense whether they are quantitative. There must be a direction measuring
> whether a statement is quantitative in a model for general purpose.

The distinction has a standard name — Stevens' **levels of measurement**:
"heavier than" is *ordinal*, "α times heavier" is *ratio*. Same relation,
different scale type. With a pleasing self-application: the ladder
nominal < ordinal < interval < ratio is itself only ordinal. *How quantitative*
a statement is, is not a quantity.

The claim that this should appear as a **direction** is the linear
representation hypothesis, and it is testable with difference-in-means probes
over minimal pairs. It cannot be tested here — DINOv3 has no propositions — and
wants a language model and pairs like "α times heavier" against "heavier". That
experiment does not belong in this repository.

## 7. Measured: features are directions, and thin ones

What *can* be tested here is whether a semantic feature is linear at all. A
single difference-in-means direction, nothing trained, held out on half the data:

| separation | AUC | images |
|---|---:|---:|
| bird vs scenery | 0.9962 | 20,106 |
| bird vs people | 0.9988 | 17,181 |
| people vs scenery | 0.9515 | 10,057 |

That direction carries **1.8% of the total variance**. So a feature is a
direction, not a principal component — thin, nearly perfectly separating, and
invisible to anything that ranks axes by variance.

## 8. Measured: what the model does not encode

Miles predicted, from the training objective alone, that the model would
separate bird species but not human identity:

> I expected to see for the "cluster all" experiment is, while the model well
> distinguishes birds, it might not distinguish different individuals in people
> category. This was proved in the output of the experiment: as expected, I
> didn't see the same person is clustered together; rather different people in
> the same scene are clustered together.

Quantified as agreement between cluster and trip, within each category — does a
cluster mean *a kind of thing*, or *an occasion*?

| category | images | clusters | AMI(cluster, trip) | single-trip clusters |
|---|---:|---:|---:|---:|
| bird | 16,867 | 1,207 | 0.5859 | **49%** |
| people | 4,048 | 637 | 0.7177 | **84%** |
| scenery | 6,292 | 868 | 0.6742 | 62% |
| animal | 1,244 | 149 | 0.7143 | 54% |

A bird cluster routinely gathers one species across different trips and years —
it means a kind. A people cluster is overwhelmingly one afternoon. DINO's
objective asks two augmented views of *one image* to agree; nothing ever asked
two photographs of one person to agree, so identity is not represented and
people fall back to scene.

This is the cleanest demonstration in the project of section 1: the model's
notion of similarity is the training objective's, and it can be predicted in
advance from the objective rather than discovered by surprise.

## 9. The seriation failure was this fact, arriving early

`ideas/03` recorded that seriation orders but does not group — it recovers
well-separated clusters as contiguous blocks only up to about eight, and more
separation does not help. Miles, on rereading it here:

> I thought I found something that totally ordered the semantics of the
> similarity, but was soon proved wrong.

The symptom and the cause are now both written down. Seriation seeks a
permutation making the similarity matrix Robinson, which exists only if the
structure is genuinely one-dimensional; a partial order has no linear extension
preserving *nearness*. Section 7 puts a number on it: if semantics lived on one
axis, that axis would dominate, and the separating direction carries 1.8% of the
variance. There are many thin, roughly independent directions, and one
permutation cannot lay them flat.

What survived was right. Seriation remains the **display** order for the
Lightroom export, and the whole review workflow rests on it. A total order is a
serviceable presentation of a structure that has none — it simply cannot be
mistaken for the structure.

## What is not settled

- Whether the semantic order is countable *and* dense *and* partial is assumed,
  not shown. Only countability has an argument behind it.
- The quantitativeness direction is untested, and untestable in this repository.
- The Cantor-set analogy has no proposed test.
- Every measurement above is against the pipeline's own labels, so it inherits
  their noise. The four-way `category` is coarse, and `trip` is a proxy for
  "occasion" that conflates place, date and outing.
