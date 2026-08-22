# 02 — What the embedding is a model of

*Arrived 2026-08-21, in conversation while the species clusters waited for expert
identification. Miles's argument throughout; the measurements were run against
the whole-library embedding to check it. §10 leaves this project's own subject
matter and is about learning in general — kept here because it is the same
argument continued, and because it is where the framework first predicted
something instead of describing it. **Interpretive, with evidence. Not a
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

*Refined by §5a: dense is right, but only half of it.*

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

## 5. Negation is a cut; the space is dense *and* codense

> A word 'not' digs a hole in the semantic space.

A Dedekind cut *is* a partition into a downward-closed set and its complement,
which is what "not P" names. The completion of ℚ then fills those holes with the
irrationals: positions fully determined by a partition and denoted by no word.

This lands unexpectedly well on the actual artifact. ℝ⁷⁶⁸ is already complete, so
density in it is automatic and says nothing; the content of "between any two
there is a third" is a claim about **nameable** meanings — a countable dense
subset — while the space supplies the continuum around them. The embedding hands
us the completion and 49,224 samples of the dense subset.

### 5a. Dense *and* codense: the semantic space is co-formed

The Cantor set was raised as a candidate for the same phenomenon, and refining
it produced the sharpest statement in this document. Miles:

> Any concept is not an open set; there is always some direction (dimension in a
> linear model) in which two elements are widely separated.

and, on why:

> Semantics space is coformal. Any part of it is co-formed with other parts, no
> matter how small or large the areas are (quite much like rational and real
> numbers), since the language can be used in such a way.

Two claims that look like rivals are one property. **Density** says something
nearby *belongs*; **non-openness** says something nearby *does not*. ℚ in ℝ has
both at once: between any two rationals there is a rational, and between any two
rationals there is an irrational. Neither set contains an interval; each is
dense in the other's gaps. The standard name is **dense and codense** — dense,
with dense complement, equivalently dense with empty interior — and it holds at
every scale, in every interval however short, which is exactly "no matter how
small or large the areas are".

The Cantor set is a good hint that **overshoots**. It captures non-openness, but
buys it by giving up density and acquiring jumps: the endpoints of each removed
interval are both in it with nothing of it between them. Nowhere dense is too
strong. Dense-and-codense is the property wanted, and ℚ ⊂ ℝ is the faithful
picture. (An earlier draft of this conversation argued the finite alphabet
*forces* the Cantor set, via Σ^ℕ. It forces codensity; Cantor is one way to
obtain it and not the right one.)

**Why it holds is generativity, not geometry.** Whatever region you point at,
the language can articulate a distinction inside it, so no region is homogeneous
and therefore none is open. The same richness that always supplies another term
also always supplies a dividing one — density and codensity have one source.

A consequence that follows and is uncomfortable: Baire category separates the
two sides even though both are dense. ℚ is **meagre**, the irrationals
**comeagre**. Carried across, the nameable meanings are dense — every meaning is
approximable by one you can say — and yet topologically negligible, while the
meanings no word reaches are the generic case. Language is everywhere and almost
nowhere at once. That is a stronger statement than the hole metaphor: the holes
are not exceptional, they are almost all of it.

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

### 7a. Measured: closeness does not survive projection

§5a predicts something checkable: pairs that are close *overall* should still be
separated along the thin direction, because the aggregate metric averages over
768 coordinates and hides a large difference in one. Ranking held-out images by
how close they sit to the opposite class's centre, and scoring only the hardest:

| subset | n | AUC of that one direction |
|---|---:|---:|
| all held out | 20,077 | 0.9959 |
| the 50% closest to the other class | 10,038 | 0.9871 |
| the 20% closest | 4,015 | 0.9540 |
| the 5% closest — most confusable | 1,003 | 0.7966 |

The mean cosine between a bird and a scenery image is **+0.019** — all but
orthogonal — while the gap along the direction is **0.244**. The aggregate
metric says "everything is unrelated to everything" and one thin direction
carries the distinction.

That near-orthogonality is the high-dimensional concentration phenomenon Miles
pointed at, in its limit form: for `u, v` uniform on the sphere in `n`
dimensions, `⟨u,v⟩` has mean 0 and variance `1/n`, with
`P(|⟨u,v⟩| ≥ ε) ≤ 2e^{−nε²/2}`. Summable, so Borel–Cantelli gives almost-sure
convergence to 0 along `n`. Worth stating as the limit rather than as a property
of an infinite-dimensional object: a separable Hilbert space carries no uniform
measure on its unit sphere, so "random unit vectors are a.s. orthogonal" is a
statement about the sequence, not about a draw.

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

The symptom and the cause are now both written down, and the cause is narrower
than "it was impossible".

Seriation seeks a permutation making the similarity matrix Robinson — entries
decreasing away from the diagonal — which requires the structure to be
one-dimensional in a metric sense. §7 puts a number on why it is not: the
separating direction carries 1.8% of the variance, so there are many thin,
roughly independent directions and no permutation lays them flat.

But a **faithful linear arrangement does exist**, which is why the idea felt so
nearly right. Every zero-dimensional compact metrizable space embeds in the
Cantor set and hence in ℝ; with compactness, total disconnectedness gives
zero-dimensionality, so a totally disconnected semantic space *does* fit on a
line without losing a point. Miles's own image for it:

> Cantor set also hints an incorrect one dimensional illusion, but a good hint
> of it: Cantor set doesn't contain an open set.

The Cantor set is the model of that situation: genuinely inside ℝ, genuinely
totally ordered by it, and genuinely not one-dimensional — it is 2^ℕ wearing a
line as a costume. So the correct epitaph is **not** that no linear
representation exists. It is that the linear representations which exist are
topologically faithful and **metrically unfaithful**, and seriation's premise is
that adjacency means similarity, which is a metric claim.

What survived was right. Seriation remains the **display** order for the
Lightroom export, and the whole review workflow rests on it. A line is a
serviceable presentation of a structure that is not linear — it simply cannot be
mistaken for the structure.

## 10. Codensity is the formal shape of overfitting

> An LLM implements a function. A theorem says that any continuous function can
> be approximated by a neural network. This gave us confidence that neural
> networks can do any reasoning: a reasoning is a function of what is given as
> the premises or input. This is exactly the coformality of semantics: a finer
> structure always exists to make the language reach any concept one is to
> represent without a prescribed upper bound.

The parallel is exact rather than figurative. **Universal approximation is
literally a density theorem** — Cybenko (1989), Hornik (1991): finite networks
are dense in `C(K)` under the sup norm. And the second half of §5a applies as
well: they are also **meagre** in it. Dense and codense in function space, the
same ℚ-in-ℝ structure one level up. Always closer, never arrived.

That is the setting for overfitting, and the two claims are the same claim.
**If a finer distinction always exists, then for any finite labelled set there
is a hypothesis separating it** — which is shattering, hence infinite VC
dimension, hence no distribution-free generalisation guarantee. So codensity,
read as a property of a hypothesis class, is exactly overfittability.

Which is why the theorem never explained the success it was credited with:

> However, LLM's success lays the other side: generativity. It captures the
> meaning in our daily reasoning in a reasonable way.

Capacity arguments say why fitting *anything* is possible and are silent about
why the fitted thing generalises. Richness is the permission, not the reason.
The universal approximation theorem, used as an account of why neural networks
work, is an argument that they should overfit.

### 10a. Why topology hands over to probability

*Read with §11: "hands over to probability" is about the **description**, not the
object. Nothing here says semantics is vague or non-deterministic.*

> Semantics as a topology object is more in a theoretical modeling; in practice,
> probabilistic description is more a useful description.

There is a sharper reason than "topology is qualitative". Topological
genericity and probabilistic typicality are different notions that **routinely
point in opposite directions**. The Liouville numbers are comeagre — a dense
G_δ, topologically almost all of ℝ — and have Lebesgue measure zero. Comeagre
and almost sure are not approximations of each other.

This qualifies §5a's uncomfortable consequence. The nameable meanings being
dense but meagre may carry no practical weight at all, because the meanings that
actually occur are drawn from a distribution rather than sampled by Baire
category. The asymmetry can be topologically true and probabilistically
irrelevant, and nothing here decides which.

### 10b. What the order picture buys: freedom in the parts that do not matter

> The topological nature of order relation renders us freedom in choosing the
> activation functions and potentially many other parameters, and quite a lot of
> tolerance to our quantization.

Both halves have precise backing.

**Activations.** Leshno, Lin, Pinkus and Schocken (1993) sharpened universal
approximation to an iff: a network with continuous activation is dense in `C(K)`
**exactly when the activation is not a polynomial.** The freedom is not merely
wide, it is characterised — everything but polynomials — which is why the choice
has always felt unimportant in practice.

**Quantization follows from §2 rather than illustrating it.** If what carries
meaning is order, then any monotone map preserves it, and quantization is a
monotone step function: order survives, magnitude degrades. So "semantics wants
order while the model supplies a metric" *predicts* that most of the metric
precision is surplus and can be thrown away — which int8 and int4 quantization
show empirically — and it predicts the failure mode too, since tasks needing
fine magnitude discrimination should degrade first.

That is the one place so far where this framework predicted something before
being told it, which is the only kind of evidence an interpretive document like
this can offer for itself.

## 11. What kind of thing the probability is

This is the summary of why the account moves from topology to probability, and
it corrects a reading the earlier sections invite.

> Saying that semantics is more on probability is not saying that semantics is
> non deterministic or in any sense vague. It could be vague, but using
> probability is not about its possibly vagueness; vagueness is how people take
> it as. By placing a sentence in a fine enough context, the meaning is
> fine-enoughly defined. What probability plays in semantics description is
> highly formal.

**The probability is a measure on the observer, not on the object.** That is the
whole of it, and it is why probability can sit on top of a fully determinate
semantics without making it fuzzy.

The model for this is cryptographic, and the correspondence is an identification
rather than an analogy — the term of art is literally **semantic security**
(Goldwasser and Micali, 1984): a ciphertext is secure iff whatever an adversary
can compute about the plaintext given it, they could compute nearly as well
without it. Nothing in that definition suggests the plaintext is vague. The
message is perfectly determinate; the probability measures *access* to it. The
formal object is **advantage** — `Pr[success | resource] − Pr[success | baseline]`.

> The security of a crypto protocol is how little the cypher message's gain in
> probability towards a successful decrypting the message. Similarly, knowledge
> is the probability gain towards a correct response to a question in the
> domain. Similarly, capability is interpreted, or formulated in a similar way:
> the gain to do the thing right (for a micro task of one bit complexity), or
> better (for a complex task).

| | resource | success |
|---|---|---|
| security | the ciphertext | recovering a function of the message |
| knowledge | the model, or the text | a correct response over a question distribution |
| capability | the same | doing the task right, or better |

One form, three baselines. And it dissolves the apparent tension with §1: the
object of study is the model, and what is measured is what the model gives
access to.

### 11a. The one-bit case is where it is clean, and that is not incidental

At one bit, success is binary, the baseline is ½, and advantage is well defined
with no scale on outcomes at all. "Or better, for a complex task" quietly needs
more: comparing outcomes requires at least an **ordinal** scale, and taking
expectations requires an **interval** one.

So §6's measurement ladder governs how far this formulation can be pushed.
Advantage is a ratio-scale quantity resting on a nominal success predicate, and
it degrades the moment success stops being binary — which is a real limit on
"capability" as a number, and plausibly why capability benchmarks are so much
shakier than security proofs.

### 11b. The missing third: semantics itself

Knowledge and capability have the form. Semantics does not yet, and this is
recorded as unfinished rather than filled in. Miles:

> I stated knowledge and capability, but didn't give a counterpart of semantics,
> because it has not yet fully formalized in my mind, with the direction,
> formation and relationship in mind. The probability space might be the way
> people take a sentence as their own meaning (huge space).

So the direction is: the probability space is over **uptakes** — how a sentence
is taken — and it is large.

The pattern of the other two *suggests* a shape, offered here as a candidate and
not as his position: meaning as the advantage an utterance confers toward the
intended interpretation, i.e. how far it moves a listener's distribution over
uptakes from prior to posterior. That is the same `Pr[· | resource] − Pr[· | baseline]`
form, with the utterance as resource and the interpretation as success. Whether
that is the right object is exactly what is not settled, and his three words name
the gaps: **direction** (whose probability — speaker's, listener's, or a
convention over both), **formation** (how the space of uptakes is constituted at
all), **relationship** (how it stands to knowledge and capability, which are
defined against it).

### 11c. Vagueness is uptake, and codensity licenses the repair

"Vagueness is how people take it as" places the indeterminacy in the uptake
rather than in the meaning, and "a fine enough context" is the repair. Codensity
is what makes the repair always available: the generativity that always supplies
a dividing term (§5a) also always supplies a finer context.

But then determinacy is a **limit** property — reached in the refinement, held
by no actual finite utterance. That is the third appearance of a structure that
keeps recurring here: always finer available, never a final one.

A caution against over-reading that recurrence. In §5a and §10 it is a precise
mathematical statement — dense and codense, dense and meagre. For contexts and
for uptakes it is so far an analogy, and the resemblance may be doing less work
than it appears to.

## What is not settled

- Whether the semantic order is countable *and* dense *and* codense *and*
  partial is assumed, not shown. Only countability has an argument behind it —
  from language being finite strings. Codensity has a *reason* (generativity,
  §5a) but not a demonstration.
- **The Baire asymmetry is a consequence nobody has examined, and §10a casts
  doubt on whether it is examinable.** If nameable meanings are dense but
  meagre, most meanings are unreachable by any expression — but comeagre and
  almost sure diverge routinely (the Liouville numbers are comeagre and null),
  so the asymmetry may be topologically true and probabilistically irrelevant.
  Nothing here decides which, and the question may not have an empirical form.
- The quantitativeness direction is untested, and untestable in this repository:
  DINOv3 has no propositions. It wants a language model and minimal pairs.
- §7a shows separation surviving to the 5% most confusable pairs at AUC 0.80,
  which is consistent with codensity but does not establish it. Codensity is a
  claim about *every* pair at *every* scale; a measurement can only ever report
  the pairs it has.
- **The semantics counterpart of §11 does not exist.** Knowledge and capability
  have the advantage form; semantics does not, and the candidate offered in §11b
  is the pattern's suggestion rather than anyone's position. Its three gaps are
  named there: direction, formation, relationship.
- **§11a bounds the whole probabilistic account.** Advantage is clean at one bit
  and needs an ordinal scale on outcomes beyond it, so "capability" as a number
  is only as well founded as the outcome scale it rests on.
- Every measurement here is against the pipeline's own labels, so it inherits
  their noise. The four-way `category` is coarse, and `trip` is a proxy for
  "occasion" that conflates place, date and outing.

A dropped question, recorded because it was wrong rather than because it was
answered: an earlier turn of this conversation posed "between two meanings, is
there always another, or always a hole?" as a fork between the ℚ picture and the
Cantor picture. It is not a fork. §5a is the resolution — both, and that
conjunction is the property.
