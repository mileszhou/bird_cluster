# 01 — A variation criterion for partitions

## Abstract

Huygens' theorem makes the total variation $T = W + B$ a constant of the data:
every partition spends the same budget, and the two extremes — everything in one
cluster, everything alone — spend all of it in opposite halves. No undeformed
combination of the within and between terms can tell them apart, which is why
minimising $W$ needs $k$ fixed from outside.

Discounting the *between* term by a weight $f(n) \le n$ with $f(1) = 1$ breaks
the symmetry. The resulting $J_f = W + \sum_i f(n_i)\|\mu_i - \mu\|^2$ is at or
below $T$ everywhere and equal to it at **both** extremes, for two different
reasons — the trivial partition has no between-variation to discount, the
discrete one earns no discount — so the minimiser is interior. A partition is
paid only for structure it has captured in groups larger than one.

What follows from that: no $k$, no `min_cluster_size`, no noise class, no
thresholds of any kind — cluster count, cluster sizes and what is left alone are
all consequences of one inequality. The move differences close in a per-point
*credibility* coordinate $\gamma(n) = 1 - f(n)/n$, where merging reduces to
Ward's own rule plus a bonus for consolidation. And there is a canonical
one-parameter family whose parameter is a within/between variance ratio, so it
could be estimated rather than chosen; $f \equiv 1$ is the point where the two
are equal.

Open: no bound on how far a descent falls short of the lattice optimum, and two
structural choices — which form of $B$ to deform, and how to propose a split.

## Setup

$S \subset \mathbb{R}^m$ finite, $|S| = N$, grand mean $\mu$. A partition
$\mathcal P = \{C_1,\dots,C_k\}$ with $|C_i| = n_i$, centroid $\mu_i$.

$$W(\mathcal P) = \sum_i \sum_{x\in C_i}\|x-\mu_i\|^2, \qquad
  B(\mathcal P) = \sum_i n_i\|\mu_i-\mu\|^2, \qquad
  T = \sum_{x\in S}\|x-\mu\|^2$$

## The invariant it is built on

$$\boxed{\;T \;=\; W(\mathcal P) \;+\; B(\mathcal P)\quad\text{for every }\mathcal P\;}$$

Huygens' theorem. The total variation is a constant of the **data**, not of the
partition; what a partition does is decide how that fixed budget is *split*
between inner and inter. The two extremes sit at opposite ends of the same
budget — the trivial partition ($k=1$) puts all of $T$ in $W$, the discrete
partition ($k=N$) puts all of it in $B$ — and neither is distinguishable from the
other by any function of the total.

This is why no unmodified combination of $W$ and $B$ can select a partition. It
is also the hook: deform one side and the constant becomes a scale to measure
against.

## The deformation

Call $f:\mathbb{N}\to\mathbb{R}_{\ge0}$ **admissible** when

$$f(n) \le n \quad\text{for all } n, \qquad f(1) = 1$$

and define the **modified total**

$$J_f(\mathcal P) \;=\; W(\mathcal P) \;+\; \sum_i f(n_i)\,\|\mu_i-\mu\|^2
  \;=\; T \;-\; R_f(\mathcal P)$$

$$R_f(\mathcal P) \;=\; \sum_i \bigl(n_i - f(n_i)\bigr)\,\|\mu_i-\mu\|^2 \;\ge\; 0$$

The inter term keeps its structure and loses weight. A partition is **rewarded**
for between-variation, but only in proportion to how much of it sits in groups
larger than a single point. Minimise $J_f$; equivalently, maximise $R_f$.

### Proposition 1 (both extremes score exactly $T$)

$J_f(\mathcal P) \le T$ for every $\mathcal P$, with equality iff every $C_i$
satisfies $f(n_i) = n_i$ or $\mu_i = \mu$. In particular $J_f = T$ at both
extremes, for two different reasons:

- **Trivial**: $k=1$ and $\mu_1 = \mu$, so there is no inter-variation to
  discount. The discount is real but there is nothing to spend it on.
- **Discrete**: $n_i = 1$ and $f(1)=1$, so the discount is zero. There is
  everything to spend it on and no discount to spend.

*Proof.* $T - J_f = R_f = \sum_i (n_i-f(n_i))\|\mu_i-\mu\|^2$, a sum of
non-negative terms by admissibility. ∎

The two conditions in the definition are therefore not stylistic. $f(n)\le n$
stops the trivial partition being beaten by nothing; $f(1)=1$ stops the discrete
partition winning outright. Relax the second to $f(1)<1$ and the discrete
partition scores $(1-f(1))\,T$ below the maximum, which for small $f(1)$ beats
any interior partition. **The normalisation is a knife-edge, not a range.**

Because both extremes are pinned at the same value and the interior is strictly
below it, the minimiser is interior whenever any two-point subset has a centroid
away from $\mu$ — which is to say, always, for data that is not a single point.

### Proposition 2 (it is exact on separated data)

Let $S$ consist of two point masses of $n/2$ each, at distance $d$. Then

$$J_f(\text{natural}) \;=\; f(n/2)\,\frac{d^2}{2}, \qquad T \;=\; \frac{n d^2}{4},
  \qquad \frac{J_f}{T} \;=\; \frac{2 f(n/2)}{n}$$

and for $f\equiv 1$ this is exactly $2/n$. The natural partition beats both
extremes iff $f(n/2) < n/2$ — that is, **iff $n > 2$**, for every admissible $f$
that is strictly sub-diagonal above 1. The margin is a factor of $n/2$, so it
grows with the data.

$n>2$ is an exact threshold, not an asymptotic one, and it is the smallest
possible: two points are one point each, and a partition of two singletons *is*
the discrete partition.

More generally, for $k$ point masses the natural partition is the minimiser once
the multiplicities are large enough that the merge cost $\frac{n_in_j}{n_i+n_j}
\|p_i-p_j\|^2$ — which grows linearly in the multiplicities — exceeds the
displacement saved by merging, which is $O(1)$ in them.

### The split rule, and what $f$ actually controls

Split a cluster of size $n$ at displacement $d^2 = \|\mu_C - \mu\|^2$ into halves
whose centroids separate by $\delta$. Using Ward's identity for the drop in $W$:

$$\Delta J_f \;=\; -\frac{n}{4}\|\delta\|^2 \;+\; f(n/2)\Bigl(2d^2 + \tfrac12\|\delta\|^2\Bigr) \;-\; f(n)\,d^2$$

$$\text{split} \iff \frac{\|\delta\|^2}{d^2} \;>\; \frac{4\bigl(2f(n/2)-f(n)\bigr)}{\,n - 2f(n/2)\,}$$

For $f\equiv1$ this is $\|\delta\|^2/d^2 > 4/(n-2)$, i.e. $n - 2 > 4d^2/\|\delta\|^2$.

Read it as: **separation must beat displacement by a margin that shrinks as the
cluster grows.** On perfectly separated data $\delta = 0$ within a true cluster
and nothing splits at any $n$ — Proposition 2 is this rule at its clean limit.
Under perturbation the rule trades the real structure $\|\delta\|$ against the
price of one more representative.

The equivalent scale-free form is $\tau(n) = g(n)/2g(n/2)$ with $g = n - f$, the
factor by which mean displacement must rise to justify a split:

| $n$ | $f\equiv1$ | $f=\sqrt n$ | ratio of excess over 1 |
|---:|---:|---:|---:|
| 10 | 1.125 | 1.237 | 1.9× |
| 50 | 1.021 | 1.073 | 3.5× |
| 100 | 1.010 | 1.048 | 4.7× |
| 500 | 1.002 | 1.020 | 9.9× |

Both tend to 1 — provably, since $\tau$ bounded below by $1+\epsilon$ would force
$g$ superlinear while $g \le n$ by construction. **So no admissible $f$ gives a
splitting threshold that survives to infinity.** But the *excess* over 1, which is
what decides an actual split, differs by two to ten times across the range real
clusters occupy, and that is the range that matters. The asymptotic statement is
true and irrelevant; $f$ is a genuine resolution control at working scale, and a
coarser $f$ (larger $f$, smaller discount) selects coarser partitions.

## Two forms of $B$, and why the choice is not free

$$\sum_i n_i\|\mu_i-\mu\|^2 \;=\; \frac{1}{2N}\sum_{i,j} n_i n_j \|\mu_i-\mu_j\|^2$$

The two expressions are **equal**. Their deformations are **not**. With unit
weights and $\bar\mu = \frac1k\sum_i\mu_i$, $S_\ast = \sum_i\|\mu_i-\bar\mu\|^2$:

$$\underbrace{\sum_i\|\mu_i-\mu\|^2}_{\text{anchored}} = S_\ast + k\|\bar\mu-\mu\|^2,
  \qquad
  \underbrace{\frac{1}{2N}\sum_{i,j}\|\mu_i-\mu_j\|^2}_{\text{pairwise}} = \frac{k}{N}\,S_\ast$$

Neither is centre-free. The identity holds *only* when $\mu$ is the mass centre —
for any other $c$, the parallel-axis theorem gives $\sum_i n_i\|\mu_i-c\|^2 =
\sum_i n_i\|\mu_i-\mu\|^2 + N\|c-\mu\|^2$, so the anchored form is minimised at
$c=\mu$ and the pairwise form is that minimum. The centre is implied, not absent.

What the choice does change:

- The pairwise form multiplies the dispersion by $k$. That is what stops a
  cluster sitting at the centre from being free — under the anchored form a
  representative placed at $\mu$ costs nothing at all.
- The anchored form carries a term $k\|\bar\mu-\mu\|^2$, which is non-zero
  exactly when cluster size correlates with cluster position. It therefore
  quietly prefers partitions whose large clusters sit near the centre of mass.
  Nobody asked for that; it is a side effect of which identity was deformed.

## What the criterion does not contain

**Within-cluster scatter does not affect the ranking.** Since $T = W+B$
identically, $J_f = T - R_f$ and $R_f$ depends only on centroids and sizes. The
inner term is real in the accounting and cancels from every comparison. Two
partitions with the same centroids and sizes score identically however tight or
diffuse their clusters are. This is not a defect introduced by a bad choice of
$f$ — the invariant forces it, for any deformation applied only to $B$.

That is less blind than it sounds, because a partition that groups incoherently
produces centroids near $\mu$ and is scored at zero. But it is blind to
compactness at fixed centroid configuration.

**There is no algorithm here.** Minimising $J_f$ over all partitions is a search
over the partition lattice. Everything above says what the answer is worth, not
how to reach it.

## Noise is not a primitive

The criterion needs no noise class, no minimum cluster size and no outlier
threshold, and this is not an omission to be repaired later. **A point left out is
a cluster of size one**, which is an ordinary member of the partition lattice, and
$g(1)=0$ means it earns nothing — so excluding a point is never free. There is
nothing to declare because there is nothing to exempt.

Proposition 1 says the same thing from the other side: the two extremes are
*maximisers* of $J_f$, which is to say the worst partitions available. Any grouping
at all does better, and the only question is which one does best.

(Precisely: they are tied with the degenerate partitions whose every non-singleton
cluster is centred at $\mu$. That tie set is not an accident and it reappears
below.)

With the descent this stops being a remark and becomes the operation. Absorbing a
lone point into a cluster is **`merge_delta` with $n_a = 1$** — the identical
formula, verified against recomputation, with no special case anywhere. A noise
point is a cluster of one, and the merge rule already decides its fate.

At $f \equiv 1$ the rule reads

$$\text{absorb} \iff \frac{\|u_b\|^2}{n_b} + \|v\|^2 \;>\; \frac{n_b}{n_b+1}\,\|u_b - v\|^2$$

and for large $n_b$ that is simply $\|v\| > \|u_b - v\|$: **absorb the point iff it
is further from the grand mean than from the cluster centroid.** The boundary is the
perpendicular bisector of $\mu$ and $\mu_b$, which a numerical sweep confirms exactly.

### And the residual class comes out inverted

That rule places the *outliers inside clusters* and leaves the *central points
alone* — the opposite population from what a density-based method calls noise.

| lone point at | distance to $\mu$ | to centroid | verdict |
|---|---:|---:|---|
| far beyond the cluster | 5.00 | 4.00 | absorb |
| at the centroid | 1.00 | 0.00 | absorb |
| on the bisector | 0.50 | 0.50 | absorb |
| at the grand mean | 0.00 | 1.00 | **alone** |
| behind the grand mean | 1.00 | 2.00 | **alone** |

It is consistent — a far-out point carries displacement mass and is worth
crediting to a group; a point at $\mu$ carries none and is worth nothing to
anybody — but it is not what "noise" usually means, and anyone reading the output
of a descent on this criterion needs to know which population is being set aside.

It is also the **third** appearance of one defect. A singleton at $\mu$ is free
because the quadratic vanishes at its anchor; a cluster at $\mu$ is free for the
same reason; and the worst-score tie set above is exactly the partitions built out
of such clusters. All three are the anchored form of $B$, and all three would be
charged by the pairwise form, where $k$ multiplies the dispersion. Which form to
deform is not a presentational choice, and this is the first place where it changes
what the answer *looks like* rather than only what it scores.

## Nothing to tune

The list of things this does *not* ask anyone to choose is most of the list a
clustering usually has: no $k$, no `min_cluster_size`, no `max_leaves`, no
`min_samples`, no distance cutoff, no outlier threshold. Every one of them is a
*consequence* here — cluster count, cluster sizes and what gets left alone all
fall out of the same inequality. There is no boundary condition because there is
no boundary.

What remains is $f$, and it is worth being exact about what kind of object that
is. It maps an integer in $[1,N]$ to a real number. It has no units, no dependence
on the dimension, the scale of the data, or $N$; `min_cluster_size = 15` means
something different in 1,000 points than in 100,000, and $f$ does not. Its
normalisation is **fixed by the theory** rather than left open — $f(1)=1$ and
$f(n)\le n$ are forced by Proposition 1, where the $\lambda$ in $W + \lambda k$
is a free scalar with no anchor at all. And it is a modelling belief about how
credible a group's mean is at a given size, not a threshold fitted against the
data it will then be used to describe.

### A canonical family, whose parameter is estimable

The credibility coordinate suggests where to look. Credibility theory shrinks a
group mean by $\gamma(n) = n/(n+\kappa)$ with $\kappa$ the ratio of within-group
to between-group variance — and that form is **inadmissible** here: it gives
$f(1)=\kappa/(1+\kappa) < 1$, a singleton with non-zero credibility, and the
discrete partition then wins outright.

Replacing $n$ by the degrees of freedom $n-1$ lands exactly:

$$\gamma_\kappa(n) = \frac{n-1}{n-1+\kappa}, \qquad
  f_\kappa(n) = \frac{\kappa\,n}{n-1+\kappa}, \qquad \kappa > 0$$

This is admissible for every $\kappa>0$ — $f_\kappa(1)=1$ and $f_\kappa(n)\le n$
identically, with $\gamma$ increasing throughout — and

$$\kappa = 1 \quad\text{is exactly}\quad f \equiv 1$$

so the "each cluster counts once" weight is not an arbitrary pick but the point
where within-group and between-group variance are equal. Larger $\kappa$ shrinks
harder, rewards less, and selects coarser.

The $n-1$ is the same degrees-of-freedom correction that $\sum_i (n_i-1) = N-k$
already produced, which is the within-groups df of the ANOVA table this whole
construction started from. Two routes to the same correction is mild evidence it
is the right one.

**And $\kappa$ is a variance ratio, so it can be estimated rather than chosen.**
$\kappa = MS_{\text{within}}/MS_{\text{between}} = 1/F$, the reciprocal of the
ANOVA F-statistic of the partition. That makes the criterion parameter-free in the
strong sense — nothing is set by hand — at the cost of a fixed point, since $F$
depends on the partition being scored. Whether that iteration converges, and to
what, is not known and is the most interesting thing left open here.

### What is still a choice

Three things, and none of them is a tuning knob:

- **The split proposer.** Relocate and merge are exhaustive; split is not, and
  something has to nominate the bisection.
- **Anchored or pairwise $B$.** A structural choice with visible consequences
  (see the residual class above), not a parameter.
- **The metric.** Huygens is an identity of *squared Euclidean* distance; the
  whole construction rests on it and does not transfer to an arbitrary
  dissimilarity. It does apply to L2-normalised embeddings, where squared
  Euclidean and cosine are affinely related — which is this project's case, and
  the reason any of it is usable here at all.

## Where it sits among known methods

| | relation |
|---|---|
| **k-means / $W$** | Minimising $W$ alone is degenerate (discrete partition, $W=0$); k-means escapes by *fixing* $k$. This escapes by deforming $B$ instead, and gets $k$ out rather than putting it in. |
| **Ward** | Ward greedily minimises the increase in $W$ — it is exactly a descent on the inner term, and its merge cost $\frac{n_an_b}{n_a+n_b}\|\mu_a-\mu_b\|^2$ is the quantity in the split rule above. But it has no stopping condition: it runs to $k=1$. $J_f$ is precisely the missing stopping rule. |
| **`cluster2.py` here** | Runs *unweighted* Ward over leaf medoids, so every leaf counts once regardless of size — which is $f\equiv1$, appearing in the inner slot rather than as a discount on the inter term. The cut is `--max-leaves 27`, a display constraint. So the pipeline implements half of this by accident and none of it on purpose. |
| **HDBSCAN** | A different family: density-based, no global functional, and it returns a *partial* partition — 15% of the population unassigned at `mcs3`, 44% at `mcs40`. $J_f$ is defined over a partition of all of $S$, so applying one to the other requires a convention (below). |
| **$W + \lambda k$** (DP-means, BIC, MDL) | The nearest relative, and the contrast is sharp: there the price of a cluster is a **constant** $\lambda$; here it is $f(n_i)\|\mu_i-\mu\|^2$ — a price that depends on how large the cluster is and where it sits. And $\lambda$ must be chosen from outside, whereas $f$'s normalisation is fixed by Proposition 1. |

I have not identified this exact form in the literature, which is a statement
about how hard I looked rather than about the literature.

## Using it as a criterion, on partitions that were not optimising it

`tools/audit_partition_reward.py` scores a partition and ranks a sweep.

```bash
python3 -m tools.audit_partition_reward --arm <label> <cluster-dir> --check
```

`--check` verifies the two identities on the arm's own vectors before any
comparison: Huygens exact, and $J_f/T = 1$ at both extremes.

**Noise is a modelling decision, not an implementation detail.** HDBSCAN declines
points and $J_f$ wants all of $S$, so three conventions are reported and none is
a default worth trusting alone. `singleton` (each declined point its own cluster)
keeps one population across a sweep and is therefore comparable — but it charges
every declined point, so it largely measures *how many* a setting refused.
`exclude` scores each setting over its own population, so $T$ differs per row.
`common` fixes the population to what every setting assigns, at the cost of
truncating clusters so the $n_i$ are not the ones the method chose.

On this project's two clusterings across five `min_cluster_size` values, the
criterion selects the **finest** partition in every `unit` and every pairwise
column, under all three conventions. Under `singleton` it wins by a wide margin,
but that convention is partly scoring the refusal rate; under the other two the
criterion is nearly flat and the margin is under 1% relative. Switching to
$f=\sqrt n$ moves the selection **coarser** in four of six cases and never finer,
which is the split rule's sign showing up in data.

Family-purity lift against an external checklist prefers coarser still, and
$f=\sqrt n$ moves toward it — the clearest evidence so far that $f$ is a real knob
rather than a free parameter. Numbers are in the run artifacts; they are
illustration, not the content of this document.

$J_f/T$ is dimensionless but **not comparable across embeddings**. It is a ratio
within one space, and two spaces at different concentrations will produce
different ratios for reasons that have nothing to do with partition quality —
the same trap that cost `findings/04` its proposed mechanism.

## A descent formulation

Because $T$ is fixed, minimising $J_f$ is maximising

$$R_f(\mathcal P) = \sum_i g(n_i)\,\|u_i\|^2, \qquad g = n - f, \qquad u_i = \mu_i - \mu$$

and **$R_f$ is a function of the centroids and the counts alone** — the points
themselves never appear, and $W$ never has to be computed. Working in displacement
coordinates $u_i$ (the grand mean is the natural origin of the whole criterion, and
it does not move when points are reassigned), each primitive move has an exact
closed form costing $O(d)$ regardless of cluster size.

The differences are *finite* differences — the lattice is discrete, so there is no
differential — but the objective is quadratic in the centroids and the counts enter
only through $g$, so they close in a form simpler than the definition.

The right coordinate is the **per-point credibility**

$$\gamma(n) \;=\; \frac{g(n)}{n} \;=\; 1 - \frac{f(n)}{n} \;\in\; [0,1)$$

with $\gamma(1) = 0$ by admissibility, and $\gamma$ non-decreasing whenever $f$ is
concave (then $f(n)/n$ decreases). It is the weight an empirical-Bayes argument
would put on a group mean, reached here from the opposite direction: a small
cluster's displacement is not credible, so it is not paid for.

**Merge** $C_a$ and $C_b$, $n = n_a + n_b$. Substituting Ward's identity
$n_a\|u_a\|^2 + n_b\|u_b\|^2 - n\|u_c\|^2 = \frac{n_an_b}{n}\|\mu_a-\mu_b\|^2$
collapses the definition to

$$\Delta R_f \;=\; \underbrace{\sum_{i\in\{a,b\}} n_i\bigl(\gamma(n)-\gamma(n_i)\bigr)\|u_i\|^2}_{\text{credibility gained}} \;-\; \underbrace{\gamma(n)\,\frac{n_an_b}{n}\|\mu_a-\mu_b\|^2}_{\gamma(n)\,\times\,\text{Ward's merge cost}}$$

**Merge iff the credibility gained exceeds $\gamma(n)$ times what Ward charges.**
Both sides are non-negative — merging always raises each part's per-point
credibility, and always pays Ward's cost — so the criterion is a clean trade-off
with one term of each sign, and it is Ward's own rule with a bonus for
consolidation.

For $f \equiv 1$, where $\gamma(n) = 1 - 1/n$, it reduces to

$$\text{merge} \iff \frac{\|u_a\|^2}{n_a} + \frac{\|u_b\|^2}{n_b} \;>\; \frac{n-1}{n}\,\|\mu_a-\mu_b\|^2$$

— displacement *per point* against squared separation, with nothing else in it.
On two point masses this gives equality at $n=2$ and refuses to merge for every
$n>2$, which is Proposition 2 recovered along an entirely different route.

**Relocate** $y$ from $C_a$ to $C_b$ ($n_a \ge 2$), with $v = y-\mu$, so that
$\|u_i - v\|^2$ is the point's squared distance to centroid $i$ and no absolute
coordinate appears:

$$\Delta R_f = n_a\bigl(\gamma(n_a{-}1)-\gamma(n_a)\bigr)\|u_a\|^2
             + n_b\bigl(\gamma(n_b{+}1)-\gamma(n_b)\bigr)\|u_b\|^2
             + \bigl(\gamma(n_b{+}1)-\gamma(n_a{-}1)\bigr)\|v\|^2$$
$$\qquad\qquad + \;\gamma(n_a{-}1)\,\frac{n_a}{n_a-1}\|u_a-v\|^2
             \;-\; \gamma(n_b{+}1)\,\frac{n_b}{n_b+1}\|u_b-v\|^2$$

The last two terms are **Hartigan's classic test** — does the move reduce $W$ —
with each side weighted by the credibility of the cluster it lands in. The first
three are the count effects, which Hartigan has no equivalent of: $a$ shrinks and
loses credibility, $b$ grows and gains it, and the point's own displacement is
re-credited at the destination's rate instead of the source's.

**Split** $C_a$ into $A, B$ is the one move needing the points, since a bisection
must be proposed before it can be scored:

$$\Delta R_f = g(n_A)\|u_A\|^2 + g(n_B)\|u_B\|^2 - g(n_a)\|u_a\|^2$$

All three are implemented as `merge_delta`, `relocate_delta` and `credibility` in
`tools/audit_partition_reward.py`, and each is tested against recomputing $R_f$
from scratch over random partitions at three choices of $f$ — which is what keeps
the formulae above honest rather than merely written down.

Accept any move with $\Delta R_f > 0$. The objective strictly increases and the
partition lattice is finite, so the search terminates at a local maximum. This is
Hartigan's exact-incremental scheme rather than Lloyd's alternating one — Lloyd
does not apply here, because a nearest-centroid assignment does not account for
the $g(n_i)$ weights changing as the counts change.

Three things follow that are worth stating before anyone implements it.

**$k$ is free, but not equally free in both directions.** Relocation deletes a
cluster when its last point leaves, so $k$ falls on its own. It cannot rise: to
split, the search must pass through a singleton, and $g(1)=0$ makes that move
strictly losing. Growth in $k$ therefore requires an explicit split move — which
is the same structure as X-means or G-means, and the same reason those exist.

**The Ward hierarchy gives the whole path in one sweep.** A dendrogram is a
sequence of $N-1$ merges, consecutive cuts differ by exactly one, and the merge
delta above is $O(d)$. So evaluating $R_f$ at *every* cut costs $O(Nd)$ total, and
the argmax is a principled cut where `cluster2.py` currently has `--max-leaves 27`.
That is the cheapest useful thing here and it needs no new clustering — only the
linkage matrix, which `linkage()` already returns and the pipeline currently
discards.

**Two of the three moves are a method; the third is a proposal problem.**
Relocate and merge are closed-form and exhaustive — every candidate can be scored.
Split cannot be, because "split $C_a$" is not one move but $2^{n_a-1}-1$ of them,
so a bisection has to be *proposed* before the exact delta can score it. That is
where an outside heuristic enters (2-means, a principal direction, the existing
dendrogram), and it is the only place it does. So what the differences buy is a
method modulo a split proposer, which is a smaller gap than "no method" and a
larger one than none.

**A descent is a heuristic, not a solver.** k-means is NP-hard even with $k$ fixed,
in the plane for general $k$ and in general dimension for $k=2$; there is no reason
to expect this to be easier. Local maxima are real, the initialisation matters, and
the dendrogram sweep is valuable mainly as a good starting point rather than as an
answer.

## Open

1. **How far the descent falls short.** The moves above are exact and the
   dendrogram sweep is cheap, but nothing bounds the gap between a local maximum
   and the lattice optimum. On synthetic data with a known answer that gap is
   directly measurable, and it has not been measured.
2. **Which form of $B$ to deform**, given that the anchored one carries an
   unwanted size-position term and the pairwise one charges $k$ times dispersion.
   Both are defensible; they are not the same criterion.
3. **Whether the family is richer than one parameter.** $f$ enters only through
   $g = n - f$, and the split rule depends on $g$ only through $\tau(n)$, so two
   different $f$ with the same $\tau$ are the same criterion for splitting. The
   $\kappa$-family above is one principled curve through the admissible set; how
   much of that set it misses, and whether anything outside it behaves
   differently, is unknown.
4. **Whether the inverted residual class is a defect or a result.** The
   criterion sets aside central points rather than outlying ones (above). Under
   the pairwise form of $B$ it would not, and which behaviour is wanted is a
   question about the problem, not about the criterion. Note this is *not* an
   open question about noise in general: the three conventions in
   `audit_partition_reward.py` exist only to score partitions handed over by a
   method that declined some points, and are an interface to foreign output
   rather than a gap in the theory.

## What this is, and what it is not

This defines **what a good partition is**. It does not say how to find one.

That distinction is the whole point of filing it separately. A clustering
*method* — HDBSCAN, Ward, k-means — is a procedure; a *criterion* is a function
that scores a partition, and the two are independent. k-means the objective and
Lloyd's algorithm are routinely confused because they arrived together; here they
did not. Everything above can be evaluated on the output of any method,
including methods that were not trying to optimise it, which is exactly how it
was used on the `min_cluster_size` sweep.

**That separation lasted about an hour.** Once the move differences were worked
out they turned out cheap and exact, and a criterion with cheap exact move
differences *is* a method: accept any move that improves it. The gap was never
going to be wide, and for a reason visible in hindsight — what makes $R_f$ cheap
to *evaluate* (it depends on centroids and counts alone) is exactly what makes it
cheap to *difference*. Lloyd falls out of k-means, Louvain out of modularity,
Hartigan out of $W$, by the same route. A criterion whose objective decomposes
over local moves is a method that has not been written down yet.

Two things survive the collapse, and they are why this is still filed apart from
a method.

A criterion scores partitions produced by **anything**, including procedures that
were not optimising it and could not have been — which is how it was used on
HDBSCAN's output. No method can do that; a method returns a partition, it
does not evaluate one. And a descent reaches a *local* optimum, so "what is
optimal" and "what the search returns" remain different objects with a gap nobody
has bounded. What has to be withdrawn is only the implication that the distance
between criterion and method was large.

It is also not a finding. Nothing here was discovered *in* the data — the idea
arrived while thinking about why a two-level clustering behaves as it does, and
it would be equally true of a dataset of anything else. `findings/` records what
was measured about this library; this records something that would still hold if
the library were deleted. That also makes it the kind of document that belongs in
a public repository without a second thought: it means something to a stranger
with no photographs.
