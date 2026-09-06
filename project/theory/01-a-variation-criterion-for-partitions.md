# 01 — A variation criterion for partitions

## What this is, and what it is not

This defines **what a good partition is**. It does not say how to find one.

That distinction is the whole point of filing it separately. A clustering
*method* — HDBSCAN, Ward, k-means — is a procedure; a *criterion* is a function
that scores a partition, and the two are independent. k-means the objective and
Lloyd's algorithm are routinely confused because they arrived together; here they
did not. What follows can be evaluated on the output of any method, including
methods that were not trying to optimise it, which is exactly how it gets used
below.

It is also not a finding. Nothing here was discovered *in* the data — the idea
arrived while thinking about why a two-level clustering behaves as it does, and
it would be equally true of a dataset of anything else. `findings/` records what
was measured about this library; this records something that would still hold if
the library were deleted. That also makes it the kind of document that belongs in
a public repository without a second thought: it means something to a stranger
with no photographs.

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

## Open

1. **No algorithm.** The obvious one is to evaluate $J_f$ at every cut of a Ward
   dendrogram and take the minimum, which is $O(k)$ evaluations over a hierarchy
   already computed. That is a heuristic for the lattice search, not a solution
   to it, and how far it falls short is unknown.
2. **Which form of $B$ to deform**, given that the anchored one carries an
   unwanted size-position term and the pairwise one charges $k$ times dispersion.
   Both are defensible; they are not the same criterion.
3. **Whether the family is richer than one parameter.** $f$ enters only through
   $g = n - f$, and the split rule depends on $g$ only through $\tau(n)$. Two
   different $f$ with the same $\tau$ are the same criterion for splitting
   purposes, so the effective dimension of the family is unclear.
4. **Behaviour under a non-partition.** Everything assumes a partition of $S$.
   Extending to a partial partition — which is what density-based methods
   actually return — needs the noise class to be *modelled*, not conventioned
   around.
