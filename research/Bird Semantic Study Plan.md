---
title: Bird Semantic Topology — Research Plan
tags: [semantop, birds, topology, embeddings, clustering, research-plan]
status: draft
created: 2026-07-16
---

# Bird Semantic — Research Plan

---

## 1. Core conceptual model (the topology this study measures)

Finite photos are a **probe** of the closure operation, not a search for a prototype. The
"violations" are the data:

- **Non-uniqueness of limit points = disconnected support = β₀ > 1.** One species with
  several plumage-modes (male/female/juvenile/breeding) accumulates onto several points. This
  is a genuine invariant (rank of H₀), stable under metric deformation. Hausdorff uniqueness
  applies *per convergent subsequence* (per mode), **not** per species.

- **Non-Cauchy / non-convergent behavior = incompleteness.** Samples "Cauchy in intent"
  (one species) that won't settle either split into modes (β₀ case) or point at a **hole** —
  a semantic distinction the embedding geometry can't represent. An incompleteness detector,
  and a finding about the *model*.

- **Failure of Hausdorff separation between two labels = non-separated species.** Confusable
  pairs (empidonax, fall warblers) are where the label-quotient fails T₂. The **non-Hausdorff
  locus of the quotient is the confusion set** — computable.

- **Persistent β₁ (a loop) = ring species / cyclic plumage cline** (*Larus* gulls,
  *Ensatina*). Invisible to any centroid or clustering; visible only to persistent homology.
  If the data ever exhibits one, it is the crown-jewel result.

**Existence / completeness / uniqueness do three different jobs** — don't fold them together:
compactness of the unit sphere (normalized embeddings) gives *existence* of an accumulation
point (Bolzano–Weierstrass, "at least one"); completeness keeps the limit *in the space*;
Hausdorff gives *uniqueness per convergent subsequence*.

**Finite-sample bridge.** The point a finite algorithm emits (centroid / medoid / density
mode) is a **consistent estimator** of a mode, not a proof of a limit. The practical stopping
signal is an **empirical Cauchy condition**: successive estimates stop moving as *n* grows.
Not *the* limit — a point past which more evidence doesn't move you.

---
## 2. The pipeline

Ordered steps, with the two-loop structure made explicit.

**0) Fix the evaluation spine (before anything).** A small held-out set of cases where you
and the expert already agree (common species, obvious splits), carried **unchanged** through
every iteration. Without a fixed yardstick, "fine-tuning improved things" is unfalsifiable
across steps 5 and 9 because the thing you'd measure against is also moving.

**1) Embed.** Choose a starting *image embedding* (see §5). Provisional by design.

**2) Cluster — DISCOVERY.** Partition + representative extraction. Existential, generative,
**recall-oriented**. Job: surface *every* candidate structure; do not interpret. Err toward
**over-segmentation** — a missed structure is invisible downstream; an extra cluster is a
cheap human merge. Emits assignment + **centroid, medoid, and density-peak** per cluster
(see §5). "How many clusters?" is *not* answered here.

**3) Cluster statistics.** Overall insight: cluster count, size distribution, noise fraction,
centroid–medoid gaps, inter-center distances. First read on whether the space is sane.

**4) Re-group into Lightroom.** Tag, **don't move** (see §5). Bring the expert in. Human
effort now scales with number of *species-modes*, not number of photos — O(clusters).

**5) Refine clustering (INNER LOOP, cheap).** On **frozen embeddings**. Retune
scale / min-cluster-size based on stats + visual inspection. Seconds-to-minutes; iterate
freely. Loop 2↔5 until the partition stabilizes.

**6) Topological analysis — ANALYSIS.** Characterizes the discoveries from 2. Precision-
oriented, interpretation lives here. Density profiles, persistence (condensed tree), β₀ per
label, boundary structure. **Allowed** to conclude "two discoveries are one thing" or "this
cluster is an artifact" — it has the statistics that step 2 lacks. This is what makes findings
*evidential* rather than self-fulfilling.

**7) Isolate doubt clusters → online-model panel.** Doubt = HDBSCAN noise set ∪ high
distance-to-centroid ∪ small-margin-between-two-centroids ∪ human-flagged. Send **only these**
(boundary of the support, ∂S) to the VLM panel as a **pairwise oracle** ("same species?"),
generating must-link / cannot-link constraints → constrained re-cluster. This is where cloud
budget is spent, and it unifies with §step-2 (clustering and adjudication become one
mechanism at the boundary).

**8) Theoretical insight.** Interpret the discoveries against §2–§3. What tore, where, which
cause. Log per-image label provenance (model / expert / external checklist) so the three
disagreement signals stay separated.

**9) Fine-tune embedding → back to 1 (OUTER LOOP, expensive).** Supervised-contrastive /
ArcFace margin objective on verified (photo → species) pairs. Changes the geometry underneath
everything; invalidates prior cluster IDs and doubt labels. **Enter only when** the inner loop
has converged *and* the residual disagreement is judged **embedding-limited** (hole/distortion
cases) rather than **clustering-limited** (boundary cases). The boundary-vs-hole distinction
tells you which loop to spend on.

**10) Exit + summarize.** See §6 — the exit criterion is the weakest link and is specified
there.

**External data (optional enrichment, any iteration).** Two distinct modes — carry provenance
as an explicit axis (see §5).

---

## 3. Design choices & detailed suggestions

### 5.1 Starting embedding (step 1) — the relocated opening question

The opening question was **relocated and demoted**: no longer "the ground-truth labeler the
architecture rests on," but "a provisional seed for step 1 the architecture is robust to." A
worse start costs extra outer-loop iterations; it cannot corrupt findings, because
encoder-specific structure is filtered by the invariance test. **So pick fast.**

**The model class changed** — do not carry the top-of-thread answer into this slot by inertia.
Step 1 needs an **image embedding / vision encoder** (metric geometry for visual identity),
**not** a generative VLM (built to emit text).

- **DINOv3 — primary discovery embedding.** Self-supervised, **taxonomy-blind**, strongest
  general instance-discrimination geometry. Clusters on *appearance*, which is exactly what a
  phenetic embedding should do. **This is the default.**
- **SigLIP 2** — image-text; stronger semantic grouping, more text-alignment artifacts. Use
  if you want language handles on clusters.
- **BioCLIP-class (bio/taxonomic encoder)** — has taxonomy *baked in*. Using it as the
  *primary* discovery space is **circular** for this study: it would agree with human taxonomy
  partly because it was told the taxonomy. Belongs on the **analysis** side as a comparison /
  invariance axis, **not** the discovery side.

> **Rule:** primary step-1 embedding must be one that *never saw the taxonomy* → DINOv3.
> Keep generative VLMs (Qwen3-VL-32B) for the **step-7 oracle** (it *names*), never step 1
> (its pooled embedding is trained for image-text alignment, risks clustering by
> pose/background).

### 5.2 Clustering algorithm (step 2)

**HDBSCAN.** Forced by what steps 6–7 need, not chosen locally:

- Density-based → matches the topological framing (rules out k-means: convex isotropic blobs,
  forces every point into a cluster, needs *k* — wrong on all three counts).
- No *k*.
- **Explicit noise set** → *is* the step-7 doubt population, for free.
- **Condensed tree** → *is* the persistence structure step 6 wants.

Fallback: agglomerative with a distance threshold, only if you need dendrogram control
HDBSCAN doesn't give.

### 5.3 Cluster "center" — emit three, not one

Centroid, medoid, and density-peak **coincide only for a clean convex isotropic blob** and
diverge exactly on the interesting clusters. Which one step 2 emits silently determines what
step 6 can see.

- **Centroid** (mean) can land in **low-density space** (a bird that doesn't exist) for
  non-convex / multi-modal clusters → biased anchor.
- **Medoid** (most central actual image) — always in-distribution. Best thing to *show the
  expert* (step 4) and to anchor distance stats.
- **Density peak / mode** — the topologically honest center for density analysis; the actual
  accumulation point.

> Emit **all three per cluster.** The **centroid–medoid gap is itself a step-6 statistic**: a
> large gap is a cheap, free **β₀ > 1 / merge-candidate flag** before you even run persistence.

### 5.4 Lightroom integration (step 4) — tag, don't move

Physically relocating files breaks the LrC catalog. Write `cluster/NN` as a **hierarchical
keyword into XMP sidecars** with `exiftool`, then *Metadata → Read Metadata from Files* in
LrC. Instant filter / group-by-cluster in the Keyword List panel, non-destructive, **survives
re-clustering** (overwrite the keyword). Fits the existing openrsync→the-nas workflow — photos
never move, only metadata changes.

### 5.5 Persistence as the instrument (step 6)

Persistent homology is the one tool that reads topological invariants off **finite** point
samples with a **stability guarantee** (bottleneck distance): finite bird photos license
claims about the arena's topology, *with error bars*.

- HDBSCAN's condensed tree **is** the 0-dimensional persistence of a density estimate.
- **β₀ over the density filtration** = polymorphism / merge structure; persistent components
  are real species, short-lived ones are pose/lighting artifacts — the **signal-vs-noise
  criterion you'd otherwise hand-tune**.
- **β₁** would flag a ring species / closed cline — no centroid could ever reveal it.

**Concrete first experiment:** plot doubt-rate against distance-to-boundary. Empirically test
whether the ambiguity really lives at ∂S (closure minus interior).

### 5.6 Clustering's limits (hold these when interpreting step 6)

A clustering is a **specific, lossy functor**, not a generic approximation of the topology:

- **Local in scale** — one clustering = one horizontal slice of the persistence diagram.
  Change ε and β₀ changes. Persistence integrates over all scales; clustering is one
  evaluation of it.
- **Local in dimension** — clustering sees **only β₀**. Structurally blind to β₁+. The ring
  is invisible to any clustering.
- **Concrete ≠ faithful** — discreteness of clusters can import a false picture of a
  *continuous* arena (clines, the ring) as disconnected. Clustering *reifies* a partition the
  topology may not have. Redeemed only by checking which features persist across **scale** and
  across **encoder**.

### 5.7 External datasets (optional, any iteration)

Two distinct operations — do not conflate:

- **Densification** (same distribution) — sharpens estimation and β₀; resolves thin necks;
  tightens the persistence stability. Clean win.
- **Extension** (new distribution) — reaches plumages/ranges/geometries your camera never
  visits; can **complete holes** and **expose β₁** (a ring only shows if you sample the whole
  ring). More valuable *and* more dangerous.

**Net-positive only if:**
- Provenance carried as an **explicit axis**; demand invariance across it. A mode that splits
  along provenance and vanishes under a provenance-invariant embedding is an **artifact**
  ("dataset look"); one that survives is real. Same test as cross-encoder invariance.
- **External labels logged as a separate taxonomic reference** (Clements / IOC / eBird
  disagree on exactly the split/lump cases you care about) — a second human axis, not merged.
- **Targeted ingestion** — pull external images for sparse / boundary / non-Hausdorff clusters
  only. Online data is presence- and popularity-biased; naive pooling *worsens* density where
  you're weakest and floods where you're already strong.

### 5.8 Two nested loops — keep them separate

- **Inner (2↔5):** frozen embeddings, cheap, iterate freely. Fixes **boundary** cases.
- **Outer (through 9):** re-fine the embedding, expensive, changes the geometry. Fixes
  **hole/distortion** cases.
- **Diagnostic for which loop:** is the residual doubt a *boundary* case (inner) or a
  *hole/distortion* case (outer)? — the connectedness-vs-completeness distinction doing
  operational work.

### 5.9 Version the space (critical for step 9)

Every embedding refinement moves the manifold → β₀ / persistence / boundary stats in
embedding-vₖ are **not comparable** to vₖ₊₁. Without versioning, "the topology is stabilizing"
is uninterpretable (can't tell topology-change from coordinate-change).

> **Snapshot per outer iteration:** (embedding weights, cluster assignments, persistence
> diagram, doubt set). Compare **topological invariants** across versions — bottleneck distance
> between persistence diagrams, ARI / AMI between assignments — **never raw distances.** This
> is what makes the study *evidential*: "β₀ for species X held at 2 across three independent
> refinements" is a finding; "the clusters looked stable" is not. Tag each iteration; keep
> artifacts on the-nas; diff the invariants.

---

## 4. Exit criterion (step 10) — the weakest link, specified

Three candidate criteria answering different questions:

1. **Metric convergence** — doubt set stops shrinking; persistence diagram stops moving
   (bottleneck distance between successive iterations < ε). The empirical-Cauchy criterion,
   now at the level of the whole topology. Clean, checkable, natural default.
2. **Irreducibility** — remaining doubt survives *both* loops *and* the online panel *and*
   cross-embedding invariance. These are the **findings** (genuine non-Hausdorff loci, real
   holes, phenetic/phylogenetic tears). The loop exits **reporting** these, not eliminating
   them.
3. **Diminishing theoretical yield** — step 8 stops producing new insight per iteration.
   Softer, human-judged; the honest one for a research (not production) goal.

> **The trap:** do **not** treat a shrinking doubt set as the goal. The residue *is the object
> of study*. Driving doubt → 0 means metric-learning the interesting topology *away* (collapse
> a real ring into a blob; brute-force-separate a genuinely non-Hausdorff pair). Optimizing for
> a small residue and optimizing for *understanding* the residue pull in opposite directions.

**Exit = (metric convergence) AND (residual doubt characterized as irreducible).** Stop when
the topology stops moving *and* the leftover is classified as real structure rather than
unfinished business.

---

## 5. First concrete brick (unblocks everything)

The one thing the whole plan rests on is whether the embedding is even in the right space —
the cheapest thing to check:

```
sample N images
  → DINOv3 embed
  → HDBSCAN
  → dump per cluster: medoid + centroid + density-peak
  → dump: noise set + condensed-tree persistence (β₀ over filtration)
  → eyeball: do clusters track species (not pose/background)?
```

If yes → build the loop. If the clusters track pose/background → the embedding is wrong for
discovery, swap before investing further. This single script answers the one question the
architecture cannot proceed without.

---

## 6. Component summary

| Role                                          | Choice                                                | Why                                                            |
| --------------------------------------------- | ----------------------------------------------------- | -------------------------------------------------------------- |
| Primary embedding (discovery)                 | **DINOv3**                                            | Self-supervised, taxonomy-blind, appearance geometry           |
| Secondary embeddings (invariance)             | SigLIP 2, BioCLIP-class                               | Cross-encoder invariance test; BioCLIP on analysis side only   |
| Clustering                                    | **HDBSCAN**                                           | Density-based; noise set = doubt; condensed tree = persistence |
| Cluster center                                | Centroid + medoid + density-peak                      | Gap = free β₀>1 flag; medoid = safe anchor                     |
| Visualization                                 | Lightroom + exiftool XMP keywords                     | Non-destructive, survives re-clustering                        |
| Topology instrument                           | Persistent homology                                   | Invariants off finite samples with stability guarantee         |
| Boundary oracle (step 7)                      | Qwen3-VL-32B + cloud panel (Gemini 3.1 Pro / GPT-5.6) | Pairwise "same species?" on ∂S only                            |
| Third reference (needed for cause (2) vs (3)) | Molecular tree / checklist authority                  | Separates real divergence from unsettled taxonomy              |
