# 03 — Seriation as a way to discover clusters

*Raised 2026-08-11 while looking for a picture of the embedding. Explored
2026-08-11 to -14. **Rejected for discovery, kept for presentation.***

*Why it could not have worked is in `findings/02` §9: the structure is a partial
order, and a partial order has no linear extension preserving nearness. The
symptom was found here in August; the reason was written down on 2026-08-21.*

## The idea

Order the images so that similar ones are adjacent, and the clusters should
appear as blocks on the diagonal of the similarity matrix. If that worked, the
ordering would be doing the clustering — no `min_cluster_size`, no density
threshold, just a permutation and whatever structure fell out of it.

The appeal was that it needs no parameter. Every clustering result in this
project is conditional on a setting nobody can justify from first principles;
a permutation has nothing to tune.

Atkins, Boman and Hendrickson (1998), *A spectral algorithm for seriation and
the consecutive ones problem*, makes it concrete and cheap: build the Laplacian
`L = D − W`, take the eigenvector of its second-smallest eigenvalue, sort by it.
Their theorem is exact — if a permutation exists that makes the matrix Robinson
(entries decreasing away from the diagonal), sorting by the Fiedler vector finds
one.

## Why it does not discover clusters

**One eigenvector carries one bipartition.** That is the whole of it. Sorting by
a single scalar per item can split a set in two; it cannot express which of *k*
groups each item belongs to. Recovering *k* groups needs *k* eigenvectors, which
is spectral clustering — a different algorithm that happens to share the
Laplacian.

Measured, on synthetic Gaussian blobs 40σ apart, which is far enough that
nothing could plausibly interleave. 60 points per blob, so a perfect result is a
mean run of 60:

    k=2    mean run 60.0   perfect
    k=8    mean run 60.0   perfect
    k=16   mean run 13.3
    k=32   mean run  6.5
    k=146  mean run  2.6

It works, exactly and reliably, up to about 8 clusters. By 16 it is gone. **More
separation does not help** — 20σ and 60σ give the same collapse. Nor does the
choice of how to make the similarity non-negative: clipping at zero (what ABH
assume) and adding a constant (which preserves the eigenvectors and keeps the
matrix low-rank) both fail at the same place, neither consistently better.

On the real data, with 146 clusters, the ordering crosses a cluster boundary at
**91.7%** of adjacent pairs — against 98.6% for a random order and 1.7% if the
clusters were contiguous. Barely better than shuffling.

**The theorem is not violated; it is about something else.** A Robinson matrix
is a *chain* — one connected gradient, similarity falling off monotonically as
you move away. A set of separated blobs is not a chain, and its Laplacian has
*k−1* small eigenvalues where a chain has one well-separated λ₂. There is no
unique Fiedler vector to sort by.

The wrong explanation, recorded because it is tempting: *spectral degeneracy*.
It is real but not the cause — the k=2..8 cases are degenerate too and still
come out perfect. The limit is capacity, not conditioning.

## What it is good for

**Presentation, and it earns its place there.** Seriating the 146 cluster
*centroids* asks for a 1-D arrangement of 146 points along a gradient, which is
a chain problem and which it does well: band enrichment 1.78× against 1.02× for
the size ordering the plots used before, and a 1.93× ceiling for a matrix that
is one-dimensional by construction.

That ordering is what makes `plot_matrix` and `plot_adjacency` readable. The
mcs40 over-merge was a dim smear until the centroids were ordered; with them,
86 of the 145 largest adjacency steps land on real cluster boundaries — 59%
against 1.7% by chance.

So: a display order, applied to groups that clustering has already found. Not a
way of finding them.

## What it cost, and what else came out

Three days, and three things worth keeping beyond the negative result.

**A bug fix.** `seriate()` was using the normalised Laplacian; ABH's theorem is
for the unnormalised one. On a synthetic Robinson matrix the unnormalised form
recovers the true order exactly at every n from 20 to 500 while the normalised
one fails from 50 up. Worth 1.67× → 1.78× on the real centroids.

**A cheap full-scale implementation.** `W = XX^T` is rank 768, so `Wv = X(Xᵀv)`
costs O(nd) and the n×n matrix is never formed — the Fiedler vector for all
27,194 images takes 3 s against a 5.5 GiB matrix that would not fit anywhere
sensible. Iterations are flat in n (39 at n=1,000, 56 at n=27,194); the √n bound
is worst-case, for graphs with a tiny spectral gap.

**A fact about the library, not about seriation.** Image-level structure is not
one-dimensional, now confirmed three independent ways: 1.34× band enrichment at
image level against 1.78× at cluster level; the ±25% band nearly at the 1-D
ceiling while ±5% is not; and this contiguity result. The coarse arrangement of
the collection is a line. The fine structure is not, and no ordering will make
it so.

## If anyone revisits this

The honest version of the original hope is **spectral clustering** — the same
Laplacian, but *k* eigenvectors instead of one, then k-means in that space. That
does recover k groups, and would be a genuine alternative to HDBSCAN rather than
a picture of it. It reintroduces a parameter (*k*), which was the thing this idea
was trying to avoid, so it is not obviously a win.

Related: [[02-adaptive-cluster-resolution]] — the parameter problem this was
trying to sidestep.
