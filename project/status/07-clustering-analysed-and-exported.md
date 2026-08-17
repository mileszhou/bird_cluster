# 07 — Clustering swept, analysed, and exported for visual study

Branch `main` (was `cluster`; renamed and made default on GitHub 2026-08-12).
Supersedes `06-label-run-complete.md` for anything after the labelling run.
Written 2026-08-14 as a session handoff.

## Where the work is

Steps 1 and 2 of the plan are done and tagged. Step 3 — "learn from the
clusters" — was planned as `stats.py`, never written under that name, and
arrived instead as `plot_matrix`, `plot_radial`, `subcluster_probe`,
`plot_adjacency` and the species-discovery report.

    v1.0-embedding-complete    step 1
    v2.0-clustering-complete   step 2

**The sweep is complete and on disk.** `output/cluster/mcs{3,5,8,15,40}`, all
five with `run.json`, so they are protected — re-running a size now stops rather
than clobbering, and `--force` is the deliberate override.

| mcs | clusters | noise | median | largest |
|---|---|---|---|---|
| 3 | 1,090 | 43.8% | 7 | 452 |
| 5 | 528 | 52.1% | 12 | 396 |
| 8 | 286 | 59.3% | 23 | 408 |
| 15 | 146 | 68.7% | 32 | 427 |
| 40 | 30 | 66.7% | 90 | 5,469 |

Parallel and sequential runs give bit-identical results; that was checked.

## What was learned

**Species are discoverable without labels.** Inside the 5,469-image over-merge,
holding the labels back entirely and letting `relative_validity_` choose the
resolution recovers them at AMI 0.754 (0.839 on labels with ≥20 images), against
a supervised 1-NN ceiling of 62.5% over 68 classes. Full account in
`project/reports/species_discovery_report.md`.

**Effective dimension predicts which clusters are worth subdividing** — +0.72
with the AMI gained by splitting, +0.64 partialling out size, and it needs no
labels at all. `tools/plot_radial.py` computes it.

**Seriation orders; it does not group.** Explored over three days and rejected
for discovery — see `project/ideas/03`. It recovers well-separated clusters as
contiguous blocks only up to about 8 of them; by 16 it is gone, and more
separation does not help. One eigenvector carries one bipartition. It stays as
the *display* order for the 146 centroids, which is what makes the matrix and
adjacency pictures readable.

**Image-level structure is not one-dimensional**, confirmed three ways: band
enrichment 1.34× at image level against 1.78× at cluster level; the ±25% band
near the 1-D ceiling while ±5% is not; and 91.7% of adjacent pairs crossing a
cluster boundary under a Fiedler ordering, against 98.6% random.

## The export, and what to do with it

`output/lightroom/jpg/mcs15/` — 8,425 copies, 1.0 GB, folder structure preserved,
capture time rewritten so **Lightroom's default sort walks the seriation**:

    day     one date per cluster, in seriated order (2000-01-01 .. 2000-05-25)
    minute  position within the cluster
    second  always :00, free for inserting test images between neighbours

**146 dates, one per cluster, nothing pooled** (`--min-cluster 15`, the default
since 2026-08-15). Dates follow the *seriation*, not cluster size, so adjacent
dates are adjacent clusters and the date sequence lines up with the adjacency
curve — the first days run warblers → robins → fulvettas → parrotbills.

**Next action: import that folder into a *new* Lightroom catalogue** and study
the clusters visually, using `output/cluster/mcs15/adjacency-cluster.csv` as the
guide — the curve's walls are cluster boundaries (86 of the 145 largest steps
are real ones, 59% against 1.7% by chance) and the bumps *inside* a cluster are
what wants a human eye.

`output/lightroom/jpg/mcs15/index.csv` joins the two: seq, capture_time,
cluster_id, species, key.

**The species label is embedded in each JPEG** as a `dc:subject` keyword
(`jbly-极北柳莺-arctic warbler(98%)`), written by `tools/export_seriated.py` via `code/lib/jpg_meta.py`
(the two were separate tools until 2026-08-17).
It has to be embedded rather than deposited in a sidecar: Lightroom Classic
reads `.xmp` sidecars only for raw formats, so the mechanism the labeller uses
cannot reach a JPEG. Hand-written keywords in the export (`bhl-百花岭`,
`cn-翠鸟-kingfisher` — 3,682 files carry them) are preserved; only the stale
labels from the earlier pipeline era (846 files) are replaced. The `(NN%)`
suffix is kept as the marker that a machine wrote the keyword, which is what
`split_keywords()` recognises. Pixels verified byte-identical to `data/jpg`
across all 8,425.

**A keyword lives in two places in a JPEG, and the first pass only wrote one.**
XMP `dc:subject` was set and verified everywhere, and Lightroom still showed the
photos unlabelled: these files also carry legacy IPTC IIM keywords in a
Photoshop `APP13` resource (`8BIM 0x0404`, dataset 2:25), which is what it
displayed — nothing on 3,897, the user's own on 3,706, a stale label on 822.
The capture time had the same split (`piexif` writes EXIF; IIM `2:55`/`2:60`
still held the true date in 8,421). Both stores are now written together, so
neither can win an argument. The general lesson: **verifying the field you just
wrote proves the write, not the outcome.**

**The pooling threshold was wrong and is fixed.** `--min-cluster` defaulted to
100, which pooled 121 of the 146 clusters — **51% of the export** — into one
undifferentiated three-date run, and `index.csv` recorded those rows as `tail`,
discarding which cluster each belonged to. Both are corrected: the default is
now 15 (at mcs15 nothing is pooled, since no cluster is smaller than the
`min_cluster_size` it was clustered with), and a pooled row keeps its own
`cluster_id`. Miles's rule for when pooling is legitimate: a cluster is worth a
date of its own, and combining only makes sense when one is genuinely tiny —
around 5. Raise the threshold for the mcs3 and mcs5 sweeps, which do have such
clusters. `--dry-run` prints the shape.

Nothing here touches `data/` or the Lightroom master. Copies, never hardlinks,
deliberately.

## State of the repository

Branch `main`, one commit unpushed at the time of writing (`db35d10`). Remotes:
`github` (public, renamed to bird_cluster) and `origin` (the private NAS). Both
were force-pushed on 2026-08-12 after three history rewrites; both are on the
scrubbed history.

**The repository went briefly public before being checked**, and the cleanup
that followed is worth knowing about: gitleaks was clean and correct, but a
photo project's exposure is personal data, not credentials. GPS at metre
precision in 5,491 sidecars on the old `master` branch (deleted), camera serials
in the sample dataset (scrubbed and republished as a single commit), and place
names in folder names and prose (purged from all history). Standing rules that
came out of it are in CLAUDE.md: **this repository is the software; research on
the data lives elsewhere**, and `local/` is the one gitignored home for anything
about the library.

## Open, in rough priority order

1. **Look at the export.** Everything above is instrumentation; the reason for
   it is a person looking at photographs.
2. **The `--min-cluster` threshold** wants choosing by eye, not by argument.
3. **`species_discovery_report.md`** is the one report still tracked. By the
   policy it is data research and belongs elsewhere; it is also the most
   compelling artifact in a public repo. Undecided on purpose.
4. **Spectral clustering** is the honest version of what seriation could not do —
   same Laplacian, k eigenvectors instead of one. It reintroduces k as a
   parameter, which is what the seriation idea was trying to avoid, so it is not
   obviously a win. `project/ideas/03` records the reasoning.
5. **`--model` still defaults to `gpt-4o`** while `--approach` defaults to
   `vllm`. Harmless — the server probe is authoritative — but a dry run prints a
   misleading line.
