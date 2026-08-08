# 06 — Full labelling run complete, dataset carries `data/label/`

Branch `cluster`. Supersedes `04-jpg-centric-labeler.md` (which was the pre-run baseline) and
closes `05-post-dedup-checklist.md`. The first complete JPEG-driven pass is done, audited, and
committed to the dataset.

Full analysis: **`project/reports/label_run_audit.md`**. This document is the state and the
decisions that follow from it.

## The run

| | |
|---|---|
| dataset | `data/` at `a902bf0a9`, post-dedup |
| model | `/data/models/Qwen3-VL-32B-Instruct` (directory renamed before the run — provenance is correct) |
| batch size | 32 |
| started / ended | 2026-08-08 00:18 → 06:05 |
| duration | 20,848 s ≈ **5.8 hours**, 2.36 img/s |
| rows | **49,224** — one per exported JPEG, 0 failures |

Faster than the 11–18 h estimated from batch-16 smoke runs; batch 32 roughly doubled throughput.

## Dataset state

`data/label/` holds the curated run — the first thing the project has put into the submodule,
under the "a person curates into `data/`" rule:

```
data/label/args.json                        run provenance
data/label/bird_identification_output.csv   49,224 rows, 17 MB
data/label/bird_label.log                   5.3 MB
data/label/processed.txt                    49,224 keys
data/label/raw/                             43,683 labelled sidecars
```

The pre-dedup Lightroom round trip is no longer on the critical path: the category now reaches
the embedding step through `data/label/` rather than through the library.

**One caveat on the working tree.** `data/label/raw/` was overwritten by hand with a
non-`data/xmp` source to produce a reverse diff for review. That source carries 190 sidecars
`data/xmp` does not and differs in `xmp:MetadataDate` / `xmpMM:InstanceID`, so the reverse diff
mixes label-removal with library drift. The committed tree is the good one; restore it with
`git checkout -- label/raw` inside `data/` before doing anything else there.

## Audit result: the write was clean

Committed `data/label/raw/` against `data/xmp`, same 43,683 paths:

- **0 changes outside the keyword block.** Every `crs:` setting and `xmpMM:History` block
  survived.
- 5,904 keywords removed across 5,822 sidecars — **all** prior pipeline output (5,579 `(NN%)`
  labels, 181 `_nb`, 81 categories, 59 early GPT-4o free text, 4 early categories).
  **0 hand-written keywords, 0 single tokens.**

The text-surgery writer and `split_keywords()` both held at full scale.

## What the model produced

| category | rows |
|---|---|
| bird | 27,188 |
| scenery | 12,950 |
| people | 7,198 |
| animal | 1,888 |

**The clustering set is 27,742**, not 27,188 — 554 photos this run called non-bird remain
`bird` in the library under the never-demote rule. Select on
`'bird' in (category, prior_category)`.

**569 of the 5,229 sidecar-less JPEGs are birds** (10.9%). That population did not exist as far
as the old sidecar-driven design was concerned.

## The finding that shapes the next stage

306 captures were exported twice, giving a free measurement of the model's self-consistency on
near-identical pixels:

- **category: 99% agreement** (302/306)
- **species label: 58% agreement** (179/306)

`herring gull` vs `snowy owl` on one frame; `golden pheasant` vs `copper pheasant` on another.
Category is trustworthy. Species is close to a coin flip at the margin.

Miles's own read, independently: the labels are **too specific, and common birds get rare
species names**. The data shows the mechanism — 345 cases of a one-off species sharing its head
noun with an abundant one *in the same trip*, frequently from the wrong continent:

- 158× great crested grebe + 1× **western grebe** (Nearctic), Rivertown
- 104× white-tailed eagle + 1× **bald eagle** (Nearctic), Norwegian fjord
- 88× pheasant-tailed jacana + 1× **northern jacana** (Neotropical)

2,825 distinct names, 1,255 of them appearing exactly once. The true count is far lower.
**To be reviewed with domain experts** before the species labels are trusted outside this
project.

Two further quality notes: 418 bird labels (1.5%) are scene descriptions rather than species
("grey heron rookery in leafless trees", "common gull and mandarin duck"), and confidence
remains uncalibrated — 99% of rows sit at 0.95 or 0.98, and the herring-gull/snowy-owl pair are
both 0.98, so confidence cannot filter the noise.

## Next — Miles's side: getting the labels into Lightroom

Independent of the pipeline work below, and in progress at the time of writing.

1. **Restore `data/label/raw`** in the submodule working tree — `cd data && git checkout -- label/raw` (see the caveat above).
2. **rsync `data/label/raw/` into the library**, then in Lightroom select the photos and run
   *Metadata → **Read Metadata from File***. This step is easy to skip and silently does
   nothing if skipped: the writer deliberately does not bump `xmp:MetadataDate`, so Lightroom
   has no signal that the sidecars changed and will not offer to reload them. The labels sit on
   disk unread until told.
3. The point of it: see the label beside the bird while browsing, and drive **smart
   collections** off the keywords. This is why the sidecar leg exists at all — principle 2, the
   sidecar as a bridge to Lightroom, not as the pipeline's transport.

**Recovery point.** `project/bookeeping/restic-snapshot.md` records the restic snapshot of the
full Lightroom library — repo `rest:http://nas4:/media`, hash `eeb1c598`, tags `lr` and
`_dedup`. It is the rollback if the apply-back goes wrong, and it holds sidecars the curated
`data/xmp` no longer does, which is what Part 4 of the audit report untangles.

Restic is the library's own provenance record. The library itself is far too large and too
binary for git, but snapshots make **file movement** — renames, re-filings, deletions across
dedup rounds — recoverable and diffable, which is exactly the class of change that has caused
trouble in this project. `data/xmp` versions the *curated* sidecars; restic versions the
*library*. Neither substitutes for the other.

## Next — pipeline

1. **Repoint `embed.py` at `data/label/`.** It still scans sidecars and reads `labels.is_bird`,
   so it cannot see the 569 sidecar-less birds and it re-derives a filter that is already
   recorded. Its JSONL must key on the JPEG path relative to `data/jpg`.
2. **Resolve the effective category at curation time**, so the embed stage reads a two-column
   contract and never learns the never-demote rule exists.
3. Exclude or flag the 418 scene-description labels before any species-level analysis.
4. Then embed 27,742 photos and cluster.

**Environment note:** `.venv` was absent from the working box at the end of this session —
rebuild with `./venv base client test cluster` before running the tests
(`cd test && ../.venv/bin/python -m pytest .`, 170 passing as of `2292d47`). The analysis in
this document and the audit report was done with system `python3` and stdlib only.

## Open, unchanged from 04

- **Confidence is not calibrated** and cannot be used as a filter. The prompt asks for "a float
  between 0.0 and 1.0" with no rubric. Fixing it would need a rubric or a different signal
  entirely — worth doing before any future labelling pass, not worth re-running this one for.
- The model-directory misnaming from 04 is **resolved** — `args.json` records the real model.
