# 01 — Rediscover mislabelled photos by embedding everything

*Raised 2026-08-09, mid-way through deciding what to do about stale bird labels.
Not a plan: no design, no schedule.*

## The idea

Embed the **whole library**, not just the birds, and let clustering surface the
photos the labeller got wrong. A wombat labelled `common myna` should land among
mammals, not among mynas. A penguin filed as `animal` should form a tight group
of its own, sitting nearer the birds than the mammals.

If that works, it is a finding rather than a convenience: the project's premise
is that a vector carries more than a label does, and *recovering the label's own
mistakes from the vectors* is the most direct demonstration of it available.

## Why it is worth doing

Selecting the embedding set by the label bounds what can be learned from it.
Filtering to `category == bird` makes label noise **inside** the bird set
measurable and false negatives **outside** it invisible — they are excluded by
construction, so no amount of clustering can find them. Embedding everything is
what closes that loop.

It also has a use beyond the demonstration. Bird protection was given up on
2026-08-09 knowing it would lose penguins and the occasional owl (see
`set_keywords_in_xmp()`), on the reasoning that finding those properly would
need several models, calibration and voting — a different project. Clustering
offers a cheaper route to the same place: not a better label, but a *shortlist*
of photos whose neighbours disagree with them.

## The part that makes it testable

There is already a validation set, produced by an unrelated method.

- **66 proven collisions.** `tools/stale_bird_labels.py` found sidecars whose
  bird label provably belongs to a same-stem twin, from the basename-keyed era:
  `Photos-24/2024-04-27 Craddle Mountain/_D5D5533.jpg` carries `common myna`
  because `Photos-24/2024-04-15 Sydney/_D5D5533.jpg` really is one. Each has a
  known-correct answer that owes nothing to any embedding.
- **~120 category slips.** Photos the run called `animal` while giving them a
  bird name it used for birds elsewhere — `great horned owl`, `king penguin`,
  `great curassow`. Expected to cluster tightly and away from mammals.

So the question is not "does clustering find something interesting" but "what
fraction of a known set does it recover", which is a much stronger claim.

The validation set is in **`data/label`** itself, and needs no superseded run
kept beside it. `prior_labels()` reads from `data/xmp`, which the pipeline never
writes, so `prior_category` / `prior_label` carry the *original* labels rather
than any previous pass's. Both sets recompute from the current CSV alone —
**559** rows whose prior was `bird` against a current non-bird verdict, of which
**65** are provable same-stem collisions.

That is arguably the cleaner arrangement: what is being tested (the old label)
sits in a different column from what is being trusted (the current one), rather
than in a different directory.

## What would have to be true

- **The boundary will not be clean.** DINOv3 is a general visual backbone;
  clusters form around composition, background and pose as much as taxonomy. A
  distant bird on water may sit nearer other water scenes than a frame-filling
  portrait of its own species. "Cluster is mostly bird, one member is `scenery`"
  will need a threshold, not a rule — and the shape of that threshold is most of
  the actual work.
- **Duplicates distort density**, and HDBSCAN is density-based. 306 captures
  were exported more than once; the alternate edits are near-identical vectors.
  Dedupe before clustering.
- Whether a *mixed* cluster means a mislabel or just a visually coherent scene
  type is the open question. It is not obviously answerable without looking at
  photos.

## Cost

Cheap enough not to need justifying. `--categories all` is implemented:

```
./run-embed --years all --categories all
```

49,224 images against 27,742 — under an hour of GPU, ~450 MB of JSONL versus
~255 MB. The composition, for reference:

| category | images |
|---|---|
| bird | 27,742 |
| scenery | 13,159 |
| people | 6,721 |
| animal | 1,456 |
| unknown | 146 |

## Related

- `set_keywords_in_xmp()` — why bird protection was given up, and what it cost.
- `tools/stale_bird_labels.py` — produced the validation set.
- `project/reports/label_run_audit.md` — the 58% species self-agreement figure,
  which is the other half of the argument that labels need explaining rather
  than trusting.
- The **manifest** idea in CLAUDE.md (an inclusion list with predicate syntax for
  selecting embedding subsets) is the general form of `--categories`; deferred
  until the requirements are clearer. Its mini version now exists as
  `--include-from` / `--exclude-from` (`code/lib/path_filter.py`) — plain path
  lists, no predicates — which is enough to scope an experiment without
  designing the language first. Worth using to drop captive collections: the
  City Zoo trip alone is 405 birds whose species mix reflects the
  collection rather than the region.
