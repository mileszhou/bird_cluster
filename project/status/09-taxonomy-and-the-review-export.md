# 09 — A taxonomy, and an export that carries every opinion

Sessions of 2026-08-25 to 08-28, on `main`. One thread with two halves: a second
backbone was added and used to obtain the taxonomy that `status/08` listed as the
top blocked item, and the browsing export was rebuilt to carry every source's
verdict at once so a human review can actually compare them.

18 commits, `de7636b` through `3a49c69`. Everything below is committed.

## Where things are

    output/embed/                DINOv3 512, 27,194 vectors     (unchanged)
    output/embed-bioclip/        BioCLIP 2.5 224 crop, 27,194   -- NEW
    output/embed-bioclip-squash/ the same, squashed             -- NEW, and a null result
    output/taxa/                 predictions + the label mapping -- NEW
    output/cluster2/mcs3/        layout.csv, 70 branches
    output/lightroom/jpg/        cluster2-mcs3 (DINOv3)
    local/lightroom/jpg/         cluster2-mcs3-bc (BioCLIP), imported to a catalog

## Thread 1 — a second backbone, and what it did not show

`hf-hub:imageomics/bioclip-2.5-vith14`: ViT-H/14, 1024-dim, supervised on
Linnaean names over TreeOfLife-200M, MIT and ungated. The server takes a backend
from the model id, so `MODEL=` is the only knob.

**They are two instruments, not two candidates.** DINOv3 is self-supervised and
is therefore the only thing here that can referee a comparison between two
*labellings*; BioCLIP is the only one with a text tower, which is what makes a
taxonomy obtainable at all. Neither substitutes for the other.

**Accuracy did not separate them, and that is the finding.** Three arms —
DINOv3 at 512, BioCLIP cropped, BioCLIP squashed — land within 0.003 of each
other at ~0.535 1-NN, and the one gap testable formally is insignificant
(McNemar p=0.21). All three are wrong on ~39.6% of images *simultaneously*, which
sits on `findings/03`'s independently derived 33.6–36.6% species error.

So **`audit_embed_quality` has saturated**: above ~0.53 it is describing the
labels, not the vectors. Reading a small 1-NN difference as "this embedding is
better" is now a mistake. Recorded in CLAUDE.md beside the measure itself rather
than in a finding, because a future session meets the tool before any finding.

The label-free comparison still has room. Asked whether two embeddings pick the
*same nearest neighbour* — no labels anywhere — DINOv3 and BioCLIP agree 42.5%
against a chance rate of 0.0037%, which is 84% of the agreement between one model
with itself under a preprocessing change. On images the labels call wrong for
both, they still agree 46.2% of the time, 89% of the rate when the labels call
them right. Genuine model failure does not look like that.

A caveat that colours the whole measure and was not known before: **78.5% of
nearest neighbours come from the same trip.** For four images in five, 1-NN is
scoring within-session consistency rather than species discrimination.

**`--resize-mode` measured flat.** crop and squash both score 0.5334 while moving
the vectors a mean cosine of 0.96 apart. Take the default; do not spend a run on
it again.

## Thread 2 — the taxonomy

TreeOfLife-200M's own taxon list, filtered to one class: **11,131 Aves taxa**,
283 families, 43 orders. It is the vocabulary BioCLIP was trained against, which
is the right label space for zero-shot.

- **`tools/predict_taxa.py`** — image vector · text vector. **No pass over
  pixels**: the stored vectors are already `encode_image` output, so the whole set
  classifies in under two minutes and a different checklist is a different
  question asked of the same vectors. Both knobs were measured and neither
  matters (four text forms within 0.002; prompt ensembling slightly worse).
- **`tools/map_label_taxa.py`** — gives the *existing labelling* a taxonomy by
  two routes blind in opposite directions: the checklist name, and BioCLIP's
  modal call over the images carrying that label. Where both fire they agree on
  family for 85.8% of images though only 63.4% of labels. 99.6% of images end
  with a family.
- **`tools/audit_rank_alignment.py`** — size-weighted modal purity against each
  rank, with a shuffled null.

### What it answered

`findings/01` asked whether a level-2 branch is a rank above species or a bag of
look-alikes. Over checklist-named labels only — no per-image model call anywhere
in the taxonomy — leaves are 0.6914 species-pure; branches are **0.6066
family-pure against a null of 0.0857, but only 0.2998 species-pure**.

A branch cannot be named with a species and can be named with a family. That is a
rank above species, and it partly overturns the visual verdict in the direction
that finding's own caveat predicted. The sweep says the same from the other side:
species purity falls from 0.6914 at mcs3 to 0.5008 at mcs40 while family purity
barely moves — larger clusters are not getting worse, they are drifting up a rank.

**`margin` is a calibrated confidence, which this project has never had.** Top-1
minus runner-up cosine, rising monotonically across all ten deciles from 0.0758
to 0.6776 agreement with the labels, against the VLM's own `confidence` column
which averages 0.968 on a ~35% error rate.

## Thread 3 — the export, rebuilt around comparison

The export captioned with the species and nothing else, so the review it exists
to support had to be done from memory. It now carries every source, each saying
who is speaking. Glossary has the table.

    (Q) (G)                          each labelling, tagged by the model that wrote it
    sp: sp-sci: ord: fam: gen:       derived FROM the label -- restates it, cannot check it
    bc: bc-sci: bc-ord: … bc-conf:   BioCLIP, read off the pixels -- the independent one
    (unprefixed, no parenthesis)     hand-written, preserved untouched

Three corrections, each from Miles looking at real output:

1. **The rank keywords restate the labelling.** For 81% of images they are looked
   up from the species string, so where the labeller is wrong they are
   confidently wrong the same way. Reviewing against them cannot find a bad
   label. Hence `--taxonomy-source {label,image,both}` and the `bc-` family.
2. **A photo could carry a self-contradiction.** For the 19% resolved by
   BioCLIP's vote, the genus came from pixels while the species keyword came from
   the labeller — one export carried a species from one genus beside a `gen:` from
   another. Both are now labelled with their source.
3. **The confidence was splitting the keyword panel.** 97% of species with 20+
   photos were listed under more than one keyword — 1,232 entries for 330 birds,
   one bird across 35 — because `(99%)` and `(95%)` are different strings. The
   suffix now names the labeller, read from each run's `args.json`. Each species
   collapses to one entry, and the number stays in the CSV.

`--label-dir` is repeatable and the keywords are additive, so two labellings put
two tagged keywords on one photo. **No CSV schema change, and none needed**: each
labelling keeps its own file and they are joined on `jpg` at export time.

## What the review has shown so far

Preliminary, and Miles is still looking. One species appears split across several
genera, and a large genus often holds visibly different species. The mapping is
provably consistent — no label produces two genera — so the splitting is *label
noise made visible*: the same bird named several ways, each name resolving
differently. The genus keywords being a projection of the labels is why they
cannot arbitrate; `bc-` can.

`warbler` spread over 34 genera is **correct**, not a defect: Old World and New
World warblers are unrelated birds sharing a folk name. That is the
folk-versus-Linnaean gap showing up properly.

## Bugs worth not re-introducing

1. **`split_keywords` must learn every new keyword prefix, in the same commit.**
   A prefix it does not recognise is preserved as the user's and then written
   again, so every `--labels-only` pass doubles it — silently, visible only in a
   keyword panel. This happened three times in one session. Tests now pin all of
   them, including that the rule never claims a user's `(juvenile)`.
2. **Sample randomly from the embedding JSONL.** It is written in path order, so
   the head of the file is one or two years and a third of the species. A
   text-form comparison scored on it read 0.45 where the true figure is 0.29, and
   even ranked the options differently.
3. **A default path must be `PROJECT_ROOT`-anchored, not cwd-relative.** A
   relative default silently found no taxonomy from any other directory, and the
   export then completed looking entirely normal.
4. **`./clean` takes a description, not flags.** `./clean --help` archived a live
   run as `output_012_help`. Now refused, with `--help` answering properly.

## Smaller things

- `docs/running-a-study.md` — the staged manual loop, for a reader with the code
  and no photographs. `run-all` is the demo; this is how the pipeline is used.
- `_local/` — a tracked template to `cp -r` into `local/`, carrying `runbook`:
  a full pass written as one script and deliberately **not executable**. Each
  step prints the command it actually ran, so the scrollback records the run.
- `./venv` is the ground truth; `requirements.txt` is a compatibility guard
  regenerated by `./venv freeze`. The old one pinned 343 `file:///` paths from
  somebody's conda environment and installed nowhere. `scipy` and
  `huggingface_hub` are now declared rather than arriving as somebody's
  dependency.
- Shebangs are `env bash` throughout — macOS has no `/usr/bin/bash`, and its
  `/bin/bash` is frozen at 3.2.
- Server probes now say where the host came from. A container with its own
  checkout has no `config.local.toml` and falls back to `localhost`, which inside
  a container is the container.

## Open, in rough priority order

1. **The review verdict.** `findings/01` is written up to the numbers but wants
   Miles's eye on whether branches read as families. Half of what that finding
   should say is not measurable.
2. **An expert.** Both verdicts in `findings/01` remain provisional on someone who
   can say whether species in one branch are close relatives. The checklist got
   us a rank; it cannot referee a species call.
3. **`bird_label.py:39`** imports torch through `transformers_engine`, a backend
   unreachable since `--approach` never listed it. It is the only thing stopping
   `./run-label` on a GPU-less box; the fix is moving one import into its function.
4. **The clustering-methods stability study**, designed in `status/08`, not started.
5. **The `label_cn` composer** still joins three fields with `-` where two can
   contain one (`status/08` bug 3). ~1,655 bird rows.
6. **`stats.py`**, still unwritten.
