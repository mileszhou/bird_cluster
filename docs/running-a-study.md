# Running a study

`./run-all` is the demo: five stages, one command, a sample dataset, and it
proves the install works. It is not how the pipeline is used.

Real work here is **staged and run by hand**, because the parameters are not
settings to get right once — they are the experiment. How the structure changes
between `min_cluster_size` 3, 15 and 40 is itself the result, and a group that
survives all three is a different kind of object than one that dissolves at 10.
So runs are kept side by side rather than overwritten, each carrying the
parameters and the input that produced it, and you look at each stage's output
before starting the next.

This document is the loop. It assumes you have the README's quickstart working —
a dataset that resolves, and the embed server reachable.

## `local/` is your side of the repository

Start by copying the template, which carries a runbook of the whole pass below:

    cp -r _local local

`local/` is gitignored wholesale and is where your working material lives:
command lists for whatever you are trying this week, `local/manifests/` for scope
lists, and copies of generated reports. Nothing in it is published, and nothing
in the repository reads it.

The convention is worth following rather than inventing your own, because it is
what makes a run reproducible six months later:

**A file in `local/` is a list of commands in order.** Read it, and run it a line
at a time. Control flow does not belong there — it is the one directory that is
untested and unreviewed, so a program living in it is a program nobody can
check. Two things do belong.

**The environment, in full.** There is no wrapper underneath doing it, and the
file will be run from anywhere:

```bash
#!/usr/bin/env bash
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"
[ -d .venv ] && source .venv/bin/activate
```

**Defaults, written as visible parameters.** A default at the top is not logic;
it is documentation of the knob. It says both what the file does today and
exactly what to set for a run that differs — `MCS=3,5 ./local/my-study` — which
is the thing a bare command list cannot tell you:

```bash
EMBED="${EMBED:-output/embed}"          # which vectors
MCS="${MCS:-3,5,8,15,40}"               # the level-1 sweep
MAX_LEAVES="${MAX_LEAVES:-27}"          # leaves per branch at level 2
```

Only the parameters that actually vary between runs. A variable for every
argument turns the list back into a program and buries the one or two things
worth changing; if it does not vary, write it inline where it is used.

## The loop

`local/runbook` (from the template above) is this sequence as one file, meant to
be walked a line at a time rather than executed — it is deliberately not
executable. What follows is why each step is there; the runbook is the steps
themselves, so edit that rather than copying commands out of here.

### 0. Know your dataset

```bash
./tool-audit
```

Read-only, about twelve seconds. It reports what the pipeline can and cannot
see: sidecars no JPEG reaches (should be zero — those are photos the pipeline
never even enumerates), JPEGs with no sidecar (ordinary, they are labelled to
the CSV alone), and sidecars several JPEGs claim. Run it after any re-export.
The report is written where you can diff it against last time; the counts are a
fingerprint of the export's shape, so a jump means something changed about how
your images were emitted.

### 1. Start a backbone

One server serves one backbone at one resolution, deliberately: both are
properties of the vectors, so letting them vary per request would let one file
hold vectors that cannot be compared. Restart to change either.

```bash
./server-embed                                             # DINOv3, self-supervised
MODEL=hf-hub:imageomics/bioclip-2.5-vith14 ./server-embed   # BioCLIP, taxonomy-supervised
./server-embed --image-size 512                            # DINOv3 at a resolution it likes
```

**Which backbone is not a free choice, and not a ranking.** The two here differ
in kind rather than quality, and the defaults encode what has been measured:

*DINOv3* is self-supervised — it never saw a species name. That is what makes it
a legitimate referee for judging a *labelling*, since it cannot be accused of
having authored either side. Nothing else here can do that job. It wants 512:
224→512 was worth +0.0156 of 1-NN, measured.

*BioCLIP* is supervised on Linnaean names, and its text tower is what turns
stored vectors into a taxonomy for the cost of one matmul — the step that makes
rank questions askable at all. For the same reason it cannot referee a
labelling: it has a stake. It is native at 224, and how the frame is fitted to
that square (`--resize-mode crop` or `squash`) measured flat, so take the
default.

What is *not* a reason to choose between them is accuracy at species. Over 27k
images they score 0.5363 and 0.5334 on 1-NN, a difference that is not
significant (McNemar p=0.21), while being wrong together on 39.6% of images —
which is roughly the independently-derived error rate of the labels themselves.
Two backbones sharing no architecture and no supervision agreeing that closely
is the measure telling you it has stopped describing the vectors. Pick the one
whose *kind* suits the question.

`--image-size` matters more than it looks. The image processor carries a resize
in its own config and applies it silently — for DINOv3 ViT-B/16 that is 224×224,
so a 1024px export reaches the backbone as a 14×14 grid and the detail
separating two similar birds is gone before the model sees it. Whether more
pixels help is a question to measure, not assume; what is not optional is that
the resolution gets recorded, and it now is.

### 2. Embed

```bash
./run-embed --output-dir output/embed --dry-run     # resolve the scope, write nothing
./run-embed --output-dir output/embed
```

Always dry-run first. It prints the population — how many images, by year, how
many distinct species — and writes nothing. A stage that does no work and
succeeds is indistinguishable from a stage that had nothing to do, so an empty
selection is an error here rather than a cheerful zero.

The run is resumable: the checkpoint is the set of keys already in the JSONL, so
an interrupted run continues where it stopped. It **refuses to append when the
server serves a different model or resolution** than the rows already present —
vectors from two backbones, or one backbone at two resolutions, are not in the
same space and nothing downstream could detect the mixture. To change either,
use a fresh `--output-dir`.

Narrow the scope with a manifest rather than by moving files:
`--include-from` / `--exclude-from`, see `manifests/README.md`. Both paths are
copied into `run.json`, so a result carries the scope that produced it.

### 3. Judge the vectors before clustering them

```bash
python3 -m tools.audit_embed_quality --run output/embed --run output/embed-other
```

Leave-one-out 1-NN accuracy against the labels: for each image, is its nearest
neighbour the same species? A property of the vectors alone, with no clustering
parameter in it — comparing clusterings instead would confound the embedding
with HDBSCAN's settings. Runs are intersected on key first, so two runs are
judged on the images they share.

Read the *difference* between runs, not the absolute value. The labels are not
ground truth, and this measure has a ceiling: two embeddings that share no
architecture and no supervision can land within noise of each other while both
being "wrong" on the same 40% of images, at which point the number is describing
the labels rather than the vectors.

### 4. Cluster, as a sweep

```bash
./run-cluster --embeddings output/embed/embeddings.jsonl \
              --output-dir output/cluster --min-cluster-size 3,5,8,15,40
```

One directory per size. Each writes only its own `mcs<N>/` and clears it first,
so different sizes can run concurrently; the same size twice at once is a user
error.

A cluster's stable name is its **medoid's image key**, not an integer — HDBSCAN's
labels do not survive a re-run, since cluster 47 at `min_cluster_size=5` is
unrelated to cluster 47 at 15, while the medoid key is meaningful to a human and
joins back to the label CSV.

### 5. Group the clusters, and look

Level 2 groups level-1 **leaves** into **branches** — Ward over the leaf
medoids, cut adaptively until no branch holds more than `--max-leaves`. It takes
one run rather than a sweep, because branches over one clustering's leaves say
nothing about another's.

```bash
./run-cluster2 --run output/cluster/mcs3 --max-leaves 27
python3 -m tools.export_seriated --layout output/cluster2/mcs3/layout.csv \
        --label-dir output/label
```

`layout.csv` is the structural output and carries no labels: which leaf each
image is in, which branch each leaf is in, and the time encoding of both. The
render chooses the caption and owns it — a species column copied into a
clustering artifact is a second copy of a fact owned elsewhere, and goes stale
the moment a different labelling runs.

**Two levels of time carry the two levels of structure** into a photo manager:
month per branch, date per leaf, minute per image, second left free. Import the
exported JPEGs and sort by capture time — picking a month gives you a branch,
picking a day gives you a leaf. That is the review surface, and looking at it is
not optional: the metrics can say the grouping is real and coarser than species,
and only your eye can say what it is made of.

### 6. Ask which rank the clusters sit at

This needs a taxonomy, which the labels do not carry — they are common names
with no rank structure, so nothing can ask whether two species in one cluster
are *related*, only whether their strings differ.

```bash
python3 -m tools.predict_taxa --run output/embed-bioclip --taxon-class Aves
python3 -m tools.map_label_taxa
python3 -m tools.audit_rank_alignment --cluster-dir output/cluster \
        --layout output/cluster2/mcs3/layout.csv
```

The first needs BioCLIP vectors specifically — it classifies against the text
tower of the same model, and image and text vectors from two different models
share no space. It does no pass over pixels: the stored vectors are already the
classifier's input, so a different checklist is a different question asked of
the same vectors rather than another GPU hour.

Read the alignment as a *shape*, not a level. Purity rises with coarseness for
free, so what carries information is the shuffled null at the same rank and the
gain from species to family: a cluster already pure at species gains nothing by
coarsening, while one that is family-pure and species-mixed gains a lot.

## Where things go, and who writes them

```
data/      input. The pipeline NEVER writes here; a person curates into it
output/    the live run, one subdirectory per stage. Gitignored
output_NNN/  archived runs. ./clean moves output/ aside; nothing is deleted
local/     your command lists, manifests and report copies. Gitignored
```

`./clean` archives `output/` to `output_NNN/` and gives you an empty one.
`./clean my-description` names the archive, which matters more than it sounds:
**a bare number means nothing was learned there.** Name a run you want to
compare against later, leave a failed one bare, and six months on the naming
tells you what is safe to delete.

A run worth keeping carries a `FINDINGS.md` written before you clean — what the
run was, what it was measured against, the numbers, and what they mean. Generate
it from the artifacts rather than transcribing; a hand-copied figure eventually
disagrees with the file beside it. `run.json` preserves the parameters, but what
they *bought* survives nowhere else, and by the time anyone asks, the run it was
compared against has itself been archived.

**Curation is deliberately manual and there is no script for it.** Choosing which
run graduates into `data/` is the entire content of the step; automating it would
make it routine, which is the one property it must not have. `cp` is the
interface.

## Things that will bite

- **A fresh clone points at `localhost`, and a container makes that a trap.**
  Server hosts come from `config.toml`, overridden by `config.local.toml` —
  which is gitignored, so a new checkout has none and falls back to the tracked
  `localhost`. On one machine that is right and invisible. From a container it
  is wrong in a way that reads as "the server is down", because `localhost`
  inside a container is the container. `cp _config.local.toml config.local.toml`
  is a setup step worth doing before the first run rather than after the first
  confusing refusal.
- **A stage that selects nothing exits non-zero.** By design. A stale scope that
  quietly embedded a seventh of a library and reported success is why.
- **Never key anything by filename.** Stems repeat across folders as camera
  counters wrap. Every key here is a path relative to the image root.
- **CSVs are written UTF-8 with BOM** so a spreadsheet reads non-Latin folder
  names correctly. Read them back with `encoding="utf-8-sig"` or the BOM glues
  itself to your first field name.
- **A failed model probe is fatal**, and should be: the probe is how a run learns
  what it is talking to, and that answer becomes the recorded provenance of every
  row.
