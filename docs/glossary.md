# Glossary

Terms and abbreviations used across this repository, with what they mean *here*
rather than in general. Three sections: **abbreviations**, **vocabulary** —
ordinary-looking words used in a narrow sense — and **the measures**, which are
the easiest thing to misread because most of them are unitless numbers between
0 and 1.

**Borrowed terms are marked, and the distinction matters.** Some of this
vocabulary the project coined and it means nothing elsewhere — `key`, `leaf`,
`branch`, `claim`, `applied`, `orphan`, `never-demote`. The rest is standard
terminology taken intact from clustering, statistics or archaeology — `Ward`,
`medoid`, `centroid`, `seriation`, `dendrogram`, `noise` — and those are worth
looking up, because the literature on them is real and this document is only a
gloss. Entries below say which is which.

## Abbreviations

### The pipeline

| | |
|---|---|
| **VLM** | Vision-language model. The thing that looks at a photo and answers with a species. |
| **LLM** | Large language model. A VLM is one with an image encoder attached. |
| **vLLM** | The inference server the labeller talks to over an OpenAI-compatible HTTP API. A serving engine, not a model — the name nods to virtual memory, after the paging trick it uses for attention. Lowercase v, and easy to misread as "VLM". |
| **HF** | Hugging Face. Where the embedding model is downloaded from; `HF_TOKEN` gates it. |
| **API** | Here, almost always the OpenAI-compatible HTTP interface that both vLLM and llama.cpp expose. |
| **GPU / CUDA / VRAM** | The graphics card, NVIDIA's programming interface for it, and its memory. Only the model servers need one. |
| **SIGINT** | The POSIX interrupt signal — what Ctrl-C sends. Both long-running stages catch it and finish the current batch rather than dying mid-write. |

### Models

| | |
|---|---|
| **DINOv3** | The embedding model. DINO is "self-**DI**stillation with **NO** labels" — it is trained without any labels at all, which is exactly why it can referee a labelling. |
| **ViT-B/16** | Vision Transformer, **B**ase size, **16**×16-pixel patches. The DINOv3 variant used here: a 512×512 image becomes a 32×32 grid of patches. |
| **L2 / L2-normalised** | The ordinary Euclidean length of a vector. The embedding server divides every vector by its own length, so all of them sit on the unit sphere — which makes cosine similarity and Euclidean distance rank things identically, and lets a dot product stand in for either. |
| **CLS token** | The "classification" token — one extra position the transformer carries alongside the image patches, whose output vector is used as the whole-image embedding. |
| **Qwen3-VL, gemma** | The two labelling models compared in `project/findings/03`. `-32B`, `-31B` are parameter counts in billions. |
| **GPT-4o** | OpenAI's model, used by the earliest labelling runs. Its verdicts survive only in the CSV's `prior_label` column. |

### Files and formats

| | |
|---|---|
| **CSV** | Comma-separated values. The labelling run's CSV *is* its output, not a report about it. |
| **JSONL** | JSON Lines — one JSON object per line. The embedding format: appendable, greppable, one row per image. |
| **XMP** | Extensible Metadata Platform, Adobe's sidecar format. An XML file next to a raw file, holding keywords and edits. |
| **RDF** | Resource Description Framework, the XML dialect inside an XMP file. `rdf:Bag` and `rdf:Seq` are its unordered and ordered list containers — Lightroom writes keywords in a Bag, older runs of this project wrote a Seq, and both must be read. |
| **IPTC / IIM** | International Press Telecommunications Council, and its Information Interchange Model — the *legacy* metadata block inside a JPEG. Lightroom needs keywords in both this and XMP or it shows neither. |
| **EXIF** | Exchangeable Image File Format: the camera's own metadata block. `DateTimeOriginal` lives here, which is the field the seriated export overwrites to control sort order. |
| **BOM** | Byte order mark. Three bytes at the start of a UTF-8 file that tell Excel it is UTF-8; without one, Excel mangles Chinese folder names. Read such files back with `encoding="utf-8-sig"`. |
| **DNG / NEF** | Adobe's Digital Negative, and Nikon Electronic Format — two raw formats. A DNG carries its metadata internally and so has no sidecar. |
| **xmpMM** | The XMP **M**edia **M**anagement namespace. `xmpMM:OriginalDocumentID` identifies the *capture*, and is how duplicate sidecars are found; `xmpMM:DocumentID` identifies the *edit*, so virtual copies of one capture share the first and differ in the second. |
| **TOML** | The config format. `config.toml` is tracked and must work for a stranger; `config.local.toml` is not. |

### Filename decorations

Suffixes Lightroom adds on export. The claim rule strips them to find the
sidecar a JPEG belongs to.

| | |
|---|---|
| **`-2`, `-3`** | A **virtual copy** — an alternate edit of one capture, exported under its copy name. |
| **`-Enhanced-NR`** | **N**oise **R**eduction: Lightroom's AI Denoise render, a separate DNG. |
| **`-HDR`, `-Pano`, `-Edit`** | High dynamic range merge, panorama stitch, and an external edit. |

### Keyword prefixes in a browsing export

What `tools/export_seriated.py` writes into the exported JPEGs, so a photo
manager's keyword panel can filter by any of them. **The prefix says which
source is speaking**, which is the whole point: a photo carries two opinions and
they frequently differ.

The prefixes sort together in a keyword list, so one source's view stays
contiguous while you browse.

| | |
|---|---|
| *(no prefix)* | The labelling itself, composed: `{pinyin_initials}-{chinese_name}-{english_name}({tag})`, where the tag names the labeller — `Q`, `G`. One keyword per labelling, so a photo can carry several and they can be told apart. A hand-written keyword has the same shape **without** any parenthesis. |
| **`sp:`** | The labeller's species, repeated plainly so it can be read and filtered beside the others. |
| **`sp-sci:`** | The scientific binomial for that species — **only** where the checklist matched the label by name. Absent where the label was resolved by BioCLIP's vote, because that binomial is BioCLIP's opinion and printing it here would show one opinion twice. |
| **`ord:` `fam:` `gen:`** | Order, family, genus **derived from the labeller's species string** via the checklist. These restate the labelling; they do not check it. |
| **`bc:`** | **B**io**C**LIP's species, as a common name. Read off the pixels, owing the labelling nothing — the independent second opinion. |
| **`bc-sci:`** | The same call as a scientific binomial, for an expert's eye. |
| **`bc-ord:` `bc-fam:` `bc-gen:`** | BioCLIP's ranks, on the same footing as its species. |
| **`bc-conf:`** | `high` / `mid` / `low`, bucketing **margin** (see The measures). The only calibrated confidence here, and the right thing to sort a review by. |
| **`<TAG>-conf:`** | `A`–`D`, banding *that labeller's own* confidence — `Q-conf:A`. A letter rather than a number, and a keyword rather than part of the label, because a confidence inside the label fragments the species: `kingfisher(99%)` and `(95%)` are two entries for one bird. The cuts follow where disagreement measurably steps rather than round numbers. |

The labeller's own confidence **is not written into a keyword**, and used to be.
It said nothing — it averages 0.968 against a measured ~35% error — and it did
harm: a species labelled at 99% on some photos and 95% on others is two
different strings, so a keyword panel listed one bird several times and no entry
held all of it. Over the species with 20 or more photos, 97% were split this
way, into 1,232 entries for 330 birds. The tag collapses each to one. The number
survives in the CSV, which is where something nobody should read at a glance
belongs. Older exports carry `(NN%)`; both forms are still read.

So one photo reads, schematically:

    ord:<Order>  fam:<Family>  gen:<GenusA>  sp:<the labeller's name>  sp-sci:<GenusA epithet>
    bc-ord:<Order>  bc-fam:<Family>  bc-gen:<GenusB>  bc:<BioCLIP's name>
    bc-sci:<GenusB epithet>  bc-conf:high

The shape to look for is that one: agreement at order and family, divergence at
genus and species, and `bc-conf:high` saying the second opinion is a confident
one. That is a disagreement a photograph can settle, which is what the export
is for. Sorting a review by `bc-conf` puts those first.

### Shorthand in paths and flags

| | |
|---|---|
| **mcs** | `min_cluster_size`, HDBSCAN's one significant parameter. `output/cluster/mcs3/` is that clustering at `min_cluster_size=3`. **Level 2 has no mcs** — it is Ward, which has no such parameter. |
| **LR** | Lightroom Classic. |
| **jpg / xmp** | Used as *key spaces*, not just formats: a `jpg` key is a path relative to `data/jpg`, an `xmp` key relative to `data/xmp`. See **key** below. |

## Vocabulary

**key** — a string that locates an item, and the only notion of identity here.
A path relative to an agreed-upon root, `data/jpg` everywhere downstream of
labelling. Two keys are the same item when the strings are equal. Never a
basename: camera counters wrap, so stems repeat across trips.

**sidecar** — the XMP file beside a raw. An *acceptor* for a label, not an
identity: thousands of JPEGs have no sidecar and are ordinary members of the
population.

**claim** — which sidecar a given JPEG's label writes into. The rule is local
(it looks only at that JPEG's name and the sidecar tree) and deterministic, so
a resumed run reproduces the same assignment.

**orphan / derived** — a JPEG no sidecar claims. *Orphan*: there was never a raw
behind it. *Derived*: it is a decorated export of a capture already matched.

**manifest** — a list of keys or folders naming which images a run should
include or exclude. Paths, never patterns. Lives inside the repository so a
run's recorded scope is not a dangling reference.

**run root / archive** — `output/` is the live run; `./clean` moves it to
`output_NNN_<description>/` and starts a fresh one. Nothing is deleted. A bare
`output_NNN` with no description means nothing was learned there.

**never-demote** — a retired rule by which a `bird` verdict could not be
overturned by a later non-bird one. Retired, but its traces are permanent: rows
marked `kept-existing` still exist in old CSVs, so **effective category** and
**effective species** resolve what the library actually holds rather than what
the run last said.

**applied** — what reached the sidecar for one row: `written`, `csv-only` (the
image has no sidecar), `failed`, or the historical `kept-existing`.

**prior_category / prior_label** — the label the *previous* run left. The CSV is
the only record of it, which is what makes the pipeline a model-comparison
instrument, and why old CSVs are not discarded.

**leaf / branch** — the two clustering levels. A **leaf** is a level-1 HDBSCAN
cluster of images; a **branch** is a level-2 Ward group of leaves. Both are
named by a **medoid** key, so the name survives a re-run — HDBSCAN's integer
ids do not.

**medoid / centroid / density peak** *(borrowed)* — three ways to say "the middle" of a
cluster. The **medoid** is an actual member closest to the centre and is what
this project names clusters by; the **centroid** is the mean vector, which is
shortened by internal disagreement; the **density peak** is the member in the
densest neighbourhood.

**noise** *(borrowed)* — HDBSCAN's label for a point in no cluster. Not an error and not a
bad photograph: it means the point had no dense neighbourhood at that
`min_cluster_size`. Raising the parameter makes *more* noise, not less.

**seriation** *(borrowed)* — ordering things so that neighbours in the order are
similar. Not a coinage: it is the archaeologists' term for arranging artefacts
into a sequence by similarity in order to date them relatively, and the same
word is used in combinatorial data analysis for reordering the rows and columns
of a matrix to bring structure onto the diagonal. Both senses are the one used
here. In this project it is presentation only — it was tried as a way to
*discover* clusters and rejected, because an ordering is one-dimensional and
the structure is not.

**Ward** *(borrowed)* — Ward's method, a way of building a hierarchy of
clusters, and an eponym rather than an acronym: Joe H. Ward Jr., 1963. It is
**agglomerative** — start with every item its own cluster and repeatedly merge
the pair whose merger increases the total within-cluster variance the least,
until one cluster remains. Three properties are why level 2 uses it instead of
HDBSCAN again: it has **no parameter**, it produces **every level at once** as a
tree rather than one partition, and it has **no noise class**, so nothing is
left out — which a taxonomy cannot tolerate.

**linkage / dendrogram** *(borrowed)* — the two artifacts Ward produces. The
**linkage** is the merge history: which pair joined, and at what cost. The
**dendrogram** is that history drawn as a tree. Cutting it at a given height
yields a partition, so a dendrogram is not *a* clustering but all of them, and
choosing a cut is a separate decision from building the tree.

**layout vs index** — `layout.csv` is the second-level clustering's output:
structure and time, no labels. `index.csv` is written by the *export* beside the
JPEGs: the layout it was given plus the species it captioned with. One owner per
fact.

**pool / tail** — clusters too small to deserve a date of their own in an
export, sharing one and alternating colour label so their boundaries stay
visible.

## The measures

All of these are computed against the pipeline's own labels, which are **not
ground truth**. Read differences between runs, not absolute levels.

| | |
|---|---|
| **1-NN** | Leave-one-out nearest-neighbour accuracy: for each image, is its closest neighbour in vector space the same species? A property of the vectors and labels alone, with no clustering parameter in it. |
| **purity** | The dominant species' share of a cluster. Rises automatically as clusters get smaller, so it means little without a null to compare against. |
| **homogeneity / completeness** | The two halves of the same question. *Homogeneity*: does each cluster contain one species? *Completeness*: is each species in one cluster? Splitting a species hurts completeness only; mixing two hurts homogeneity only. |
| **AMI** | Adjusted Mutual Information. How much knowing one grouping tells you about the other, adjusted so that chance scores 0. The workhorse here, because it survives having thousands of classes. |
| **NMI** | Normalized Mutual Information — the same thing *unadjusted*, and badly inflated when there are thousands of classes. Quoted only to show the inflation. |
| **ARI** | Adjusted Rand Index. Pair-counting rather than information: do the two groupings agree about which *pairs* belong together? |
| **VI** | Variation of Information, in bits. A true metric — 0 means identical — and it splits into a *splitting* term and a *merging* term, which is how you tell which grouping is finer. |
| **effective species** | `exp(entropy)` of a cluster's species mix: "this cluster is effectively 6.2 species". Sees the whole distribution, where purity sees only the top of it. |
| **McNemar** | A paired test for "did this change help?" It counts only the cases where the two alternatives disagree, which is the right denominator when both are mostly right about the same easy cases. |
| **null** | What a measure scores when the structure is removed — usually by shuffling labels while keeping group sizes. Without one, a purity of 0.42 is unreadable. |
| **σ (sigma)** | Standard deviations away from that null. |
| **p90 / p99** | Percentiles: the value 90% or 99% of the data falls below. Used where a mean hides a tail. |
