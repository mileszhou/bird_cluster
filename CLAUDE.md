# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Automated bird species identification for photo libraries. Processes JPEG images through a vision-language model to identify birds, generate English/Chinese species names with confidence scores, update XMP sidecar metadata, and produce CSV result files.

## Running the Labeler

Set up environment first:
```bash
cp _env .env
# Add OPENAI_API_KEY to .env if using chatgpt approach
```

`config.toml` also carries **`current_working_output`** — the run directory the analysis tools
work on by default. Runs get archived under new names (`./clean` moves `output/` to
`output_NNN_<description>/`), so a post-processor that hardcoded `./output` would analyse
whatever happened to be sitting there. Point the setting at an archive to re-analyse an old
run and every tool follows without a flag; `--output-dir` still wins per invocation.

Non-secret configuration (which host each backend server runs on, etc.) lives in `config.toml`
at the repo root, checked into git — see `[servers.*]` entries, read via
`code/lib/config.py:server_url()`. Keep it separate from `.env`: `.env` is for secrets only and
is gitignored.

**vLLM is the only backend fast enough for a real run.** `chatgpt` and
`llama.cpp` work and are kept — the paired-verdict design makes this project a
model-comparison instrument, and the early GPT-4o labels in `prior_label` came
through that path — but neither is a way to label 49k images. `--approach`
defaults to `vllm` accordingly; it used to default to `llama.cpp`, so a direct
`python -m code.bird_label` quietly took the slow path.

**`./run-all`** runs the three stages back to back on `./sample_data`: the
one-command demo, and the integration test, since a change to one stage has
repeatedly broken another. It **does not curate** — the two hand-offs are staged
into `output/curated/` and the `cp` a person would run is printed instead. It
refuses `--data-dir ./data` without `--force`, refuses a non-empty `./output`
(`./clean` first), and preflights both model servers before doing any work.

**Three prefixes, by what a script does to the world.** `run-` is a pipeline
stage writing to `output/` (`run-label`, `run-embed`, `run-cluster`, `run-all`);
`tool-` inspects and reports without changing anything (`tool-audit`,
`tool-dedup`, `tool-decode-jpg`); `server-` starts a long-running process
(`server-embed`). All were `run-` before 2026-08-11, which put a read-only
report and an 11-hour labelling run behind the same verb.

Every wrapper must either forward `"$@"` or reject arguments it does not know;
`test/tools/test_run_scripts.py` asserts it across all three prefixes, after
`./run-vllm --years` silently relabelled the whole library.

The backend is a flag, not a script:
`run-gpt` / `run-cpp` / `run-tf` were four wrappers differing only in
`--approach`, which put a deployment detail in the command name and let
`run-tf` sit there broken (its approach was never a valid choice). Deleted
2026-08-11.

Run via convenience scripts or directly:
```bash
./run-label                       # vLLM server, host from config.toml [servers.vllm]
./run-label --approach chatgpt    # GPT-4o (requires OPENAI_API_KEY)
./run-label --approach llama.cpp  # llama.cpp server, host from [servers.llama_cpp]

# Or directly with options:
python3 -m code.bird_label --approach vllm --model "Qwen/Qwen3-VL-132B-Instruct" --conf-threshold 0.6
```

Key CLI flags:
- `--approach` — `chatgpt`, `llama.cpp` or `vllm`. Not `transformer`: `transformers_engine.py` exists but was never added to the choices, so `--approach transformer` has always been rejected by argparse
- `--vllm-url URL` — vLLM OpenAI-compatible server endpoint (default: `config.toml` `[servers.vllm]`, vllm approach only)
- `--conf-threshold FLOAT` — confidence below which to flag as low-confidence (default 0.6)
- `--no-bird FLOAT` — confidence below which to mark as "no bird" (default 0.2)
- `--filter-csv PATH` — re-process only "animal" or low-confidence rows from a prior run's CSV
- `--run-label TEXT` — tag this run in the output CSV
- `--batch-size INT` — number of images processed concurrently against the vLLM server (default 1; 8 is a reasonable default — each unit is a concurrent HTTP request, the server does its own continuous batching)
- `--include-from` / `--exclude-from PATH` — a manifest under `manifests/`; `manifests/exclude-captive.txt` is the form to type, since it tab-completes and matches what is on disk. A bare name works too. Same mechanism and same keys as `embed.py`, so one list scopes both stages
- `--dry-run` — resolve the scope and report it, then stop. No model probe, no sidecar copy, no writes

**A failed model probe is fatal.** For the `vllm` and `llama.cpp` backends the run resolves what
the server actually serves before doing anything else, and exits if it cannot. The probe is not
a convenience for fixing up `--model` — it is how the run learns what it is talking to, and that
answer becomes the recorded provenance of every row. It used to warn and carry on with the
requested name, which is how a whole run came to be attributed to a 132B model that was never
loaded; an empty `data` list did not even warn.

Start a fresh run: **`./clean`** — it *archives* `output/` to `output_NNN/` (one beyond the
highest present) and creates an empty one. Nothing is deleted. `./clean first-full-run` appends
a description, and an archive can be renamed by hand later (`output_003_qwen32b`) without
breaking the numbering — only the leading digits are read. `output*/` is gitignored, so the
archives are too. `raw/` is a full copy of the sidecar tree (739 MB) and dominates each
archive's size; it is safe to drop from an archive once its labels have been rsynced into the
library, keeping the CSV, log, `args.json` and checkpoint.

`run-label` defaults to `--approach vllm`, which talks to a vLLM OpenAI-compatible server (`--vllm-url`, default from `config.toml` `[servers.vllm]`) rather than loading the model in-process, so start the server separately before running this. (`run-server-vllm` used to sit here as an example; it was deleted 2026-08-11, since the server is launched from outside this repo and the wrapper had drifted to naming a model that is not the one being served.) On startup the script probes `{vllm-url}/models` and swaps in whatever model the server actually has loaded if `--model` doesn't match, so the requested model name doesn't need to be exact.

## Architecture

### What the project is

**Building clusters, and learning from them.**

The plural is deliberate and has consequences. A clustering is not one artifact with an `id`
column; it is *many clusters*, each of which is a thing to look at, compare, name and argue
about. That makes a cluster an addressable object, and objects need identities that survive
a re-run — HDBSCAN's integer labels do not, since cluster 47 at `min_cluster_size=5` is
unrelated to cluster 47 at 15. A cluster's stable name is its **medoid's image key**: already
computed, meaningful to a human, and the same string that joins back to the label CSV.

The second consequence is that **a clustering run is an experiment, not a build step**. The
parameters are not settings to get right once — how the structure changes between
`min_cluster_size` 5, 15 and 40 is itself something to learn from, and a group that survives
all three is a different kind of object than one that dissolves at 10. So runs are kept side
by side rather than overwritten, each carrying the parameters and the input commit that
produced it.

And the learning is the deliverable. `stats.py` is not a report tacked onto the end; if the
point is what the clusters teach, the cluster statistics and the condensed tree *are* the
result, and `discover.py` exists to feed them.

This is also why the labels are not ground truth here (see `project/ideas/01`): the premise is
that a vector carries more than a label does, so clusters are used to *study* the labelling
rather than be scored against it.

### The JPEG is the centre

Four principles. Most of the design decisions below follow from them, and a change that
contradicts one is probably wrong.

1. **The JPEG is what the project is about, and the thing that gets embedded.** Everything
   downstream — the vectors, the clustering, the species structure — is computed from pixels.
   So the JPEG is the unit of work and the key: the labeler walks `data/jpg`, one row per
   image, and the checkpoint and the embedding are keyed the same way.
2. **The sidecar is an acceptor, not an identity.** It is a place a run's label is *deposited*
   as a by-product — the real output is the curated `data/label/` — and it happens to be how
   Lightroom is reached. 5,229 JPEGs have no acceptor at all and are ordinary members of the
   population; a photo is not less real for having no XMP to carry its keyword. Nothing keys on
   a sidecar: a **key is a path relative to an agreed-upon root**, `data/jpg` downstream of
   labelling, which is what the CSV's `jpg` column, `processed.txt` and the embedding JSONL all
   hold.
3. **Filter the CSV; do not manipulate the Lightroom database by hand.** Deduplicating,
   re-scoping, or re-selecting is a query over a text file. Doing the equivalent in Lightroom
   means manual surgery and a re-extract, and it cannot be replayed or reviewed.
4. **Anything that conflicts with the JPEG-centric view is audited and flagged**, and fixed only
   where it matters. `./tool-audit` is where that lives: sidecars no JPEG reaches (0 today — the
   blind spot of walking the image tree), JPEGs claiming a sidecar another already claimed (307,
   resolved deterministically), JPEGs with no sidecar (5,229, expected).

### What a key is

**A key is a string that locates an item.** It has structure — slashes, a library, a trip, a
stem — but that structure is a *convention some consumers read*, not what the key is. Identity
is string identity: two keys are the same item when the strings are equal. Nothing derives
meaning from a key that it could not equally get from a column.

The string is a path relative to an agreed-upon root: `data/jpg` everywhere downstream of
labelling, which is what the CSV's `jpg` column, `output/label/processed.txt` and the embedding
JSONL all hold. The root is a convention between stages, not a lookup — `code/lib/path_filter.py`
compares keys without opening anything.

Two consumers do read the structure, and it is worth knowing which: `path_filter` splits on
`/` so a folder line can stand for its subtree, and `embed.py` derives `year` / `library` /
`trip` / `stem` into columns so the JSONL can be grouped without re-parsing. Both are
conveniences layered on the string, and neither is the key's identity.

### Stages, and who writes `data/`

> **The pipeline never writes `data/`. A person curates into it.**

Every stage reads from `data/`, writes to `output/`, and a human decides what graduates to
become the next stage's input. `data/` is not read-only because writing there is dangerous — it
is read-only *to the code*. "Only curated data is copied in" was always the rule; the curating
is the part done by hand.

```
                 reads            writes           curated by hand into
labelling        data/jpg         output/          data/label/
                 data/xmp
embedding        data/jpg         output/embed/    data/embed/     (when it exists)
                 data/label/
clustering       data/embed/      output/          —
```

`data/xmp` and `data/jpg` arrived exactly this way — a manual Lightroom export, copied in.
`data/label/` is the same act at the next boundary, so adding it does not weaken the rule; it
shows the rule was about *who writes*, not about whether anything is ever added.

**Curation is deliberately manual, and there is no script for it.** Choosing which run
graduates is the entire content of the step; automating it would make it routine, which is the
one property it must not have. `./clean` archives each run to `output_NNN_<description>/` so
the candidates survive to be chosen between; `cp` is the interface. `data/label/` is not a
renamed output folder — it is built by cherry-picking what is actually wanted, and a scripted
*transformation* of the picked data may be added later if a need appears.

Because `data/` is a submodule, anything curated in must be committed there and the pointer
bumped in the parent, or it exists only on one machine. Keep it small: a run's CSV is ~5 MB at
49k rows, but `output/label/raw` is 739 MB and does not belong in the dataset — the labelled sidecars
belong in the Lightroom library, which is where they are rsynced anyway.

**Why this matters more than tidiness.** `embed.py` currently takes its bird filter from
`labels.is_bird` — it reads the *sidecar keywords*. So today a category travels from labelling
to embedding like this:

```
label → output/label/raw/*.xmp → rsync → Lightroom → re-export → data/xmp → embed reads is_bird
```

A boolean makes a round trip through a foreign database to get between two stages of this
project, which contradicts principle 2 outright. `data/label/` short-circuits it:
`label → output_NNN → curate → data/label/ → embed`. The sidecar write still happens, for its
own reason — smart collections and browsing in Lightroom — it just stops being load-bearing.

**Resolve the effective category at curation time.** The run CSV's `category` is *this run's
verdict*; where `applied` is `kept-existing` the library actually keeps `prior_category`. If
`data/label/` carries the run CSV verbatim (for provenance — it is the only record of the
previous labels) *plus* a narrow derived selection with that already resolved, then `embed.py`
reads a two-column contract (`jpg`, `category`) and never needs to know the never-demote rule
exists. Leaving it unresolved means every future consumer re-implements it, and the first one
to write `category == 'bird'` silently drops the set the rule was written to protect.

A corollary about labels: **embedding does not depend on labels.** The premise is that a vector
carries more than a label does, so embeddings are used to *study* the labelling, not the reverse.
Labelling only scopes the set — it filters out the non-birds. Selecting the embedding set by
label therefore bounds what can be learned from it: label noise *within* the bird set is
measurable, false negatives are not, since they are excluded by construction. Embedding the whole
library is what would close that loop, and it is cheap (~453 MB, well under an hour of GPU
against ~11 hours to label). Left for when the requirements are clearer.

A **manifest** — an inclusion list naming the rows to embed, with room for collection or
predicate syntax so different studies can select different subsets — is the intended shape for
that selection. Deferred deliberately: a selector language designed before the first clusters
exist would be a guess. The one thing to keep forward-compatible is the **key**: the JSONL must
be keyed by the JPEG path relative to `data/jpg`, because re-keying stored vectors means
re-embedding, while any filter can be added later without touching a vector already written.

**Data flow:**
1. Input: `data/xmp/<Photos-YY>/<trip>/*.xmp` sidecars; the JPEG export **mirrors the same
   folder structure** at `data/jpg/<Photos-YY>/<trip>/*.jpg`.
2. `code/bird_label.py` copies the sidecar tree to `output/label/raw/` and works on the copy —
   **the pipeline never writes `data/`** (see Stages above). Labels therefore land in
   `output/label/raw/`, and getting them back into the Lightroom library is a separate, manual step.
3. It walks **`data/jpg`** — one work item per exported JPEG — and looks up the sidecar each
   one writes into (see JPEG-driven walk below)
4. Each backend sends a system prompt + base64-encoded JPEG to the model (vLLM/llama.cpp backends call an OpenAI-compatible HTTP server; chatgpt calls OpenAI directly)
5. Model returns JSON: `{category, label, label_cn, confidence}`
6. `code/lib/label_generator.py` formats a compact label with pinyin initials and confidence
7. A CSV row is appended and the checkpoint updated — **the CSV is the output of labelling**.
   A keyword is also deposited in the sidecar when the image has one, but that is a by-product
   for Lightroom's benefit, not the result: 5,229 images have no sidecar and are labelled just
   the same, and it is the curated CSV in `data/label/` that the embedding step reads

**The JPEG is the unit of work.** The walk is over `data/jpg`, not the sidecar tree: the JPEG is
what the model sees and what gets embedded downstream, and 5,229 exports have no sidecar at all
yet are ordinary members of the population. A sidecar is a *destination* for the label, not the
thing being enumerated. One row per JPEG — 49,270 today: 43,728 write to a sidecar, 5,542 to the
CSV alone.

**The claim rule** (`code/lib/jpg_claim.py`, `SidecarClaims.claim()`) is **local** — it looks only
at the JPEG's own name and the sidecar tree, never at another JPEG, so per-photo processing stays
independent. Exact stem first; otherwise strip trailing `-<decoration>` segments, longest base
first, and take the first sidecar that exists. Prefix-based, so a new decoration needs no code
change: `-2`/`-3` (Lightroom virtual copies), `-Enhanced-NR` (AI Denoise), `-Pano`, `-HDR`,
`-Edit`.

307 sidecars are claimed by more than one JPEG (`X.jpg` and `X-2.jpg` both reach `X.xmp`).
**First claimant in stem-sorted order wins, and that is always the exact match** — `X` is a proper
prefix of `X-2`, so it sorts first; verified for every one of the 307 groups. So this is not a
tie-break heuristic, it is the right answer reached without comparing candidates. The 313 later
claimants are alternate edits of a capture whose label already reached its sidecar; they keep
their CSV row as `csv-only`. Sort on the **stem**, never the filename: `X-2.jpg` sorts *before*
`X.jpg` ('-' is 0x2D, '.' is 0x2E), which would hand the sidecar to the virtual copy.

Claims are assigned over the whole sorted tree **before** the checkpoint filters anything, so
`build_items()` is a pure function of the two trees and a resumed run reproduces the same
assignment. Tracking claims only within one process would hand a sidecar to a virtual copy after
a Ctrl-C.

**The blind spot** of walking the image tree: a sidecar no JPEG reaches is not skipped with a
warning, it is never enumerated. That is 0 today — every one of the 43,728 is reached — but it is
a property of the current export, not an invariant, so `process_folder()` counts unreached
sidecars and warns. A partial re-export would otherwise drop those photos in silence.

**JPEG-only rows** — 5,229 exports never had a raw behind them (phone shots, in-camera JPEGs,
raws lost to a filing mistake). They have nowhere to carry a keyword, so their label goes to the
CSV alone as `applied=csv-only`. Do not assume they are non-birds: 607 come from
interchangeable-lens bodies (Nikon Z9/D500/D850/D5/Z8, Canon R5, Sony) and cluster in birding
trips — `2025-04-25.1 Birding Birds` alone holds 118.

**Never key anything by basename** — not the checkpoint, not the CSV, not a filter set. Stems
repeat across trips as camera counters wrap (10,832 of the sidecars share a basename with
another). `output/label/processed.txt` and the CSV's `jpg` column both hold the JPEG's path relative to
`data/jpg`; the CSV's `filename` column is kept for readability only. `--filter-csv` refuses a CSV
with no `jpg` column for the same reason.

## Clustering pipeline (branch `cluster`)

Downstream of the labeler: embed the identified bird photos and discover species
structure from appearance rather than trusting the VLM's per-photo guess. Theory in
`research/Bird Semantic Study Plan.md`; design in
`project/plans/2026-07-29-embed-cluster-stats.md`.

**Dataset layout** — `data/` is a git submodule holding the organised dataset. Only curated
data is copied in and tracked, and **the copying is done by hand**, never by the pipeline (see
Stages above). The Lightroom export mirrors the photo library exactly, so both trees have the
same shape:

```
data/xmp/<Photos-YY>/<trip>/*.xmp     # Photos-19/2019-01-13 山公园/_D8S0025.xmp
data/jpg/<Photos-YY>/<trip>/*.jpg     # same folder, same stem
data/jpg/export.report.txt            # what the exporter skipped, and why
data/label/                           # curated from a labelling run -- input to embedding
data/embed/                           # curated from an embedding run -- input to clustering
```

**The `data/` submodule is ~6.5 GB, and `jpg/` is nearly all of it.** That is deliberate, and
the obvious-looking economy — drop the JPEGs, they are only a Lightroom export — is wrong.

The export **is** the ground truth. It is the thing embedded, the thing clustered, and the
population every result is computed over; the library it came from is not in git at all. Nor is
it reliably re-derivable: re-exporting yields *an* export, not *the* export a set of vectors was
computed against. This project has already watched it move — a dedup round changed which files
exist, decorated names (`-Enhanced-NR`, `-2`) come and go, and a different Lightroom version or
export setting changes the pixels. A result whose population cannot be reconstructed is not
interpretable, which is the same argument that makes `manifests/` versioned, one level down: a
manifest *names* a population, the JPEG tree *is* one.

So the two provenance systems divide by what they are good at. **Restic snapshots the library**
(`project/bookeeping/`) — large, binary, and interesting mainly for *file movement*. **Git
versions the export** — the fixed substrate that results are anchored to. Neither substitutes
for the other. JPEGs are already compressed, so the pack is essentially the file bytes and no
amount of gc will shrink it; that is the cost of pinning the substrate, and it is worth paying.

(If a clone is ever unbearable, `git clone --filter=blob:none` or a shallow submodule fetch gets
the sidecars and history without the images. Removing them from history does not.)

`data/label/` is cherry-picked from an archived run (`output_NNN_<description>/`), not a
renamed copy of one. It exists so a category reaches the embedding step directly instead of
travelling out through Lightroom and back. `data/embed/` is the same act at the next boundary.

Both carry a `PROVENANCE.md`: what produced them, what was selected, and why they are versioned
at all. For `data/embed/` that last question has a different answer than for `data/jpg` — the
vectors *are* re-derivable, but re-derivable is not identical (kernel selection, batching and
library versions move the last decimal places), and run-to-run variation in the embedder is
itself a question that can only be asked against a stored run.

Trip folders match verbatim between the two photo trees, so resolving a JPEG is a lookup inside
one folder. Nothing is hardcoded per year — any dataset in this shape works.

**Scope is a manifest and nothing else.** `--years` was removed on 2026-08-09: it did the same
job through a second mechanism, and a manifest expresses everything it could (`--years
2024,2025` is a two-line file). One way to narrow means one place for it to be wrong. The
default is everything, which is the rsync convention — an empty include list means "no
restriction", not "nothing".

The one thing `--years` gave for free was refusing a selection that matched no library. A
manifest cannot do that, so **both stages now exit on an empty selection** rather than
reporting a finished run: a stage that does no work and succeeds is indistinguishable from a
stage that had nothing to do, which is exactly how a stale `--years 2019` default came to embed
a seventh of the library and call it done.

A full survey of the current dataset is in `project/reports/data_preview_report.md`,
regenerated by **`./tool-audit`** (read-only, ~12s). It also writes the worklists next to the
report: `missing_jpg.csv`, `extra_jpg.csv`, `unprocessed_sidecars.csv`, `multi_claim.csv`.

`multi_claim.csv` is the principle-4 check for the JPEG-driven walk: sidecars several JPEGs
reach (307 today, 313 JPEGs downgraded to `csv-only`). Nothing to repair — the resolution is
deterministic — but the count is a fingerprint of the export's shape, so a jump after a
re-export means something changed about how virtual copies were emitted. `exact_match_wins` is
false only where the master was never exported and every candidate is decorated (1 today).

**Two key spaces, deliberately.** The audit tools ask questions *about sidecars*, so their
worklists carry a `path` column — the sidecar's path relative to `data/xmp`. The labeler and the
embedding step ask questions *about images*, so they key on the JPEG's path relative to
`data/jpg`: the labeler's `jpg` column, `output/label/processed.txt`, and `embed.py`'s JSONL. The
labeler's CSV carries **both** (`jpg` and `xmp`, the latter empty where the JPEG has no sidecar),
so it is the bridge between the two spaces. Nothing is ever keyed by basename (see above).

Note `project/reports/sidecar_duplicates.csv` writes its `path` with the `data/xmp/` prefix
rather than relative to it — the one CSV deviating from the convention; strip it when joining.

`missing_jpg.csv` additionally carries `raw_name` (the
raw as the exporter named it — `X-Enhanced-NR.dng` and `X.nef` are different files on disk even
though one sidecar covers both) and `expected_jpg` (where the JPEG would have landed, so a
re-export can be checked). The worklists are written **UTF-8 with BOM**: trip folders are named
in Chinese and Excel reads a BOM-less UTF-8 CSV as the system codepage, mangling every one.
Read them back with `encoding="utf-8-sig"`.

**The two discrepancies the audit exists to find.** *Sidecar with no JPEG* — a photo the pipeline
cannot see, and since the labeler walks the image tree, one it never even enumerates. **0 of
43,728 today**, after the dataset reconciliation; it was 105 before. This is the number to watch
after any re-export. *JPEG with no sidecar* — 5,229, not a defect: the source was never a raw
file. These are labelled to the CSV alone rather than counted and skipped.

**Derived exports are not missing JPEGs.** 189 sidecars have no JPEG under their own name but
do have a decorated one in the same folder: `-2`/`-3` is a Lightroom **virtual copy** exported
under its copy name (the only export when the master was not in the export set), and
`-Enhanced-NR` is the **AI Denoise** render — a separate DNG carrying its metadata internally,
so it has no sidecar of its own. Both are the right photo. `JpgIndex` resolves a whole folder
at a time with exact hits claimed first, so where sidecars `A` and `A-2` both exist, `A-2.jpg`
goes to `A-2` rather than reading as `A`'s virtual copy. Treating these as missing would report
189 discrepancies that are not there. (`JpgIndex` answers sidecar → JPEG for the audit and the
embedding step; `jpg_claim.SidecarClaims` answers the reverse for the labeler.)

The report's filename never changes, so `git diff` after a run shows exactly what moved in the
dataset — keep it that way. The worklist CSVs are gitignored: they are large and fully
regenerated every run. `./tool-audit --snapshot <label>` additionally files a numbered copy in
`project/reports/archive/NN-data_preview_report-<label>.md` when a specific run is worth
keeping to compare against later.

**Where documents go:** `project/` holds everything about the work in progress — `plans/`,
`status/` handoffs, `reports/` (analysis output and worklists), `messages/` (correspondence
with the user, named `YYYY-MM-DD.NN <who>:-<topic>.md`; reply by filling in the placeholder
file they leave), `bookeeping/` (pointers to state held outside this repo), `ideas/`. `docs/` is
reserved for product documentation, i.e. output meant for whoever uses the result rather than
notes about building it.

**`project/ideas/` is not `plans/`.** A plan is a commitment to an approach; an idea is a
thought that surfaced while doing something else and would be lost by the time the project is
deep enough to act on it — most arrive mid-task, with no room to chase them. Write one down
rather than carrying it: what it is, why it looked promising *at the time*, what would have to
be true, and what it would cost — but do not invent a design it does not have. Promote to
`plans/` if it becomes a commitment; keep the ones that turn out wrong, annotated, since a road
looked at and rejected is worth as much as the ones taken.

**The library's own provenance is restic**, recorded in `project/bookeeping/`. The Lightroom
library is far too large and too binary for git, but snapshots make *file movement* — renames,
re-filings, deletions across dedup rounds — recoverable and diffable, which is the class of
change that has repeatedly caused trouble here. `data/xmp` versions the **curated** sidecars;
restic versions the **library**. Neither substitutes for the other, and a snapshot taken before
a dedup will legitimately hold sidecars `data/xmp` no longer has.

**Where run artifacts go:** `output/` is the live run and is gitignored. `./clean` archives it
to `output_NNN_<description>/`, also gitignored — every past run stays on disk, nothing is
deleted. `data/label/` is the small, hand-picked subset promoted from one of those archives to
serve as the next stage's input, and it is versioned in the submodule. The distinction is
deliberate: archives are *everything a run produced*, `data/label/` is *what was chosen*.

**Sidecar deduplication** (`code/lib/sidecar_meta.py`, `tools/dedup_sidecars.py`,
**`./tool-dedup`**) — operates on `data/xmp`. The same photo was imported into
two trip folders in places, which is what made neighbouring trips look like frame-counter
collisions during splitting. Two sidecars are one capture when they share
`xmpMM:OriginalDocumentID`; **do not compare `exif:DateTimeOriginal` as a string** — only ~10%
of sidecars carry sub-second precision, timezone corrections move a capture across calendar
days, and import clashes rename one copy, so the literal date+time+filename rule finds just
265 of 534 redundant sidecars. Lightroom virtual copies (`-2`, `-3` stems) share an original
but have their own `xmpMM:DocumentID` — deliberate alternate edits, reported separately and
never offered for deletion. Report: `project/reports/sidecar_dedup_report.md`.

`./tool-dedup` **reports; it never deletes.** Real deduplication has to happen in Lightroom, the
database of record — acting on the extracted files would be undone by the next export. The
worklist therefore records a *proposal*, not an action, which is why it is gitignored and the
report is overwritten: whatever was actually decided shows up in `data/` on the next bump.

A first pass was cleared on 2026-08-02, but the reconciliation that followed reintroduced some:
as of 2026-08-06 the report shows **45 cross-trip duplicates** (the same capture filed in two
trips, sharing both `OriginalDocumentID` and `DocumentID`), 0 within-trip, and 5 virtual copies.
Deliberately left in place — at 45 of 43,728 they are not worth manual surgery in Lightroom, and
the labeler will simply produce two rows for those captures. Deduplicate on the CSV when it
matters; the clustering step has to dedupe identical vectors regardless.

**Wraparound splitting** (`tools/analyze_wraparound.py`) — **superseded**, and its `./run-split` wrapper was deleted 2026-08-11. The script stays; recover the wrapper from history if it is ever wanted.
It proposed cutting each year into date-contiguous segments whose photo filenames are unique,
so a *flat* jpg export folder could be keyed by filename. The Windows re-export mirrors the
library's folder structure instead, so filename collisions no longer matter and nothing
downstream consumes `split_plan.csv`. Kept only as the way to answer "which frame numbers
repeat, and where". Method: `docs/wraparound-splitting.md`.

**Deciding whether a filename collision is a duplicate** (`code/lib/trips.py`): trip folders
are named `YYYY-MM-DD <place>`, so sorting them gives a timeline. A frame counter cannot
legitimately repeat within one trip, so the same frame number in *neighbouring* trips is a
candidate duplicate — one photo filed twice — while a collision far apart in time is ordinary
wraparound. `frame_id()` collapses the filename decorations first, so `20190113-_D8S0025-2`
and `_D8S0025` count as one frame. This is a *prefilter only* — 16 of 227 candidates sat in
neighbouring trips, some the same day, yet were demonstrably different photos, so confirm with
the pixels before acting. No longer used by the audit (the mirrored export removed the
question); still the right tool if cross-trip duplication comes up again.

**JPEG resolution** (`code/lib/jpg_index.py`, `JpgIndex`) — shared by the labeler, the audit
and the embedding step. `JpgIndex(xmp_dir, jpg_dir).resolve(xmp)` returns a `JpgMatch` with a
verdict of `ok` (exact stem in the mirrored folder), `derived` (only a decorated export
exists — see above), `no_jpg` (folder exported, this photo was not) or `no_folder` (no
exported folder mirrors this trip). `extras()` reports the other direction: JPEGs no sidecar
claims, split into `orphan` (never a raw file) and `derived` (an additional virtual copy of a
sidecar that already matched).

The old `MatchPolicy` (`expected` / `same_year` / `any`) is **gone**. It existed to arbitrate
whole-tree stem fallbacks against a flat export, where a cross-year stem match returned a photo
from an unrelated shoot. With the folder in the key there is nothing to arbitrate.

Earlier pixel-hashing over that flat export found the embedding set contained exact duplicates
(60 groups / 117 rows in 2019's 3,860). Since HDBSCAN is density-based, **the clustering step
must still dedupe identical vectors first** — a capture exported more than once, or genuinely
repeated frames, will otherwise distort local density. Duplicate sidecars sometimes disagree on
species for identical pixels, which is a useful direct measure of VLM label noise.

**Steps:**
1. `code/embedding/embed_server.py` — DINOv3 on the GPU host, `POST /embed` (base64 JPEGs →
   L2-normalised vectors, so cosine distance == euclidean downstream). `./server-embed`;
   needs `HF_TOKEN` with the gated DINOv3 licence **accepted for the account** — a valid token
   alone gives a 403 on file fetches, and `model_info()` succeeds regardless since gated repos
   expose metadata publicly. On this box `~/.cache/huggingface/hub/.locks` is root-owned, so
   `server-embed` falls back to a private `HF_HUB_CACHE`; fix with
   `sudo chown -R "$USER" ~/.cache/huggingface/hub/.locks`.
2. `code/embedding/embed.py` — non-GPU client; POSTs batches and appends to
   `output/embed/embeddings.jsonl`. `./run-embed`. Resumable (checkpoint is the set of `key`s
   already in the JSONL) and `--dry-run` does the whole scan/resolve with no network calls.
   Deferred SIGINT, same as `bird_label.py`. Every row records the producing `model`, and the
   client **refuses to append when the server serves a different one** — two backbones' vectors
   are not in the same space and nothing downstream could detect the mixture. Use a fresh
   `--output-dir` to switch models.

   **The CSV is the guide; nothing here reads a sidecar.** The label set comes from
   `--label-dir`'s `bird_identification_output.csv` (default `data/label/`), one row per
   exported image, and the `jpg` column names the file relative to `data/jpg`. That is what lets
   the **569 birds with no sidecar** be embedded at all — a sidecar walk cannot enumerate them.
   It also stops a second resolver re-deriving which file backs each capture, which the
   labelling run already recorded.

   **The JSONL `key` is the image path relative to `data/jpg`** — the only identifier spanning
   sidecar-backed and sidecar-less images. Any JSONL written before this change is keyed by
   sidecar path: regenerate rather than resume. `xmp` rides along as a column so alternate edits
   of one capture stay groupable.

   **`effective_category()` and `effective_species()` resolve the never-demote rule here, once.**
   The CSV's `category` is *this run's* verdict; where `applied` is `kept-existing` the library
   kept `prior_category`, and the species is then in `prior_label`, not `label`. Getting the
   category right but not the label is the worse bug of the two — it silently contributes 294
   scene descriptions to the species vocabulary. Effective bird set: **27,742**.

   A caveat that falls out of the rule: 554 never-demoted rows enter the set, and 398 of them are
   ones this run called `animal`. Some are this run's own miscategorisations (`great horned owl`,
   `lesser rhea` — birds called `animal`), but others are the *old* pipeline's over-calls kept
   alive: `common myna` over a wombat, `silver gull` over a husky, `western grebe` over a black
   bear. ~1.4% of the embedding set. Left in deliberately — clustering isolating them is a test
   of the premise, not a failure — but do not read those species labels as ground truth.

   **Scope is a list, not an arrangement** (`code/lib/path_filter.py`; lists live in
   `manifests/`, versioned because a `run.json` records the *path* to one and a dangling
   reference makes the scope irreproducible).
   `--include-from` / `--exclude-from` take a path that must land inside `manifests/` once
   resolved — checked on the resolved location, not the leading segment, so `manifests/../x`
   is refused rather than merely looking compliant. `manifests/exclude-captive.txt` is the form
   worth typing (it tab-completes, and matches disk and `args.json`); a bare name is read as
   manifest-relative, and subdirectories work either way. Where scope lists live is not the
   caller's choice: a list read from `/tmp` makes a `run.json` a dangling reference and the
   population behind a result unrecoverable. The file holds **paths, not patterns** — no globs, no regex — relative to
   `data/jpg`, one per line, `#` comments allowed. A line is either a key or
   a folder standing for every key beneath it, at any depth (`a/b/c` is an ordinary folder line,
   and a parent takes nested children with it). Comparison is on whole path segments, so
   `Photos-2` does not swallow `Photos-24`. Exclude wins over include. The
   alternative — moving folders out of `data/` so a run cannot see them — mutates the dataset
   for every consumer, leaves no record of what was excluded, cannot express two scopes at
   once, and is a submodule commit each time. Both paths are copied into `run.json`, so a
   result carries the scope that produced it. This is the *mini* manifest, deliberately: no
   predicates, no set algebra. `manifests/exclude-captive.txt` drops 12 zoo and aviary trips
   (1,090 birds) — a collection's species mix is an artefact of the collection rather than a
   place or season, and the enclosure is a background the model can learn instead of the bird.
   It deliberately keeps the Safari "Safari" trips, which are a waterhole in the park National
   Park and therefore wild; a keyword match on `safari` would have dropped 143 wild wild
   records, which is the argument for writing these by hand and commenting them.

   Output is a single flat `output/embed/embeddings.jsonl` plus `run.json`; the folder structure
   survives only as the `year` / `library` / `trip` / `stem` columns. ~255 MB for the bird set at
   768 dims, against ~85 MB as float32 — the price of being appendable and inspectable, which is
   what makes resumption and the single-model check cheap. If load time hurts, cache a derived
   `.npy` beside it rather than changing the write format.
3. `code/cluster/discover.py` / `stats.py` — HDBSCAN + cluster statistics. Not yet written.

Environment: `./venv` takes stage arguments so a non-GPU box need not pull torch —
`./venv base client test cluster`, and `./venv server` adds torch/transformers/fastapi.
Tests run from within `test/` (`../.venv/bin/python -m pytest lib/`). `test/conftest.py`
exists because the project package is named `code`, which shadows the stdlib module of the
same name once pytest preloads it.

**Outputs** (in `output/label/`). Each stage writes its own subdirectory of the
run root -- `output/label/`, `output/embed/`, `output/cluster/` -- so one `./clean`
archives a whole pipeline pass together. Labelling used to write the run root
directly, which put its `raw/`, `args.json` and checkpoint beside the other
stages' folders. The CSV is *the* output; `raw/` is a by-product:
- `bird_identification_output.csv` — the result (jpg, xmp, filename, category, label, label_cn, confidence, note, prior_category, prior_label, applied, run_label, response_json). **`jpg` is the key** — the JPEG's path relative to `data/jpg`, one row per image. `xmp` is the sidecar the label went into, relative to `data/xmp`, empty when the image has none. `filename` is a bare basename kept for readability and must never be used to look a row up. `category` is its own column — `note` still embeds it as `"bird (0.90)"`, but parse the column, not the string. **`prior_category` / `prior_label` hold the label the previous run left**, mostly early paid GPT-4o — this CSV is the *only* record of it, so don't discard old CSVs. `applied` says what reached the sidecar: `written`, `kept-existing` (a non-bird result deferring to the prior category), `csv-only` (no sidecar to write — either the image never had a raw, or its capture's sidecar went to the exact-stem export) or `failed`.
  - **`category` is this run's verdict, not the library's state.** Where `applied` is `kept-existing`, the sidecar keeps `prior_category` and the new verdict was overruled. Anything filtering on the *effective* label — e.g. picking the bird set to embed — must read `'bird' in (category, prior_category)`, lowercased, or it silently drops every photo the never-demote rule was written to protect.
  - A schema change makes the CSV un-appendable; `check_csv_schema()` refuses to resume against a mismatched header rather than shifting every new row one field left.
- `args.json` — CLI arguments for reproducibility
- `processed.txt` — checkpoint, one `jpg` key per line; delete to reprocess all images. Written per batch, *after* the CSV is flushed, so a hard kill can never mark a photo done without a row
- `raw/` — the working copy of `data/xmp`, with keywords deposited into it. A by-product, not the result: it exists so Lightroom can show the label beside the photo and drive smart collections. Getting it back into the library is a separate manual step, and `data/` is never written to by the pipeline. Nothing downstream reads it — the embedding step reads the CSV

**Backend modules** (all in `code/bird_label.py`):
- `predict_with_vllm()` / `predict_with_vllm_batch()` — vLLM server via OpenAI-compatible API (`--vllm-url`); batch mode fires concurrent HTTP requests via a thread pool so the server's continuous batching handles them together
- `predict_with_gpt4o()` — OpenAI cloud API
- `predict_with_llamacpp()` — llama.cpp OpenAI-compatible server

**Supporting library** (`code/lib/`):
- `label_generator.py` — formats `{pinyin_initials}-{chinese_name}-{english_name}({confidence}%)` labels
- `jpg_claim.py` — which sidecar a JPEG's label writes into; the local claim rule and its ordering
- `xmp_write.py` — sets keywords by editing the sidecar text, so the diff stays reviewable
- `transformers_engine.py` — HuggingFace Transformers backend, unreachable: `transformer` is not one of `--approach`'s choices, and the `run-tf` wrapper that pretended otherwise was deleted 2026-08-11 rather than wired up

## Re-labelling an already-labelled library

`./clean && ./run-label` relabels everything — the checkpoint is what causes skipping, and
`./clean` moves it aside with the rest of `output/`. Nothing keys off whether a sidecar already
carries a label.

**Sidecars must stay clean**: they go back into Lightroom, so no archive property, no versioned
keywords, nothing Lightroom would show in its keyword list. The previous label is preserved in
the CSV's `prior_category` / `prior_label` columns instead, and `data/xmp` (never written by the
pipeline, versioned) still holds every original.

**The labeller records; it does not arbitrate.** `set_keywords_in_xmp(xmp, category, label)`
writes this run's verdict unconditionally. Whatever was there goes to the CSV's
`prior_category` / `prior_label` and is then replaced. No result is discarded on the grounds
that an older one looked better — that judgement belongs to whatever consumes the output, which
has the whole population in front of it and a question to answer. Neither is true of a function
looking at one sidecar.

**Two guards were tried here and both inverted**, which is why there is now none.

- **`bird` was protected** so a fresh false negative could not drop a photo from the clustering
  set. Of the 554 rows it held, 66 are provably a same-stem twin's label from the basename-keyed
  era — a wombat carrying `common myna` because a Sydney photo with the same stem
  really is one (`tools/stale_bird_labels.py` buckets them by evidence). Given up 2026-08-09.
- **`scenery` was protected** because the early GPT-4o pass wrote specific descriptions where a
  fresh run wrote a bare `scenery`. By the time it was measured across its 2,401 rows the
  current prompt was asking for the landmark *by name*, so the guard was keeping generic text
  over identifying text: "historic building with columns and flag" kept over "brisbane city hall
  facade". It was also blocking 363 category corrections. Given up the same day.

The pattern is the lesson: a guard encodes an assumption about the labeller it was written for,
the labeller improves, and the guard silently starts preserving the worse answer. It cannot
notice, because it compares categories rather than quality.

**This makes the pipeline a model-comparison instrument.** Run a second model over a tree the
first one labelled and every row carries both verdicts side by side — `category`/`label` against
`prior_category`/`prior_label` — a paired comparison over the whole library, for free, with no
arbitration baked in. `prior_labels()` reads from `PRISTINE_XMP_DIR`, so pointing a run's
`--data-dir` at a tree curated from a previous run's `raw/` is what sets the first model up as
the "existing" one.

`applied` is now only `written`, `csv-only` or `failed`. **`kept-existing` is still produced by
nothing but understood by everything** — `data/label` and every archived run carry 3,693 such
rows, and `embed.py`'s `effective_category()` keys off `applied` precisely so it is a no-op on
new CSVs and load-bearing on old ones.

**Re-running part of a library:** a partial run's `raw/` is a *full* copy of `data/xmp` with only
the selected years touched, so merging it back is not a directory copy — taking `raw/` wholesale
would replace every other year's labelled sidecar with an unlabelled one.
`tools/merge_label_run.py` takes a sidecar from the overlay only where the overlay's CSV has a
row for it, and writes a new label directory rather than editing either input.

`set_keywords_in_xmp()` preserves the user's own keywords and is idempotent on re-run. Four
things it has to get right:

- **The rest of the file must not move** (`code/lib/xmp_write.py`) — the sidecar is read with
  the XML parser but edited as *text*. Parse → mutate → `tree.write()` is correct XML and still
  rewrites every line: ElementTree keeps no formatting, so Lightroom's one-attribute-per-line
  layout collapses, `xmlns:` declarations get hoisted to the root and sorted, and any prefix not
  passed to `register_namespace` is renamed (`x:xmpmeta` → `ns0:xmpmeta`). A 348-line sidecar came
  back as 120 and `diff` showed the whole file changed. Registering the source's own prefixes
  fixes the names but not the reflow — the loss is inherent to the round-trip. Since these files
  are rsynced back into the Lightroom library, an unreadable diff is an unreviewable change.
  Text surgery makes a `dc:subject` insertion a 6-line diff. `verify_only_keywords_changed()`
  re-parses every edit and asserts nothing outside the keywords moved, so the shortcut fails loudly
  rather than corrupting a file; a failure is recorded as `applied=failed`. 1,749 sidecars in
  `data/xmp` still carry the old `ns0:` prefixes from before this change.
- **`rdf:Bag` vs `rdf:Seq`** — Lightroom writes `dc:subject` as a Bag (17,590 sidecars), earlier
  runs of this script wrote a Seq (1,748). Read either; reuse whichever is present. Adding a
  second container under one `dc:subject` makes the keyword list depend on which the reader picks.
  (One sidecar has both and needs fixing by hand.)
- **`lr:hierarchicalSubject` holds keyword *paths*, not a flat mirror** — `People|Family|Miles` is
  one keyword naming a position in Lightroom's keyword tree. An earlier version mirrored the
  flattened `dc:subject` list into it, which would have turned that into three unrelated top-level
  keywords on import; 71 sidecars carry such paths, 12 of them with no category to defer to, so a
  full run would have stripped them. `merge_hierarchical()` replaces only entries whose *leaf* is
  one of ours, leaving the user's paths intact. It is never created where absent — a stale mirror
  resurrects old keywords on import, but an absent one is not a gap to fill.
- **Which keywords are ours** (`split_keywords()` in `code/lib/xmp_labels.py`) — three
  generations exist: bare categories, `py-cn-en(NN%)` labels, and early GPT-4o free text with no
  confidence at all (`mountain landscape with glacier`). Note the `(NN%)` suffix is the *only*
  common marker: scenery/people labels are a bare description, so `LABEL_RE` does not match them.
  Early free text is claimed on two signals together — the sidecar carries a category (only this
  pipeline writes one) *and* the text is a descriptive phrase. Single tokens (`Family`,
  `Rivertown`) and the `HAND_WRITTEN_RE` species shape (`xs-小隼-Kestrel`, `qq-昵称`) are always
  the user's. Err towards leaving a stale keyword, never towards deleting hand-written work.

`prior_labels()` reads the previous label from `data/xmp`, **not** from the working copy —
otherwise a resumed run records its own fresh labels as the prior ones.

## Key Behaviors

- **Checkpoint resumption:** Already-processed filenames in `output/label/processed.txt` are skipped on re-run; CSV is opened in append mode so prior results are preserved
- **XMP permissions:** Script calls `chmod` on XMP files before writing keywords (needed for library-managed files)
- **JSON parsing:** Model may wrap response in markdown fences or return Chinese in the `label` field; post-processing strips fences and moves Chinese characters from `label` to `label_cn` automatically
- **vLLM prompt constraints:** `category: "bird"` is strictly class Aves; insects/butterflies must be `animal`. `label` must use Latin characters only; `label_cn` is Mandarin. For `scenery`, the prompt asks for the specific subject of the scene (landmark, landscape feature, or activity — e.g. "Sunset over Lofoten fjord") rather than a generic description.
- **Filter mode:** `--filter-csv` reads a prior output CSV and only reprocesses rows where `note` contains "animal" or confidence is below threshold
- **Graceful Ctrl-C:** SIGINT is intercepted with a deferred handler — the signal sets a flag rather than raising immediately, so the current batch always completes fully (inference, XMP write, CSV write, checkpoint) before the loop exits. This synchronises the async signal with the program's batch boundary, ensuring no partial writes.
