# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Automated bird species identification for photo libraries. Processes JPEG images through a vision-language model to identify birds, generate English/Chinese species names with confidence scores, update XMP sidecar metadata, and produce CSV result files.

## Running the Labeler

Set up environment first:
```bash
cp _env .env
# Add OPENAI_API_KEY to .env if using the openai approach
```

`config.toml` also carries **`current_working_output`** — the run directory the analysis tools
work on by default. Runs get archived under new names (`./clean` moves `output/` to
`output_NNN_<description>/`), so a post-processor that hardcoded `./output` would analyse
whatever happened to be sitting there. Point the setting at an archive to re-analyse an old
run and every tool follows without a flag; `--output-dir` still wins per invocation.

Non-secret configuration lives in `config.toml` at the repo root, checked into git — see
`[servers.*]`, read via `code/lib/config.py:server_url()`. **Everything tracked there must be
a value that works for a stranger**, so the hosts say `localhost`; they used to name the
author's own machines, which meant a public clone either failed to resolve them or
reached somebody else's machine, and the only fix was editing a tracked file.

Machine-specific values go in **`config.local.toml`** (gitignored, copy `_config.local.toml`),
layered over `config.toml` recursively — naming a host alone keeps the tracked port. It holds
**everything local except secrets**: the `[servers.*]` hosts and `data_dir`. A `.datapath` file
did the dataset half for a day and was folded in on 2026-08-11 — two mechanisms for "local
settings" is exactly the second mechanism this project keeps removing.

`.env` stays separate for a reason that is not tidiness: the run scripts `source` it as shell,
which a TOML file cannot be, and a leaked API key is a different problem from a leaked
hostname. So: two ignored files, one boundary, and nobody edits a versioned file to point the
project at their own machine.

**vLLM is the only backend fast enough for a real run.** `openai` and
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

**The root is the front door, and stays short.** A wrapper there is one of the
handful of operations you would name to describe what this project does. Anything
else lives in `tools/` and is invoked as `python3 -m tools.$name` — that is the
home for a script with real logic that is not a project-level operation, and it
needs no wrapper to be usable. `tool-embed-server` existed for about an hour
before being deleted on those grounds: asking a server what it has loaded is a
diagnostic, not a stage, and every wrapper added for convenience makes `ls run-*
tool-* server-*` a worse answer to "what can I run here?".

Below that, **`local/` is a list of commands in order** — read, and often run a
line at a time rather than as a batch. Control flow does not belong there: it is
the one directory that is gitignored, untested and unshared, so a program living
in it is a program nobody can review.

Two things that do belong. **The environment**, in full — `PROJECT_ROOT`, `cd`,
`source .venv/bin/activate`, `.env` where a secret is needed — because there is
no wrapper underneath doing it and the script will be run from anywhere. And
**defaults, written as visible parameters**:

```bash
SIZE="${SIZE:-1024}"                    # square resolution to embed at
MCS="${MCS:-3,5,8,15,40}"               # the cluster sweep
```

A default at the top is not logic; it is documentation of the knob. It says both
what the script does today and exactly what to set for a run that differs —
`SIZE=512 ./local/embed-resolutions` — which is the thing a bare command list
cannot tell you.

**Only the main parameters**, though. A variable for every argument turns the
list back into a program and buries the one or two things actually worth
changing. If it does not vary between runs, write it inline where it is used.

The backend is a flag, not a script:
`run-gpt` / `run-cpp` / `run-tf` were four wrappers differing only in
`--approach`, which put a deployment detail in the command name and let
`run-tf` sit there broken (its approach was never a valid choice). Deleted
2026-08-11.

Run via convenience scripts or directly:
```bash
./run-label                       # vLLM server, host from config.toml [servers.vllm]
./run-label --approach openai     # a hosted OpenAI-protocol API (needs OPENAI_API_KEY)
./run-label --approach llama.cpp  # llama.cpp server, host from [servers.llama_cpp]

# Or directly with options:
python3 -m code.bird_label --approach vllm --model "Qwen/Qwen3-VL-132B-Instruct" --conf-threshold 0.6
```

Key CLI flags:
- `--approach` — `openai`, `llama.cpp` or `vllm`. Named for the *protocol*, not a
  product: all three speak the OpenAI chat API over **one** transport and one
  prompt, and `$OPENAI_BASE_URL` points the hosted one at any endpoint that
  speaks it (Azure, a gateway, a proxy) — which is also what lets it be tested
  end to end against a stub. Renamed from `chatgpt` on 2026-09-04 with no alias,
  costing nothing: it had raised `NameError` on the first image of every run
  since the module split, so no working script could have used the old name.
  That single transport is the fix for what broke it — the cloud path kept its
  own copy, which drifted until it carried a prompt from before `bird` meant
  class Aves, a `max_tokens` too small for the JSON now asked for, and
  `json.JSONDecode_decodeError`, an attribute that does not exist, so every
  code-fenced reply silently became `scenery/unknown`. The **model for `openai`
  comes from `config.toml` `[models]`**, not a literal — models get retired, and
  a name buried in code is one nobody edits until a run fails. `vllm` and
  `llama.cpp` are deliberately absent from that table: they probe their server,
  and a configured name would be a second unchecked source of the same fact. Not
  `transformer` either — `transformers_engine.py` exists but was never added to
  the choices, so `--approach transformer` has always been rejected by argparse
- `--vllm-url URL` — vLLM OpenAI-compatible server endpoint (default: `config.toml` `[servers.vllm]`, vllm approach only)
- `--conf-threshold FLOAT` — confidence below which to flag as low-confidence (default 0.6)
- `--no-bird FLOAT` — confidence below which to mark as "no bird" (default 0.2)
- `--filter-csv PATH` — re-process only "animal" or low-confidence rows from a prior run's CSV. The rule is hardcoded; to select on anything else, generate a manifest (below)
- `--prior-labels DIR` — where `prior_category` / `prior_label` come from (default `data/label`). **The CSV, not the sidecars**: a sidecar exists only for a photo that had a raw, so the sidecar route covered 9,918 of 49,224 rows and never the 5,229 with no raw at all — meaning the paired-verdict comparison, which is the point of the design, was a comparison of nothing over a fifth of the library. Fixed 2026-09-04; sidecars remain the fallback when no such CSV exists
- `--run-label TEXT` — tag this run in the output CSV
- `--batch-size INT` — number of images processed concurrently against the vLLM server (default 1; 8 is a reasonable default — each unit is a concurrent HTTP request, the server does its own continuous batching)
- `--include-from` / `--exclude-from PATH` — a manifest under `manifests/`; `local/manifests/exclude-captive.txt` is the form to type, since it tab-completes and matches what is on disk. A bare name works too. Same mechanism and same keys as `embed.py`, so one list scopes both stages
- `--limit INT` — stop after N images, matching `embed.py`. **Not a scoping mechanism**: which images you get depends on walk order, so a run narrowed this way cannot be described or reproduced, and `--include-from` remains the only way to name a population. It exists so a paid backend can be tried for five API calls instead of 47,908. Applied after the scope and the checkpoint, and before the dry-run report, so `--dry-run --limit 5` says 5
- `--dry-run` — resolve the scope and report it, then stop. No model probe, no sidecar copy, no writes

**A run that labels nothing exits non-zero**, and a photo the model could not answer
for gets no row, no sidecar edit and **no checkpoint entry** — so a re-run retries it.
Both were fixed 2026-09-04. Every per-image failure is caught and logged, and the
caller used to print "Run complete" regardless: that is how the `openai` backend
stayed broken for months, raising on the first image of every run and reporting
success each time. Worse, a transport failure returned the predictor's
`scenery/unknown/0.00` defaults, which read as a *verdict* rather than a gap — one
dead server or timeout became a permanent wrong label on a real bird. The
predictors now put the error in `response_json` under `_prediction_failed` and the
loop skips those rows entirely.

**A reasoning model needs room to answer, and asking for less thinking.**
`max_completion_tokens` covers reasoning *plus* output, and reasoning comes
first: at the old `max_tokens: 200` budget a GPT-5 run spent the whole allowance
thinking and returned an empty string with `finish_reason: "length"` — which
surfaced as "No JSON found" over a blank response and sent the reader hunting a
parsing bug that was not there. 18 of 20 images failed that way. Raising the
budget alone would be the wrong fix, since reasoning tokens are **billed** and a
generous budget across 49k images is money spent on deliberation nobody reads. So
the same adaptation that renames the parameter also sets `REASONING_BUDGET` and
`reasoning_effort=minimal`, and an empty reply now names the token limit and the
reasoning tokens consumed rather than blaming the parser.

**The cloud backend adapts to what a model will accept.** GPT-5 and the o-series
reject `max_tokens` for `max_completion_tokens` and reject a non-default
`temperature`; a hardcoded list of which model wants which spelling is a list that
goes stale, and the failure is a 400 nobody sees until a paid run dies. So the 400
is read and the payload corrected — once per endpoint and model, cached for the run,
looping until accepted so the *first* image succeeds. Same spirit as probing a vLLM
server rather than assuming what it serves.

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
result, and `cluster.py` exists to feed them.

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
clustering       output/embed/    output/          —
                 (data/embed/ by name)
```

`data/xmp` and `data/jpg` arrived exactly this way — a manual Lightroom export, copied in.
`data/label/` is the same act at the next boundary, so adding it does not weaken the rule; it
shows the rule was about *who writes*, not about whether anything is ever added.

**Clustering is the one stage that reads `output/` by default, and deliberately.** It used to
default to `data/embed/`, from when there was only ever one embedding set. That made a fresh
embedding run impossible to cluster without first curating it into `data/` — promoting a run
in order to find out whether it deserved promoting, when the deciding *is* the curation. So
`./run-cluster` looks at the live run and the curated set is a path you type. The rule it
does not break is the one that matters: nothing here writes `data/`.

The analysis tools go further and do not default to a path at all
(`code/lib/csv_post.py`, `embeddings_for()`): each reads whatever the run it is describing
recorded as its `source`, falling back to the `embed/` beside that run's own root when
`./clean` has moved the archive and stranded the absolute path. A fixed default cannot
promise that the vectors being read are the ones the clustering was computed from, and with
runs at several resolutions the wrong ones produce a description that looks entirely
reasonable.

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
4. Each backend sends a system prompt + base64-encoded JPEG to the model (all three speak the OpenAI chat protocol over one transport; `openai` adds an Authorization header and defaults to the hosted API)
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
trips — one birding trip alone holds 118.

**Never key anything by basename** — not the checkpoint, not the CSV, not a filter set. Stems
repeat across trips as camera counters wrap (10,832 of the sidecars share a basename with
another). `output/label/processed.txt` and the CSV's `jpg` column both hold the JPEG's path relative to
`data/jpg`; the CSV's `filename` column is kept for readability only. `--filter-csv` refuses a CSV
with no `jpg` column for the same reason.

## Clustering pipeline

Downstream of the labeler: embed the identified bird photos and discover species
structure from appearance rather than trusting the VLM's per-photo guess. Theory in
`research/Bird Semantic Study Plan.md`; design in
`project/plans/2026-07-29-embed-cluster-stats.md`.

**Dataset layout** — `data/` is a git submodule holding the organised dataset. Only curated
data is copied in and tracked, and **the copying is done by hand**, never by the pipeline (see
Stages above). The Lightroom export mirrors the photo library exactly, so both trees have the
same shape:

```
data/xmp/<Photos-YY>/<trip>/*.xmp     # Photos-19/<YYYY-MM-DD trip>/_D8S0025.xmp
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

**`_local/` is the tracked template for it** — `cp -r _local local`, the same shape as `_env`
→ `.env`. It carries `runbook`, a full pass written as one coherent script and deliberately
**not executable**: every step has something worth reading before the next starts, and a
runbook that can be executed whole is one that will be. Its defaults are the project's, so it
runs as shipped and the point is that you change them. `docs/running-a-study.md` explains why
each step is there; the runbook is the steps, and neither repeats the other.

**`local/` is everything local, and gitignored.** Added 2026-08-12: ad-hoc run
commands for whatever is being tried this week, `local/manifests/` for scope
lists that name real trips, and working copies of the generated reports. None of
it is permanent or shareable — an ad-hoc command is stale within days, and a
manifest listing `Photos-16/2016-07-12 <zoo>` publishes a travel history.
Scripts here should open with `cd "$(dirname "$0")/.."` so they act on the repo
root whatever directory they are invoked from, the same idiom the `run-`
wrappers use.

**A manifest can therefore be in one of two places**, and `resolve()` accepts
both: `manifests/` for lists that mean the same to anyone (the `inclusion-*.txt`
name only libraries), `local/manifests/` for lists that name places. Both are
inside the repository, which is the part worth keeping — a list read from `/tmp`
makes a run's recorded scope a dangling reference. The cost is accepted: a
`run.json` naming a `local/` manifest is reproducible only for the person who
has it.

**Selecting by a previous run's category** — `tools/manifest_from_labels.py`
writes a manifest naming the images a labelling put in a category, resolving the
never-demote rule the way `embed.py` does. It exists as a *generator* rather than
a `--category` flag so that scope stays one mechanism: the output is an ordinary
manifest, recorded in `run.json`, diffable and reproducible. The category is the
part of a labelling worth trusting — two independent labellers agree on it 0.9626
against 0.288 on species — and this is where that difference is worth spending:
`--category bird` selects 27,194 of 49,224, so a paid run skips 45% of its cost
without touching a bird.

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

The report's filename never changes, so `diff` between runs shows exactly what moved in the
dataset. It is **no longer tracked** (2026-08-12): it is trip-by-trip counts of a private
library, and it described the data rather than the software. Compare against a copy you keep
in `local/` instead. The worklist CSVs are gitignored: they are large and fully
regenerated every run. `./tool-audit --snapshot <label>` additionally files a numbered copy in
`project/reports/archive/NN-data_preview_report-<label>.md` when a specific run is worth
keeping to compare against later.

**This repository is the software. Research on the data goes elsewhere.** Settled
2026-08-12, after the repo was briefly public and a check found what it was carrying.
Anything *computed from the library* — audit reports, dedup proposals, cluster statistics,
species findings — describes a private photo collection: trip names, places, dates,
per-trip counts. It is also not what a reader of this repo came for. It belongs in a
separate private repository, or purely locally. What stays here is code, tests, the design
record, and documents that would still make sense to someone who has never seen the
library.

The line is *what the document is about*, not who wrote it: a plan for how clustering
should work is software; a table of which of your trips are captive collections is data.
When in doubt, ask whether the document would mean anything to a stranger with the code and
no photographs.

**A generated report is written into `local/` and only *copied* into `project/reports/`
after it has been checked** — by eye, or by a program that can vouch for it. Settled
2026-08-16, and it is a change of default rather than a new rule: a tool that writes
straight into a tracked directory is safe only while someone remembers to add a
`.gitignore` line for each new report, and the repository is public. Writing to `local/`,
which is ignored wholesale, means the unsafe direction requires a deliberate act. The check
is the act. `tools/audit_embed_quality.py` is the first to work this way; new report
writers should default their output path the same. Reports that were already generated into
`project/reports/` keep their per-file `.gitignore` entries — those entries are the record
of which files are known to be data, and deleting them would quietly re-arm the problem.

**`project/` is working material, not deliverable.** The intent has always been that on a
release day its contents go; `docs/` is what survives. `project/messages/` went further and
left the repository on 2026-08-12 — correspondence between the two of us is team
communication, not project content, and it reads as neither to anyone else. It lives in
`local/messages/` now.

**`docs/glossary.md`** defines the abbreviations, the project's own vocabulary — `key`,
`leaf`, `branch`, `medoid`, `applied`, `noise` — and the measures. Written for someone
arriving without the conversation around the code, which is also the test for whether
anything else belongs in `docs/`.

**Where documents go:** `project/` holds everything about the work in progress — `plans/`,
`status/` handoffs, `reports/` (analysis output — generated ones are gitignored, see
above), `plans/`, `ideas/`, `findings/`. Correspondence is `local/messages/`, named
`YYYY-MM-DD.NN <who>:-<topic>.md`; reply by filling in the placeholder file they leave. `local/` is gitignored and holds everything local: period-specific run commands,
`local/manifests/` for scope lists naming real trips, and working copies of generated
reports. `docs/` is reserved for product documentation, i.e. output meant for whoever uses
the result rather than notes about building it.

**`project/findings/` is measured; `ideas/` is not.** A finding is a result —
what was measured, on what, and what it supports — and obliges nobody to act,
which is what separates it from a plan. An idea that gets measured becomes a
finding and moves; `ideas/04` did that within an afternoon and is now
`findings/01`. Generate the numbers from the artifacts rather than transcribing
them, and state the caveats as plainly as the result. A finding about *one run*
belongs in that run's own `FINDINGS.md` instead, travelling with its artifacts;
a finding about the *collection* is data research and belongs in `local/` or the
private repository, not here.

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

**A run worth keeping carries a `FINDINGS.md` at its root, written before `./clean`.** A run
that cannot say what was learned from it is a pile of vectors: the parameters survive in
`run.json`, but what they *bought* survives nowhere, and by the time anyone asks, the run it
was compared against has itself been archived. So the findings travel with the artifacts they
describe — what the run was, what it was measured against, the numbers, and what they mean.
**Generate it from the artifacts rather than transcribing it**; a hand-copied figure is a
figure that will eventually disagree with the file beside it. Identifying detail does not go
in it — species names and the like stay in `local/`, which the document may point to. Check it
with `python3 -m tools.audit_report_safety` first. `output_009_image-size-1024/FINDINGS.md` is
the first, and the shape to copy.

**A failed or throwaway run gets none of that** — just a bare `./clean` with no description,
leaving `output_NNN/`, to be garbage-collected by hand later. The naming already carries the
distinction and always has: `output_004_cluster-analysis-session` and
`output_008_image-size-224` are named for what they are, while `output_005`, `006` and `007`
are bare numbers holding 85, 85 and 7,488 vectors from runs that went nowhere. **A bare number
means nothing was learned here**, which is exactly the signal you want when deciding what to
delete six months later.

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
1. `code/embedding/embed_server.py` — the backbone on the GPU host, `POST /embed` (base64
   JPEGs → L2-normalised vectors, so cosine distance == euclidean downstream). `./server-embed`;
   `MODEL=` picks the backbone, one per server like one resolution;
   needs `HF_TOKEN` with the gated DINOv3 licence **accepted for the account** — a valid token
   alone gives a 403 on file fetches, and `model_info()` succeeds regardless since gated repos
   expose metadata publicly. On this box `~/.cache/huggingface/hub/.locks` is root-owned, so
   `server-embed` falls back to a private `HF_HUB_CACHE`; fix with
   `sudo chown -R "$USER" ~/.cache/huggingface/hub/.locks`.

   **Two backbones, and they are two instruments rather than two candidates.**
   `facebook/dinov3-vitb16-pretrain-lvd1689m` is **self-supervised** — it never saw a species
   name, which is the whole reason it can referee a comparison between two *labellings*
   (`findings/03`). Nothing else here can do that job. `hf-hub:imageomics/bioclip-2.5-vith14`
   is **supervised on Linnaean names** over TreeOfLife-200M, ViT-H/14, 1024-dim, MIT and
   ungated; it is the only one with a **text tower**, which is what makes a taxonomy
   obtainable at all (see below). For the same reason it cannot referee a labelling: it has a
   stake in the answer. Backend follows from the model id — `hf-hub:` or a name containing
   `clip` needs open_clip, everything else transformers — and `--backend` overrides.

   **Accuracy at species is not what separates them.** Over the same 27,194 images DINOv3 at
   512 scores 1-NN 0.5363 and BioCLIP at 224 scores 0.5334, a difference that is **not
   significant** (McNemar p=0.21), while the two are wrong *together* on 39.6%. Choose on
   kind, not score.

   **`--resize-mode {crop,squash}`, open_clip only.** open_clip's own transform resizes the
   shortest edge and centre-crops, so a 3:2 frame loses its outer thirds; DINOv3's processor
   squashes the whole frame instead. Recorded in the `image_size` string (`224x224/crop`) so
   two runs cannot be confused. Measured flat — both score 0.5334 despite moving the vectors
   substantially (mean cosine 0.9602 between the two encodings of one image) — so take the
   default and do not spend a run on it again.

   **`POST /embed_text`** embeds label text into the same space, for zero-shot use. It refuses
   on the transformers backend rather than inventing a vector: DINOv3 is vision-only, and a
   plausible nonsense vector would be undetectable downstream.

   **`--image-size` is the resolution the backbone actually sees, and it is not the size of
   your JPEG.** `AutoImageProcessor` carries a resize in its own config and applies it
   silently: for DINOv3 ViT-B/16 that is **224×224**, so a 1024px export arrives as a 14×14
   grid — 201 tokens, 1 CLS + 4 registers + 196 patches — and `default_to_square` means it is
   *squashed*, not cropped, so the aspect ratio goes too. The whole first embedding run went
   through that way, and nothing recorded it. Pass `--image-size 1024` and the same image is
   4101 tokens over a 64×64 grid, at ~37× the GPU time (measured: 0.0041 s/img at 224 against
   0.1516 at 1024, ~1 min against ~1h10m for 27k birds, under 1 GB either way). One server
   serves **one** resolution, like one backbone; restart to change it. Whether more pixels
   actually help is an open question — position embeddings are interpolated far beyond the
   training resolution — so it is a thing to measure, not assume.
2. `code/embedding/embed.py` — non-GPU client; POSTs batches and appends to
   `output/embed/embeddings.jsonl`. `./run-embed`. Resumable (checkpoint is the set of `key`s
   already in the JSONL) and `--dry-run` does the whole scan/resolve with no network calls.
   Deferred SIGINT, same as `bird_label.py`. Every row records the producing `model` **and
   `image_size`**, and the client **refuses to append when the server serves a different
   pair** — vectors from two backbones, or from one backbone at two resolutions, are not in
   the same space and nothing downstream could detect the mixture. Use a fresh `--output-dir`
   to change either. Rows written before `image_size` was recorded report `(unrecorded)` and
   are treated as foreign, which is what the model check has always done with a row that has
   no model; `data/embed/embeddings.jsonl`'s 27,194 rows are exactly that case, and they are
   known to be 224 (the archive `output_008_image-size-224/` is named for it).

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
   is refused rather than merely looking compliant. `local/manifests/exclude-captive.txt` is the form
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
   predicates, no set algebra. `local/manifests/exclude-captive.txt` drops 12 zoo and aviary trips
   (1,090 birds) — a collection's species mix is an artefact of the collection rather than a
   place or season, and the enclosure is a background the model can learn instead of the bird.
   It deliberately keeps several "Safari" trips, which are a waterhole in a national
   park and therefore wild; a keyword match on `safari` would have dropped 143 wild
   records, which is the argument for writing these by hand and commenting them.

   Output is a single flat `output/embed/embeddings.jsonl` plus `run.json`; the folder structure
   survives only as the `year` / `library` / `trip` / `stem` columns. ~255 MB for the bird set at
   768 dims, against ~85 MB as float32 — the price of being appendable and inspectable, which is
   what makes resumption and the single-model check cheap. If load time hurts, cache a derived
   `.npy` beside it rather than changing the write format.
3. `code/cluster/cluster.py` — HDBSCAN over the frozen vectors, one directory per
   `min_cluster_size`. Named for what it does: it was `discover.py`, which described
   an intention rather than an operation, and the intention is the whole pipeline's,
   not this stage's. `stats.py` is still unwritten.
4. `code/cluster/cluster2.py` — the second level: Ward over the level-1 **medoids**,
   grouping *clusters* rather than images. The vocabulary is **leaf** for a level-1
   cluster and **branch** for a group of them. Ward rather than HDBSCAN again, and
   medoids rather than centroids, both on the evidence in `project/findings/01`; the
   cut is adaptive — descend until no branch holds more than `--max-leaves` leaves —
   because the constraint that matters is a display one and a fixed `k` cannot promise
   it. **`./run-cluster2`**, which takes `--run` rather than sweeping: a level-2
   grouping is *of* one level-1 run, and branches over mcs3's leaves say nothing about
   mcs15's, so there is no sensible default across a sweep.

   **`layout.csv` is this stage's output, and it carries no labels.** Which leaf each
   image is in, which branch each leaf is in, and the time encoding both — a few
   hundred KB that can be read and diffed, against 2.7 GB of copied pixels that
   decides nothing structural. `export_seriated --layout <path>` renders it and
   writes `index.csv` beside the JPEGs — the layout it was given, plus what it
   captioned with — and a `run.json` recording the labellings and their tags, the
   taxonomy source, the embedding the clustering came from, and what was written.
   **The folder is named for the clustering *and its backbone*** —
   `cluster2-mcs3-dinov3-512`, `cluster2-mcs3-bioclip-224crop` — because `mcs3`
   under two backbones is two different groupings with one name, and a hand-added
   suffix survives only as long as the person who invented it. The slug is
   resolved by walking the level-2 `run.json` back to the embedding run's. A render's inputs are not recoverable
   from pixels afterwards: a photo tagged `(Q)` does not say which directory `Q`
   was.

   **The render is a view organizer: it decides nothing and records everything
   it presents.** The grouping came from the clustering, the names from the
   labellers, the ranks from a checklist; this stage arranges them and is
   obliged to say where each part came from. A folder of photographs is the most
   persuasive artifact here — you look and form a belief — so a mislabelled one
   misleads faster than a wrong number. Nothing in it is named by hand: the
   backbone is read from what the clustering recorded and omitted when it cannot
   be, each labelling tags its own keywords, and `run.json` carries the rest.

   **The label is chosen by the render, and by nothing earlier.** Clustering does not
   read a labelling, because it does not depend on one — the vectors are
   self-supervised and the grouping is geometry. A species column copied into a
   clustering artifact is a second copy of a fact owned elsewhere, and it goes stale
   the moment a different labeller runs: that is exactly what `assignments.csv`'s
   frozen `species` did, putting one bird in a JPEG and another in the index beside it
   across 70% of an export. One owner per fact, so there is nothing to disagree. A
   guard was tried first and then deleted — a guard exists because two things *can*
   disagree, and the better fix is that only one of them holds the fact.

   This re-splits the "one command, not two" that `export_seriated`'s docstring
   argues for, and safely: the old trap was `index.csv` being written *during* the
   copy so a reader could catch it half-done, whereas the layout is complete and
   renamed into place before any copying starts. `local/exp-jpg2` runs the pair.

   **Two levels of time carry two levels of structure.** Month per branch, date per
   leaf, minute per image within a leaf, second left free. Lightroom sorts by capture
   time and filters by date, so picking a month gives a branch and picking a day gives
   a leaf. 28 days are used in every month — `--max-leaves` 27 plus a pool date — so
   February needs no special case, and at the default cut the pool never fills. A leaf
   larger than the 1,440 minutes in a day is a hard error, not an overflow: the next
   date belongs to the next leaf, so spilling would corrupt the encoding rather than
   crowd it.

### Taxonomy, from the vectors already on disk

**The checklist arrived, and it was the top of the open list.** `project/status/08` records two
findings blocked on one artifact; TreeOfLife-200M's taxon list supplies it — **11,131 Aves
taxa**, 283 families, 43 orders, 98% carrying a common name, filtered from 867,455 by class.
It is the vocabulary BioCLIP was *trained* against, which is the right label space for
zero-shot: a name the text tower never saw is a name it cannot rank fairly. Restricting to one
class is the single largest accuracy lever and costs nothing.

- `tools/predict_taxa.py` — image vector · text vector, so **no pass over pixels**: the stored
  vectors are already `encode_image` output and the whole set classifies in under two minutes.
  A different checklist is a different question asked of the same vectors, not another GPU
  hour. Writes `output/taxa/taxa_predictions.csv`, keyed by `jpg`. One guard, the one that
  matters: the vectors must come from the model the server serves, since image and text
  vectors from two models share no space and the result would look fine and mean nothing.
  Both knobs were measured and neither matters — four text forms agree within 0.002, and
  prompt ensembling over the 80 OpenAI templates makes it slightly *worse*, since BioCLIP was
  fine-tuned on taxonomic strings. **Measure on a random sample if this is revisited**: the
  JSONL is written in path order, so the head of the file is one or two years and reads ~0.45
  where the true figure is 0.29.
- `tools/map_label_taxa.py` — gives the *existing labelling* a taxonomy, so a cluster can be
  described by family rather than by a list of species strings. Two routes, blind in opposite
  directions: the checklist name (precise, blind to a life stage the checklist does not carry
  — `<bird> duckling` — and to a spelling it lists differently)
  and BioCLIP's modal call over the images carrying that label (sees pixels, not words, lands
  somewhere for everything including the junk). Where both fire they agree on family for
  **85.8% of images** though only 63.4% of labels; disagreement falls from 53.6% on one-image
  labels to 10.0% at 20+, so `agrees=no` on a large label is a lead. 99.6% of images end with
  a family. Writes `vocabulary_taxa.csv` and `label_taxonomy.csv`.
- `tools/audit_rank_alignment.py` — size-weighted modal purity of clusters against each rank,
  with a shuffled null. **Read the shape, not the level**: purity rises with coarseness for
  free, so what carries information is the null and the *gain* from species to family.
  `--source checklist` (the default) drops the ~20% of images whose taxonomy came from
  BioCLIP's vote, so no embedding marks its own homework.

**What it answered.** `findings/01` asked whether a level-2 branch is a rank above species or a
bag of look-alikes. Over checklist-named labels only — no per-image model call anywhere in the
taxonomy — leaves are 0.6914 species-pure and branches are **0.6066 family-pure against a
shuffled null of 0.0857, but only 0.2998 species-pure**. A branch cannot be named with a
species and can be named with a family. That is a rank above species, and it partly overturns
the visual verdict in the direction that finding's own caveat predicted: a branch of four
congeners looks like a mixture and is a genus. The sweep says the same thing from the other
side — species purity falls from 0.6914 at mcs3 to 0.5008 at mcs40 while family purity barely
moves, so larger clusters are not getting worse, they are drifting up a rank.

**`margin` is a calibrated confidence, which this project has never had.** Top-1 minus
runner-up cosine, rising monotonically across all ten deciles from 0.0758 to 0.6776 agreement
with the existing labels. Set that against the VLM's own `confidence` column, which averages
0.968 against a measured ~35% error. It is the triage order for a review.

**Judging a change to the embedding** — `tools/audit_embed_quality.py` compares runs by
leave-one-out **1-NN accuracy** against the pipeline's own species labels: for each image, is
its nearest neighbour the same species? A property of the vectors alone, with no
`min_cluster_size` in it — comparing clusterings instead would confound the embedding with
HDBSCAN's parameters. Runs are intersected on `key` first, so the numbers are over one
population. Three figures, because micro alone hides the tail: micro over everything, plus
micro and macro over species with ≥20 images. The restriction matters — a single-image species
scores zero by construction and 1,246 of the 2,820 species have exactly one image, so an
all-species macro would mostly measure how long the tail is. **Baseline at 224px: 1-NN
0.5207, macro≥20 0.5567, ≥20 0.6259** over 27,194 images.

**This measure has since saturated, and that is a result rather than a defect.** Three arms —
DINOv3 ViT-B/16 at 512 (86M params, self-supervised, squashed), BioCLIP ViT-H/14 at 224 (632M,
taxonomy-supervised, cropped) and the same BioCLIP squashed — land within 0.003 of each other
at ~0.535, and the one gap testable formally is insignificant. Meanwhile all three are wrong on
roughly 39.6% of images *simultaneously*, which sits on top of `findings/03`'s independently
derived 33.6–36.6% species error. Two backbones sharing no architecture, no training data and
no supervision regime cannot agree that closely by accident: above ~0.53 the number is
describing the **labels**, not the vectors. The instrument is not blind — 224→512 was worth
+0.0156 and was detectable — it has simply run out of headroom against a ~35%-wrong reference.

Two things follow. Reading a small 1-NN difference as "this embedding is better" is now a
mistake. And the label-free comparison is the one with room left: asked whether two embeddings
pick the *same nearest neighbour*, DINOv3 and BioCLIP agree 42.5% of the time against a chance
rate of 0.0037%, which is 84% of the agreement between one model with itself under a
preprocessing change. On the images the labels call wrong for both, they still agree with each
other 46.2% of the time — 89% of the rate when the labels call them right. Genuine model
failure would not look like that. A caveat that colours the whole measure: **78.5% of nearest
neighbours come from the same trip**, so for four images in five 1-NN is scoring within-session
label consistency rather than species discrimination.

The report goes to a fixed path so it can be diffed in an editor between runs, and carries no
timestamp for the same reason; `--snapshot [LABEL]` files a numbered copy alongside. Both land
in **`local/reports/`** rather than `project/reports/`, because this repository is public and
the report is computed from a private collection — `local/` is gitignored wholesale, which is
a stronger guarantee than remembering to add a per-file rule. `--anonymise` replaces species
names with a stable digest for a copy that is going to leave the machine.

**The same measure runs the other way round, and the labeller wins by more.** Hold the
vectors fixed and swap the *labels*, and 1-NN scores the labelling instead of the
embedding — the referee is legitimate in both directions because the embedding is
self-supervised and saw neither. 224px→512px was worth 0.0156 of 1-NN; swapping
`Qwen3-VL-32B` for `gemma-4-31B` costs 0.09. Anything reading the species column is far
more exposed to which model wrote it than to how the pixels were fed in. The tool cannot
do this as it stands — it takes species from the JSONL, so a second labelling has to be
joined on `key` from its CSV.

Environment: **`./venv` is the ground truth for what gets installed**, and takes stage
arguments so a non-GPU box need not pull torch — `./venv base client test cluster`, and
`./venv server` adds torch/transformers/fastapi/open_clip_torch.

**`requirements.txt` is a compatibility guard, not the install path**: a pinned snapshot for
the day a major version jump breaks something and a known-good set is wanted. `./venv freeze`
regenerates it, deliberately absent from the default stage list — writing a tracked file as a
side effect of "set my environment up" is how the previous one came to describe somebody's
conda environment, pinning 343 `file:///private/var/folders/...` paths and installing nowhere,
including here. The freeze drops `nvidia-*`/`triton` (transitive CUDA wheels torch resolves
for itself; pinning this box's set would make the file installable only on a box exactly like
it) and records what is *installed*, a superset of what `./venv` installs — so read the diff
before committing a re-freeze.

**Declare what arrives by luck.** `scipy` (cluster2's Ward linkage) and `huggingface_hub`
(predict_taxa's checklist) are imported directly but used to arrive only as hdbscan's and
transformers' dependencies. A GPU-less box skips `server`, may skip hdbscan since it is the one
package that compiles, and then both fail on an import nothing asked for.
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
  era — a wombat carrying `common myna` because a city photo with the same stem
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

**It has been used once, and the useful output was not a winner.**
`google/gemma-4-31B-it` relabelled the whole library on 2026-08-24, against
`data/label/`'s `Qwen3-VL-32B-Instruct`. The two agree on the **category** (0.9626 over
49,224 images) and disagree on the **species** — 0.288 exact agreement on the 26,520
images both call birds, 0.380 after matching the two vocabularies optimally, so nine
points of the gap is wording and sixty-two is not.

**The new labelling lost.** Judged against the 512px DINOv3 embedding — which neither
labeller went into, so it is a referee neither authored — leave-one-out 1-NN is 0.5371
against 0.4455 (McNemar p ≈ 5.6e-127), and the old labelling also wins on AMI against the
cluster partition at all five `min_cluster_size` values and on per-cluster purity. The
mechanism is **self-consistency, not granularity**: same vocabulary size, same label
length, but on images whose nearest neighbour is a cosine of 0.99 away the old run repeats
its own name 0.94 of the time and the new one 0.72.

**What the pair bought is a bound on the winner's error, which no single run can give.**
Two routes sharing no assumption: the two labellings' irreducible disagreement puts the
better one at ~33.6% species error, and `1 - purity` at the finest clustering — read as a
measure of the labels, once the clusters are confirmed single-species by eye — puts it at
36.6%. The `confidence` column averages 0.968, implying 3.2%. It is not an error estimate,
and one run could never have shown that. Method in **`project/findings/03`**; numbers in
that run's own `FINDINGS.md`; per-species detail, which names birds, in
`local/reports/label_vs_cluster_report.md`.

**So a second labelling is worth running even when it loses** — the losing comparison is
what carries the bound — and for calibration the useful second model is the most
*decorrelated* one, not the best one, since shared errors inflate agreement and inflated
agreement bounds less. Promoting the winner was never the point, and this run should not
be promoted into `data/label/`.

Two things this does *not* settle. The referee is **appearance, not taxonomy**: a labeller
that is confidently and consistently wrong scores well on both routes, which is exactly the
shape of the inherited over-calls already known to be in the old set — only a reference
checklist or an expert closes that. And the comparison runs inside a bird set **selected by
the old labels**, so the category boundary is only tested where they overlap.

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
  a place name) and the `HAND_WRITTEN_RE` species shape (`xs-小隼-Kestrel`, `pp-昵称`) are always
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
