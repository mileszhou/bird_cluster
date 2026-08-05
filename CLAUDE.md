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

Non-secret configuration (which host each backend server runs on, etc.) lives in `config.toml`
at the repo root, checked into git — see `[servers.*]` entries, read via
`code/lib/config.py:server_url()`. Keep it separate from `.env`: `.env` is for secrets only and
is gitignored.

Run via convenience scripts or directly:
```bash
./run-gpt                         # GPT-4o (requires OPENAI_API_KEY)
./run-cpp                         # llama.cpp server, host from config.toml [servers.llama_cpp]
./run-vllm                        # vLLM server, host from config.toml [servers.vllm] (recommended)
./run-tf                          # HuggingFace Transformers

# Or directly with options:
python3 -m code.bird_label --approach vllm --model "Qwen/Qwen3-VL-132B-Instruct" --conf-threshold 0.6
```

Key CLI flags:
- `--approach` — `chatgpt`, `llama.cpp`, `vllm`, or `transformer`
- `--vllm-url URL` — vLLM OpenAI-compatible server endpoint (default: `config.toml` `[servers.vllm]`, vllm approach only)
- `--conf-threshold FLOAT` — confidence below which to flag as low-confidence (default 0.6)
- `--no-bird FLOAT` — confidence below which to mark as "no bird" (default 0.2)
- `--filter-csv PATH` — re-process only "animal" or low-confidence rows from a prior run's CSV
- `--run-label TEXT` — tag this run in the output CSV
- `--batch-size INT` — number of images processed concurrently against the vLLM server (default 1; 8 is a reasonable default — each unit is a concurrent HTTP request, the server does its own continuous batching)
- `--years LIST` — label only these years, e.g. `--years 2019,2021` (default `all`). Selects library folders by year: `2019` → `Photos-19`

Reset output: `./clean`

`run-vllm` is the recommended approach — it talks to a vLLM OpenAI-compatible server (`--vllm-url`, default from `config.toml` `[servers.vllm]`) rather than loading the model in-process, so start the server separately (see `run-server-vllm` for an example) before running this. On startup the script probes `{vllm-url}/models` and swaps in whatever model the server actually has loaded if `--model` doesn't match, so the requested model name doesn't need to be exact.

## Architecture

**Data flow:**
1. Input: `data/xmp/<Photos-YY>/<trip>/*.xmp` sidecars; the JPEG export **mirrors the same
   folder structure** at `data/jpg/<Photos-YY>/<trip>/*.jpg`.
2. `code/bird_label.py` copies the sidecar tree to `output/raw/` and works on the copy —
   **`data/` is treated as read-only** (it is a git submodule holding the organised dataset;
   only curated data is copied in and tracked). Labels therefore land in `output/raw/`, and
   getting them back into the Lightroom library is a separate, manual step.
3. It iterates the copied XMP files, finds the matching JPEG (see JPEG matching below), calls
   the selected backend
4. Each backend sends a system prompt + base64-encoded JPEG to the model (vLLM/llama.cpp backends call an OpenAI-compatible HTTP server; chatgpt calls OpenAI directly)
5. Model returns JSON: `{category, label, label_cn, confidence}`
6. `code/lib/label_generator.py` formats a compact label with pinyin initials and confidence
7. XMP sidecar gets keywords injected; CSV row appended; checkpoint updated

**JPEG matching** (`build_items()` in `code/bird_label.py`, backed by `code/lib/jpg_index.py`
— the same resolver the embedding step uses): the export mirrors the library, so a sidecar's
JPEG is a lookup inside its own trip folder. Camera counters wrap and stems repeat across the
library, but with the folder part of the key that creates no ambiguity, so the whole-tree stem
fallback the old flat export needed (and the cross-shoot mismatches it caused) is gone.

**Decorated exports.** A JPEG whose stem is `<sidecar stem>-<something>` is a derived export:
`-2`/`-3` (Lightroom silently renaming on an unnecessary re-import — *not* virtual copies),
`-Enhanced-NR` (AI Denoise), `-Pano`, `-HDR`, `-Edit`. Matching is prefix-based, so a new
decoration needs no code change. Two cases, treated oppositely: where the plain `X.jpg` is
**absent** the decorated file is the capture's only export and is used (5,990 sidecars); where
`X.jpg` **exists** the decorated file is surplus and ignored (193, all with an intact pair).

**JPEG-only items** (`--include-orphan-jpg`) — 4,857 exported JPEGs have no sidecar, the source
never having been raw. They have nowhere to carry a keyword, so their label goes to the CSV alone,
flagged `source=jpg-only` / `applied=csv-only` and keyed by the path relative to `data/jpg` (the
extension keeps those keys from colliding with sidecar keys). Do not assume they are non-birds:
607 come from interchangeable-lens bodies (Nikon Z9/D500/D850/D5/Z8, Canon R5, Sony) and cluster
in birding trips — `2025-04-25.1 Birding Birds` alone holds 118.

**Never key anything by basename** — not the checkpoint, not the CSV, not a filter set. 10,832
of the 34,160 sidecars share a basename with another. `output/processed.txt` and the CSV's
`path` column both hold the sidecar's path relative to the sidecar root (`xmp_key()`); the
CSV's `filename` column is kept for readability only. A basename-keyed checkpoint marks ~5,845
unlabelled photos as already done; `--filter-csv` refuses a CSV with no `path` column for the
same reason.

## Clustering pipeline (branch `cluster`)

Downstream of the labeler: embed the identified bird photos and discover species
structure from appearance rather than trusting the VLM's per-photo guess. Theory in
`research/Bird Semantic Study Plan.md`; design in
`project/plans/2026-07-29-embed-cluster-stats.md`.

**Dataset layout** — `data/` is a git submodule holding the organised dataset, treated as
**read-only**: only curated data is copied in and tracked. The Lightroom export mirrors the
photo library exactly, so both trees have the same shape:

```
data/xmp/<Photos-YY>/<trip>/*.xmp     # Photos-19/2019-01-13 山公园/_D8S0025.xmp
data/jpg/<Photos-YY>/<trip>/*.jpg     # same folder, same stem
data/jpg/export.report.txt            # what the exporter skipped, and why
```

Trip folders match verbatim between the trees, so resolving a JPEG is a lookup inside one
folder. Nothing is hardcoded per year — any dataset in this shape works. `--years` limits the
range (`--years 2019`, `--years 2019,2021`, `--years all`); it selects library folders by year,
`2019` → `Photos-19`.

A full survey of the current dataset is in `project/reports/data_preview_report.md`,
regenerated by **`./run-audit`** (read-only, ~12s). It also writes the worklists next to the
report: `missing_jpg.csv`, `extra_jpg.csv`, `unprocessed_sidecars.csv`.

**Every CSV in this project carries a `path` column** — the sidecar's path relative to
`data/xmp` — and that is the join key across all of them: the audit worklists, the labeler's
`bird_identification_output.csv`, `output/processed.txt`, and `embed.py`'s JSONL `key`. Nothing
is ever keyed by basename (see above). `missing_jpg.csv` additionally carries `raw_name` (the
raw as the exporter named it — `X-Enhanced-NR.dng` and `X.nef` are different files on disk even
though one sidecar covers both) and `expected_jpg` (where the JPEG would have landed, so a
re-export can be checked). The worklists are written **UTF-8 with BOM**: trip folders are named
in Chinese and Excel reads a BOM-less UTF-8 CSV as the system codepage, mangling every one.
Read them back with `encoding="utf-8-sig"`.

**The two discrepancies the audit exists to find.** *Sidecar with no JPEG* — a photo the
pipeline cannot see; 105 of 34,160 today. `data/jpg/export.report.txt` attributes them: 93 the
exporter itself reported as "The file could not be found." (a dangling sidecar whose raw is
gone from disk), 12 it never mentioned (a technical gap worth chasing). *JPEG with no sidecar*
— 4,857, harmless and counted only: the source was never a raw file, so it never had a sidecar.

**Derived exports are not missing JPEGs.** 5,990 sidecars have no JPEG under their own name but
do have a decorated one in the same folder: `-2`/`-3` is a Lightroom **virtual copy** exported
under its copy name (the only export when the master was not in the export set), and
`-Enhanced-NR` is the **AI Denoise** render — a separate DNG carrying its metadata internally,
so it has no sidecar of its own. Both are the right photo. `JpgIndex` resolves a whole folder
at a time with exact hits claimed first, so where sidecars `A` and `A-2` both exist, `A-2.jpg`
goes to `A-2` rather than reading as `A`'s virtual copy. Treating these as missing would report
6,095 instead of 105.

The report's filename never changes, so `git diff` after a run shows exactly what moved in the
dataset — keep it that way. The worklist CSVs are gitignored: they are large and fully
regenerated every run. `./run-audit --snapshot <label>` additionally files a numbered copy in
`project/reports/archive/NN-data_preview_report-<label>.md` when a specific run is worth
keeping to compare against later.

**Where documents go:** `project/` holds everything about the work in progress — `plans/`,
`status/` handoffs, `reports/` (analysis output and worklists), `messages/` (correspondence
with the user, named `YYYY-MM-DD.NN <who>:-<topic>.md`; reply by filling in the placeholder
file they leave). `docs/` is reserved for product documentation, i.e. output meant for whoever
uses the result rather than notes about building it.

**Sidecar deduplication** (`code/lib/sidecar_meta.py`, `tools/dedup_sidecars.py`,
**`./run-dedup`**) — operates on `data/xmp`. The same photo was imported into
two trip folders in places, which is what made neighbouring trips look like frame-counter
collisions during splitting. Two sidecars are one capture when they share
`xmpMM:OriginalDocumentID`; **do not compare `exif:DateTimeOriginal` as a string** — only ~10%
of sidecars carry sub-second precision, timezone corrections move a capture across calendar
days, and import clashes rename one copy, so the literal date+time+filename rule finds just
265 of 534 redundant sidecars. Lightroom virtual copies (`-2`, `-3` stems) share an original
but have their own `xmpMM:DocumentID` — deliberate alternate edits, reported separately and
never offered for deletion. Report: `project/reports/sidecar_dedup_report.md`. The
duplicates were cleared on 2026-08-02; `./run-dedup` now reports 0 cross-trip duplicates.

**Wraparound splitting** (`tools/analyze_wraparound.py`, **`./run-split`**) — **superseded.**
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
   L2-normalised vectors, so cosine distance == euclidean downstream). `./run-server-embed`;
   needs `HF_TOKEN` with the gated DINOv3 licence **accepted for the account** — a valid token
   alone gives a 403 on file fetches, and `model_info()` succeeds regardless since gated repos
   expose metadata publicly. On this box `~/.cache/huggingface/hub/.locks` is root-owned, so
   `run-server-embed` falls back to a private `HF_HUB_CACHE`; fix with
   `sudo chown -R "$USER" ~/.cache/huggingface/hub/.locks`.
2. `code/embedding/embed.py` — non-GPU client; scans sidecars, filters to `bird`, resolves
   JPEGs, POSTs batches, appends to `output/embed/embeddings.jsonl`. `./run-embed`.
   Resumable (checkpoint is the set of `key`s already in the JSONL) and `--dry-run` does the
   whole scan/resolve with no network calls. Deferred SIGINT, same as `bird_label.py`.
   Every row records the producing `model`, and the client **refuses to append when the server
   serves a different one** — two backbones' vectors are not in the same space and nothing
   downstream could detect the mixture. Use a fresh `--output-dir` to switch models.
   Where a capture was exported more than once (master plus virtual copies), only the first
   export is embedded — the copies are alternate edits of one frame. The `key` is the sidecar's
   path relative to `data/xmp`, so any JSONL written before the mirrored export is keyed the old
   way *and* points at the old jpg tree: regenerate rather than resume.
3. `code/cluster/discover.py` / `stats.py` — HDBSCAN + cluster statistics. Not yet written.

Environment: `./venv` takes stage arguments so a non-GPU box need not pull torch —
`./venv base client test cluster`, and `./venv server` adds torch/transformers/fastapi.
Tests run from within `test/` (`../.venv/bin/python -m pytest lib/`). `test/conftest.py`
exists because the project package is named `code`, which shadows the stdlib module of the
same name once pytest preloads it.

**Outputs** (in `output/`):
- `bird_identification_output.csv` — main results (path, source, filename, category, label, label_cn, confidence, note, prior_category, prior_label, run_label, response_json). `path` is the key; `filename` is a bare basename kept for readability and must never be used to look a row up. `category` is its own column — `note` still embeds it as `"bird (0.90)"`, but parse the column, not the string. **`prior_category` / `prior_label` hold the label the previous run left**, mostly early paid GPT-4o — this CSV is the *only* record of it, so don't discard old CSVs. `applied` says what reached the sidecar: `written`, `kept-existing` (a non-bird result deferring to the prior category), `csv-only` (no sidecar exists) or `failed`. `source` says what `path` refers to: `xmp` (relative to `data/xmp`) or `jpg-only` (relative to `data/jpg`)
- `args.json` — CLI arguments for reproducibility
- `processed.txt` — checkpoint, one `path` per line; delete to reprocess all images
- `raw/` — the working copy of `data/xmp`, with keywords added. Getting labels back into the Lightroom library is a separate manual step; `data/` is never written to

**Backend modules** (all in `code/bird_label.py`):
- `predict_with_vllm()` / `predict_with_vllm_batch()` — vLLM server via OpenAI-compatible API (`--vllm-url`); batch mode fires concurrent HTTP requests via a thread pool so the server's continuous batching handles them together
- `predict_with_gpt4o()` — OpenAI cloud API
- `predict_with_llamacpp()` — llama.cpp OpenAI-compatible server

**Supporting library** (`code/lib/`):
- `label_generator.py` — formats `{pinyin_initials}-{chinese_name}-{english_name}({confidence}%)` labels
- `transformers_engine.py` — HuggingFace Transformers backend (in progress)

## Re-labelling an already-labelled library

`./clean && ./run-vllm` relabels everything — the checkpoint is what causes skipping, and
`./clean` removes it. Nothing keys off whether a sidecar already carries a label.

**Sidecars must stay clean**: they go back into Lightroom, so no archive property, no versioned
keywords, nothing Lightroom would show in its keyword list. The previous label is preserved in
the CSV's `prior_category` / `prior_label` columns instead, and `data/xmp` (read-only, versioned)
still holds every original.

**`bird` wins; anything else defers.** `set_keywords_in_xmp(xmp, category, label)` writes the new
label only when this run said `bird`, or when the sidecar carries no category at all. A non-bird
result over an existing category leaves the sidecar untouched — the categories already there are
finer-grained than what a fresh run produces (the early GPT-4o pass wrote specific scene
descriptions, so a new bare `scenery` would lose information rather than correct anything). Bird
identification is the one axis this pipeline is authoritative on, so it is the one it overwrites.
Current dataset: 24,783 sidecars have no category and always take the new label; 9,377 are
protected (6,363 `bird`, 2,349 `scenery`, 336 `animal`, 179 `People`, 150 `Unknown`).

A consequence worth knowing: a photo already called `bird` is **never demoted** by a non-bird
result. That follows from the rule and guards the clustering set against a false negative, but it
also preserves any genuine over-call from the old run. The CSV's `category` (this run) against
`prior_category` (previous) plus `applied` (`written` / `kept-existing`) makes every disagreement
recoverable without reading the sidecars.

`set_keywords_in_xmp()` preserves the user's own keywords and is idempotent on re-run. Three
things it has to get right:

- **`rdf:Bag` vs `rdf:Seq`** — Lightroom writes `dc:subject` as a Bag (13,879 sidecars), earlier
  runs of this script wrote a Seq (1,794). Read either; reuse whichever is present. Adding a
  second container under one `dc:subject` makes the keyword list depend on which the reader picks.
- **`lr:hierarchicalSubject`** — Lightroom mirrors `dc:subject` there and will resurrect old
  keywords from it on import. Write both together or not at all.
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

- **Checkpoint resumption:** Already-processed filenames in `output/processed.txt` are skipped on re-run; CSV is opened in append mode so prior results are preserved
- **XMP permissions:** Script calls `chmod` on XMP files before writing keywords (needed for library-managed files)
- **JSON parsing:** Model may wrap response in markdown fences or return Chinese in the `label` field; post-processing strips fences and moves Chinese characters from `label` to `label_cn` automatically
- **vLLM prompt constraints:** `category: "bird"` is strictly class Aves; insects/butterflies must be `animal`. `label` must use Latin characters only; `label_cn` is Mandarin. For `scenery`, the prompt asks for the specific subject of the scene (landmark, landscape feature, or activity — e.g. "Sunset over Lofoten fjord") rather than a generic description.
- **Filter mode:** `--filter-csv` reads a prior output CSV and only reprocesses rows where `note` contains "animal" or confidence is below threshold
- **Graceful Ctrl-C:** SIGINT is intercepted with a deferred handler — the signal sets a flag rather than raising immediately, so the current batch always completes fully (inference, XMP write, CSV write, checkpoint) before the loop exits. This synchronises the async signal with the program's batch boundary, ensuring no partial writes.
