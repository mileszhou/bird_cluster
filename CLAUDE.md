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

Reset output: `./clean`

`run-vllm` is the recommended approach — it talks to a vLLM OpenAI-compatible server (`--vllm-url`, default from `config.toml` `[servers.vllm]`) rather than loading the model in-process, so start the server separately (see `run-server-vllm` for an example) before running this. On startup the script probes `{vllm-url}/models` and swaps in whatever model the server actually has loaded if `--model` doesn't match, so the requested model name doesn't need to be exact.

## Architecture

**Data flow:**
1. Input: `data/raw/<Photos-YY.NN.xmp>/<trip folder>/*.xmp` sidecar files, nested by half-year and then by trip; matching JPEGs live flat (no trip-level nesting) under `data/jpg/<Photos-YYYY.NN>/*.jpg`. `code/bird_label.py` recursively scans `data/raw` only — folders of the same shape sitting directly under `data/` are not picked up.
2. `code/bird_label.py` iterates XMP files, finds the matching JPEG (see JPEG matching below), calls the selected backend
3. Each backend sends a system prompt + base64-encoded JPEG to the model (vLLM/llama.cpp backends call an OpenAI-compatible HTTP server; chatgpt calls OpenAI directly)
4. Model returns JSON: `{category, label, label_cn, confidence}`
5. `code/lib/label_generator.py` formats a compact label with pinyin initials and confidence
6. XMP sidecar gets keywords injected; CSV row appended; checkpoint updated

**JPEG matching** (`find_jpg_for_xmp` in `code/bird_label.py`): half-year folder names don't always line up 1:1 between `raw/` and `jpg/` (a trip filed under one raw half-year can have its JPEGs exported into an adjacent half's jpg folder), and camera filenames can collide across unrelated shoots (counter wraparound). Matching therefore: (1) tries the expected jpg folder derived from the raw folder name (`Photos-23.01.xmp` → `Photos-2023.01`); (2) falls back to a filename-stem index built over the whole `jpg/` tree; (3) if the stem is found in more than one place outside the expected folder, treats it as ambiguous and skips rather than guessing.

## Clustering pipeline (branch `cluster`)

Downstream of the labeler: embed the identified bird photos and discover species
structure from appearance rather than trusting the VLM's per-photo guess. Theory in
`research/Bird Semantic Study Plan.md`; design in
`project/plans/2026-07-29-embed-cluster-stats.md`.

**Dataset layout** — this is a *labeler output tree*, not the labeler's own input tree:

```
data/xmp/result-<YYYY>/bird_identification_output.csv
data/xmp/result-<YYYY>/raw/<half-year>/<trip>/*.xmp     # keywords already injected
data/jpg/<half-year>/*.jpg                              # flat, no trip nesting
```

Half-year folder names match verbatim between `raw/` and `jpg/` (`raw/2023.1/<trip>/_X.xmp`
→ `jpg/2023.1/_X.jpg`). Nothing is hardcoded per year — any dataset in this shape works.
`--years` limits the range (`--years 2019`, `--years 2019,2021`, `--years all`).
A full survey of the current dataset is in `project/reports/data_preview_report.md`,
regenerated by **`./run-audit`** (read-only, ~15s). It also writes the manual-repair worklists
next to the report: `duplicate_frames.csv`, `photos_in_doubt.csv`, `unprocessed_sidecars.csv`,
`colliding_jpg_stems.csv`.

The report's filename never changes, so `git diff` after a run shows exactly what moved in the
dataset — keep it that way. The worklist CSVs are gitignored: they are large and fully
regenerated every run. `./run-audit --snapshot <label>` additionally files a numbered copy in
`project/reports/archive/NN-data_preview_report-<label>.md` when a specific run is worth
keeping to compare against later.

**Where documents go:** `project/` holds everything about the work in progress — `plans/`,
`status/` handoffs, `reports/` (analysis output and worklists). `docs/` is reserved for
product documentation, i.e. output meant for whoever uses the result rather than notes about
building it.

**Deciding whether a filename collision is a duplicate** (`code/lib/trips.py`): trip folders
are named `YYYY-MM-DD <place>`, so sorting them gives a timeline. A frame counter cannot
legitimately repeat within one trip, so the same frame number in *neighbouring* trips is a
candidate duplicate — one photo filed twice — while a collision far apart in time is ordinary
wraparound. `frame_id()` collapses the filename decorations first, so `20190113-_D8S0025-2`
and `_D8S0025` count as one frame. The timeline is only a prefilter: the audit confirms each
candidate by decoding both JPEGs and hashing the pixels (on by default; `--no-verify-pixels`
to skip), because in this dataset 16 of 227 candidates sit in neighbouring trips — some the
same day — yet are demonstrably different photos. Act on the CSV's `action` column, which
reflects the pixel evidence, not on the timeline verdict alone.

**Never key the result CSV by basename.** The CSV's `filename` column is a bare basename
with no folder, and basenames repeat within a single year (camera counter wraparound):
2025's CSV has 484 duplicate keys, 2024's 347, 2021's 429 — and 119 / 153 / 81 of those
respectively have duplicates that *disagree on category*. A basename-keyed lookup silently
attaches the wrong row and inflates bird counts. Read category and species from each
sidecar's own `dc:subject` keywords instead — `code/lib/xmp_labels.py`. Keep the CSV for
`response_json` and audit work only.

**JPEG resolution** (`code/lib/jpg_index.py`, `JpgIndex`) — 4,371 of 27,043 jpg stems appear
in more than one folder, so a whole-tree stem fallback silently returns a photo from an
unrelated shoot. Off-folder matches come in two kinds needing opposite treatment: *same
year, adjacent folder* (`2019.7` → `2019.8`) is an export-boundary offset — same shoot, right
photo; *different year* (`2018.1` → `2024.5`) is counter wraparound — wrong photo. Hence
`--match-policy`: `expected` (strictest), `same_year` (default: also takes a unique same-year
match), `any` (the loose behaviour the labeler used — accepts cross-year, unsafe). Ambiguous
matches are never accepted under any policy. On the current dataset the default yields 14,979
usable bird photos while refusing 868 cross-year and 214 ambiguous. Note the labels already in
the sidecars were produced under loose matching, so those 868 carry a species guess made from
the wrong photo.

Verified by hashing decoded pixels across a 300-stem sample of the collisions: 84% are
different photos sharing a frame number, 11% are the same image exported into several folders,
and every same-image case was *same-year* — no cross-year collision was ever the same image.
File bytes cannot detect this (all colliding stems are byte-distinct even when the pixels
match), so dedup must work on decoded pixels or on the embedding. Consequence: the embedding
set contains exact duplicates — 60 groups / 117 rows in 2019's 3,860 — and since HDBSCAN is
density-based, **the clustering step must dedupe identical vectors first**. Those duplicate
sidecars sometimes disagree on species for identical pixels, which is a useful direct measure
of VLM label noise.

2018 has no `data/jpg/2018.*` export at all, so it self-excludes (0 usable) — every apparent
match is a wraparound coincidence. 2020 originally nested trips directly under `raw/` with no
half-year level and had a flat `jpg/2020`; both were normalised to `2020.1` so it matches
every other year.

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
3. `code/cluster/discover.py` / `stats.py` — HDBSCAN + cluster statistics. Not yet written.

Environment: `./venv` takes stage arguments so a non-GPU box need not pull torch —
`./venv base client test cluster`, and `./venv server` adds torch/transformers/fastapi.
Tests run from within `test/` (`../.venv/bin/python -m pytest lib/`). `test/conftest.py`
exists because the project package is named `code`, which shadows the stdlib module of the
same name once pytest preloads it.

**Outputs** (in `output/`):
- `bird_identification_output.csv` — main results (filename, label, label_cn, confidence, note, run_label, response_json)
- `args.json` — CLI arguments for reproducibility
- `processed.txt` — checkpoint; delete to reprocess all images
- `raw/` — XMP files with keywords added

**Backend modules** (all in `code/bird_label.py`):
- `predict_with_vllm()` / `predict_with_vllm_batch()` — vLLM server via OpenAI-compatible API (`--vllm-url`); batch mode fires concurrent HTTP requests via a thread pool so the server's continuous batching handles them together
- `predict_with_gpt4o()` — OpenAI cloud API
- `predict_with_llamacpp()` — llama.cpp OpenAI-compatible server

**Supporting library** (`code/lib/`):
- `label_generator.py` — formats `{pinyin_initials}-{chinese_name}-{english_name}({confidence}%)` labels
- `transformers_engine.py` — HuggingFace Transformers backend (in progress)

## Key Behaviors

- **Checkpoint resumption:** Already-processed filenames in `output/processed.txt` are skipped on re-run; CSV is opened in append mode so prior results are preserved
- **XMP permissions:** Script calls `chmod` on XMP files before writing keywords (needed for library-managed files)
- **JSON parsing:** Model may wrap response in markdown fences or return Chinese in the `label` field; post-processing strips fences and moves Chinese characters from `label` to `label_cn` automatically
- **vLLM prompt constraints:** `category: "bird"` is strictly class Aves; insects/butterflies must be `animal`. `label` must use Latin characters only; `label_cn` is Mandarin. For `scenery`, the prompt asks for the specific subject of the scene (landmark, landscape feature, or activity — e.g. "Sunset over Lofoten fjord") rather than a generic description.
- **Filter mode:** `--filter-csv` reads a prior output CSV and only reprocesses rows where `note` contains "animal" or confidence is below threshold
- **Graceful Ctrl-C:** SIGINT is intercepted with a deferred handler — the signal sets a flag rather than raising immediately, so the current batch always completes fully (inference, XMP write, CSV write, checkpoint) before the loop exits. This synchronises the async signal with the program's batch boundary, ensuring no partial writes.
