# Bird Semantic Clustering — Steps 1-3 (Embed / Cluster / Stats)

## Context

`research/Bird Semantic Study Plan.md` §2 lays out a 10-step pipeline for discovering
bird-species structure in the photo library via image embeddings + topological clustering,
rather than trusting the VLM's noisy per-photo species guess. Two design decisions from that
doc are already locked in (§5.1/5.2, confirmed in the §6 summary table):

- **Embedding: DINOv3** (self-supervised, taxonomy-blind — clusters on appearance, not
  pre-baked taxonomy).
- **Clustering: HDBSCAN** (density-based, no *k*, gives an explicit noise set for free, and
  its condensed tree *is* the persistence structure later steps need).

This session's job is steps 1-3 only: **Embed → Cluster (discovery) → Cluster statistics** —
the doc's own §5 "first concrete brick": embed a set of bird photos, HDBSCAN them, dump
centroid/medoid/density-peak per cluster + noise set + condensed tree, and produce enough of a
report to eyeball whether clusters track species rather than pose/background.

**Environment constraint clarified by user:** this container has no GPU and no `data/`
(photo library) mounted right now. The user can restart it with GPU access from the host, but
the *target* architecture regardless of that is a **client/server split**, same pattern
`bird_label.py` already uses for `run-vllm`: a server hosts the model (on a GPU box, e.g.
galileo), and the pipeline code that runs here (or wherever) is a thin non-GPU HTTP client.
So step 1 is built as `embed_server.py` (GPU host) + `embed.py` (client, mirrors
`predict_with_vllm`'s pattern). Steps 2-3 have no GPU dependency at all and can be built and
tested end-to-end right now with synthetic data.

**Structure authority:** `README.md` fixes the layout — `code/` is organized into domain
subfolders (`embedding/`, `cluster/`, ...), library code lives in `code/lib/` ("never manually
run"), tests mirror the same domain subfolders under `test/` and are run from within `test/`,
and root-level `run-$func` scripts are the user-facing entry points. The current flat
`code/*.py` (bird_label) is slated to move under `code/bird_label/` later — new code should not
couple to that flat layout, so shared logic gets extracted to `code/lib/` now rather than
imported from `code.bird_label` directly.

## Session note

Rather than reconfiguring this container for GPU passthrough, the user will start a fresh
Claude Code session directly on a host machine that already has GPU access. This is a
dev-session convenience, not an architecture change: the client/server split below stays as
designed (galileo-style GPU host running `embed_server.py`, non-GPU client elsewhere) since
production may still have data and GPU on different boxes.

## New layout

```
code/
  lib/
    jpg_index.py        # extracted from bird_label.py: JPEG-matching helpers
  embedding/
    __init__.py
    embed_server.py      # GPU host: loads DINOv3, serves /embed over HTTP
    embed.py             # client: results/ -> bird rows -> jpg -> embed_server -> artifact
  cluster/
    __init__.py
    discover.py           # step 2: HDBSCAN over frozen embeddings
    stats.py              # step 3: cluster statistics + condensed tree dump
test/
  embedding/
  cluster/
  lib/
run-server-embed           # +x, launches embed_server.py on the GPU host
run-embed                  # +x, client convenience script (mirrors run-vllm)
run-cluster                # +x, runs discover.py then stats.py
```

`requirements.txt` gains: `hdbscan`, `fastapi`, `uvicorn` (server only needs these on the GPU
host, but the file is shared, so just add them).

**Already done (this session, no GPU/data needed):** `config.toml` (repo root) +
`code/lib/config.py`'s `server_url(name, path="")` now hold non-secret server host/port config,
separate from `.env` (secrets only, gitignored). `[servers.vllm]` (spark:8000),
`[servers.embed]` (spark:9100), `[servers.llama_cpp]` (darwin:8080) are already defined.
`code/bird_label.py`'s `--vllm-url`/`--llama-url` already resolve from this file when not
passed explicitly; `run-vllm`/`run-cpp` no longer hardcode a hostname. `embed.py`/
`embed_server.py` should use the same `server_url("embed")` convention rather than inventing a
new one.

## 1. `code/lib/jpg_index.py` — extract, don't duplicate

Pull `_jpg_stem_index`, `_jpg_subfolders_for_raw_subfolder`, and `find_jpg_for_xmp` out of
`code/bird_label.py:354-400`. Today these rely on module globals (`JPG_DIR` set at line 613);
refactor into a small `JpgIndex` class parameterized by `jpg_dir: Path`, keeping the exact same
matching semantics (expected-folder-first, then whole-tree stem index, ambiguous = skip). Update
`code/bird_label.py` to construct and use a `JpgIndex(JPG_DIR)` instead of the free functions +
globals, so behavior is unchanged there. `code/embedding/embed.py` imports the same class.

## 2. `code/embedding/embed_server.py` (runs on the GPU host)

FastAPI app, single model loaded at startup (mirrors the load-once-cache pattern in
`code/lib/transformers_engine.py:9-36`, `device = "cuda" if torch.cuda.is_available() else
"cpu"`):

- `GET /health` → `{"status": "ok", "model": model_id}` — for the client's startup probe, same
  spirit as the existing `get_actual_vllm_model_name` / llama.cpp `/models` probe in
  `code/bird_label.py:596-609`.
- `POST /embed` → body `{"images": ["<base64 jpeg>", ...]}` → `{"embeddings": [[...], ...]}`.
  Loads `facebook/dinov3-*` via `transformers.AutoModel.from_pretrained`, takes the pooled /
  CLS representation, **L2-normalizes** before returning — the study plan's whole "compactness
  of the unit sphere" argument (§1) assumes normalized embeddings, and it makes cosine distance
  == euclidean distance downstream, which is what HDBSCAN will use.
- CLI flags: `--model`, `--port`, `--host` — `--port`/`--host` default from `config.toml`
  `[servers.embed]` (currently `spark:9100`), matching `embed.py`'s client-side default.

Launched via `run-server-embed` (new root script, same shape as `run-server-vllm`).

## 3. `code/embedding/embed.py` (client — step 1 driver)

```
python3 -m code.embedding.embed \
  --results-dir ./results --data-dir ./data \
  --output-dir ./output/embed \
  --batch-size 32 [--limit N] [--years 2019,2024] [--dry-run] [--embed-url URL]
```

`--embed-url` defaults to `config.toml` `[servers.embed]` (via `code.lib.config.server_url`,
see below) rather than being hardcoded — same convention now used for `--vllm-url` /
`--llama-url` in `code/bird_label.py`. Only pass `--embed-url` to override.

Mirrors `bird_label.py`'s own iteration shape rather than the CSV:

1. Recursively scan `results/result-*/raw/**/*.xmp` (parallel to how `bird_label.py` scans
   `data/raw`).
2. For each stem, look up its row in that year's `bird_identification_output.csv` (indexed by
   `filename`) to read `category`/`note`; keep only rows where `note` starts with `bird`
   (case-insensitive — same convention as the existing `--filter-csv` matching).
3. Resolve the JPEG via `JpgIndex(data_dir / "jpg").find_jpg_for_xmp(xmp_file, raw_root=results
   /result-YYYY/raw)`.
4. Batch-encode resolved JPEGs (base64) and POST to `{embed-url}/embed`.
5. Append results to `output/embed/embeddings.jsonl`, one line per image:
   `{"filename": stem, "jpg_path": ..., "year": ..., "embedding": [...]}`. JSONL instead of a
   single `.npz` blob so it's append-friendly and resumable — checkpoint is just "stems already
   present in the file," matching the append-mode-CSV resumption `bird_label.py` already uses.
6. `--dry-run`: run steps 1-3 only and report match/no-match counts, no network call — this is
   the one thing fully testable without a GPU or embed server, since it verifies the
   traversal/filtering logic against the real `results/` CSVs already in the repo (matches will
   legitimately report "missing" if `data/jpg` isn't populated yet — that's fine).

## 4. `code/cluster/discover.py` (step 2)

```
python3 -m code.cluster.discover \
  --embeddings ./output/embed/embeddings.jsonl --output-dir ./output/cluster \
  --min-cluster-size 5 [--min-samples N]
```

- Load the JSONL into an (N × D) float array + filename list.
- `hdbscan.HDBSCAN(min_cluster_size=..., min_samples=..., metric="euclidean")` on the
  (already-normalized) vectors.
- Per non-noise cluster, emit all three centers per §5.3 of the doc:
  - **centroid** = mean vector.
  - **medoid** = member minimizing sum of distances to other members (exact, O(n²) per
    cluster — fine at this scale).
  - **density peak** = member with the highest `clusterer.probabilities_` (HDBSCAN's own
    density-based membership strength) — avoids reimplementing core-distance math for v1.
- Write `output/cluster/assignments.csv` (filename, jpg_path, cluster_id, probability,
  is_noise) and `output/cluster/centers.jsonl` (cluster_id, size, centroid vector, medoid
  filename, density_peak filename, centroid–medoid gap).
- Pickle the fitted clusterer to `output/cluster/clusterer.pkl` so `stats.py` can read
  `condensed_tree_` without refitting.

## 5. `code/cluster/stats.py` (step 3)

```
python3 -m code.cluster.stats --cluster-dir ./output/cluster
```

Loads the three artifacts above and produces:
- Cluster count, size distribution, noise fraction.
- Centroid–medoid gap stats (already per-cluster in `centers.jsonl` — aggregate/sort here to
  surface the doc's "large gap = free β₀>1 flag," §5.3).
- Inter-center pairwise distance matrix + closest-pair list (small distance between two
  clusters = confusable-pair candidate, §1's non-Hausdorff-locus idea).
- `clusterer.condensed_tree_.to_pandas()` dumped to `output/cluster/condensed_tree.csv` — the
  doc states this literally *is* the 0-D persistence structure (§5.5), so exporting it now sets
  up the later analysis step even though interpretation itself is out of scope here.
- Human-readable `output/cluster/stats_report.md`: cluster table sorted by size, noise
  fraction, flagged large-gap clusters, flagged close cluster-pairs — the artifact for
  eyeballing per the doc's "first concrete brick."

## Testing given current environment

- **`discover.py` / `stats.py`**: pure numpy/hdbscan, no GPU or data/ needed. Add
  `test/cluster/` fixtures that generate synthetic normalized embeddings (a few Gaussian blobs
  on the unit sphere, plus deliberate noise points) and assert cluster count, medoid/centroid
  sanity, and that `condensed_tree.csv` gets written.
- **`embed.py` traversal/filtering**: run `--dry-run` against the real `results/result-*`
  data already in the repo (no `data/jpg` needed for a dry run) to sanity-check the bird-row
  filter and XMP traversal counts against the per-year totals (~3.9k / 0.2k / 4.8k / 0.7k /
  0.4k / 3.3k / 1.7k bird rows for 2019/20/21/22/23/24/25).
- **`embed_server.py`**: needs GPU + DINOv3 weights + reachable `data/jpg` — verify once
  running on a host with GPU access and `data/` populated.
- Add `hdbscan`, `fastapi`, `uvicorn` to `requirements.txt`; run `./venv` to rebuild `.venv`
  before any of the above.

## Open items to flag, not block on

- Whether vLLM's own server (already running for the VLM backend) can serve DINOv3 as a
  pooling/embedding model — unclear/unlikely for a vision-only backbone; defaulting to a small
  dedicated FastAPI server instead. Worth a 5-minute check against galileo's vLLM version once
  we're back on a networked box, but not worth blocking this plan on.
- DINOv3 weights are Meta-licensed/gated on Hugging Face — first run of `embed_server.py` will
  need `HF_TOKEN` with the license accepted.
