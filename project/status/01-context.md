# 01 — Context handoff: embed/cluster/stats plan, pre-implementation

Branch `cluster`. Implementing steps 1-3 of `research/Bird Semantic Study Plan.md` §2
(Embed -> Cluster -> Cluster statistics), per the doc's own §5 "first concrete brick."

Decisions locked in (doc §5.1/5.2): DINOv3 embedding, HDBSCAN clustering.

Architecture: client/server split, mirroring `run-vllm`/`bird_label.py`'s existing pattern —
`code/embedding/embed_server.py` runs on a GPU host (loads DINOv3, serves POST /embed over
HTTP); `code/embedding/embed.py` is a non-GPU client that scans `results/result-*/raw/**/*.xmp`,
filters to `category=='bird'` rows, resolves JPEGs via a new `code/lib/jpg_index.py` (extracted
from `code/bird_label.py`'s `find_jpg_for_xmp`), and writes `output/embed/embeddings.jsonl`.
`code/cluster/discover.py` (step 2, HDBSCAN + centroid/medoid/density-peak) and
`code/cluster/stats.py` (step 3, cluster stats + condensed-tree dump) are pure numpy/hdbscan,
no GPU needed.

Full design: `project/plans/2026-07-29-embed-cluster-stats.md`.

**Deployment detail:** the vLLM server (and, per the plan, the new embed server) will run on
`spark` — the same box as the code. Hostnames are not hardcoded in scripts; they're read from
`config.toml` at the repo root (`[servers.vllm]`, `[servers.embed]`, `[servers.llama_cpp]`),
loaded via `code/lib/config.py`'s `server_url(name, path=...)`. `.env` stays secrets-only
(gitignored); `config.toml` is checked in and is where future non-secret parameters should go.

**Already implemented this session** (no GPU/data required, done and left uncommitted pending
review): `config.toml`, `code/lib/config.py`, and `code/bird_label.py`'s `--vllm-url`/
`--llama-url` now default from `config.toml` instead of hardcoded `galileo`/`darwin` — `run-vllm`
and `run-cpp` no longer pass a hardcoded URL. `CLAUDE.md` updated to match.

Status at handoff: embed/cluster/stats plan designed, not yet implemented (that part still
needs GPU + `data/`, deferred to the next session). The config.toml mechanism above is done and
should be reused by `embed.py`/`embed_server.py` (`server_url("embed")`) rather than
reinvented. Next session runs directly on a host (`spark`) with GPU + `data/` populated — start
there, build the embed/cluster/stats pipeline per the plan.

Reference: bird-category row counts per year (from `results/result-*/bird_identification_output.csv`,
`note` starting with "bird"): 2019=3893, 2020=247, 2021=4811, 2022=732, 2023=406, 2024=3287,
2025=1746 (~15.1k total).
