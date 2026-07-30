# 02 — Data populated, step 1 (embed) built

Branch `cluster`. Supersedes the environment assumptions in `01-context.md`: `data/` is now
populated on `spark` and the embed client + shared libraries are written and exercised
against the real tree. Step 1's server half is written but **not yet run** — it needs the
DINOv3 weights pulled on a GPU host.

## Layout correction

`01-context.md` and `docs/plans/2026-07-29-embed-cluster-stats.md` both assume
`results/result-*/raw/**/*.xmp` with `Photos-YYYY.NN`-style folder names. The populated
dataset is `data/xmp/result-*/raw/<half-year>/<trip>/*.xmp` with bare `2023.1`-style names
that match the jpg folder names verbatim. The `Photos-` mapping in
`bird_label.py:_jpg_subfolders_for_raw_subfolder` is legacy-only and did not need to fire.
The plan's step-1 CSV-lookup design (its step 2, "look up its row in that year's CSV") was
dropped — see below.

## Findings that changed the design

Full survey: `docs/reports/01-data_preview_report.md` (regenerate with
`tools/audit_dataset.py --report ... --issues-dir ...`).

1. **The CSV cannot be keyed by basename.** `filename` is a bare basename with no folder, and
   basenames repeat within a year. 2025 has 484 duplicate keys, 2024 347, 2021 429; of those
   119 / 153 / 81 have duplicates that disagree on category. Keying by basename silently
   attaches the wrong row — it inflated 2021 to 4,849 birds and 2024 to 3,294 before this was
   caught. Sidecar keywords give 4,811 / 3,287, matching `01-context.md`'s reference counts
   exactly. Hence `code/lib/xmp_labels.py`: category and species come from the XMP.
2. **Cross-year stem collisions are the real hazard.** 4,371 of 27,043 jpg stems appear in more
   than one folder. The labeler's single-candidate whole-tree fallback therefore returned a
   photo from an unrelated shoot for 868 bird sidecars — 729 of them in 2018, which has no jpg
   export at all, so *every* 2018 match is a coincidence. `MatchPolicy` refuses that class.
   Nothing was changed in `code/bird_label.py`: it belongs to a branch that will never merge
   here and is slated for removal (README "Evolution"), so it is not a constraint on this
   pipeline — but the labels already in the sidecars were produced under the loose matching,
   which is what the cross-year counts above measure.
3. **Off-folder matches are two different things.** Same-year adjacent (`2019.7` → `2019.8`,
   81 photos) is an export-boundary offset — right photo, safe to accept. Cross-year
   (`2018.1` → `2024.5`) is wraparound — wrong photo. `MatchPolicy.SAME_YEAR` (default) splits
   them; this is the one design decision not anticipated by the plan.
4. **Species tail is long.** 2,401 distinct English names over 19,565 labelled bird photos;
   1,028 singletons, only 253 names with ≥20 photos and 79 with ≥50. Step 2 should expect a
   large HDBSCAN noise fraction as the shape of the data, not a failure.

## Dataset changes made

- `data/jpg/2020` → `data/jpg/2020.1`, and `data/xmp/result-2020/raw/<41 trips>` →
  `data/xmp/result-2020/raw/2020.1/<41 trips>`, so 2020 has the same half-year level as every
  other year. Counts verified unchanged (1,171 jpg / 1,133 xmp). This is why `expected_folder`
  treats the trip level as optional — a dataset shaped either way resolves the same.

## Built this session

- `code/lib/xmp_labels.py` — read categories + `py-cn-en(NN%)` labels from `dc:subject`. The
  trailing confidence is load-bearing: hand-written species keywords share the py-cn-en shape
  but carry no percentage, and pre-existing user tags (`bhl-山公园`, `add=20231014`) coexist.
- `code/lib/jpg_index.py` — `JpgIndex` / `MatchPolicy` / `Verdict`. Written fresh with the
  strictness rules above rather than a straight lift of `find_jpg_for_xmp`. The plan's §1 also
  wanted `bird_label.py` refactored onto it; dropped as pointless — that file lives on a
  branch that will never merge here.
- `code/embedding/embed_server.py` + `run-server-embed` — DINOv3, `/health` + `/embed`,
  L2-normalised. Never executed yet.
- `code/embedding/embed.py` + `run-embed` — scan → filter → resolve → batch POST → JSONL.
  Resumable, `--dry-run`, `--years`, `--limit`, `--min-confidence`, deferred SIGINT.
- `tools/audit_dataset.py` — read-only dataset audit; emits the markdown report plus
  `output/audit/unresolved_bird_sidecars.csv` (per-sidecar, with year/half-year/trip/species/
  verdict) and `output/audit/colliding_jpg_stems.csv`.
- `test/conftest.py`, `test/lib/test_jpg_index.py`, `test/lib/test_xmp_labels.py` — 28 tests,
  all passing. `conftest.py` works around the project package being named `code`.
- `venv` now takes stage arguments (`base client test cluster server notebook`) so a non-GPU
  box need not pull torch. `.venv` currently has base+client+test only.

## Verified

Scan: 2019 gives 3,818 expected-folder + 42 same-year = 3,860 usable, refusing 15 cross-year
and 18 no-jpg (`--match-policy any` would take 3,875, i.e. 15 wrong photos). All years: 14,979
usable, 2018 contributing 0 as intended.

Full client/server round trip exercised on the GB10 — but with **`facebook/dinov2-base` as a
stand-in**, because DINOv3 is still gated (below). Vectors came back 768-d with L2 norm
1.000000 to six places, at ~90 img/s at `--batch-size 16`; resumption re-read the checkpoint
and produced no duplicate keys; SIGINT and the retry path were not exercised. As a smoke test
of the embedding's usefulness, mean cosine within a species was 0.839 vs 0.773 across species
on a 64-image sample — a real but small margin, which is roughly what DINOv2 at 224px on
uncropped frames should give and is itself an argument for DINOv3 plus bird-region cropping.
The dinov2 vectors were then deleted; nothing in `output/embed` is real data.

## Environment notes (spark, GB10)

- `torch 2.13.0+cu130` / `torchvision 0.28.0+cu130` from default PyPI resolve correctly for
  aarch64 + CUDA 13.0; `sm_121` confirmed. `torchvision` is a hard requirement of
  transformers' `AutoImageProcessor` and is now in `venv`'s server stage.
- `run-embed` / `run-server-embed` activate `.venv` themselves — the older `run-*` scripts
  assume an already-activated venv, which is why the first server start failed on `no module
  named torch`.
- `~/.cache/huggingface/hub/.locks` is **root-owned** on this box, so every HF download fails
  with EACCES for a normal user. `run-server-embed` detects this and falls back to
  `HF_HUB_CACHE=~/.cache/huggingface-bird_cluster/hub`. The proper fix is
  `sudo chown -R "$USER" ~/.cache/huggingface/hub/.locks`; no passwordless sudo here.

## Blocked on the user

**DINOv3 is gated and this account is not on the allow list.** `HF_TOKEN` authenticates fine
(whoami = `mileszhou`) and repo *metadata* reads, but file fetches 403 with "Access to model
facebook/dinov3-vitb16-pretrain-lvd1689m is restricted and you are not in the authorized
list." Accept the licence at
<https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m>, then
`./run-server-embed` (its `--model` default is already the DINOv3 id) and `./run-embed`.
Note `api.model_info()` succeeding is not evidence of access — gated repos expose metadata
publicly; only a file fetch proves it.

## Next

1. Real DINOv3 run once the licence lands: `./run-server-embed`, then `./run-embed`
   (defaults to `--years 2019`, 3,860 images; ~1 min at the rate measured above).
2. Steps 2-3: `code/cluster/discover.py` + `stats.py` per the plan's §4/§5, with
   `test/cluster/` synthetic-blob fixtures. `./venv cluster` already installs the deps.
3. `requirements.txt` is a stale freeze from the labeler era and is being disregarded for now;
   generate a fresh one from `.venv` when the pipeline is close to done.

## Guard worth keeping in mind

`embeddings.jsonl` records the producing `model` on every row, and `embed.py` refuses to append
when the server serves a different one — mixing backbones would corrupt every distance
downstream with nothing able to detect it. Rows written before this guard read as
`(unrecorded)` and are also refused. To switch models, use a fresh `--output-dir` or delete the
file.

## Open, owned by the user

- Locating the 2018 jpg export (4,426 bird sidecars currently unusable) and the cross-year
  cases in other years. Skipped for the first run by design.
- Re-running the labeler on the 324 `missing JPEG` rows that resolve now (241 in 2020, 69 in
  2024), plus 2024's 1,010 `scenery (0.00)` rows with empty `response_json`.
