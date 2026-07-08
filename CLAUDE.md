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

Run via convenience scripts or directly:
```bash
./run-gpt                         # GPT-4o (requires OPENAI_API_KEY)
./run-cpp                         # llama.cpp server at http://darwin:8080
./run-vllm                        # vLLM server at http://galileo:8000/v1 (recommended)
./run-tf                          # HuggingFace Transformers

# Or directly with options:
python3 -m code.bird_label --approach vllm --vllm-url http://galileo:8000/v1 --model "Qwen/Qwen3-VL-132B-Instruct" --conf-threshold 0.6
```

Key CLI flags:
- `--approach` — `chatgpt`, `llama.cpp`, `vllm`, or `transformer`
- `--vllm-url URL` — vLLM OpenAI-compatible server endpoint (default `http://galileo:8000/v1`, vllm approach only)
- `--conf-threshold FLOAT` — confidence below which to flag as low-confidence (default 0.6)
- `--no-bird FLOAT` — confidence below which to mark as "no bird" (default 0.2)
- `--filter-csv PATH` — re-process only "animal" or low-confidence rows from a prior run's CSV
- `--run-label TEXT` — tag this run in the output CSV
- `--batch-size INT` — number of images processed concurrently against the vLLM server (default 1; 8 is a reasonable default — each unit is a concurrent HTTP request, the server does its own continuous batching)

Reset output: `./clean`

`run-vllm` is the recommended approach — it talks to a vLLM OpenAI-compatible server (`--vllm-url`, default `http://galileo:8000/v1`) rather than loading the model in-process, so start the server separately (see `run-server-vllm` for an example) before running this. On startup the script probes `{vllm-url}/models` and swaps in whatever model the server actually has loaded if `--model` doesn't match, so the requested model name doesn't need to be exact.

## Architecture

**Data flow:**
1. Input: `data/raw/<Photos-YY.NN.xmp>/<trip folder>/*.xmp` sidecar files, nested by half-year and then by trip; matching JPEGs live flat (no trip-level nesting) under `data/jpg/<Photos-YYYY.NN>/*.jpg`. `code/bird_label.py` recursively scans `data/raw` only — folders of the same shape sitting directly under `data/` are not picked up.
2. `code/bird_label.py` iterates XMP files, finds the matching JPEG (see JPEG matching below), calls the selected backend
3. Each backend sends a system prompt + base64-encoded JPEG to the model (vLLM/llama.cpp backends call an OpenAI-compatible HTTP server; chatgpt calls OpenAI directly)
4. Model returns JSON: `{category, label, label_cn, confidence}`
5. `code/lib/label_generator.py` formats a compact label with pinyin initials and confidence
6. XMP sidecar gets keywords injected; CSV row appended; checkpoint updated

**JPEG matching** (`find_jpg_for_xmp` in `code/bird_label.py`): half-year folder names don't always line up 1:1 between `raw/` and `jpg/` (a trip filed under one raw half-year can have its JPEGs exported into an adjacent half's jpg folder), and camera filenames can collide across unrelated shoots (counter wraparound). Matching therefore: (1) tries the expected jpg folder derived from the raw folder name (`Photos-23.01.xmp` → `Photos-2023.01`); (2) falls back to a filename-stem index built over the whole `jpg/` tree; (3) if the stem is found in more than one place outside the expected folder, treats it as ambiguous and skips rather than guessing.

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
