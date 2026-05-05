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
./rungpt                          # GPT-4o (requires OPENAI_API_KEY)
./runcpp                          # llama.cpp server at http://darwin:8080
./vllm.run                        # vLLM with Qwen2-VL-7B-Instruct
./runtf                           # HuggingFace Transformers

# Or directly with options:
python3 -m code.bird_label --approach vllm --model "Qwen/Qwen2-VL-7B-Instruct" --conf-threshold 0.6
```

Key CLI flags:
- `--approach` — `chatgpt`, `llama.cpp`, `vllm`, or `transformer`
- `--conf-threshold FLOAT` — confidence below which to flag as low-confidence (default 0.6)
- `--no-bird FLOAT` — confidence below which to mark as "no bird" (default 0.2)
- `--filter-csv PATH` — re-process only "animal" or low-confidence rows from a prior run's CSV
- `--run-label TEXT` — tag this run in the output CSV
- `--batch-size INT` — number of images per vLLM batch (default 1; 8 recommended for 32B model on 2 GPUs)

Reset output: `./clean`

`vllm.run` is the recommended approach — significantly faster than llama.cpp, fully occupies the GPU. The vLLM engine is created once before the processing loop (expensive CUDA kernel compilation on first run, cached in `~/.cache/vllm/` afterward).

## Architecture

**Data flow:**
1. Input: `data/jpg/*.jpg` paired with `data/raw/*.xmp` sidecar files
2. `code/bird_label.py` iterates XMP files, finds matching JPEG, calls the selected backend
3. Each backend sends a system prompt + PIL image via vLLM's chat template
4. Model returns JSON: `{category, label, label_cn, confidence}`
5. `code/lib/label_generator.py` formats a compact label with pinyin initials and confidence
6. XMP sidecar gets keywords injected; CSV row appended; checkpoint updated

**Outputs** (in `output/`):
- `bird_identification_output.csv` — main results (filename, label, label_cn, confidence, note, run_label, response_json)
- `args.json` — CLI arguments for reproducibility
- `processed.txt` — checkpoint; delete to reprocess all images
- `raw/` — XMP files with keywords added

**Backend modules** (all in `code/bird_label.py`):
- `predict_with_vllm()` — GPU inference via vLLM OpenAI-compatible API
- `predict_with_gpt4o()` — OpenAI cloud API
- `predict_with_llamacpp()` — llama.cpp OpenAI-compatible server

**Supporting library** (`code/lib/`):
- `label_generator.py` — formats `{pinyin_initials}-{chinese_name}-{english_name}({confidence}%)` labels
- `transformers_engine.py` — HuggingFace Transformers backend (in progress)

## Key Behaviors

- **Checkpoint resumption:** Already-processed filenames in `output/processed.txt` are skipped on re-run; CSV is opened in append mode so prior results are preserved
- **XMP permissions:** Script calls `chmod` on XMP files before writing keywords (needed for library-managed files)
- **JSON parsing:** Model may wrap response in markdown fences or return Chinese in the `label` field; post-processing strips fences and moves Chinese characters from `label` to `label_cn` automatically
- **vLLM prompt constraints:** `category: "bird"` is strictly class Aves; insects/butterflies must be `animal`. `label` must use Latin characters only; `label_cn` is Mandarin.
- **Filter mode:** `--filter-csv` reads a prior output CSV and only reprocesses rows where `note` contains "animal" or confidence is below threshold
- **Graceful Ctrl-C:** SIGINT is intercepted with a deferred handler — the signal sets a flag rather than raising immediately, so the current batch always completes fully (inference, XMP write, CSV write, checkpoint) before the loop exits. This synchronises the async signal with the program's batch boundary, ensuring no partial writes.
