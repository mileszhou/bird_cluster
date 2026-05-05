#!/usr/bin/env python3
"""run_hf_bird_model_chatgpt.py – Bird identification using OpenAI GPT‑4o (Vision).

- Uses the OpenAI Vision API to extract **English** and **Chinese** bird names, plus confidence.
- Output CSV includes: `filename`, `label` (English), `label_cn` (Chinese), `confidence`, `note`, `response_json`.
- No logits are available from the OpenAI API (as noted).

Run with the same CLI flags as before, for example:
```
python3 run_hf_bird_model_chatgpt.py --conf-threshold 0.6 --no-bird 0.2
```
"""

import argparse
import base64
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
import csv
import urllib.request
import re
from PIL import Image

from code.lib.label_generator import pinyin_initials
import code.run_hf_bird_model_llamacpp

# ------------------------------------------------------------
# Helper: read an image and encode as base64 for the OpenAI API.
# ------------------------------------------------------------

def read_image_base64(image_path: Path) -> str:
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

# ------------------------------------------------------------
# Helper: call OpenAI Chat Completion API using only stdlib.
# ------------------------------------------------------------
def _openai_chat_completion(messages, model_name):
    """Send a request to OpenAI's /v1/chat/completions endpoint using urllib."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY not set in environment')
    payload = {
        'model': model_name,
        'messages': messages,
        'max_tokens': 80,
        'temperature': 0.0,
    }
    request = urllib.request.Request(
        url='https://api.openai.com/v1/chat/completions',
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        }
    )
    with urllib.request.urlopen(request) as response:
        resp_body = response.read().decode('utf-8')
        return json.loads(resp_body)

# ------------------------------------------------------------
# XMP keyword handling (unchanged from original).
# ------------------------------------------------------------

def add_keywords_to_xmp(xmp_path: Path, keywords):
    """Add keywords to an XMP side‑car file.
    This version catches permission errors (e.g., when files are read‑only) and
    attempts to fix the file mode before retrying. If it still fails, it logs a
    warning and skips the file so the overall processing does not abort.
    """
    ns = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "dc": "http://purl.org/dc/elements/1.1/",
    }
    # Ensure namespaces are registered for new files.
    import xml.etree.ElementTree as ET
    for prefix, uri in ns.items():
        ET.register_namespace(prefix, uri)

    try:
        tree = ET.parse(xmp_path)
        root = tree.getroot()

        # Find or create the RDF Description block
        desc = root.find('.//rdf:Description', ns)
        if desc is None:
            rdf_elem = root.find('rdf:RDF', ns)
            if rdf_elem is None:
                rdf_elem = ET.SubElement(root, f"{{{ns['rdf']}}}RDF")
            desc = ET.SubElement(rdf_elem, f"{{{ns['rdf']}}}Description")

        # Find or create dc:subject
        subject = desc.find('dc:subject', ns)
        if subject is None:
            subject = ET.SubElement(desc, f"{{{ns['dc']}}}subject")

        # Find or create rdf:Seq within subject
        seq = subject.find('rdf:Seq', ns)
        if seq is None:
            seq = ET.SubElement(subject, f"{{{ns['rdf']}}}Seq")

        existing = {li.text for li in seq.findall('rdf:li', ns) if li.text}
        for kw in keywords:
            if kw not in existing:
                li = ET.SubElement(seq, f"{{{ns['rdf']}}}li")
                li.text = kw

        # Write back, handling PermissionError.
        try:
            tree.write(xmp_path, encoding='utf-8', xml_declaration=True)
        except PermissionError:
            # Try to make file writable and retry once.
            try:
                xmp_path.chmod(0o666)
                tree.write(xmp_path, encoding='utf-8', xml_declaration=True)
                print(f"⚠️  Fixed permissions and updated {xmp_path.name}")
            except Exception as e2:
                print(f"⚠️  Could not write XMP file {xmp_path.name}: {e2}. Skipping.")
    except Exception as e:
        print(f"⚠️  Failed to process XMP file {xmp_path.name}: {e}. Skipping.")

# ------------------------------------------------------------
# vLLM query.
# ------------------------------------------------------------
def predict_with_vllm(image_path: Path, llm, conf_threshold: float, no_bird_conf: float):
    """Returns (category, label, label_cn, confidence, raw_json) from vLLM.
    The model is asked to return a JSON object with the following fields:
    - `category` – 'bird', 'animal', 'people', or 'scenery'
    - `label` – English name or description
    - `label_cn` – Chinese name
    - `confidence` – float 0.0‑1.0
    """
    system_prompt = (
        "You are an expert bird and wild animal identification system. "
        "For the given image, output a JSON object with the following fields: "
        "`category` – one of: 'bird' (only class Aves — actual birds), 'animal' (all other animals: insects, butterflies, mammals, reptiles, etc.), 'people', or 'scenery'. "
        "`label` – the English common name using Latin characters only (e.g. 'Grey Wagtail'). "
        "`label_cn` – the standard Chinese (Mandarin) name (e.g. '灰鶺鸰'). "
        "`confidence` – a float between 0.0 and 1.0. "
        "Output ONLY a JSON object, no other text."
    )

    # Prepare defaults in case of failure
    category = "scenery"
    label = "unknown"
    label_cn = ""
    confidence = 0.0
    raw_json = "{}"
    content = None
    try:
        image = Image.open(image_path)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Identify this image."}]}
        ]
        from vllm import SamplingParams
        tokenizer = llm.get_tokenizer()
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        sampling_params = SamplingParams(max_tokens=200, temperature=0.0)
        response = llm.generate({"prompt": prompt, "multi_modal_data": {"image": image}}, sampling_params, use_tqdm=False)
        content = response[0].outputs[0].text
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Extract JSON block if there is surrounding text
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if not match:
                raise ValueError('No JSON found in vLLM response')
            data = json.loads(match.group())
        raw_json = json.dumps(data, ensure_ascii=False)
        label = data.get('label', '').lower()
        label_cn = data.get('label_cn', '')
        confidence = float(data.get('confidence', 0.0))
        category = data.get('category', 'scenery')
        # If model put Chinese in label field, move it to label_cn
        if any('\u4e00' <= c <= '\u9fff' for c in label):
            if not label_cn:
                label_cn = label
            label = ''
        label = label or 'unknown'
    except Exception as e:
        print(f"⚠️  vLLM request failed for {image_path.name}: {e}")
        print(f"    raw response: {repr(content)}")
    # Return the collected values (defaults may be unchanged if an error occurred)
    return category, label, label_cn, confidence, raw_json


def predict_with_vllm_batch(image_paths: list, llm, conf_threshold: float, no_bird_conf: float):
    """Batch vLLM inference. Returns list of (category, label, label_cn, confidence, raw_json)."""
    system_prompt = (
        "You are an expert bird and wild animal identification system. "
        "For the given image, output a JSON object with the following fields: "
        "`category` – one of: 'bird' (only class Aves — actual birds), 'animal' (all other animals: insects, butterflies, mammals, reptiles, etc.), 'people', or 'scenery'. "
        "`label` – the English common name using Latin characters only (e.g. 'Grey Wagtail'). "
        "`label_cn` – the standard Chinese (Mandarin) name (e.g. '灰鶺鸰'). "
        "`confidence` – a float between 0.0 and 1.0. "
        "Output ONLY a JSON object, no other text."
    )
    results = [("scenery", "unknown", "", 0.0, "{}") for _ in image_paths]

    from vllm import SamplingParams
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(max_tokens=200, temperature=0.0)

    inputs = []
    valid_indices = []
    for i, image_path in enumerate(image_paths):
        try:
            image = Image.open(image_path)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Identify this image."}]}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})
            valid_indices.append(i)
        except Exception as e:
            print(f"⚠️  Could not load image {image_path.name}: {e}")

    if not inputs:
        return results

    responses = llm.generate(inputs, sampling_params, use_tqdm=False)

    for idx, response in zip(valid_indices, responses):
        content = None
        try:
            content = response.outputs[0].text
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if not match:
                    raise ValueError('No JSON found in vLLM response')
                data = json.loads(match.group())
            raw_json = json.dumps(data, ensure_ascii=False)
            label = data.get('label', '').lower()
            label_cn = data.get('label_cn', '')
            confidence = float(data.get('confidence', 0.0))
            category = data.get('category', 'scenery')
            if any('一' <= c <= '鿿' for c in label):
                if not label_cn:
                    label_cn = label
                label = ''
            label = label or 'unknown'
            results[idx] = (category, label, label_cn, confidence, raw_json)
        except Exception as e:
            print(f"⚠️  vLLM parse failed for {image_paths[idx].name}: {e}")
            print(f"    raw response: {repr(content)}")

    return results

# ------------------------------------------------------------
# Helper to build a compact label string for animal entries.
# ------------------------------------------------------------
def mk_label(category: str, label_en: str, label_cn: str, conf: float) -> str:
    """Construct a label like 'zs-zhangsan (95%)'.
    `label_en` – English name, `label_cn` – Chinese name, `conf` – confidence (0‑1).
    """
    ret = ""
    if category in ("bird", "animal"):
        c_init = pinyin_initials(label_cn)
        ret = f"{c_init}-{label_cn}-{label_en}({conf * 100:.0f}%)"
    else:
        ret = f"{label_en}({conf * 100:.0f}%)"
    if conf<=0.2:
        ret = f"_{ret}"  # low‑confidence marker for non‑bird
    return ret

# ------------------------------------------------------------
# Process a single XMP file (extracted for notebook testing)
# ------------------------------------------------------------

def process_single_xmp(xmp_file: Path, csv_writer, args, llm=None) -> None:
    """Process one XMP side‑car file.
    - Finds the matching JPEG.
    - Calls the GPT‑4o model.
    - Updates the XMP with keyword tags.
    - Writes a row to the provided CSV writer.
    - Prints a concise status line.
    """
    base = xmp_file.stem
    jpg_file = JPG_DIR / f"{base}.jpg"
    # Avoid early return: handle missing JPEG with an else block.
    if not jpg_file.is_file():
        print(f"⚠️  JPEG missing for {xmp_file.name}, skipping.")
        # Populate placeholder values so the CSV row can still be written if desired.
        label = "unknown"
        label_cn = "未知"
        conf = 0.0
        raw_json = "{}"
        keywords = []
        note = "missing JPEG"
    else:
        switch = args.approach
        if switch == "llama.cpp":
            category, label, label_cn, conf, raw_json = code.run_hf_bird_model_llamacpp.predict_with_llamacpp(jpg_file, args.model, args.conf_threshold, args.no_bird, args.llama_url)
        elif switch == "chatgpt":
            category, label, label_cn, conf, raw_json = predict_with_gpt4o(jpg_file, args.model, args.conf_threshold, args.no_bird)
        elif switch == "vllm":
            category, label, label_cn, conf, raw_json = predict_with_vllm(jpg_file, llm, args.conf_threshold, args.no_bird)
        else: # should not happen due to argparse choices, but handle gracefully:else: # should not happen due to argparse choices, but handle gracefully:
            print(f"⚠️  Unknown approach '{switch}' for {xmp_file.name}, skipping.")
            category, label, label_cn, conf, raw_json = "scenery", "unknown", "未知", 0.0, "{}"

        # Choose keywords based on the returned category
        keywords = []
        keywords.append(category)
        spec = mk_label(category, label, label_cn, conf)
        keywords.append(spec)
        note = f"{category} ({conf:.2f})"

        # Update XMP file (handles permission issues internally)
        add_keywords_to_xmp(xmp_file, keywords)

    # Write CSV row (common for both branches)
    csv_writer.writerow([
        xmp_file.name,
        label,
        label_cn,
        f"{conf:.2f}",
        note,
        args.run_label,
        raw_json,
    ])
    print(f"✅ {xmp_file.name} → {', '.join(keywords)} (conf={conf:.2f})")


# ------------------------------------------------------------
# Process all side‑car XMP files, generate CSV, update XMP keywords.
# ------------------------------------------------------------

def load_filter_set(filter_csv: Path, conf_threshold: float) -> set | None:
    """Return a set of xmp filenames to process, based on a prior run's CSV.
    Includes images where category == 'animal' or confidence < conf_threshold.
    Returns None if no filter is requested."""
    if not filter_csv:
        return None
    filenames = set()
    with open(filter_csv, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            note = row.get('note', '')
            category = note.split('(')[0].strip() if '(' in note else ''
            try:
                conf = float(row.get('confidence', 1.0))
            except ValueError:
                conf = 1.0
            if category == 'animal' or conf < conf_threshold:
                # CSV has jpg filename; derive xmp name
                stem = Path(row['filename']).stem
                filenames.add(stem + '.xmp')
    return filenames

def process_folder(xmp_root: Path, csv_path: Path, args, llm=None) -> None:
    # Determine a deterministic order for processing files
    xmp_files = sorted(xmp_root.rglob('*.xmp'), key=lambda p: p.as_posix())
    filter_set = load_filter_set(getattr(args, 'filter_csv', None), args.conf_threshold)
    if filter_set is not None:
        print(f"⚙️  Filter active: {len(filter_set)} images selected from prior CSV.")
    # Checkpoint file to record processed filenames
    checkpoint_path = csv_path.parent / "processed.txt"
    processed: set = set()
    if checkpoint_path.is_file():
        processed = set(line.strip() for line in checkpoint_path.read_text().splitlines())

    pending = [
        xmp for xmp in xmp_files
        if xmp.name not in processed and (filter_set is None or xmp.name in filter_set)
    ]

    batch_size = getattr(args, 'batch_size', 1)
    use_batch = args.approach == "vllm" and batch_size > 1 and llm is not None
    if use_batch:
        print(f"⚙️  vLLM batch mode: batch_size={batch_size}, {len(pending)} images to process.")

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'label', 'label_cn', 'confidence', 'note', 'run_label', 'response_json'])

        step = batch_size if use_batch else 1
        for i in range(0, len(pending), step):
            if use_batch:
                batch = pending[i:i + batch_size]
                # Separate missing JPEGs from valid ones
                valid_items, missing = [], []
                for xmp in batch:
                    jpg = JPG_DIR / f"{xmp.stem}.jpg"
                    (valid_items if jpg.is_file() else missing).append((xmp, jpg))
                for xmp, _ in missing:
                    writer.writerow([xmp.name, "unknown", "未知", "0.00", "missing JPEG", args.run_label, "{}"])
                    print(f"⚠️  JPEG missing for {xmp.name}, skipping.")
                if valid_items:
                    try:
                        batch_results = predict_with_vllm_batch(
                            [jpg for _, jpg in valid_items], llm, args.conf_threshold, args.no_bird
                        )
                        for (xmp_file, _), (category, label, label_cn, conf, raw_json) in zip(valid_items, batch_results):
                            keywords = [category, mk_label(category, label, label_cn, conf)]
                            note = f"{category} ({conf:.2f})"
                            add_keywords_to_xmp(xmp_file, keywords)
                            writer.writerow([xmp_file.name, label, label_cn, f"{conf:.2f}", note, args.run_label, raw_json])
                            print(f"✅ {xmp_file.name} → {', '.join(keywords)} (conf={conf:.2f})")
                    except Exception as e:
                        print(f"⚠️  Batch error at index {i}: {e}. Stopping to preserve checkpoint.")
                        break
                # Checkpoint entire batch (including skipped missing-JPEG files)
                with open(checkpoint_path, 'a', encoding='utf-8') as cp:
                    for xmp in batch:
                        cp.write(f"{xmp.name}\n")
            else:
                xmp_file = pending[i]
                try:
                    process_single_xmp(xmp_file, writer, args, llm=llm)
                    with open(checkpoint_path, 'a', encoding='utf-8') as cp:
                        cp.write(f"{xmp_file.name}\n")
                except Exception as e:
                    print(f"⚠️  Error processing {xmp_file.name}: {e}. Stopping batch to preserve checkpoint.")
                    break

# ------------------------------------------------------------
# Main script execution
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bird ID using GPT‑4o (Vision) with Chinese name support")
    parser.add_argument("--run-label", default="", help="Label for this run (e.g., 'first successful run')")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model (default gpt-4o)")
    parser.add_argument("--conf-threshold", type=float, default=0.6, help="Low‑confidence threshold for special keyword (default 0.6)")
    parser.add_argument("--no-bird", type=float, default=0.2, help="Confidence below which we label as 'no bird' (default 0.2)")
    parser.add_argument("--output-dir", default="./output", help="Directory for run outputs")
    parser.add_argument("--data-dir", default="./data", help="Root data directory (contains jpg/ raw)")
    parser.add_argument("--approach", choices=["chatgpt", "llama.cpp", "vllm"], default="llama.cpp", help="Use LLaMA.cpp API instead of OpenAI (ignored in this script)")
    parser.add_argument("--llama-url", default="", help="URL for LLaMA.cpp API (ignored in this script)")
    parser.add_argument("--filter-csv", default="", help="Path to a prior run's CSV; only reprocess 'animal' category or low-confidence rows")
    parser.add_argument("--tensor-parallel", type=int, default=1, help="Number of GPUs for tensor parallelism (default 1)")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of images per vLLM batch (default 1, vllm only)")
    args = parser.parse_args()

    # Update args with actual model name if using llama.cpp to ensure args.json is accurate
    if args.approach == "llama.cpp" and args.llama_url:
        try:
            probe_request = urllib.request.Request(f'{args.llama_url.rstrip("/")}/v1/models')
            with urllib.request.urlopen(probe_request, timeout=5) as probe_resp:
                models_data = json.loads(probe_resp.read().decode('utf-8'))
                available_models = [m['id'] for m in models_data.get('data', [])]
                if available_models and args.model not in available_models:
                    print(f"ℹ️  [args.json fix] Server mismatch: requested '{args.model}', but using '{available_models[0]}'")
                    args.model = available_models[0]
        except Exception as e:
            print(f"⚠️  [args.json fix] Could not probe models for args.json: {e}")

    # Paths
    DATA_DIR = Path(args.data_dir)
    JPG_DIR = DATA_DIR / "jpg"
    RAW_DIR = DATA_DIR / "raw"
    OUTPUT_DIR = Path(args.output_dir)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use a single output folder (no per‑run subdirectory)
    RUN_DIR = OUTPUT_DIR
    RAW_OUT = RUN_DIR / "raw"
    CSV_PATH = RUN_DIR / "bird_identification_output.csv"

    # Ensure output directories exist; copy raw data only if not already present
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_OUT.exists():
        shutil.copytree(RAW_DIR, RAW_OUT)
    else:
        print(f"⚙️  Raw output folder {RAW_OUT} already exists; reusing existing files.")
    # Save command‑line args for reproducibility (overwrites previous args.json)
    import subprocess
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_hash = "unknown"
    with open(RUN_DIR / "args.json", "w", encoding="utf-8") as f:
        json.dump({**vars(args), "git_commit": git_hash}, f, indent=2)

    # Create the vLLM engine once, outside the processing loop
    llm_engine = None
    if args.approach == "vllm":
        from vllm import LLM
        print(f"\n🔧 Loading vLLM engine for {args.model}…")
        llm_engine = LLM(model=args.model,
                         max_model_len=8192,
                         gpu_memory_utilization=0.95,
                         limit_mm_per_prompt={"image": 1},
                         tensor_parallel_size=args.tensor_parallel
                         )

    # Process and generate CSV
    print("\n🔧 Processing side‑car XMP files…")
    process_folder(RAW_OUT, CSV_PATH, args, llm=llm_engine)
    print("\n✅ Run complete. Output stored in:", RUN_DIR)
