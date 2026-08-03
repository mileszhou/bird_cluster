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
import logging
import os
import shutil
import signal
import sys
from datetime import datetime
from pathlib import Path
import csv
import urllib.request
import re
from PIL import Image
import time

from code.lib.label_generator import pinyin_initials
from code.lib.config import server_url
from code.lib.jpg_index import JpgIndex, Verdict, library_year
import code.lib.run_hf_bird_model_llamacpp

logger = logging.getLogger("bird_label")


def setup_logging(log_path: Path) -> None:
    """Mirror stdout's status lines into an INFO-level log file at log_path,
    appending so a resumed run's log stays contiguous with prior runs."""
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

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
                logger.info(f"⚠️  Fixed permissions and updated {xmp_path.name}")
            except Exception as e2:
                logger.info(f"⚠️  Could not write XMP file {xmp_path.name}: {e2}. Skipping.")
    except Exception as e:
        logger.info(f"⚠️  Failed to process XMP file {xmp_path.name}: {e}. Skipping.")


_INJECTED_CATEGORIES = {"bird", "animal", "people", "scenery"}
_INJECTED_LABEL_RE = re.compile(r'^_?.+\(\d+%\)$')


def _is_injected_keyword(text: str) -> bool:
    return text in _INJECTED_CATEGORIES or bool(_INJECTED_LABEL_RE.match(text or ""))


def strip_injected_keywords(raw_out: Path) -> int:
    """Remove keywords previously written by add_keywords_to_xmp() from a fresh
    copy of the raw XMP tree, so re-running against leftover output doesn't
    treat a prior run's own labels as pre-existing metadata. Only removes the
    rdf:Seq (or its parent dc:subject, if the Seq was its only content) that
    matches the exact category/label shapes add_keywords_to_xmp() writes;
    any other pre-existing dc:subject content (e.g. rdf:Bag, rdf:Alt) is left
    untouched. Returns the number of files modified.
    """
    ns = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "dc": "http://purl.org/dc/elements/1.1/",
    }
    import xml.etree.ElementTree as ET
    for prefix, uri in ns.items():
        ET.register_namespace(prefix, uri)

    modified = 0
    for xmp_path in raw_out.rglob('*.xmp'):
        if not xmp_path.is_file():
            continue
        try:
            tree = ET.parse(xmp_path)
        except Exception as e:
            logger.info(f"⚠️  Could not parse {xmp_path.name} while stripping keywords: {e}. Skipping.")
            continue
        root = tree.getroot()
        changed = False
        for desc in root.findall('.//rdf:Description', ns):
            subject = desc.find('dc:subject', ns)
            if subject is None:
                continue
            for seq in subject.findall('rdf:Seq', ns):
                lis = seq.findall('rdf:li', ns)
                if lis and all(_is_injected_keyword(li.text) for li in lis):
                    subject.remove(seq)
                    changed = True
            if len(subject) == 0:
                desc.remove(subject)
                changed = True
        if changed:
            tree.write(xmp_path, encoding='utf-8', xml_declaration=True)
            modified += 1
    return modified

# ------------------------------------------------------------
# vLLM query (via OpenAI-compatible vLLM server, e.g. http://galileo:8000/v1).
# ------------------------------------------------------------
VLLM_SYSTEM_PROMPT = (
    "You are an expert bird, wild animal, and scene identification system. "
    "For the given image, output a JSON object with the following fields: "
    "`category` \u2013 one of: 'bird' (only class Aves \u2014 actual birds), 'animal' (all other animals: insects, butterflies, mammals, reptiles, etc.), 'people', or 'scenery'. "
    "`label` \u2013 for 'bird'/'animal', the English common name using Latin characters only (e.g. 'Grey Wagtail'); "
    "for 'scenery', a concise description naming the specific subject of the scene \u2014 the landmark, landscape feature, or activity in view (e.g. 'Sunset over Lofoten fjord', 'Sahara sand dunes', 'Gothic cathedral facade', 'Hikers on mountain trail') rather than a generic word like 'landscape' or 'outdoor scene'; "
    "for 'people', a brief description of who/what they're doing. "
    "`label_cn` \u2013 the standard Chinese (Mandarin) name or translation of `label` (e.g. '\u7070\u9d3a\u9dcc'). "
    "`confidence` \u2013 a float between 0.0 and 1.0. "
    "Output ONLY a JSON object, no other text."
)


def get_actual_vllm_model_name(vllm_url: str, requested_model: str) -> str:
    """Probes the vLLM OpenAI-compatible server to find the actual loaded model name."""
    base_url = vllm_url.rstrip('/').replace('/v1', '')
    try:
        probe_request = urllib.request.Request(f'{base_url}/v1/models')
        with urllib.request.urlopen(probe_request, timeout=10) as probe_resp:
            models_data = json.loads(probe_resp.read().decode('utf-8'))
            available_models = [m['id'] for m in models_data.get('data', [])]
            if available_models and requested_model not in available_models:
                actual = available_models[0]
                logger.info(f"\u2139\ufe0f  Server mismatch: requested '{requested_model}', but using '{actual}'")
                return actual
    except Exception as e:
        logger.info(f"\u26a0\ufe0f  Could not probe models at {base_url}/v1: {e}")
    return requested_model


def _vllm_chat_completion(messages, model_name: str, vllm_url: str, timeout: int = 120):
    base_url = vllm_url.rstrip('/').replace('/v1', '')
    payload = {
        'model': model_name,
        'messages': messages,
        'max_tokens': 200,
        'temperature': 0.0,
    }
    request = urllib.request.Request(
        url=f'{base_url}/v1/chat/completions',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


def predict_with_vllm(image_path: Path, vllm_url: str, model_name: str, conf_threshold: float, no_bird_conf: float):
    """Returns (category, label, label_cn, confidence, raw_json) from a vLLM server.
    The model is asked to return a JSON object with the following fields:
    - `category` \u2013 'bird', 'animal', 'people', or 'scenery'
    - `label` \u2013 English name or description
    - `label_cn` \u2013 Chinese name
    - `confidence` \u2013 float 0.0\u20111.0
    """
    category = "scenery"
    label = "unknown"
    label_cn = ""
    confidence = 0.0
    raw_json = "{}"
    content = None
    try:
        img_b64 = read_image_base64(image_path)
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": VLLM_SYSTEM_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ]}
        ]
        response = _vllm_chat_completion(messages, model_name, vllm_url)
        msg = response['choices'][0]['message']
        content = msg.get('content') or msg.get('reasoning_content', '')
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
        logger.info(f"\u26a0\ufe0f  vLLM request failed for {image_path.name}: {e}")
        logger.info(f"    raw response: {repr(content)}")
    # Return the collected values (defaults may be unchanged if an error occurred)
    return category, label, label_cn, confidence, raw_json


def predict_with_vllm_batch(image_paths: list, vllm_url: str, model_name: str, conf_threshold: float, no_bird_conf: float):
    """Batch vLLM inference against a vLLM server. Fires requests concurrently so the
    server's continuous batching handles them together. Returns a list of
    (category, label, label_cn, confidence, raw_json), one per input image, in order."""
    results = [("scenery", "unknown", "", 0.0, "{}") for _ in image_paths]
    if not image_paths:
        return results

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(image_paths)) as pool:
        futures = {
            pool.submit(predict_with_vllm, path, vllm_url, model_name, conf_threshold, no_bird_conf): i
            for i, path in enumerate(image_paths)
        }
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.info(f"\u26a0\ufe0f  vLLM request failed for {image_paths[idx].name}: {e}")

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
# Locate the JPEG matching an XMP sidecar.
#
# The export mirrors the photo library, so a sidecar's JPEG sits in the jpg
# tree at the same library/trip folder under the same stem:
#   xmp/Photos-19/2019-01-13 山公园/_D8S0025.xmp
#   jpg/Photos-19/2019-01-13 山公园/_D8S0025.jpg
# Camera counters wrap, so stems repeat across the library -- but with the
# folder part of the key that no longer creates ambiguity, and the whole-tree
# stem fallback the flat export needed (which returned photos from unrelated
# shoots) is gone. See code/lib/jpg_index.py, shared with the embedding step,
# for the derived-export cases: a capture whose only export is a virtual copy
# (`-2`) or an AI Denoise render (`-Enhanced-NR`).
#
# The index is built over RAW_OUT, the working copy of the sidecar tree, which
# mirrors data/xmp exactly -- so the folder keys line up either way.
# ------------------------------------------------------------

_JPG_INDEX = None


def jpg_index(raw_root: Path) -> JpgIndex:
    global _JPG_INDEX
    if _JPG_INDEX is None:
        _JPG_INDEX = JpgIndex(raw_root, JPG_DIR)
    return _JPG_INDEX


def find_jpg_for_xmp(xmp_file: Path, raw_root: Path):
    match = jpg_index(raw_root).resolve(xmp_file)
    if match.ok:
        return match.path
    if match.verdict is Verdict.NO_FOLDER:
        logger.info(f"⚠️  No exported folder for {xmp_file.parent.name}; skipping {xmp_file.name}.")
    return None


def xmp_key(xmp_file: Path, raw_root: Path) -> str:
    """Identity of a sidecar for the checkpoint and the CSV.

    The path relative to the sidecar root, never the bare filename: 10,832 of
    the 34,160 sidecars share a basename with another (counter wraparound
    across trips), so a basename-keyed checkpoint marks thousands of unlabelled
    photos as already done, and a basename-keyed CSV attaches rows to the wrong
    photo.
    """
    try:
        return xmp_file.relative_to(raw_root).as_posix()
    except ValueError:
        return xmp_file.as_posix()


# ------------------------------------------------------------
# Process a single XMP file (extracted for notebook testing)
# ------------------------------------------------------------

def process_single_xmp(xmp_file: Path, csv_writer, args, raw_root: Path) -> None:
    """Process one XMP side‑car file.
    - Finds the matching JPEG.
    - Calls the GPT‑4o model.
    - Updates the XMP with keyword tags.
    - Writes a row to the provided CSV writer.
    - Prints a concise status line.
    """
    jpg_file = find_jpg_for_xmp(xmp_file, raw_root)
    # Avoid early return: handle missing JPEG with an else block.
    if jpg_file is None:
        logger.info(f"⚠️  JPEG missing for {xmp_file.name}, skipping.")
        # Populate placeholder values so the CSV row can still be written if desired.
        label = "unknown"
        label_cn = "未知"
        conf = 0.0
        raw_json = "{}"
        keywords = []
        note = "missing JPEG"
        category = ""
    else:
        switch = args.approach
        if switch == "llama.cpp":
            category, label, label_cn, conf, raw_json = code.lib.run_hf_bird_model_llamacpp.predict_with_llamacpp(jpg_file, args.model, args.conf_threshold, args.no_bird, args.llama_url)
        elif switch == "chatgpt":
            category, label, label_cn, conf, raw_json = predict_with_gpt4o(jpg_file, args.model, args.conf_threshold, args.no_bird)
        elif switch == "vllm":
            category, label, label_cn, conf, raw_json = predict_with_vllm(jpg_file, args.vllm_url, args.model, args.conf_threshold, args.no_bird)
        else: # should not happen due to argparse choices, but handle gracefully:else: # should not happen due to argparse choices, but handle gracefully:
            logger.info(f"⚠️  Unknown approach '{switch}' for {xmp_file.name}, skipping.")
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
        xmp_key(xmp_file, raw_root),
        xmp_file.name,
        category,
        label,
        label_cn,
        f"{conf:.2f}",
        note,
        args.run_label,
        raw_json,
    ])
    logger.info(f"✅ {xmp_file.name} → {', '.join(keywords)} (conf={conf:.2f})")


# ------------------------------------------------------------
# Process all side‑car XMP files, generate CSV, update XMP keywords.
# ------------------------------------------------------------

def load_filter_set(filter_csv: Path, conf_threshold: float) -> set | None:
    """Return a set of sidecar keys to process, based on a prior run's CSV.
    Includes images where category == 'animal' or confidence < conf_threshold.
    Returns None if no filter is requested.

    Keyed by the CSV's `path` column -- the sidecar's path relative to the
    sidecar root. The `filename` column is a bare basename and repeats across
    trips, so it cannot identify a photo.

    Reads the `category` column directly; older rows only encoded it inside
    `note` ("bird (0.90)"), which is parsed as a fallback."""
    if not filter_csv:
        return None
    keys = set()
    # utf-8-sig: tolerates the BOM a spreadsheet adds when the CSV is edited by
    # hand, which would otherwise turn the first header name into "﻿path".
    with open(filter_csv, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if 'path' not in (reader.fieldnames or []):
            sys.exit(f"error: {filter_csv} has no 'path' column -- it predates the mirrored "
                     f"export and its basename keys cannot identify a photo. Re-run without "
                     f"--filter-csv, or filter against a CSV from a current run.")
        for row in reader:
            note = row.get('note', '')
            category = (row.get('category')
                        or (note.split('(')[0].strip() if '(' in note else '')).strip()
            try:
                conf = float(row.get('confidence', 1.0))
            except ValueError:
                conf = 1.0
            if category == 'animal' or conf < conf_threshold:
                keys.add(row['path'])
    return keys


def select_libraries(xmp_root: Path, years: list[str] | None) -> list[Path]:
    """Library folders under the sidecar root, filtered by --years."""
    libraries = sorted(d for d in xmp_root.iterdir() if d.is_dir())
    if years is None:
        return libraries
    kept = [d for d in libraries if library_year(d.name) in years]
    if not kept:
        sys.exit(f"error: --years {','.join(years)} matched no library under {xmp_root} "
                 f"(found: {', '.join(d.name for d in libraries) or 'nothing'})")
    return kept


def process_folder(xmp_root: Path, csv_path: Path, args) -> None:
    # Determine a deterministic order for processing files
    xmp_files = sorted(
        (p for d in select_libraries(xmp_root, getattr(args, 'years', None))
         for p in d.rglob('*.xmp') if p.is_file()),
        key=lambda p: p.as_posix())
    filter_set = load_filter_set(getattr(args, 'filter_csv', None), args.conf_threshold)
    if filter_set is not None:
        logger.info(f"⚙️  Filter active: {len(filter_set)} images selected from prior CSV.")
    # Checkpoint keyed by path relative to xmp_root -- see xmp_key() for why a
    # bare filename silently skips thousands of sidecars.
    checkpoint_path = csv_path.parent / "processed.txt"
    processed: set = set()
    if checkpoint_path.is_file():
        processed = set(line.strip() for line in checkpoint_path.read_text().splitlines())

    keys = {xmp: xmp_key(xmp, xmp_root) for xmp in xmp_files}
    pending = [
        xmp for xmp in xmp_files
        if keys[xmp] not in processed and (filter_set is None or keys[xmp] in filter_set)
    ]

    batch_size = getattr(args, 'batch_size', 1)
    use_batch = args.approach == "vllm" and batch_size > 1
    if use_batch:
        logger.info(f"⚙️  vLLM batch mode: batch_size={batch_size}, {len(pending)} images to process.")

    resuming = bool(processed)
    csv_mode = 'a' if resuming else 'w'
    if resuming:
        logger.info(f"⚙️  Resuming: {len(processed)} already processed, {len(pending)} remaining.")
    with open(csv_path, csv_mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not resuming:
            writer.writerow(['path', 'filename', 'category', 'label', 'label_cn',
                             'confidence', 'note', 'run_label', 'response_json'])

        interrupted = False
        def _sigint_handler(sig, frame):
            nonlocal interrupted
            if not interrupted:
                logger.info("\nInterrupt received — finishing current batch before stopping...")
                interrupted = True
        original_sigint = signal.signal(signal.SIGINT, _sigint_handler)

        step = batch_size if use_batch else 1
        try:
            for i in range(0, len(pending), step):
                if use_batch:
                    batch = pending[i:i + batch_size]
                    # Separate missing JPEGs from valid ones
                    valid_items, missing = [], []
                    for xmp in batch:
                        jpg = find_jpg_for_xmp(xmp, xmp_root)
                        (valid_items if jpg is not None else missing).append((xmp, jpg))
                    for xmp, _ in missing:
                        writer.writerow([keys[xmp], xmp.name, "", "unknown", "未知", "0.00",
                                         "missing JPEG", args.run_label, "{}"])
                        logger.info(f"⚠️  JPEG missing for {xmp.name}, skipping.")
                    if valid_items:
                        try:
                            batch_results = predict_with_vllm_batch(
                                [jpg for _, jpg in valid_items], args.vllm_url, args.model, args.conf_threshold, args.no_bird
                            )
                            for (xmp_file, _), (category, label, label_cn, conf, raw_json) in zip(valid_items, batch_results):
                                keywords = [category, mk_label(category, label, label_cn, conf)]
                                note = f"{category} ({conf:.2f})"
                                add_keywords_to_xmp(xmp_file, keywords)
                                writer.writerow([keys[xmp_file], xmp_file.name, category,
                                                 label, label_cn, f"{conf:.2f}", note,
                                                 args.run_label, raw_json])
                                logger.info(f"✅ {xmp_file.name} → {', '.join(keywords)} (conf={conf:.2f})")
                        except Exception as e:
                            logger.info(f"⚠️  Batch error at index {i}: {e}. Stopping to preserve checkpoint.")
                            break
                    # Checkpoint entire batch (including skipped missing-JPEG files)
                    with open(checkpoint_path, 'a', encoding='utf-8') as cp:
                        for xmp in batch:
                            cp.write(f"{keys[xmp]}\n")
                else:
                    xmp_file = pending[i]
                    try:
                        process_single_xmp(xmp_file, writer, args, raw_root=xmp_root)
                        with open(checkpoint_path, 'a', encoding='utf-8') as cp:
                            cp.write(f"{keys[xmp_file]}\n")
                    except Exception as e:
                        logger.info(f"⚠️  Error processing {xmp_file.name}: {e}. Stopping batch to preserve checkpoint.")
                        break
                if interrupted:
                    logger.info("Stopped cleanly. Re-run to resume.")
                    break
        finally:
            signal.signal(signal.SIGINT, original_sigint)

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
    parser.add_argument("--llama-url", default="", help="URL for LLaMA.cpp API (llama.cpp approach only; default: from config.toml [servers.llama_cpp])")
    parser.add_argument("--vllm-url", default="", help="URL for the vLLM OpenAI-compatible server (vllm approach only; default: from config.toml [servers.vllm])")
    parser.add_argument("--filter-csv", default="", help="Path to a prior run's CSV; only reprocess 'animal' category or low-confidence rows")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of images per vLLM batch (default 1, vllm only)")
    parser.add_argument("--years", default="all", help="Comma-separated years to label, or 'all' (default). Selects library folders by year: 2019 -> Photos-19")
    args = parser.parse_args()

    args.years = None if args.years.strip().lower() == "all" else [
        y.strip() for y in args.years.split(",") if y.strip()
    ]

    if args.approach == "vllm" and not args.vllm_url:
        args.vllm_url = server_url("vllm", path="/v1")
    if args.approach == "llama.cpp" and not args.llama_url:
        args.llama_url = server_url("llama_cpp", path="/v1")

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(OUTPUT_DIR / "bird_label.log")

    # Update args with actual model name if using llama.cpp to ensure args.json is accurate
    if args.approach == "llama.cpp" and args.llama_url:
        try:
            probe_request = urllib.request.Request(f'{args.llama_url.rstrip("/")}/models')
            with urllib.request.urlopen(probe_request, timeout=5) as probe_resp:
                models_data = json.loads(probe_resp.read().decode('utf-8'))
                available_models = [m['id'] for m in models_data.get('data', [])]
                if available_models and args.model not in available_models:
                    logger.info(f"ℹ️  [args.json fix] Server mismatch: requested '{args.model}', but using '{available_models[0]}'")
                    args.model = available_models[0]
        except Exception as e:
            logger.info(f"⚠️  [args.json fix] Could not probe models for args.json: {e}")
    elif args.approach == "vllm" and args.vllm_url:
        args.model = get_actual_vllm_model_name(args.vllm_url, args.model)

    # Paths. `data/` is treated as read-only: the sidecar tree is copied into
    # output/raw and keywords are injected there, never back into the submodule.
    DATA_DIR = Path(args.data_dir)
    JPG_DIR = DATA_DIR / "jpg"
    RAW_DIR = DATA_DIR / "xmp"
    if not RAW_DIR.is_dir() or not JPG_DIR.is_dir():
        sys.exit(f"error: expected {RAW_DIR} and {JPG_DIR} to exist")

    # Use a single output folder (no per‑run subdirectory)
    RUN_DIR = OUTPUT_DIR
    RAW_OUT = RUN_DIR / "raw"
    CSV_PATH = RUN_DIR / "bird_identification_output.csv"

    # Ensure output directories exist; copy raw data only if not already present
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_OUT.exists():
        shutil.copytree(RAW_DIR, RAW_OUT)
        stripped = strip_injected_keywords(RAW_OUT)
        if stripped:
            logger.info(f"🧹 Stripped leftover injected keywords from {stripped} XMP file(s) in {RAW_OUT}.")
    else:
        logger.info(f"⚙️  Raw output folder {RAW_OUT} already exists; reusing existing files.")
    # Save command‑line args for reproducibility (overwrites previous args.json)
    import subprocess
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_hash = "unknown"
    with open(RUN_DIR / "args.json", "w", encoding="utf-8") as f:
        json.dump({**vars(args), "git_commit": git_hash}, f, indent=2)

    start_time = time.perf_counter()

    if args.approach == "vllm":
        logger.info(f"🔧 Using vLLM server at {args.vllm_url} with model '{args.model}'.")

    init_time = time.perf_counter()
    init_elapsed = init_time - start_time
    logger.info(f"⏱️  Initialization complete in {init_elapsed:.1f} seconds.")

    # Process and generate CSV
    logger.info("\n🔧 Processing side‑car XMP files…")
    process_folder(RAW_OUT, CSV_PATH, args)

    end_time = time.perf_counter()
    processing_elapsed = end_time - init_time
    logger.info(f"⏱️  Initialization complete in {init_elapsed:.1f} seconds.")
    logger.info(f"⏱️  Processing time: {processing_elapsed:.1f} seconds.")
    
    logger.info(f"\n✅ Run complete. Output stored in: {RUN_DIR}")

