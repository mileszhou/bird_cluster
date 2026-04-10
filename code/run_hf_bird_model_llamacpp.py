#!/usr/bin/env python3
"""run_hf_bird_model_llamacpp.py – Bird identification using local llama.cpp or Transformers.

- Uses llama.cpp server or local Transformers library to extract **English** and **Chinese** bird names, plus confidence.
- Output CSV includes: `filename`, `label` (English), `label_cn` (Chinese), `confidence`, `note`, `run_label`, `response_json`.
- Run with: python3 run_hf_bird_model_llamacpp.py --use-llamacpp --llama-url http://localhost:8080
"""

from code.lib.transformers_engine import predict_with_transformers
from code.lib.label_generator import pinyin_initials
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

# ------------------------------------------------------------
# Helper: read an image and encode as base64.
# ------------------------------------------------------------

def read_image_base64(image_path: Path) -> str:
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

# ------------------------------------------------------------
# OpenAI API helper (existing)
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
# llama.cpp API helper (NEW)
# ------------------------------------------------------------

def _llamacpp_chat_completion(messages, model_name, base_url: str):
    """Send a request to a local llama.cpp server's /v1/chat/completions endpoint."""
    payload = {
        'model': model_name,
        'messages': messages,
        'max_tokens': 4096,
        'temperature': 0.0,
    }
    request = urllib.request.Request(
        url=f'{base_url.rstrip("/")}/chat/completions',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            resp_body = response.read().decode('utf-8')
            return json.loads(resp_body)
    except urllib.error.URLError as e:
        raise RuntimeError(f'Failed to connect to llama.cpp server at {base_url}: {e}')

# ------------------------------------------------------------
# XMP keyword handling (unchanged from original).
# ------------------------------------------------------------

def add_keywords_to_xmp(xmp_path: Path, keywords):
    ns = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "dc": "http://purl.org/dc/elements/1.1/",
    }
    import xml.etree.ElementTree as ET
    for prefix, uri in ns.items():
        ET.register_namespace(prefix, uri)

    try:
        tree = ET.parse(xmp_path)
        root = tree.getroot()

        desc = root.find('.//rdf:Description', ns)
        if desc is None:
            rdf_elem = root.find('rdf:RDF', ns)
            if rdf_elem is None:
                rdf_elem = ET.SubElement(root, f"{{{ns['rdf']}}}RDF")
            desc = ET.SubElement(rdf_elem, f"{{{ns['rdf']}}}Description")

        subject = desc.find('dc:subject', ns)
        if subject is None:
            subject = ET.SubElement(desc, f"{{{ns['dc']}}}subject")

        seq = subject.find('rdf:Seq', ns)
        if seq is None:
            seq = ET.SubElement(subject, f"{{{ns['rdf']}}}Seq")

        existing = {li.text for li_elem in seq.findall('rdf:li', ns) if (li := li_elem.text)}
        for kw in keywords:
            if kw not in existing:
                li = ET.SubElement(seq, f"{{{ns['rdf']}}}li")
                li.text = kw

        try:
            tree.write(xmp_path, encoding='utf-8', xml_declaration=True)
        except PermissionError:
            try:
                xmp_path.chmod(0o666)
                tree.write(xmp_path, encoding='utf-8', xml_declaration=True)
                print(f"⚠️  Fixed permissions and updated {xmp_path.name}")
            except Exception as e2:
                print(f"⚠️  Could not write XMP file {xmp_path.name}: {e2}. Skipping.")
    except Exception as e:
        print(f"⚠️  Failed to process XMP file {xmp_path.name}: {e}. Skipping.")

# ------------------------------------------------------------
# OpenAI GPT‑4o Vision query (existing).
# ------------------------------------------------------------

def predict_with_gpt4o(image_path: Path, model_name: str, conf_threshold: float, no_bird_conf: float):
    img_b64 = read_image_base64(image_path)
    system_prompt = (
        "You are an expert bird and wild animal identification system. "
        "For the given image, output a JSON object with the following fields: "
        "`category` – a string that must be one of: 'bird', 'animal', 'people', or 'scenery'. "
        "`label` – the English name of the bird/animal, or a brief English description if the category is 'people' or 'scenery'."
        "`label_cn` – the Chinese name corresponding to `label`."
        "`confidence` – a float between 0.0 and 1.0 indicating the model's confidence. "
        "If the image contains no recognizable animal, set `category` to 'people' or 'scenery' as appropriate and provide an appropriate English description, leaving `label_cn` blank."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}
    ]

    category = "scenery"
    label = "unknown"
    label_cn = ""
    confidence = 0.0
    raw_json = "{}"
    try:
        response = _openai_chat_completion(messages, model_name)
        content = response['choices'][0]['message']['content']
        try:
            data = json.loads(content)
        except json.JSONDecode_decodeError:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if not match:
                raise ValueError('No JSON found in OpenAI response')
            data = json.loads(match.group())
        raw_json = json.dumps(data, ensure_ascii=False)
        label = data.get('label', 'unknown').lower()
        label_cn = data.get('label_cn', '未知')
        confidence = float(data.get('confidence', 0.0))
        category = data.get('category', 'scenery')
    except Exception as e:
        print(f"⚠️  OpenAI request failed for {image_path.name}: {e}")
    return category, label, label_cn, confidence, raw_json

# ------------------------------------------------------------
# llama.cpp Vision query (NEW)
# ------------------------------------------------------------

def predict_with_llamacpp(image_path: Path, model_name: str, conf_threshold: float, no_bird_conf: float, llama_url: str):
    img_b64 = read_image_base64(image_path)
    system_prompt = (
        "You are an expert ornithologist and wildlife photographer specializing in birds of China and East Asia. "
        "Identify the subject in the image to the most specific taxonomic level possible — always prefer full species name over genus or family. "
        "Use plumage details, body shape, beak, tail, eye markings, and habitat cues visible in the image. "
        "If the bird is common in China, use the standard Chinese species name. "
        "Output ONLY a JSON object with these fields: "
        "`category` – one of: 'bird', 'animal', 'people', or 'scenery'. "
        "`label` – full English species name (e.g. 'Light-vented Bulbul', not just 'Bulbul'). "
        "`label_cn` – standard Chinese species name (e.g. '白头鹎', not just '鹎'). "
        "`confidence` – float 0.0–1.0 reflecting how certain you are of the species identification. "
        "If the image contains no recognizable animal, set `category` to 'people' or 'scenery' and leave `label_cn` blank. "
        "Do not explain your reasoning. Output the JSON object only."
    )
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": system_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ]}
    ]
    category = "scenery"
    label = "unknown"
    label_cn = ""
    confidence = 0.0
    raw_json = "{}"
    try:
        response = _llamacpp_chat_completion(messages, model_name, llama_url)
        msg = response['choices'][0]['message']
        content = msg.get('content') or msg.get('reasoning_content', '')
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if not match:
                raise ValueError('No JSON found in llama.cpp response')
            data = json.loads(match.group())
        raw_json = json.dumps(data, ensure_ascii=False)
        label = data.get('label', 'unknown').lower()
        label_cn = data.get('label_cn', '未知')
        confidence = float(data.get('confidence', 0.0))
        category = data.get('category', 'scenery')
    except Exception as e:
        print(f"⚠️  llama.cpp request failed for {image_path.name}: {e}")
    return category, label, label_cn, confidence, raw_json

# (Rest of script logic omitted for brevity in this thought block, but will be included in the write)