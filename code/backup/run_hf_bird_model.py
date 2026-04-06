#!/usr/bin/env python3
"""
run_hf_bird_model_output.py

This version implements the **standardised workflow** you described:

1. **Data layout**
   - Raw side‑card XMP files live in ``./data/raw`` (parallel to ``./data/jpg``).
   - The script copies only the ``raw`` folder (plus the JPG folder for image access) into a per‑run output directory.
2. **Per‑run output**
   - Each execution creates ``./output/run_YYYYMMDD_HHMMSS/``.
   - Inside that folder you’ll find:
        * ``raw/`` – the processed side‑cards (modified XMP files).
        * ``bird_identification_results.csv`` – summary of predictions.
        * ``args.json`` – a record of the arguments used for this run.
3. **Bird‑identification**
   - Uses the Hugging Face model ``chriamue/bird-species-classifier``.
   - Bird name is stored in **lower‑case** (``sparrow`` → ``s‑unknown‑sparrow``).
   - Low‑confidence predictions (< ``CONF_THRESHOLD``) are prefixed with an underscore (e.g. ``_s‑unknown‑sparrow``).
   - Images with confidence below ``NO_BIRD_CONF`` are labelled ``_nb``.
4. **Extensible arguments** – you can pass ``--threshold`` and ``--no-bird`` on the command line; they are saved to ``args.json``.

The script is meant to be edited in‑place for future development cycles.
"""

import argparse
import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import xml.etree.ElementTree as ET

# ------------------------------------------------------------------
# Default configuration (can be overridden via CLI args)
# ------------------------------------------------------------------
DEFAULT_MODEL = "chriamue/bird-species-classifier"
DEFAULT_CONF_THRESHOLD = 0.70   # below this → low‑confidence marker
DEFAULT_NO_BIRD_CONF = 0.30     # below this → treat as no bird

# ------------------------------------------------------------------
# Argument parsing (allows easy tweaking for future runs)
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Batch bird identification with HF model")
parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model name")
parser.add_argument("--conf-threshold", type=float, default=DEFAULT_CONF_THRESHOLD,
                    help="Confidence above which the label is considered high confidence")
parser.add_argument("--no-bird-conf", type=float, default=DEFAULT_NO_BIRD_CONF,
                    help="Confidence below which the image is considered to contain no bird")
parser.add_argument("--output-root", default="./output",
                    help="Root folder where per‑run directories are created")
parser.add_argument("--data-root", default="./data",
                    help="Root data folder containing 'jpg' and 'raw' subfolders")
args = parser.parse_args()

# ------------------------------------------------------------------
# Paths derived from arguments
# ------------------------------------------------------------------
DATA_ROOT   = Path(args.data_root)
RAW_ROOT    = DATA_ROOT / "raw"          # side‑card XMP files
JPG_ROOT    = DATA_ROOT / "jpg"          # original images (read‑only)
OUTPUT_ROOT = Path(args.output_root)

# Create a timestamped run directory
run_timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
RUN_DIR = OUTPUT_ROOT / run_timestamp
RAW_OUT = RUN_DIR / "raw"               # where processed XMPs will live
CSV_OUT = RUN_DIR / "bird_identification_results.csv"
ARGS_OUT = RUN_DIR / "args.json"

# --------------------------
# Device
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ------------------------------------------------------------------
# Helper: copy raw side‑cards (only) into the run folder
# ------------------------------------------------------------------
def copy_raw_sidecards():
    if not RAW_ROOT.is_dir():
        raise RuntimeError(f"Raw side‑card folder not found at {RAW_ROOT}")
    print(f"📂 Copying raw side‑cards to {RAW_OUT} …")
    shutil.copytree(RAW_ROOT, RAW_OUT)
    print("✅ Raw side‑cards copied.")

# ------------------------------------------------------------------
# Load the HF model once
# ------------------------------------------------------------------
def load_model(model_name: str):
    print(f"🔧 Loading model {model_name} …")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

# ------------------------------------------------------------------
# Predict a label and confidence for a single image
# ------------------------------------------------------------------
def predict(image_path: Path, processor, model, device) -> Tuple[str, float]:
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits.squeeze(0)
    probs = torch.nn.functional.softmax(logits, dim=0)
    confidence, idx = torch.max(probs, dim=0)
    label = model.config.id2label[int(idx)]
    label = label.lower()  # ensure lower‑case bird name
    return label, confidence.item()
    

# ------------------------------------------------------------------
# XMP keyword handling (adds to <dc:subject> without duplicates)
# ------------------------------------------------------------------
def add_keywords_to_xmp(xmp_path: Path, keywords: list[str]):
    ns = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "dc": "http://purl.org/dc/elements/1.1/",
    }
    ET.register_namespace("rdf", ns["rdf"])
    ET.register_namespace("dc", ns["dc"])
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
        bag = ET.SubElement(subject, f"{{{ns['rdf']}}}Seq")
    else:
        bag = subject.find('rdf:Seq', ns)
        if bag is None:
            bag = ET.SubElement(subject, f"{{{ns['rdf']}}}Seq")
    existing = {li.text for li in bag.findall('rdf:li', ns) if li.text}
    for kw in keywords:
        if kw not in existing:
            li = ET.SubElement(bag, f"{{{ns['rdf']}}}li")
            li.text = kw
    tree.write(xmp_path, encoding="utf-8", xml_declaration=True)

# ------------------------------------------------------------------
# Process a folder of XMP side‑cards (reading images from JPG_ROOT)
# ------------------------------------------------------------------
def process_folder(xmp_root: Path, csv_path: Path, processor, model, device):
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "label", "confidence", "note"])
        for xmp_path in xmp_root.rglob("*.xmp"):
            base = xmp_path.stem  # e.g. _D5C5770
            jpg_path = JPG_ROOT / f"{base}.jpg"
            if not jpg_path.is_file():
                print(f"⚠️  JPEG missing for {xmp_path.name}")
                continue
            label, conf = predict(jpg_path, processor, model, device)
            label = label.lower()  # ensure lower‑case bird name
            keywords = []
            note = ""
            if conf < args.no_bird_conf:
                keywords.append("_nb")
                note = "no bird"
            else:
                # generic "bird" keyword for every detected bird
                keywords.append("bird")
                xyz = label[0]
                specific = f"{xyz}-unknown-{label}"
                if conf < args.conf_threshold:
                    specific = f"_{specific}"  # low‑confidence marker
                keywords.append(specific)
                note = f"bird ({conf:.2f})"
            add_keywords_to_xmp(xmp_path, keywords)
            writer.writerow([xmp_path.name, label, f"{conf:.2f}", note])
            print(f"✅ {xmp_path.name} → {', '.join(keywords)} (conf={conf:.2f})")

# ------------------------------------------------------------------
# Main execution flow
# ------------------------------------------------------------------
if __name__ == "__main__":
    # 1️⃣ Prepare output directory structure
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    copy_raw_sidecards()

    # 2️⃣ Save the run arguments for reproducibility
    with open(ARGS_OUT, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # 3️⃣ Load model
    processor, model, device = load_model(args.model)

    # 4️⃣ Process the copied raw side‑cards
    print("\n🔧 Processing side‑cards …")
    process_folder(RAW_OUT, CSV_OUT, processor, model, device)

    print("\n✅ Run complete.")
    print(f"Outputs stored in: {RUN_DIR}")
