#!/usr/bin/env python3
"""
run_hf_bird_model_output.py

- Copies the relevant parts of ./data/lr-proj into ./output/lr-proj, keeping ./data read‑only.
- Processes the copies with the local model `chriamue/bird-species-classifier`.
- Writes a CSV summary (bird_identification_results.csv) in ./output.
- Updates XMP side‑cars in the copy, adding:
    * generic keyword "bird" for detected birds
    * specific keyword `xyz‑unknown‑<label>` (prefixed with '_' if low confidence)
    * "_nb" for images with no bird.
"""

import csv
import shutil
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import xml.etree.ElementTree as ET

# ------------------------------------------------------------------
# Configuration – adjust as needed
# ------------------------------------------------------------------
MODEL_NAME = "chriamue/bird-species-classifier"
CONF_THRESHOLD = 0.80   # below -> low‑confidence marker
NO_BIRD_CONF = 0.10     # below -> treat as no bird

# ------------------------------------------------------------------
# Paths (relative to workspace root)
# ------------------------------------------------------------------
DATA_ROOT   = Path("./data/lr-proj")          # read‑only source
OUTPUT_ROOT = Path("./output/lr-proj")        # writable copy

JPG_SUBDIR   = "jpg"
PHOTOS_2025  = "Photos-2025"
PHOTOS_2024  = "Photos-2024"
CSV_OUT_NAME = "bird_identification_results.csv"

# ------------------------------------------------------------------
# Helper: copy source tree (once)
# ------------------------------------------------------------------
def copy_source():
    if OUTPUT_ROOT.exists():
        print(f"⚠️  Output already exists at {OUTPUT_ROOT}, skipping copy.")
        return
    print(f"🔧 Copying source data to output …")
    shutil.copytree(DATA_ROOT, OUTPUT_ROOT)
    print("✅ Copy complete.")

# ------------------------------------------------------------------
# Load model once
# ------------------------------------------------------------------
def load_model():
    print("🔧 Loading model …")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

# ------------------------------------------------------------------
# Predict label & confidence for a JPG
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
    return label, confidence.item()

# ------------------------------------------------------------------
# Update XMP: add keyword list (avoid duplicates)
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
# Process a folder (writes CSV, updates XMP)
# ------------------------------------------------------------------
def process_folder(jpg_root: Path, xmp_root: Path, csv_path: Path, processor, model, device):
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "label", "confidence", "note"])
        for xmp_path in xmp_root.rglob("*.xmp"):
            base = xmp_path.stem
            jpg_path = jpg_root / f"{base}.jpg"
            if not jpg_path.is_file():
                print(f"⚠️  JPG missing for {xmp_path.name}")
                continue
            label, conf = predict(jpg_path, processor, model, device)
            keywords = []
            note = ""
            if conf < NO_BIRD_CONF:
                keywords.append("_nb")
                note = "no bird"
            else:
                keywords.append("bird")
                xyz = label[0].lower()
                specific = f"{xyz}-unknown-{label}"
                if conf < CONF_THRESHOLD:
                    specific = f"_{specific}"
                keywords.append(specific)
                note = f"bird ({conf:.2f})"
            add_keywords_to_xmp(xmp_path, keywords)
            writer.writerow([xmp_path.name, label, f"{conf:.2f}", note])
            print(f"✅ {xmp_path.name} → {', '.join(keywords)} (conf={conf:.2f})")

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    copy_source()
    # Paths inside the output copy
    JPG_ROOT = OUTPUT_ROOT / JPG_SUBDIR
    XMP_2025 = OUTPUT_ROOT / PHOTOS_2025
    XMP_2024 = OUTPUT_ROOT / PHOTOS_2024
    CSV_OUT = OUTPUT_ROOT / CSV_OUT_NAME
    processor, model, device = load_model()
    print("\n🔧 Processing 2025 …")
    process_folder(JPG_ROOT, XMP_2025, CSV_OUT, processor, model, device)
    print("\n🔧 Processing 2024 …")
    process_folder(JPG_ROOT, XMP_2024, CSV_OUT, processor, model, device)
    print("\n✅ Done – CSV written to", CSV_OUT)
