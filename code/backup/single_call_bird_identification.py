#!/usr/bin/env python3
"""single_call_bird_identification.py

A simplified version of the bird‑identification pipeline that sends **one** OpenAI request
containing *all* images (as base‑64 data URLs) and receives a JSON array with the
predictions for each image.

Usage:
    python3 single_call_bird_identification.py \
        --output-dir ./output \
        --data-dir ./data

Arguments are optional; the defaults match the original script.
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path
import urllib.request
import urllib.error
from datetime import datetime
import csv

# --------------------------------------------------------------------
# Helper to call OpenAI Chat Completion (no external library needed)
# --------------------------------------------------------------------
def _openai_chat_completion(messages, model_name):
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY not set in environment')
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": 2000,  # generous limit for a batch response
        "temperature": 0.0,
    }
    request = urllib.request.Request(
        url='https://api.openai.com/v1/chat/completions',
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        }
    )
    try:
        with urllib.request.urlopen(request) as resp:
            resp_body = resp.read().decode('utf-8')
            return json.loads(resp_body)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode('utf-8')
        raise RuntimeError(f"OpenAI HTTP error {e.code}: {err_body}")

# --------------------------------------------------------------------
# Encode an image as a base64 data‑URL (small helper to keep the request tidy)
# --------------------------------------------------------------------
def _image_to_data_url(image_path: Path) -> str:
    with image_path.open('rb') as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"

# --------------------------------------------------------------------
# Build a single request that contains all images and their filenames
# --------------------------------------------------------------------
def _build_messages(image_paths):
    system_prompt = (
        "You are an expert bird‑identification system. "
        "You will receive a series of image blocks, each preceded by a text block that "
        "contains the filename (e.g., 'Filename: IMG_001.jpg'). "
        "Return a **JSON array** where each element is an object with the fields: "
        "'filename' (string), 'label' (English lower‑case name or 'unknown'), "
        "'label_cn' (Chinese name or '未知'), and 'confidence' (float 0‑1). "
        "Do NOT include any extra text, just the JSON array."
    )
    # Build user content: for each image we send a text block with the filename,
    # then an image_url block with the data‑URL.
    user_content = []
    for p in image_paths:
        user_content.append({"type": "text", "text": f"Filename: {p.name}"})
        user_content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(p)}})
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    return messages

# --------------------------------------------------------------------
# Main workflow
# --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Bird identification with a **single** OpenAI request.")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model (default gpt-4o)")
    parser.add_argument("--output-dir", default="./output", help="Where to store the run folder and CSV")
    parser.add_argument("--data-dir", default="./data", help="Folder containing jpg/ and raw/ subfolders")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    jpg_dir = data_dir / "jpg"
    if not jpg_dir.is_dir():
        sys.exit(f"❌ JPEG directory not found: {jpg_dir}")
    image_paths = sorted(jpg_dir.glob('*.jpg'))
    if not image_paths:
        sys.exit("❌ No JPG files found.")

    # Build a single request
    messages = _build_messages(image_paths)
    print(f"📤 Sending a single request with {len(image_paths)} images to model {args.model}…")
    response = _openai_chat_completion(messages, args.model)
    content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
    if not content:
        sys.exit("❌ Empty response from OpenAI.")
    # Parse the JSON array
    try:
        results = json.loads(content)
    except json.JSONDecodeError:
        # Attempt to extract the first JSON array in the text
        import re
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if not match:
            sys.exit("❌ Could not parse JSON array from response.")
        results = json.loads(match.group())

    # Prepare output folder (timestamped)
    timestamp = datetime.now().strftime('run_%Y%m%d_%H%M%S')
    out_dir = Path(args.output_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "bird_identification_output.csv"

    # Write CSV
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label', 'label_cn', 'confidence'])
        for entry in results:
            # Ensure fields exist; fall back to defaults if missing
            filename = entry.get('filename', '')
            label = entry.get('label', 'unknown').lower()
            label_cn = entry.get('label_cn', '未知')
            confidence = entry.get('confidence', 0.0)
            writer.writerow([filename, label, label_cn, f"{confidence:.2f}"])

    print(f"✅ Done. CSV written to {csv_path}")

if __name__ == "__main__":
    main()
