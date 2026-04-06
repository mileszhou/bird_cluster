#!/usr/bin/env python3
"""single_call_wrapper.py

A compact helper that imports the core functions from the existing
`run_hf_bird_model_chatgpt` module and performs **one** OpenAI request for a
list of images.

The file is intentionally short so you can copy the `single_call_main`
function into a notebook later and still import the same helpers.
"""

from pathlib import Path
from typing import List

# Import the low‑level helpers defined in the main module
# (they are public functions in that script)
from run_hf_bird_model_chatgpt import read_image_base64, _openai_chat_completion


def _image_to_data_url(image_path: Path) -> str:
    """Return a `data:image/jpeg;base64,…` URL for the given image."""
    return f"data:image/jpeg;base64,{read_image_base64(image_path)}"


def single_call_main(image_paths: List[Path], model: str = "gpt-4o"):
    """Send **one** request that includes all `image_paths`.

    Returns a list of dictionaries, each with:
        - filename
        - label (english, lower‑case)
        - label_cn (chinese)
        - confidence (float 0‑1)
    """
    # Build the user content: a text block with the filename followed by the image
    user_content = []
    for p in image_paths:
        user_content.append({"type": "text", "text": f"Filename: {p.name}"})
        user_content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(p)}})

    system_prompt = (
        "You are an expert bird‑identification system. "
        "You will receive a series of image blocks, each preceded by a text block "
        "containing the filename (e.g., 'Filename: IMG_001.jpg'). "
        "Return a **JSON array** where each element has the keys: "
        "'filename', 'label' (english lower‑case or 'unknown'), "
        "'label_cn' (chinese name or '未知'), and 'confidence' (float 0‑1). "
        "Do NOT add any extra text – just the JSON array."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # Perform the request using the imported low‑level helper
    response = _openai_chat_completion(messages, model)
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not content:
        raise RuntimeError("Empty response from OpenAI")
    # Parse the JSON array – be tolerant of surrounding text
    import json, re
    try:
        results = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if not match:
            raise ValueError("Could not extract JSON array from response")
        results = json.loads(match.group())
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="One‑call bird ID wrapper")
    parser.add_argument("--data-dir", default="./data", help="Directory containing jpg/ subfolder")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use")
    args = parser.parse_args()

    jpg_dir = Path(args.data_dir) / "jpg"
    image_files = sorted(jpg_dir.glob("*.jpg"))
    if not image_files:
        raise SystemExit("No JPG files found in the specified directory.")

    results = single_call_main(image_files, model=args.model)
    # Simple pretty‑print for the console
    for r in results:
        fname = r.get("filename", "")
        label = r.get("label", "unknown")
        label_cn = r.get("label_cn", "未知")
        conf = r.get("confidence", 0.0)
        print(f"{fname:30} {label:15} {label_cn:10} {conf:.2f}")
