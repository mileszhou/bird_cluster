
#!/usr/bin/env python3

# -------------------------------------------------
# label_generator.py
# -------------------------------------------------
# Generates a label consisting of the first‑letter pinyin
# for each Chinese character in a name, followed by the
# confidence level in parentheses, e.g.   "zs (95.00%)"
# -------------------------------------------------

from __future__ import annotations
import csv, sys
from typing import List
from pypinyin import lazy_pinyin, Style

def pinyin_initials(name: str) -> str:
    """Return the concatenated first‑letter pinyin (lowercase)."""
    # `lazy_pinyin(..., style=Style.FIRST_LETTER)` returns a list of 1‑character strings
    initials = ''.join(lazy_pinyin(name, style=Style.FIRST_LETTER)).lower()
    return initials

def _format_confidence(conf: float) -> str:
    """
    Convert a confidence (0‑1 or 0‑100) to a string like '95.00%'.
    """
    # If the value looks like a fraction, convert to a percentage
    if 0 <= conf <= 1:
        conf = conf * 100
    return f"{conf:.2f}%"

def build_label(name: str, confidence: float) -> str:
    """Return the final label string."""
    initials = pinyin_initials(name)
    conf_str = _format_confidence(confidence)
    return f"{initials} ({conf_str})"

# -----------------------------------------------------------------
# Optional CSV processing (run from command line)
# -----------------------------------------------------------------
def process_csv(input_path: str, output_path: str, delimiter: str = ',') -> None:
    with open(input_path, newline='', encoding='utf-8') as csv_file:
        rows = list(csv.DictReader(csv_file))
    # Add a new column called 'label'
    for row in rows:
        name = row['name']
        confidence = float(row['confidence'])
        row['label'] = f"{pinyin_initials(name)} ({_format_confidence(confidence)})"
    # Write to output CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as out_file:
        writer = csv.DictWriter(out_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

# -----------------------------------------------------------------
# Command‑line interface
# -----------------------------------------------------------------
if __name__ == "__main__" and len(sys.argv) == 3:
    # Usage: python label_generator.py input.csv output.csv
    input_path, output_path = sys.argv[1:3]
    process_csv(input_path, output_path)
    print(f"✅  Output written to {output_path}")
else:
    # Simple usage example
    # python -c "from label_generator import build_label; print(build_label('张三', 0.94))"
    pass
