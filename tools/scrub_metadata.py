#!/usr/bin/env python3
"""Remove identifying metadata from a dataset tree before publishing it.

Written when `sample_data` went public and a check afterwards found camera
serial numbers in 90 of its 103 sidecars and 90 of its 109 JPEGs. A secret
scanner does not flag any of this -- gitleaks came back clean -- because none of
it is a credential. It is personal data, which is a different class and needs a
different check.

**What it removes.**

*Sidecars*: `aux:SerialNumber`, `aux:LensSerialNumber`, and anything matching
`GPS*` in any namespace. Body and lens serials link a photograph to every other
photograph from the same camera, including ones published elsewhere. GPS is the
serious one where present -- degrees plus decimal minutes to four places is
roughly metre-level, and a photo library is a multi-year record of where
somebody was. `sample_data` happens to carry none, but the old `results/` tree
on the `bird_label` master branch has 5,491 files that do.

*JPEGs*: the whole EXIF block, by dropping the APP1 segment.

**The pixels are not touched.** Re-saving through an image library would
re-encode and change every byte of the scan data -- for a *sample dataset* that
is worse than the problem, since the images are the thing under study. So the
JPEG is edited as a byte stream: walk the marker structure, drop APP1, copy
everything else including the entropy-coded data verbatim. `--verify` decodes
before and after and asserts the pixel arrays are identical.

Same principle as `xmp_write`: edit the file, do not rewrite it.

    python -m tools.scrub_metadata sample_data              # dry run
    python -m tools.scrub_metadata sample_data --apply --verify

Dry run by default. Reports every attribute it would remove, by name and count,
so the list can be read before anything is written.
"""

import argparse
import collections
import re
from pathlib import Path

# Namespace prefixes vary between Lightroom versions, so match on the local name.
XMP_ATTR = re.compile(
    r'\s+[A-Za-z0-9_]+:(?:Body)?(?:Lens)?SerialNumber="[^"]*"'
    r'|\s+[A-Za-z0-9_]+:GPS[A-Za-z]*="[^"]*"')

SOI, APP1, SOS, EOI = b"\xff\xd8", 0xE1, 0xDA, 0xD9


def scrub_xmp(text: str):
    """(new text, [attribute names removed])."""
    found = [m.group(0).strip().split("=")[0] for m in XMP_ATTR.finditer(text)]
    return XMP_ATTR.sub("", text), found


def scrub_jpeg(data: bytes):
    """Drop APP1 (EXIF) segments, byte for byte otherwise.

    JPEG is SOI, then marker segments (0xFF, code, 2-byte big-endian length
    including those two bytes), until SOS -- after which the entropy-coded scan
    runs to EOI and must be copied untouched. Standalone markers (RSTn, TEM)
    carry no length, hence the explicit skip list.
    """
    if not data.startswith(SOI):
        return data, 0
    out, i, removed = bytearray(SOI), 2, 0
    while i < len(data) - 1:
        if data[i] != 0xFF:
            break
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:   # no length field
            out += data[i:i + 2]
            i += 2
            continue
        if marker == SOS:
            out += data[i:]                                     # scan data, verbatim
            return bytes(out), removed
        length = int.from_bytes(data[i + 2:i + 4], "big")
        seg = data[i:i + 2 + length]
        if marker == APP1 and data[i + 4:i + 10] == b"Exif\x00\x00":
            removed += 1
        else:
            out += seg
        i += 2 + length
    out += data[i:]
    return bytes(out), removed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="a dataset tree, e.g. sample_data")
    ap.add_argument("--apply", action="store_true", help="write the changes")
    ap.add_argument("--verify", action="store_true",
                    help="decode each JPEG before and after and assert the pixels match")
    args = ap.parse_args()

    attrs = collections.Counter()
    xmp_changed = jpg_changed = 0

    for path in sorted(args.root.rglob("*.xmp")):
        text = path.read_text(encoding="utf-8")
        new, found = scrub_xmp(text)
        if not found:
            continue
        attrs.update(found)
        xmp_changed += 1
        if args.apply:
            path.write_text(new, encoding="utf-8")

    for path in sorted(args.root.rglob("*.jpg")):
        data = path.read_bytes()
        new, removed = scrub_jpeg(data)
        if not removed:
            continue
        if args.verify:
            import io
            import numpy as np
            from PIL import Image
            before = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
            after = np.asarray(Image.open(io.BytesIO(new)).convert("RGB"))
            if not np.array_equal(before, after):
                raise SystemExit(f"error: {path} pixels changed -- refusing to write")
        jpg_changed += 1
        if args.apply:
            path.write_bytes(new)

    print(f"\n  sidecars: {xmp_changed} to change")
    for name, n in attrs.most_common():
        print(f"    {n:>5}  {name}")
    print(f"  jpegs:    {jpg_changed} with an EXIF block to drop"
          f"{' (pixels verified identical)' if args.verify else ''}")
    if not args.apply:
        print("\n  Dry run. Re-run with --apply to write.")


if __name__ == "__main__":
    main()
