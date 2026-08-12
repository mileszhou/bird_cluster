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
#
# What is NOT here is as considered as what is. `crs:*` -- the develop settings,
# 13,271 attributes and ~90% of the bytes -- stays. It identifies nobody, and a
# realistic sidecar is exactly what this sample exists to be: `xmp_write`'s
# whole text-surgery design comes from these files being 330 lines of
# Lightroom-formatted settings, and a stripped one would stop exercising it.
# Timestamps stay too: the trip folder already carries the date, so removing
# them hides nothing while breaking `sidecar_meta.read_meta()`.
#
# Camera *brand* is not removable and is not pretended to be. `crs:RawFileName`
# ends in .nef / .arw / .cr3, and `read_meta()` reads that field -- so Nikon,
# Sony or Canon is legible whatever tiff:Make says. Stripping
# `photoshop:SidecarForExtension` as well would be theatre. A brand is not a
# body; the serial was the identifier and it is gone.
IDENTIFYING = (
    r'[A-Za-z0-9_]+:(?:Body)?(?:Lens)?SerialNumber',   # links to every other photo
    r'[A-Za-z0-9_]+:GPS[A-Za-z]*',                     # where somebody was
    r'tiff:Make', r'tiff:Model',                       # camera body
    r'aux:Lens', r'aux:LensID', r'aux:LensInfo',       # lens
    r'aux:Firmware',
    r'aux:ImageNumber',                                # the body's shutter count
    r'aux:LensDistortInfo',                            # a lens profile: reinstates the lens
    r'exif:LensMake',                                  # ditto, from the other namespace
    r'xmp:Rating', r'xmp:Label',                       # the photographer's own judgement
)
XMP_ATTR = re.compile("|".join(rf'\s+{a}="[^"]*"' for a in IDENTIFYING))

SOI, APP1, SOS, EOI = b"\xff\xd8", 0xE1, 0xDA, 0xD9


# --minimal: keep only what the pipeline reads, drop the rest. Local names, since
# Lightroom's prefixes move between versions.
#
# Everything else goes: 13,271 crs:* develop attributes, the exposure and lens
# EXIF, the edit history, the ratings. That loses realism -- a genuine sidecar
# is ~330 lines and `xmp_write`'s text-surgery design exists because of it -- and
# the sample stops representing what the labeller meets in the wild. Chosen
# deliberately: nobody reads a sample dataset's develop settings, and the less
# it carries the less there is to regret publishing.
KEEP_ATTRS = {
    "OriginalDocumentID", "DocumentID",   # sidecar_meta: capture identity, dedup
    "RawFileName",                        # sidecar_meta: which raw this describes
    "DateTimeOriginal", "CreateDate",     # sidecar_meta: capture time
    "about",                              # rdf:Description's own attribute
}
KEEP_ELEMENTS = {"subject", "hierarchicalSubject", "Bag", "Seq", "Alt", "li",
                 "xmpmeta", "RDF", "Description"}
ATTR_RE = re.compile(r'\s+([A-Za-z0-9_]+):([A-Za-z0-9]+)="[^"]*"')


def minimal_xmp(text: str) -> str:
    """Drop every attribute and element block outside the keep-lists.

    Text surgery rather than a parse-and-rewrite, so the container each file
    uses survives: 49 of these sidecars carry `dc:subject` as an `rdf:Bag` and
    54 have none at all, and `xmp_write` reuses whichever is present. A
    reserialisation would normalise that away and delete the case.
    """
    # Element blocks first, so their inner attributes go with them.
    for tag in ("crs:ToneCurvePV2012", "crs:ToneCurvePV2012Red",
                "crs:ToneCurvePV2012Green", "crs:ToneCurvePV2012Blue",
                "crs:ToneCurve", "crs:PointColors", "crs:ColorVariance",
                "crs:Look", "xmpMM:History", "xmpMM:DerivedFrom",
                "exif:ISOSpeedRatings", "exif:Flash", "dc:format",
                "crs:MaskGroupBasedCorrections", "crs:CircularGradientBasedCorrections",
                "crs:GradientBasedCorrections", "crs:PaintBasedCorrections",
                "crs:RetouchAreas", "crd:Look"):
        text = re.sub(rf"[ \t]*<{tag}\b[^>]*/>\n", "", text)
        text = re.sub(rf"[ \t]*<{tag}\b.*?</{tag}>\n", "", text, flags=re.S)
    # Then loose attributes. `xmlns:` declarations are attributes too and must
    # survive -- removing them leaves every remaining prefix unbound and the file
    # stops being XML at all, which is exactly what happened the first time.
    return ATTR_RE.sub(
        lambda m: m.group(0) if m.group(1) == "xmlns" or m.group(2) in KEEP_ATTRS
        else "", text)


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
    ap.add_argument("--minimal", action="store_true",
                    help="keep only the fields the pipeline reads; drop develop "
                         "settings, EXIF, history, everything else")
    ap.add_argument("--verify", action="store_true",
                    help="decode each JPEG before and after and assert the pixels match")
    args = ap.parse_args()

    attrs = collections.Counter()
    xmp_changed = jpg_changed = 0

    for path in sorted(args.root.rglob("*.xmp")):
        text = path.read_text(encoding="utf-8")
        new, found = scrub_xmp(text)
        if args.minimal:
            new = minimal_xmp(new)
        if not found and new == text:
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
