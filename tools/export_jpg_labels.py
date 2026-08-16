#!/usr/bin/env python3
"""Write each image's species label into the exported JPEG's own metadata.

**Lightroom Classic reads `.xmp` sidecars only for proprietary raw formats.**
For a JPEG it reads embedded metadata and ignores a sidecar sitting beside it.
So the mechanism this project already has -- `xmp_write.set_keywords_in_xmp()`
depositing into `output/label/raw/*.xmp` -- cannot carry a label into the
seriated export. The keyword has to go *inside* the file.

That is the whole reason this tool exists. `export_seriated.py` writes the
*order* into the capture time, because Lightroom sorts by capture time and
never by arbitrary metadata; this writes the *label* into the keywords, because
that is the one place Lightroom will look for it on a JPEG.

**A keyword lives in two places, and both must be written.** The first version
of this tool set only XMP `dc:subject`, verified it, and produced 8,425 files
that Lightroom showed as unlabelled. These JPEGs also carry a Photoshop image
resource block (`APP13`) holding **legacy IPTC IIM** keywords, and that is what
Lightroom actually displayed: photos with no IIM keyword showed nothing, photos
with the user's own showed only those, and 822 showed a *stale* label from an
earlier pipeline era. Which of the two a reader prefers is not something to
have an opinion about -- Adobe's own rule turns on the caption digest in
resource `0x0425` -- so the fix is to leave them no way to disagree. Both are
written with the same list.

The general lesson, since it has now cost a full pass over the library: a
verification that reads back the field you just wrote proves the write, not the
outcome. It cannot see a second copy of the same fact somewhere else in the file.

**The capture time has the same two-places problem.** `piexif` rewrites the
EXIF time; IIM `2:55`/`2:60` kept the *true* capture date in 8,421 of 8,425
files, so the file contradicted itself about when the photo was taken. The sort
happened to come from EXIF, but a metadata re-read could have reinstated the
real dates and scrambled the seriation. The IIM dates are now written to match.

**It edits the copies, never `data/`.** Same rule as everywhere else: the
pipeline does not write the dataset. The targets are the copies under
`output/lightroom/jpg/<run>/`, which are regenerable by re-running the export.

The run name stays in the path, so sweeps sit side by side the way the cluster
runs themselves do -- `mcs15` and `mcs50` are different resolutions of the same
question, and comparing them is the point:

    output/lightroom/jpg/mcs15/index.csv                seq, capture_time, cluster_id, species, key
    output/lightroom/jpg/mcs15/<Photos-YY>/<trip>/      the tree `key` is relative to

The manifest sits inside the folder handed to Lightroom rather than beside it,
which keeps a run self-describing; Lightroom imports image formats and ignores
the CSV.

**Separate tool rather than a flag on the export.** The export is 1.0 GB of
file copying; the keywords are a few hundred bytes each. Re-deciding the label
form should not mean re-copying 8,425 images, and this is idempotent, so it can
be run over an export that already exists -- which is the situation it was
written for.

**The label keeps its `(NN%)` suffix**, and not because the number is worth
much. It is the marker that says a machine wrote this keyword and not a person:
`split_keywords()` recognises this pipeline's output by that suffix and by
nothing else, so a label written without it would be indistinguishable from a
hand-typed one and the next run would refuse to touch it. Provenance, not
precision.

**The user's own keywords survive.** 4,528 of the export's JPEGs already carry
keywords -- hand-written species (`cn-翠鸟-kingfisher`), places (`bhl-百花岭`),
admin tags (`add=20231014`) -- alongside stale labels from an earlier pipeline
era (`ho-海鸥-seagull(95%)`, `bird`). `split_keywords()` separates the two, and
only this pipeline's own entries are replaced. Erring towards a stale keyword
over a deleted one is the standing rule; see `code/lib/xmp_labels.py`.

**Pixels are never touched.** The file is rebuilt from its marker structure --
the APP13 block changes length, so the packet cannot simply be overwritten in
place -- but everything from the start-of-scan marker onward is copied through
verbatim, and the result is checked against the source in `data/jpg`.

    python3 tools/export_jpg_labels.py --dry-run
    python3 tools/export_jpg_labels.py
    python3 tools/export_jpg_labels.py --export output/lightroom/jpg/mcs50

Lightroom reads embedded metadata **at import**. Run this before importing; on
an already-imported folder use Metadata > Read Metadata from File.
"""

import argparse
import csv
import os
import re
import struct
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.environ.get("PROJECT_ROOT")
                or str(Path(__file__).resolve().parents[1]))

from code.lib.config import PROJECT_ROOT  # noqa: E402
from code.lib.label_generator import pinyin_initials  # noqa: E402
from code.lib.xmp_labels import parse_label, split_keywords  # noqa: E402
from code.lib.xmp_write import (XmpEditError, set_subject_keywords,  # noqa: E402
                                verify_only_keywords_changed)

XMP_SIG = b"http://ns.adobe.com/xap/1.0/\x00"
PHOTOSHOP_SIG = b"Photoshop 3.0\x00"
APP1, APP13, SOS, SOI, EOI = 0xE1, 0xED, 0xDA, 0xD8, 0xD9
IPTC_RESOURCE = 0x0404          # 8BIM resource holding the IIM datasets
KEYWORD_DS = (2, 25)            # IIM 2:25 Keywords, repeatable
# 1:90 declares the coded character set; ESC % G is UTF-8. Without it a reader
# is entitled to treat the bytes as its own default encoding, which would make
# the Chinese in a label unreadable -- so it is ensured, never assumed.
UTF8_MARKER = (1, 90, b"\x1b%G")
IIM_VERSION = (2, 0, b"\x00\x04")
DATE_DS = {(2, 55): "%Y%m%d", (2, 60): "%H%M%S",     # DateCreated / TimeCreated
           (2, 62): "%Y%m%d", (2, 63): "%H%M%S"}     # DigitalCreation date / time
MAX_SEGMENT = 0xFFFF - 2

SUBJECT_RE = re.compile(r"<(?P<p>[\w.-]+):subject\b[^>]*(?<!/)>(?P<body>.*?)"
                        r"</(?P=p):subject>", re.S)
LI_RE = re.compile(r"<rdf:li[^>]*>(.*?)</rdf:li>", re.S)


class SegmentError(Exception):
    """The JPEG is not shaped the way the edit requires."""


# --- JPEG marker structure --------------------------------------------------

def parse_jpeg(data: bytes) -> tuple[list, bytes]:
    """([[marker, payload], ...], scan) -- segments before the scan, then the rest.

    The markers are walked rather than searched for, so a byte sequence inside
    the compressed scan cannot be mistaken for a segment header. `scan` is
    everything from the start-of-scan marker onward and is never interpreted.
    """
    segments, i = [], 2
    while i < len(data) - 1:
        if data[i] != 0xFF:
            raise SegmentError(f"not a marker at offset {i}")
        marker = data[i + 1]
        if marker in (SOI, EOI):
            i += 2
            continue
        if marker == SOS:
            return segments, data[i:]
        length = struct.unpack(">H", data[i + 2:i + 4])[0]
        segments.append([marker, data[i + 4:i + 2 + length]])
        i += 2 + length
    raise SegmentError("no start-of-scan marker")


def build_jpeg(segments, scan: bytes) -> bytes:
    out = [b"\xff\xd8"]
    for marker, payload in segments:
        if len(payload) > MAX_SEGMENT:
            raise SegmentError(f"segment {marker:#x} is {len(payload)} bytes, over the limit")
        out.append(bytes((0xFF, marker)) + struct.pack(">H", len(payload) + 2) + payload)
    out.append(scan)
    return b"".join(out)


def find_segment(segments, marker, signature):
    for entry in segments:
        if entry[0] == marker and entry[1].startswith(signature):
            return entry
    return None


def find_xmp(data: bytes) -> tuple[int, int]:
    """(offset, length) of the XMP packet text within `data`, for reading."""
    i = 2
    while i < len(data) - 1:
        if data[i] != 0xFF:
            raise SegmentError(f"not a marker at offset {i}")
        marker = data[i + 1]
        if marker in (SOI, EOI):
            i += 2
            continue
        if marker == SOS:
            break
        length = struct.unpack(">H", data[i + 2:i + 4])[0]
        if marker == APP1 and data[i + 4:i + 4 + len(XMP_SIG)] == XMP_SIG:
            start = i + 4 + len(XMP_SIG)
            return start, (i + 2 + length) - start
        i += 2 + length
    raise SegmentError("no XMP packet")


# --- Photoshop image resources and IPTC IIM ---------------------------------

def parse_irb(blob: bytes) -> list:
    """[[resource_id, name, data], ...] from a Photoshop image resource block."""
    out, i = [], 0
    while i + 12 <= len(blob):
        if blob[i:i + 4] != b"8BIM":
            break
        rid = struct.unpack(">H", blob[i + 4:i + 6])[0]
        i += 6
        nlen = blob[i]
        name = blob[i + 1:i + 1 + nlen]
        i += 1 + nlen
        if (1 + nlen) % 2:                 # pascal string padded to even length
            i += 1
        size = struct.unpack(">I", blob[i:i + 4])[0]
        i += 4
        out.append([rid, name, blob[i:i + size]])
        i += size + (size % 2)
    return out


def build_irb(resources) -> bytes:
    out = []
    for rid, name, data in resources:
        chunk = b"8BIM" + struct.pack(">H", rid) + bytes((len(name),)) + name
        if (1 + len(name)) % 2:
            chunk += b"\x00"
        chunk += struct.pack(">I", len(data)) + data
        if len(data) % 2:
            chunk += b"\x00"
        out.append(chunk)
    return b"".join(out)


def parse_iim(data: bytes) -> list:
    """[(record, dataset, value), ...] from an IIM stream."""
    out, i = [], 0
    while i + 5 <= len(data):
        if data[i] != 0x1C:
            raise SegmentError(f"IIM tag marker expected at {i}")
        record, dataset = data[i + 1], data[i + 2]
        length = struct.unpack(">H", data[i + 3:i + 5])[0]
        if length & 0x8000:
            raise SegmentError("extended IIM dataset length is not supported")
        i += 5
        out.append((record, dataset, data[i:i + length]))
        i += length
    return out


def build_iim(datasets) -> bytes:
    out = []
    for record, dataset, value in datasets:
        if len(value) > 0x7FFF:
            raise SegmentError("IIM value too long for a standard dataset")
        out.append(b"\x1c" + bytes((record, dataset))
                   + struct.pack(">H", len(value)) + value)
    return b"".join(out)


def sync_iim(datasets, keywords, when):
    """IIM datasets with the keywords replaced, and the dates if `when` is given.

    Datasets are emitted in ascending (record, dataset) order, which is how
    every file in the export already has them. `sorted` is stable, so repeated
    keywords keep the order they were given. The `1:90` charset marker declaring
    UTF-8 is present in all of them and is left alone -- without it the Chinese
    in a label is not legally encodable here.
    """
    drop = {KEYWORD_DS} | (set(DATE_DS) if when else set())
    kept = [d for d in datasets if (d[0], d[1]) not in drop]
    added = [(*KEYWORD_DS, k.encode("utf-8")) for k in keywords]
    if when:
        added += [(r, d, when.strftime(fmt).encode())
                  for (r, d), fmt in DATE_DS.items()]
    present = {(d[0], d[1]) for d in kept}
    added += [d for d in (UTF8_MARKER, IIM_VERSION) if (d[0], d[1]) not in present]
    return sorted(kept + added, key=lambda d: (d[0], d[1]))


def iim_keywords(datasets):
    return [v.decode("utf-8", "replace") for r, d, v in datasets if (r, d) == KEYWORD_DS]


# --- the edit ---------------------------------------------------------------

def current_subjects(text: str) -> list[str]:
    match = SUBJECT_RE.search(text)
    if not match:
        return []
    return [i.strip() for i in LI_RE.findall(match.group("body"))]


def effective_label(row) -> str | None:
    """The `py-cn-en(NN%)` keyword for a label CSV row, resolving never-demote.

    Mirrors `embed.effective_species()`, and for the same reason: where
    `applied` is `kept-existing` the run's verdict was overruled and the bird is
    in `prior_label`, while `label`/`label_cn` describe the discarded non-bird
    call. Taking `label_cn` there would pair a bird's English name with a
    landscape's Chinese one.

    On an overruled row the prior label is already in this exact shape, so it is
    reused verbatim rather than rebuilt -- there is nothing to improve on it and
    re-deriving the pinyin would only invent a way to disagree.
    """
    if row.get("applied") == "kept-existing" and row.get("prior_category"):
        for part in (row.get("prior_label") or "").split(";"):
            label = parse_label(part.strip())
            if label:
                return label.raw
        return None

    english = (row.get("label") or "").strip().lower()
    if not english:
        return None
    chinese = (row.get("label_cn") or "").strip()
    try:
        percent = int(round(float(row.get("confidence") or 0) * 100))
    except ValueError:
        return None
    if not chinese:
        return f"{english}({percent}%)"
    return f"{pinyin_initials(chinese)}-{chinese}-{english}({percent}%)"


def write_keywords(path: Path, label: str, when=None) -> str:
    """Set `label` as this JPEG's pipeline keyword, in XMP and IPTC alike."""
    original = path.read_bytes()
    segments, scan = parse_jpeg(original)

    xmp_segment = find_segment(segments, APP1, XMP_SIG)
    if xmp_segment is None:
        raise SegmentError("no XMP packet")
    before = xmp_segment[1][len(XMP_SIG):].decode("utf-8")

    ours, theirs = split_keywords(current_subjects(before))
    subjects = list(theirs) + [label]

    # hierarchical=None: no export carries lr:hierarchicalSubject, and a stale
    # mirror resurrects old keywords on import. Never create one.
    after = set_subject_keywords(before, subjects)
    verify_only_keywords_changed(before, after, subjects)
    xmp_segment[1] = XMP_SIG + after.encode("utf-8")

    photoshop = find_segment(segments, APP13, PHOTOSHOP_SIG)
    if photoshop is None:
        raise SegmentError("no Photoshop image resource block")
    resources = parse_irb(photoshop[1][len(PHOTOSHOP_SIG):])
    iptc = next((r for r in resources if r[0] == IPTC_RESOURCE), None)
    if iptc is None:
        # An APP13 block without an IIM resource: rare but real (1 of the 12,896
        # in mcs5). Create it rather than failing, otherwise that file is the one
        # left with a keyword Lightroom cannot see -- which is the whole bug this
        # tool exists to fix. Resources are kept in id order, as Photoshop writes
        # them; sync_iim() supplies the charset marker and record version.
        iptc = [IPTC_RESOURCE, b"", b""]
        resources.append(iptc)
        resources.sort(key=lambda r: r[0])
    iptc[2] = build_iim(sync_iim(parse_iim(iptc[2]), subjects, when))
    photoshop[1] = PHOTOSHOP_SIG + build_irb(resources)

    rebuilt = build_jpeg(segments, scan)
    if rebuilt == original:
        return "unchanged"

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(rebuilt)
    tmp.replace(path)                       # atomic: no half-written JPEG
    return "written"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", type=Path,
                    default=PROJECT_ROOT / "output" / "lightroom" / "jpg" / "mcs15",
                    help="a seriated export directory containing index.csv")
    ap.add_argument("--label-dir", type=Path,
                    default=PROJECT_ROOT / "data" / "label",
                    help="directory holding bird_identification_output.csv")
    ap.add_argument("--keep-iptc-dates", action="store_true",
                    help="leave IIM DateCreated alone; it then still holds the "
                         "true capture date, contradicting the EXIF time the "
                         "seriation sort depends on")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be written, touching nothing")
    args = ap.parse_args()

    index = args.export / "index.csv"
    if not index.exists():
        sys.exit(f"no index.csv in {args.export} -- is that a seriated export?")

    with open(args.label_dir / "bird_identification_output.csv",
              encoding="utf-8-sig", newline="") as fh:
        labels = {r["jpg"]: r for r in csv.DictReader(fh)}
    with open(index, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    stat, failures, distinct = Counter(), [], set()
    for n, row in enumerate(rows, 1):
        key = row["key"]
        source = labels.get(key)
        if source is None:
            stat["no label row"] += 1
            failures.append((key, "no row in the label CSV"))
            continue
        label = effective_label(source)
        if not label:
            stat["no label"] += 1
            failures.append((key, "row carries no usable label"))
            continue
        distinct.add(label)

        when = None
        if not args.keep_iptc_dates:
            try:
                when = datetime.strptime(row["capture_time"], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                stat["unreadable capture time"] += 1
                failures.append((key, f"capture_time {row['capture_time']!r}"))
                continue

        if args.dry_run:
            stat["would write"] += 1
            if n <= 3:
                print(f"    {key}\n      -> {label}")
            continue
        try:
            stat[write_keywords(args.export / key, label, when)] += 1
        except (SegmentError, XmpEditError, OSError) as exc:
            stat["failed"] += 1
            failures.append((key, str(exc)))
        if n % 1000 == 0:
            print(f"    {n:,}/{len(rows):,}", flush=True)

    print(f"  {args.export}: {len(rows):,} images, "
          f"{len(distinct):,} distinct labels")
    for what, count in sorted(stat.items()):
        print(f"    {what}: {count:,}")
    for key, why in failures[:10]:
        print(f"    ! {key}: {why}")
    if len(failures) > 10:
        print(f"    ! ... and {len(failures) - 10} more")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
