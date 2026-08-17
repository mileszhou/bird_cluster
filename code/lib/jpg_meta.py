"""Write keywords and a colour label into a JPEG's own metadata.

**Lightroom Classic reads `.xmp` sidecars only for proprietary raw formats.**
For a JPEG it reads embedded metadata and ignores a sidecar beside it, so
`xmp_write.set_keywords_in_xmp()` -- which deposits into `output/label/raw/*.xmp`
-- cannot carry a label into an export. It has to go inside the file. This
module is the inside-the-file half of that; `code/lib/xmp_write.py` remains the
sidecar half, and they share the keyword logic in `xmp_labels.py`.

**A keyword lives in two places, and both must be written.** The first version
of this set only XMP `dc:subject`, verified it, and produced 8,425 files that
Lightroom showed as unlabelled: these JPEGs also carry legacy **IPTC IIM**
keywords in a Photoshop `APP13` resource, and that is what Lightroom displayed.
Which one a reader prefers is not worth an opinion -- Adobe's rule turns on the
caption digest in resource `0x0425` -- so both are written with the same list and
left no way to disagree. The general lesson, since it cost a full pass over the
library: **reading back the field you just wrote proves the write, not the
outcome.** It cannot see a second copy of the same fact elsewhere in the file.

**Pixels are never touched.** The file is rebuilt from its marker structure --
APP13 changes length, so the packet cannot be overwritten in place -- with
everything from the start-of-scan marker copied through verbatim.

It is a library rather than a tool because it does one small thing to an
argument: `tools/export_seriated.py` is the command that uses it.
"""

import re
import struct
from pathlib import Path

from code.lib.label_generator import pinyin_initials
from code.lib.xmp_labels import parse_label, split_keywords
from code.lib.xmp_write import (XmpEditError, set_subject_keywords,
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


LABEL_ATTR_RE = re.compile(r'(\bxmp:Label=")([^"]*)(")')
LABEL_ELEM_RE = re.compile(r"(<xmp:Label>)(.*?)(</xmp:Label>)", re.S)
DESCRIPTION_RE = re.compile(r"<rdf:Description\b")


def set_color_label(text: str, colour: str) -> str:
    """Return `text` with Lightroom's colour label set to `colour`.

    Stored as `xmp:Label`, a plain string rather than a colour -- Lightroom's
    label *sets* are user-defined, which is why this export's sources carry
    values like `Safari` alongside `Red`. Writing one therefore overwrites
    whatever the photographer had; acceptable on a derived export, and the
    caller counts how often it happens.

    Attribute form first because that is what Lightroom writes and what all
    10,982 sampled files use; the element form is handled for completeness. When
    neither is present the attribute is inserted on the first `rdf:Description`,
    the same anchor `xmp_write._insert_subject` uses and for the same reason --
    property order inside a Description carries no meaning in RDF.
    """
    if LABEL_ATTR_RE.search(text):
        return LABEL_ATTR_RE.sub(lambda m: m.group(1) + colour + m.group(3), text, count=1)
    if LABEL_ELEM_RE.search(text):
        return LABEL_ELEM_RE.sub(lambda m: m.group(1) + colour + m.group(3), text, count=1)
    match = DESCRIPTION_RE.search(text)
    if not match:
        raise XmpEditError("no rdf:Description to attach xmp:Label to")
    at = match.end()
    return text[:at] + f' xmp:Label="{colour}"' + text[at:]


def write_keywords(path: Path, label: str, when=None, colour: str = "") -> str:
    """Set `label` as this JPEG's pipeline keyword, in XMP and IPTC alike.

    `colour`, when given, additionally sets the Lightroom colour label. It is a
    separate edit applied after the keyword one has been verified, because
    `verify_only_keywords_changed` asserts that *nothing* outside the keywords
    moved and would rightly reject it. Its own check is stricter and simpler:
    putting the old value back must reproduce the previous text byte for byte.
    """
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

    if colour:
        was = LABEL_ATTR_RE.search(after) or LABEL_ELEM_RE.search(after)
        previous = was.group(2) if was else None
        tinted = set_color_label(after, colour)
        restored = (set_color_label(tinted, previous) if previous is not None
                    else tinted.replace(f' xmp:Label="{colour}"', "", 1))
        if restored != after:
            raise XmpEditError("setting xmp:Label changed something else")
        after = tinted

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


