"""Capture identity read out of an XMP sidecar's own metadata.

Used to decide whether two sidecars describe the *same photo*. The library was
imported more than once in places, so the same capture can sit in two trip
folders; those copies are what made neighbouring trips look like frame-counter
collisions during splitting.

Three fields matter:

* `xmpMM:OriginalDocumentID` -- Lightroom's identifier for the original raw
  file. Survives renaming, re-dating and timezone corrections, so it is the
  primary key here.
* `exif:DateTimeOriginal` -- capture time. Usable as a key, but only after
  normalising: about 10% of sidecars carry sub-second precision and the rest do
  not, so `...T12:30:58.82` and `...T12:30:58` are the same instant written two
  ways. Comparing the raw strings silently misses those.
* `crs:RawFileName` -- the original raw filename.

A virtual copy (Lightroom's `-2`, `-3` stems) shares `OriginalDocumentID` with
its parent but has its own `xmpMM:DocumentID`. Those are deliberate alternate
edits of one capture, not accidental duplicates, so they must be reported
separately rather than offered for deletion.
"""

import re
import xml.etree.ElementTree as ET
from typing import NamedTuple, Optional

# Attribute local-names worth pulling out of the RDF description.
_WANTED = ("DateTimeOriginal", "CreateDate", "RawFileName",
           "OriginalDocumentID", "DocumentID")

_DT_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})")
_VIRTUAL_COPY_RE = re.compile(r"-\d+$")


class SidecarMeta(NamedTuple):
    captured: str              # "YYYY-MM-DDTHH:MM:SS", whole seconds, no zone
    captured_raw: str          # exactly as written, zone and sub-seconds intact
    original_document_id: str
    document_id: str
    raw_file_name: str
    is_virtual_copy: bool

    @property
    def capture_date(self) -> str:
        return self.captured[:10]


def normalise_datetime(value: str) -> str:
    """Whole-second capture time, timezone and sub-seconds dropped.

    Sub-second precision is written inconsistently between two sidecars for the
    same photo, so it cannot be part of an equality test.
    """
    if not value:
        return ""
    m = _DT_RE.match(value.strip())
    return f"{m.group(1)}T{m.group(2)}" if m else value.strip()


def read_meta(xmp_path) -> Optional[SidecarMeta]:
    """Capture identity for one sidecar, or None if it will not parse."""
    try:
        tree = ET.parse(xmp_path)
    except (ET.ParseError, OSError):
        return None

    found: dict[str, str] = {}
    for element in tree.getroot().iter():
        for key, value in element.attrib.items():
            local = key.split("}")[-1]
            if local in _WANTED and local not in found:
                found[local] = value
        local = element.tag.split("}")[-1]
        if local in _WANTED and local not in found and (element.text or "").strip():
            found[local] = element.text.strip()

    captured_raw = found.get("DateTimeOriginal") or found.get("CreateDate") or ""
    stem = getattr(xmp_path, "stem", str(xmp_path))
    return SidecarMeta(
        captured=normalise_datetime(captured_raw),
        captured_raw=captured_raw,
        original_document_id=found.get("OriginalDocumentID", ""),
        document_id=found.get("DocumentID", ""),
        raw_file_name=found.get("RawFileName", ""),
        is_virtual_copy=bool(_VIRTUAL_COPY_RE.search(stem)),
    )


def capture_key(meta: SidecarMeta, frame) -> Optional[tuple]:
    """Key identifying the capture, or None when there is nothing to key on.

    Prefers `OriginalDocumentID`: it survives the renames, re-datings and
    timezone corrections that make the time-based key miss real duplicates.
    Falls back to normalised capture time plus frame identity.
    """
    if meta.original_document_id:
        return ("odid", meta.original_document_id)
    if meta.captured and frame is not None:
        return ("time+frame", meta.captured, frame)
    if meta.captured:
        return ("time+name", meta.captured, meta.raw_file_name)
    return None
