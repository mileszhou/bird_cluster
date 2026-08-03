"""Read the exporter's own log of what it refused to write.

Lightroom leaves `data/jpg/export.report.txt` next to the export, grouping the
files it skipped under a one-line reason:

    The file could not be found. (119)
        D:\\Lightroom\\MediaFiles\\Photos\\Photos-19\\2019-06-08 山公园\\_D5V0824.nef

That log is the difference between a sidecar with no JPEG being *explained* --
the raw it describes is gone from disk, so nothing could be exported -- and it
being an unexplained gap worth chasing. Paths are Windows-side and rooted in the
library, so they are re-anchored at the `Photos-YY` component to line up with
`data/xmp`.

A raw's decorations are stripped when keying: `X-Enhanced-NR.dng` is the AI
Denoise render of `X.nef` and carries no sidecar of its own, so a failure to
find it explains a missing JPEG for `X.xmp`.
"""

import re
from pathlib import PurePosixPath, PureWindowsPath
from typing import NamedTuple, Optional

REPORT_NAME = "export.report.txt"

_COUNT_RE = re.compile(r"^(?P<reason>.*?)\s*\((?P<count>\d+)\)\s*$")
_LIBRARY_RE = re.compile(r"^Photos-\d{2}$")
# Renders Lightroom writes alongside a raw, which have no sidecar of their own.
_RENDER_RE = re.compile(r"-(?:Enhanced-NR|Enhanced|Pano|HDR|Panorama)(?:-\d+)?$")

# The one reason that explains a missing JPEG rather than merely describing a
# file that was never going to be a photo.
NOT_FOUND = "The file could not be found."


class Entry(NamedTuple):
    reason: str
    folder: str  # "Photos-19/2019-06-08 山公园"
    stem: str    # the raw's stem, decorations intact
    name: str    # the raw's filename

    @property
    def sidecar_stem(self) -> str:
        """Stem of the sidecar that covers this raw (renders have none)."""
        return _RENDER_RE.sub("", self.stem)

    @property
    def key(self) -> tuple[str, str]:
        return (self.folder, self.sidecar_stem)


def _relocate(raw_path: str) -> Optional[tuple[str, str]]:
    """`D:\\...\\Photos\\Photos-19\\<trip>\\x.nef` -> ("Photos-19/<trip>", "x.nef")."""
    parts = PureWindowsPath(raw_path.strip()).parts
    start = next((i for i, p in enumerate(parts) if _LIBRARY_RE.match(p)), None)
    if start is None or start + 1 >= len(parts):
        return None
    rest = parts[start:]
    return str(PurePosixPath(*rest[:-1])), rest[-1]


def parse(path) -> list[Entry]:
    """Every skipped file in the report. Empty list if there is no report."""
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return []

    entries, reason = [], None
    for line in text.splitlines():
        if not line.strip():
            continue
        if not line[0].isspace():
            m = _COUNT_RE.match(line.strip())
            reason = m.group("reason") if m else line.strip()
            continue
        if reason is None:
            continue
        placed = _relocate(line)
        if placed is None:
            continue
        folder, name = placed
        entries.append(Entry(reason, folder, name.rsplit(".", 1)[0], name))
    return entries


class ExportReport:
    """Lookup from (trip folder, sidecar stem) to why the export skipped it."""

    def __init__(self, entries: list[Entry]):
        self.entries = entries
        self.by_key: dict[tuple[str, str], Entry] = {}
        for entry in entries:
            # Keep the first reason seen; the log repeats a path when several
            # renders of one raw failed together.
            self.by_key.setdefault(entry.key, entry)

    @classmethod
    def load(cls, jpg_dir) -> "ExportReport":
        from pathlib import Path
        return cls(parse(Path(jpg_dir) / REPORT_NAME))

    def entry_for(self, folder: str, stem: str) -> Optional[Entry]:
        """The log entry covering this sidecar, or None if it was never mentioned.

        Carries the raw's actual filename, which is what to go looking for when
        recovering it -- `X-Enhanced-NR.dng` and `X.nef` are different files on
        disk even though one sidecar covers both.
        """
        return self.by_key.get((folder, stem))

    def reason_for(self, folder: str, stem: str) -> Optional[str]:
        entry = self.entry_for(folder, stem)
        return entry.reason if entry else None
