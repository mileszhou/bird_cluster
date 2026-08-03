"""Resolve an XMP sidecar to its exported JPEG(s).

The jpg tree mirrors the sidecar tree exactly -- same library folder, same trip
folder, same stem:

    xmp/Photos-19/2019-01-13 山公园/20190113-_D8S0025.xmp
    jpg/Photos-19/2019-01-13 山公园/20190113-_D8S0025.jpg

so a match is a plain lookup inside one folder. This replaces the half-year /
whole-tree-stem machinery the flat export needed: stems repeat across the
library (camera counter wraparound), but that no longer matters when the folder
is part of the key, so there is nothing left to be ambiguous about.

**Derived exports.** Lightroom does not always export a raw under its own name:

* a virtual copy exports as `<stem>-2`, `-3`, ... (its copy name), and when the
  master itself was not in the export set, that decorated file is the *only*
  export of the capture -- 5,990 sidecars in the current dataset;
* AI Denoise writes a separate `<stem>-Enhanced-NR.dng`, which carries its
  metadata internally and so has **no sidecar of its own**; if only that DNG was
  exported, the raw's sidecar again has no same-named JPEG.

Both are the right photo for the sidecar, so `resolve()` accepts them and says
`derived` rather than pretending the export is missing. Matching is done a whole
folder at a time because exact hits must be claimed first: where sidecars `A`
and `A-2` both exist, `A-2.jpg` belongs to `A-2`, not to `A` as a virtual copy.
Ties among prefixes go to the longest sidecar stem.

A JPEG nothing claims is an *extra*: usually a photo that was never raw (phone
shots), so there is no sidecar to have. `extras()` splits those from the
decorated siblings of an already-matched sidecar, which are ordinary virtual
copies rather than unaccounted files.
"""

import re
from enum import Enum
from pathlib import Path
from typing import NamedTuple, Optional

JPG_SUFFIXES = (".jpg", ".jpeg")
_YEAR_RE = re.compile(r"^Photos-(\d{2})$")


class _NamedByValue(str, Enum):
    """A str enum that prints as its value, so argparse renders readable choices."""

    def __str__(self):
        return self.value


class Verdict(_NamedByValue):
    OK = "ok"                # a JPEG of the same stem, in the mirrored folder
    DERIVED = "derived"      # only a decorated export exists (virtual copy / -Enhanced-NR)
    NO_JPG = "no_jpg"        # the folder was exported, this photo was not
    NO_FOLDER = "no_folder"  # no exported folder mirrors this trip at all


class ExtraKind(_NamedByValue):
    DERIVED = "derived"  # decorated sibling of a sidecar that already matched
    ORPHAN = "orphan"    # nothing claims it -- not a raw file, so never had a sidecar


class JpgMatch(NamedTuple):
    paths: tuple[Path, ...]  # more than one when several virtual copies were exported
    verdict: Verdict
    folder: str              # trip folder, relative to the tree root

    @property
    def path(self) -> Optional[Path]:
        return self.paths[0] if self.paths else None

    @property
    def ok(self) -> bool:
        return bool(self.paths)


class Extra(NamedTuple):
    path: Path
    kind: ExtraKind
    folder: str
    parent_stem: str  # the sidecar it decorates, "" for an orphan


def library_year(library: str) -> Optional[str]:
    """Full year of a library folder name (`Photos-19` -> `2019`)."""
    m = _YEAR_RE.match(library)
    return f"20{m.group(1)}" if m else None


def _relfolder(path: Path, root: Path) -> str:
    return str(path.parent.relative_to(root))


class _Folder(NamedTuple):
    matches: dict[str, tuple[Path, ...]]
    verdicts: dict[str, Verdict]
    extras: tuple[Extra, ...]


def _resolve_folder(folder: str, xmp_stems: dict[str, Path],
                    jpgs: dict[str, list[Path]]) -> _Folder:
    """Assign one trip folder's JPEGs to its sidecars: exact first, then derived."""
    matches: dict[str, tuple[Path, ...]] = {}
    verdicts: dict[str, Verdict] = {}
    claimed: set[str] = set()

    for stem in xmp_stems:
        if stem in jpgs:
            matches[stem] = tuple(jpgs[stem])
            verdicts[stem] = Verdict.OK
            claimed.add(stem)

    # Longest stem first so `A-2` claims `A-2-3.jpg` ahead of `A`. A JPEG whose
    # stem is itself a sidecar stem is never a derived export of another one.
    free = {s for s in jpgs if s not in claimed and s not in xmp_stems}
    for stem in sorted(set(xmp_stems) - claimed, key=len, reverse=True):
        taken = sorted(s for s in free if s.startswith(stem + "-"))
        if not taken:
            verdicts[stem] = Verdict.NO_JPG
            continue
        matches[stem] = tuple(p for s in taken for p in jpgs[s])
        verdicts[stem] = Verdict.DERIVED
        free -= set(taken)
        claimed |= set(taken)

    extras = []
    for stem in sorted(set(jpgs) - claimed):
        parents = [s for s in xmp_stems if stem.startswith(s + "-")]
        parent = max(parents, key=len) if parents else ""
        kind = ExtraKind.DERIVED if parent else ExtraKind.ORPHAN
        extras += [Extra(p, kind, folder, parent) for p in jpgs[stem]]

    return _Folder(matches, verdicts, tuple(extras))


class JpgIndex:
    """Sidecar <-> JPEG assignment over a mirrored `xmp/` + `jpg/` pair."""

    def __init__(self, xmp_dir, jpg_dir):
        self.xmp_dir = Path(xmp_dir)
        self.jpg_dir = Path(jpg_dir)
        self._folders: Optional[dict[str, _Folder]] = None
        self._xmp: Optional[dict[str, dict[str, Path]]] = None
        self._jpg: Optional[dict[str, dict[str, list[Path]]]] = None

    # -- scanning ---------------------------------------------------------

    def _scan(self):
        if self._folders is not None:
            return
        xmp: dict[str, dict[str, Path]] = {}
        for path in sorted(self.xmp_dir.rglob("*.xmp")):
            xmp.setdefault(_relfolder(path, self.xmp_dir), {})[path.stem] = path
        jpg: dict[str, dict[str, list[Path]]] = {}
        for path in sorted(self.jpg_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in JPG_SUFFIXES:
                folder = jpg.setdefault(_relfolder(path, self.jpg_dir), {})
                folder.setdefault(path.stem, []).append(path)
        self._xmp, self._jpg = xmp, jpg
        self._folders = {
            folder: _resolve_folder(folder, stems, jpg.get(folder, {}))
            for folder, stems in xmp.items()
        }

    @property
    def xmp_folders(self) -> dict[str, dict[str, Path]]:
        self._scan()
        return self._xmp

    @property
    def jpg_folders(self) -> dict[str, dict[str, list[Path]]]:
        self._scan()
        return self._jpg

    def sidecars(self):
        """Every sidecar path, in folder order."""
        for folder in sorted(self.xmp_folders):
            for stem in sorted(self.xmp_folders[folder]):
                yield self.xmp_folders[folder][stem]

    # -- lookups ----------------------------------------------------------

    def resolve(self, xmp_path) -> JpgMatch:
        self._scan()
        xmp_path = Path(xmp_path)
        folder = _relfolder(xmp_path, self.xmp_dir)
        if folder not in self._jpg:
            return JpgMatch((), Verdict.NO_FOLDER, folder)
        entry = self._folders[folder]
        verdict = entry.verdicts.get(xmp_path.stem, Verdict.NO_JPG)
        return JpgMatch(entry.matches.get(xmp_path.stem, ()), verdict, folder)

    def extras(self):
        """JPEGs no sidecar claims, including whole folders with no sidecars."""
        self._scan()
        for folder in sorted(self._folders):
            yield from self._folders[folder].extras
        for folder in sorted(set(self._jpg) - set(self._xmp)):
            for stem in sorted(self._jpg[folder]):
                for path in self._jpg[folder][stem]:
                    yield Extra(path, ExtraKind.ORPHAN, folder, "")

    def orphan_folders(self) -> list[str]:
        """Exported folders with no sidecar folder mirroring them."""
        self._scan()
        return sorted(set(self._jpg) - set(self._xmp))

    def unexported_folders(self) -> list[str]:
        """Sidecar folders with no exported folder mirroring them."""
        self._scan()
        return sorted(set(self._xmp) - set(self._jpg))
