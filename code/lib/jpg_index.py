"""Resolve an XMP sidecar to its exported JPEG.

Layout: sidecars are nested `raw/<half-year>/<trip>/*.xmp`, JPEGs are flat
`jpg/<half-year>/*.jpg`. The half-year folder names match, so the primary
lookup is `raw/2023.1/<trip>/_X.xmp` -> `jpg/2023.1/_X.jpg`.

Filename stems are *not* globally unique. Camera frame counters wrap around, so
in the current dataset 4371 of 27043 stems appear in more than one jpg folder.
Falling back to a whole-tree stem lookup therefore silently returns a photo
from an unrelated shoot -- in the current dataset that misresolves 949 bird
sidecars, 729 of them in a year whose JPEGs were never exported at all. Hence
`MatchPolicy`: the default refuses cross-year matches.

Two distinct things produce an off-folder match, and they need opposite
treatment:

* Same year, adjacent folder (`2019.7` -> `2019.8`): the export split fell in a
  slightly different place than the sidecar split, so the trip's JPEGs landed
  in the neighbouring half. Same shoot, right photo.
* Different year (`2018.1` -> `2024.5`): counter wraparound. Wrong photo.

`SAME_YEAR` accepts the first and rejects the second.
"""

import re
from enum import Enum
from pathlib import Path
from typing import NamedTuple, Optional

JPG_SUFFIXES = (".jpg", ".jpeg")
_YEAR_RE = re.compile(r"^(\d{4})")


class _NamedByValue(str, Enum):
    """A str enum that prints as its value, so argparse renders readable choices."""

    def __str__(self):
        return self.value


class MatchPolicy(_NamedByValue):
    EXPECTED = "expected"    # only the half-year folder matching the sidecar's
    SAME_YEAR = "same_year"  # + a unique match elsewhere in the same year
    ANY = "any"              # + a unique match anywhere (legacy; unsafe)


class Verdict(_NamedByValue):
    OK = "ok"                                # in the expected folder
    OFF_FOLDER_SAME_YEAR = "off_folder_same_year"
    OFF_FOLDER_CROSS_YEAR = "off_folder_cross_year"
    AMBIGUOUS = "ambiguous"                  # several candidates, none expected
    NO_JPG = "no_jpg"                        # stem absent from the jpg tree
    REJECTED = "rejected"                    # found, but policy refused it


class JpgMatch(NamedTuple):
    path: Optional[Path]          # None unless the match was accepted
    verdict: Verdict
    expected_folder: Optional[str]
    candidates: tuple[Path, ...]  # everything found, accepted or not

    @property
    def ok(self) -> bool:
        return self.path is not None


def folder_year(folder_name: str) -> Optional[str]:
    """Leading year of a half-year folder name ('2019.7' -> '2019')."""
    m = _YEAR_RE.match(folder_name)
    return m.group(1) if m else None


class JpgIndex:
    """Stem -> JPEG paths over one jpg tree, built lazily and cached."""

    def __init__(self, jpg_dir, policy: MatchPolicy = MatchPolicy.SAME_YEAR):
        self.jpg_dir = Path(jpg_dir)
        self.policy = MatchPolicy(policy)
        self._stems: Optional[dict[str, list[Path]]] = None

    @property
    def stems(self) -> dict[str, list[Path]]:
        if self._stems is None:
            index: dict[str, list[Path]] = {}
            for path in self.jpg_dir.rglob("*"):
                if path.is_file() and path.suffix.lower() in JPG_SUFFIXES:
                    index.setdefault(path.stem, []).append(path)
            self._stems = index
        return self._stems

    def colliding_stems(self) -> dict[str, list[Path]]:
        """Stems whose copies span more than one folder -- the wraparound set."""
        out = {}
        for stem, paths in self.stems.items():
            if len({p.parent.name for p in paths}) > 1:
                out[stem] = paths
        return out

    @staticmethod
    def expected_folder(xmp_file, raw_root) -> Optional[str]:
        """The half-year folder a sidecar's JPEG should live in.

        The first path component under `raw_root`, so both
        `raw/<half-year>/<trip>/x.xmp` and `raw/<half-year>/x.xmp` give
        `<half-year>` -- the trip level is optional. Returns None only when
        there is no folder component at all (a sidecar directly in `raw/`) or
        the sidecar is not under `raw_root`.
        """
        xmp_file, raw_root = Path(xmp_file), Path(raw_root)
        try:
            parts = xmp_file.relative_to(raw_root).parts
        except ValueError:
            return None
        return parts[0] if len(parts) > 1 else None

    def find(self, xmp_file, raw_root) -> JpgMatch:
        xmp_file = Path(xmp_file)
        stem = xmp_file.stem
        expected = self.expected_folder(xmp_file, raw_root)
        candidates = tuple(self.stems.get(stem, ()))

        if not candidates:
            return JpgMatch(None, Verdict.NO_JPG, expected, ())

        if expected is not None:
            for path in candidates:
                if path.parent.name == expected:
                    return JpgMatch(path, Verdict.OK, expected, candidates)

        if self.policy is MatchPolicy.EXPECTED:
            return JpgMatch(None, Verdict.REJECTED, expected, candidates)

        folders = {p.parent.name for p in candidates}
        if len(folders) > 1:
            return JpgMatch(None, Verdict.AMBIGUOUS, expected, candidates)

        # Exactly one folder holds this stem, and it is not the expected one.
        only = candidates[0]
        same_year = (
            expected is not None
            and folder_year(expected) is not None
            and folder_year(expected) == folder_year(only.parent.name)
        )
        verdict = Verdict.OFF_FOLDER_SAME_YEAR if same_year else Verdict.OFF_FOLDER_CROSS_YEAR
        accept = same_year or self.policy is MatchPolicy.ANY
        return JpgMatch(only if accept else None, verdict, expected, candidates)
