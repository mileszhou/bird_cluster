"""Trip folders as a time axis, and camera frame identity.

Trip folders are named `YYYY-MM-DD <place>` (a few older ones give only
`YYYY-MM`). Ordering every trip in the library by that date turns the library
into a timeline, which is what makes filename collisions decidable:

A camera's frame counter only repeats after it has wrapped through its whole
range -- thousands of frames, which takes months or years of shooting. So a
counter value cannot legitimately repeat inside one trip, nor between two trips
that are neighbours on the timeline. If the same frame number shows up in
neighbouring trips, the only explanation left is that it is the *same photo*,
filed twice. Far apart on the timeline, the same collision is wraparound: two
genuinely different photos.

`frame_id` normalises the filename variants Lightroom and filesystem sync leave
behind, so `20190113-_D8S0025-2.xmp` and `_D8S0025.xmp` are recognised as the
same frame.
"""

import re
from datetime import date
from pathlib import Path
from typing import NamedTuple, Optional

# "2023-10-05 Lake Tahoe", "2020-11 福州.鸟". "0000-test" is a scratch folder.
TRIP_DATE_RE = re.compile(r"^(\d{4})-(\d{2})(?:-(\d{2}))?")

# Filename shapes seen in this library:
#   _D8S7785            plain camera file
#   _D8S7785-2          Lightroom virtual copy
#   20181229-_D8S7785   date-disambiguated on import (name clash)
#   20190113-_D8S0025_<host>_<ts>_Conflict   filesystem sync-conflict copy
# The camera "code" is the body id immediately before a run of 4+ digits (the
# counter); that (code, counter) pair is what the camera also burns into the
# exported jpg filename.
FRAME_RE = re.compile(
    r"^(?:\d{8}-)?"              # optional date-disambiguation prefix
    r"_?"                        # leading underscore
    r"(?P<code>[A-Za-z0-9_]*?)"
    r"(?P<num>\d{4,})"
    r"(?:-\d+)?"                 # optional virtual-copy suffix
    r"(?:_.*)?$"                 # optional sync-conflict suffix
)


class Frame(NamedTuple):
    code: str   # camera body id
    num: str    # frame counter, as written (leading zeros preserved)


def frame_id(stem: str) -> Optional[Frame]:
    """Camera frame identity for a filename stem, or None if unrecognised.

    Variants of the same physical frame (virtual copies, date-disambiguated
    imports, sync conflicts) all collapse to the same Frame.
    """
    m = FRAME_RE.match(stem)
    if not m:
        return None
    return Frame(m.group("code"), m.group("num"))


def trip_date(folder_name: str) -> Optional[date]:
    """Date a trip folder name starts with, or None if it has no date prefix.

    Day-less names (`2020-11 福州.鸟`) are treated as the 1st, which is precise
    enough for ordering trips against each other. When the distinction matters
    -- e.g. asking how far a photo's capture date sits from its folder's date --
    check `trip_date_precision` first, since such a folder names a whole month
    and its photos are not "23 days late".
    """
    m = TRIP_DATE_RE.match(folder_name)
    if not m:
        return None
    year, month, day = m.groups()
    try:
        return date(int(year), int(month), int(day or 1))
    except ValueError:
        return None


def trip_date_precision(folder_name: str) -> Optional[str]:
    """`"day"`, `"month"`, or None -- how precisely the folder name is dated."""
    if trip_date(folder_name) is None:
        return None
    m = TRIP_DATE_RE.match(folder_name)
    return "day" if m.group(3) else "month"


class Trip(NamedTuple):
    name: str            # folder name, e.g. "2023-10-05 Lake Tahoe"
    year: str            # owning result-<YYYY>
    half_year: str       # owning half-year folder, e.g. "2023.1"
    path: Path
    when: Optional[date]
    rank: int            # position on the global timeline; -1 if undated

    @property
    def key(self) -> str:
        return f"{self.year}/{self.half_year}/{self.name}"


class Timeline:
    """Every trip in the dataset, ordered by date.

    Undated trips get rank -1 and are never considered anyone's neighbour --
    without a date there is no evidence either way, so they are reported as
    undecidable rather than guessed at.
    """

    def __init__(self, trips: list[Trip]):
        self.trips = trips
        self.by_key = {t.key: t for t in trips}

    @classmethod
    def from_dataset(cls, xmp_root) -> "Timeline":
        found = []
        for year_dir in sorted(Path(xmp_root).glob("result-*")):
            year = year_dir.name[len("result-"):]
            raw = year_dir / "raw"
            if not raw.is_dir():
                continue
            for trip_dir in raw.glob("*/*"):
                if trip_dir.is_dir():
                    found.append((trip_dir, year, trip_dir.parent.name))

        dated = [(t, y, h) for t, y, h in found if trip_date(t.name) is not None]
        undated = [(t, y, h) for t, y, h in found if trip_date(t.name) is None]
        dated.sort(key=lambda x: (trip_date(x[0].name), x[0].name))

        trips = [
            Trip(name=t.name, year=y, half_year=h, path=t, when=trip_date(t.name), rank=i)
            for i, (t, y, h) in enumerate(dated)
        ]
        trips += [
            Trip(name=t.name, year=y, half_year=h, path=t, when=None, rank=-1)
            for t, y, h in undated
        ]
        return cls(trips)

    def neighbours(self, a: Trip, b: Trip, window: int = 1) -> bool:
        """True if two trips are within `window` positions on the timeline.

        Undated trips (rank -1) are never neighbours: with no date there is no
        basis to call a collision either a duplicate or a wraparound.
        """
        if a.rank < 0 or b.rank < 0:
            return False
        return abs(a.rank - b.rank) <= window

    def day_gap(self, a: Trip, b: Trip) -> Optional[int]:
        if a.when is None or b.when is None:
            return None
        return abs((a.when - b.when).days)
