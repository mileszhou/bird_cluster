"""Tests for code/lib/trips.py -- run from within test/: `pytest lib/`."""
from datetime import date

import pytest

from code.lib.trips import Frame, Timeline, Trip, frame_id, trip_date


@pytest.mark.parametrize("stem,expected", [
    ("_D8S7785", Frame("D8S", "7785")),
    ("_D8S7785-2", Frame("D8S", "7785")),                 # Lightroom virtual copy
    ("20181229-_D8S7785", Frame("D8S", "7785")),          # date-disambiguated import
    ("20181229-_D8S7785-2", Frame("D8S", "7785")),        # both at once
    ("20190113-_D8S0025_host_123_Conflict", Frame("D8S", "0025")),  # sync conflict
])
def test_frame_id_collapses_filename_variants(stem, expected):
    """All the decorations name the same physical frame."""
    assert frame_id(stem) == expected


def test_frame_id_keeps_leading_zeros():
    """0025 and 25 are different counter values; the string must survive."""
    assert frame_id("_D8S0025").num == "0025"


def test_frame_id_distinguishes_bodies():
    assert frame_id("_D8S7785") != frame_id("_V9A7785")


def test_frame_id_rejects_non_camera_names():
    assert frame_id("holiday-snap") is None
    assert frame_id("IMG") is None


@pytest.mark.parametrize("name,expected", [
    ("2023-10-05 Lake Tahoe", date(2023, 10, 5)),
    ("2020-11 福州.鸟", date(2020, 11, 1)),        # day-less: 1st is precise enough to order
    ("2019-02-08 盈江2、7、8、10、5号塘", date(2019, 2, 8)),
    ("0000-test", None),
    ("no date here", None),
    ("2023-13-99 impossible", None),
])
def test_trip_date(name, expected):
    assert trip_date(name) == expected


def timeline(*names):
    trips = []
    dated = sorted((n for n in names if trip_date(n)), key=lambda n: (trip_date(n), n))
    for i, n in enumerate(dated):
        trips.append(Trip(n, "2019", "2019.1", None, trip_date(n), i))
    for n in names:
        if not trip_date(n):
            trips.append(Trip(n, "2019", "2019.1", None, None, -1))
    return Timeline(trips), {t.name: t for t in trips}


def test_adjacent_trips_are_neighbours():
    tl, t = timeline("2019-01-01 A", "2019-01-02 B", "2019-06-01 C")
    assert tl.neighbours(t["2019-01-01 A"], t["2019-01-02 B"])
    assert not tl.neighbours(t["2019-01-01 A"], t["2019-06-01 C"])


def test_window_widens_neighbourhood():
    tl, t = timeline("2019-01-01 A", "2019-01-02 B", "2019-06-01 C")
    assert tl.neighbours(t["2019-01-01 A"], t["2019-06-01 C"], window=2)


def test_ordering_is_by_date_not_folder_name():
    """Trips must sort chronologically even when the names sort differently."""
    tl, t = timeline("2019-02-01 zzz", "2019-01-01 aaa", "2019-03-01 mmm")
    assert [x.name for x in sorted(tl.trips, key=lambda x: x.rank)] == [
        "2019-01-01 aaa", "2019-02-01 zzz", "2019-03-01 mmm"]


def test_undated_trip_is_never_a_neighbour():
    """No date means no evidence -- a collision there must stay undecidable."""
    tl, t = timeline("2019-01-01 A", "0000-test")
    assert t["0000-test"].rank == -1
    assert not tl.neighbours(t["2019-01-01 A"], t["0000-test"])
    assert not tl.neighbours(t["0000-test"], t["0000-test"])


def test_day_gap():
    tl, t = timeline("2019-01-01 A", "2019-01-11 B", "0000-test")
    assert tl.day_gap(t["2019-01-01 A"], t["2019-01-11 B"]) == 10
    assert tl.day_gap(t["2019-01-01 A"], t["0000-test"]) is None


def test_from_dataset_walks_half_year_and_trip(tmp_path):
    for half, trip in [("2019.1", "2019-01-01 A"), ("2019.2", "2019-02-01 B")]:
        (tmp_path / "result-2019" / "raw" / half / trip).mkdir(parents=True)
    tl = Timeline.from_dataset(tmp_path)
    assert {t.name for t in tl.trips} == {"2019-01-01 A", "2019-02-01 B"}
    a = tl.by_key["2019/2019.1/2019-01-01 A"]
    b = tl.by_key["2019/2019.2/2019-02-01 B"]
    # Neighbouring on the timeline even though they sit in different half-years.
    assert tl.neighbours(a, b)
