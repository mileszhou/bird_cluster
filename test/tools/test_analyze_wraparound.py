"""Tests for the split proposer.

`tools/` is a directory of scripts, not a package, so the module is loaded by
path rather than imported by name.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "analyze_wraparound", ROOT / "tools" / "analyze_wraparound.py")
aw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(aw)


def trip(tmp_path, name, stems):
    folder = tmp_path / name
    folder.mkdir()
    for stem in stems:
        (folder / f"{stem}.xmp").write_text("")
    return folder


@pytest.fixture
def year(tmp_path):
    """A year folder builder: `year("2019-01-01 a", ["_D5C0001"], ...)`."""
    def build(*trips):
        return [trip(tmp_path, name, stems) for name, stems in trips]
    return build


def test_cuts_only_where_a_name_repeats(year):
    dirs = year(
        ("2019-01-01 a", ["_D5C0001", "_D5C0002"]),
        ("2019-02-01 b", ["_D5C0003"]),
        ("2019-03-01 c", ["_D5C0002"]),      # repeats a's frame -- forces the cut
        ("2019-04-01 d", ["_D5C0009"]),
    )
    trips = aw.order_trips([aw.read_trip(d, aw.STEM) for d in dirs])
    segments = aw.propose_segments("2019", trips)

    assert [s.name for s in segments] == ["2019.1", "2019.2"]
    assert [t.name for t in segments[0].trips] == ["2019-01-01 a", "2019-02-01 b"]
    assert [t.name for t in segments[1].trips] == ["2019-03-01 c", "2019-04-01 d"]
    assert segments[1].cut_by == [("_D5C0002", "2019-01-01 a")]
    assert aw.verify(segments) == []


def test_no_collision_is_one_segment(year):
    dirs = year(
        ("2022-01-01 a", ["_D5C0001"]),
        ("2022-06-01 b", ["_D5C0002"]),
    )
    trips = aw.order_trips([aw.read_trip(d, aw.STEM) for d in dirs])
    assert len(aw.propose_segments("2022", trips)) == 1


def test_import_prefix_and_virtual_copy_do_not_force_a_cut(year):
    """Lightroom already disambiguated these, and the decoration survives into
    the jpg export -- so by stem they are distinct names, though the camera
    frame underneath is the same."""
    dirs = year(
        ("2019-01-01 a", ["_D5C0001"]),
        ("2019-02-01 b", ["20190201-_D5C0001", "_D5C0001-2"]),
    )
    by_stem = aw.order_trips([aw.read_trip(d, aw.STEM) for d in dirs])
    by_frame = aw.order_trips([aw.read_trip(d, aw.FRAME) for d in dirs])

    assert len(aw.propose_segments("2019", by_stem)) == 1
    assert len(aw.propose_segments("2019", by_frame)) == 2


def test_undated_folders_sort_last(year):
    dirs = year(
        ("misc", ["_D5C0009"]),
        ("2019-05-01 b", ["_D5C0002"]),
        ("2019-01-01 a", ["_D5C0001"]),
    )
    trips = aw.order_trips([aw.read_trip(d, aw.STEM) for d in dirs])
    assert [t.name for t in trips] == ["2019-01-01 a", "2019-05-01 b", "misc"]
    assert trips[-1].when is None


def test_nested_sub_trip_counts_towards_its_parent(tmp_path):
    """A split moves the whole top-level folder, so a nested sub-trip's photos
    have to travel with the parent -- and collide on the parent's behalf."""
    parent = trip(tmp_path, "2021-03-26 outer", ["_D5C0001"])
    nested = parent / "2021-03-28 inner"
    nested.mkdir()
    (nested / "_D5C0002.xmp").write_text("")

    info = aw.read_trip(parent, aw.STEM)
    assert info.keys == {"_D5C0001", "_D5C0002"}
    assert info.sidecars == 2


def test_verify_flags_a_repeat_inside_one_trip(year):
    """Two names identical within a single folder cannot be split apart, since a
    segment boundary never runs through a trip."""
    dirs = year(("2019-01-01 a", ["_D5C0001", "20190101-_D5C0001"]))
    trips = aw.order_trips([aw.read_trip(d, aw.FRAME) for d in dirs])
    segments = aw.propose_segments("2019", trips)

    broken = aw.verify(segments)
    assert len(broken) == 1
    assert broken[0][0] == "2019.1"
    assert aw.key_label(broken[0][1]) == "_D5C0001"


def test_neighbouring_collisions_need_dated_trips(year):
    dirs = year(
        ("2019-01-01 a", ["_D5C0001"]),
        ("2019-01-02 b", ["_D5C0001"]),
        ("misc", ["_D5C0001"]),
    )
    trips = aw.order_trips([aw.read_trip(d, aw.STEM) for d in dirs])
    found = aw.neighbouring(trips, aw.collisions(trips), window=1)

    assert len(found) == 1
    key, first, second, gap = found[0]
    assert (first.name, second.name, gap) == ("2019-01-01 a", "2019-01-02 b", 1)


def test_strict_cuts_on_range_overlap_without_a_repeat(year):
    """No name repeats, but the counter ranges interleave -- a frame shot and
    never imported would still clash in a re-export."""
    dirs = year(
        ("2019-01-01 a", ["_D5C0010", "_D5C0020"]),
        ("2019-02-01 b", ["_D5C0015"]),
    )
    trips = aw.order_trips([aw.read_trip(d, aw.STEM) for d in dirs])
    assert len(aw.propose_segments("2019", trips)) == 1
    assert aw.count_segments_strict(trips) == 2


def test_year_label():
    assert aw.year_label("Photos-19") == "2019"
    assert aw.year_label("Photos-2019") == "2019"
