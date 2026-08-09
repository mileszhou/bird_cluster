"""Tests for code/lib/path_filter.py -- run from within test/: `pytest lib/`."""
import pytest

from code.lib.path_filter import PathFilter, build, read_patterns

ZOO = "Photos-16/2016-07-12 City Zoo"
KEY = f"{ZOO}/_D5S1234.jpg"


def test_no_lists_allows_everything():
    f = PathFilter()
    assert not f
    assert f.allows(KEY)


def test_a_folder_line_takes_its_subtree():
    assert not PathFilter(exclude=[ZOO]).allows(KEY)


def test_a_trailing_slash_is_the_same_folder():
    assert not PathFilter(exclude=[ZOO + "/"]).allows(KEY)


def test_an_exact_file_line():
    f = PathFilter(exclude=[KEY])
    assert not f.allows(KEY)
    assert f.allows(f"{ZOO}/_D5S9999.jpg")


def test_matching_is_on_path_segments_not_characters():
    """`Photos-2` must not swallow `Photos-24`, which a prefix test would."""
    f = PathFilter(exclude=["Photos-2"])
    assert f.allows("Photos-24/trip/a.jpg")
    assert f.allows("Photos-21/trip/a.jpg")


def test_include_restricts_to_the_listed_subtrees():
    f = PathFilter(include=["Photos-24", "Photos-25/trip b"])
    assert f.allows("Photos-24/anything/a.jpg")
    assert f.allows("Photos-25/trip b/a.jpg")
    assert not f.allows("Photos-25/trip c/a.jpg")
    assert not f.allows("Photos-16/trip/a.jpg")


def test_exclude_wins_over_include():
    """Carving a known-bad subset out of a wanted range is the usual shape."""
    f = PathFilter(include=["Photos-16"], exclude=[ZOO])
    assert f.allows("Photos-16/other trip/a.jpg")
    assert not f.allows(KEY)


# --- the file format --------------------------------------------------------

def test_read_patterns_drops_comments_and_blanks(tmp_path):
    p = tmp_path / "list.txt"
    p.write_text("# a comment\n\n"
                 f"{ZOO}   # trailing comment\n"
                 "Photos-24/trip/\n", encoding="utf-8")
    assert read_patterns(p) == [ZOO, "Photos-24/trip"]


def test_read_patterns_tolerates_a_spreadsheet_bom(tmp_path):
    """These get edited in Excel, and trip folders are Chinese."""
    p = tmp_path / "list.txt"
    p.write_text("Photos-19/2019-01-13 山公园\n", encoding="utf-8-sig")
    assert read_patterns(p) == ["Photos-19/2019-01-13 山公园"]


def test_read_patterns_normalises_windows_separators(tmp_path):
    p = tmp_path / "list.txt"
    p.write_text("Photos-24\\trip b\n", encoding="utf-8")
    assert read_patterns(p) == ["Photos-24/trip b"]


def test_build_from_files(tmp_path):
    inc, exc = tmp_path / "i.txt", tmp_path / "e.txt"
    inc.write_text("Photos-16\n", encoding="utf-8")
    exc.write_text(f"{ZOO}\n", encoding="utf-8")
    f = build(inc, exc)
    assert f.allows("Photos-16/other/a.jpg")
    assert not f.allows(KEY)
    assert "1 include" in f.describe() and "1 exclude" in f.describe()


def test_build_with_nothing_is_falsy():
    assert not build(None, None)
