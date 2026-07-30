"""Tests for code/lib/jpg_index.py -- run from within test/: `pytest lib/`."""
import pytest

from code.lib.jpg_index import JpgIndex, MatchPolicy, Verdict, folder_year


@pytest.fixture
def dataset(tmp_path):
    """A miniature dataset exercising every resolution outcome.

    jpg/
      2019.7/  a.jpg          expected-folder hit for 2019.7
      2019.8/  b.jpg          same-year off-folder (export boundary offset)
      2024.5/  c.jpg          cross-year off-folder (counter wraparound)
      2021.2/  d.jpg          } d in two folders -> ambiguous
      2022/    d.jpg          }
    """
    jpg = tmp_path / "jpg"
    for folder, names in {
        "2019.7": ["a"], "2019.8": ["b"], "2024.5": ["c"],
        "2021.2": ["d"], "2022": ["d"],
    }.items():
        (jpg / folder).mkdir(parents=True)
        for n in names:
            (jpg / folder / f"{n}.jpg").write_bytes(b"x")

    raw = tmp_path / "raw"
    for stem in ["a", "b", "c", "d", "missing"]:
        d = raw / "2019.7" / "trip"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{stem}.xmp").write_text("<x/>")
    return tmp_path


def find(dataset, stem, policy):
    idx = JpgIndex(dataset / "jpg", policy=policy)
    return idx.find(dataset / "raw" / "2019.7" / "trip" / f"{stem}.xmp", dataset / "raw")


def test_expected_folder_hit_wins(dataset):
    for policy in MatchPolicy:
        m = find(dataset, "a", policy)
        assert m.verdict is Verdict.OK
        assert m.ok and m.path.parent.name == "2019.7"


def test_no_jpg_anywhere(dataset):
    for policy in MatchPolicy:
        m = find(dataset, "missing", policy)
        assert m.verdict is Verdict.NO_JPG
        assert not m.ok and m.path is None


def test_same_year_off_folder_accepted_by_default(dataset):
    m = find(dataset, "b", MatchPolicy.SAME_YEAR)
    assert m.verdict is Verdict.OFF_FOLDER_SAME_YEAR
    assert m.ok and m.path.parent.name == "2019.8"


def test_cross_year_off_folder_refused_by_default(dataset):
    """The wraparound case: a match exists but is the wrong photo."""
    m = find(dataset, "c", MatchPolicy.SAME_YEAR)
    assert m.verdict is Verdict.OFF_FOLDER_CROSS_YEAR
    assert not m.ok and m.path is None
    # ...and is still reported as a candidate, for auditing.
    assert [p.parent.name for p in m.candidates] == ["2024.5"]


def test_cross_year_accepted_only_under_any(dataset):
    m = find(dataset, "c", MatchPolicy.ANY)
    assert m.verdict is Verdict.OFF_FOLDER_CROSS_YEAR
    assert m.ok and m.path.parent.name == "2024.5"


def test_expected_policy_refuses_all_off_folder(dataset):
    for stem in ("b", "c"):
        m = find(dataset, stem, MatchPolicy.EXPECTED)
        assert m.verdict is Verdict.REJECTED
        assert not m.ok


def test_ambiguous_never_accepted(dataset):
    for policy in MatchPolicy:
        m = find(dataset, "d", policy)
        if policy is MatchPolicy.EXPECTED:
            assert m.verdict is Verdict.REJECTED
        else:
            assert m.verdict is Verdict.AMBIGUOUS
        assert not m.ok


def test_expected_folder_trip_level_is_optional(tmp_path):
    """The half-year folder is the first component under raw/, trip or not.

    2020 originally nested its trips directly under raw/ with no half-year
    level; it has since been normalised to 2020.1, but a dataset shaped either
    way must resolve to the same half-year folder.
    """
    root = tmp_path / "raw"
    assert JpgIndex.expected_folder(root / "2019.7" / "trip" / "a.xmp", root) == "2019.7"
    assert JpgIndex.expected_folder(root / "2019.7" / "a.xmp", root) == "2019.7"


def test_expected_folder_none_without_a_folder_component(tmp_path):
    root = tmp_path / "raw"
    assert JpgIndex.expected_folder(root / "a.xmp", root) is None
    assert JpgIndex.expected_folder(tmp_path / "elsewhere" / "a.xmp", root) is None


def test_colliding_stems(dataset):
    idx = JpgIndex(dataset / "jpg")
    assert set(idx.colliding_stems()) == {"d"}


@pytest.mark.parametrize("folder,expected", [
    ("2019.7", "2019"), ("2022", "2022"), ("2020.1", "2020"), ("weird", None),
])
def test_folder_year(folder, expected):
    assert folder_year(folder) == expected
