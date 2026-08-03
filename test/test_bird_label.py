"""Tests for code/bird_label.py's dataset-facing helpers.

Run from within test/: `pytest test_bird_label.py`. Only the offline pieces are
covered -- sidecar identity, library selection and the filter CSV. The model
backends need a live server.
"""
import csv

import pytest

import code.bird_label as bl


@pytest.fixture
def tree(tmp_path):
    """Two libraries, with the same stem reused in three different trips."""
    for library, trips in {
        "Photos-19": ["2019-01-13 crane", "2019-06-08 heron"],
        "Photos-24": ["2024-10-13 santiago"],
    }.items():
        for trip in trips:
            d = tmp_path / library / trip
            d.mkdir(parents=True)
            (d / "_D8S0025.xmp").write_text("<x/>")
    return tmp_path


def test_xmp_key_is_the_path_not_the_basename(tree):
    """The whole point: one basename, three photos, three distinct keys."""
    keys = {bl.xmp_key(p, tree) for p in tree.rglob("*.xmp")}
    assert keys == {
        "Photos-19/2019-01-13 crane/_D8S0025.xmp",
        "Photos-19/2019-06-08 heron/_D8S0025.xmp",
        "Photos-24/2024-10-13 santiago/_D8S0025.xmp",
    }


def test_xmp_key_falls_back_to_the_full_path_when_outside_the_root(tmp_path):
    stray = tmp_path / "elsewhere" / "a.xmp"
    assert bl.xmp_key(stray, tmp_path / "root") == stray.as_posix()


def test_select_libraries_filters_by_year(tree):
    assert [d.name for d in bl.select_libraries(tree, None)] == ["Photos-19", "Photos-24"]
    assert [d.name for d in bl.select_libraries(tree, ["2024"])] == ["Photos-24"]


def test_select_libraries_refuses_a_year_that_matches_nothing(tree):
    """Silently labelling zero photos would look like a finished run."""
    with pytest.raises(SystemExit):
        bl.select_libraries(tree, ["1999"])


def write_csv(path, rows, fieldnames, encoding="utf-8"):
    with open(path, "w", newline="", encoding=encoding) as fh:
        wr = csv.DictWriter(fh, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(rows)


FIELDS = ["path", "filename", "category", "label", "label_cn", "confidence", "note",
          "run_label", "response_json"]


def test_filter_csv_selects_animals_and_low_confidence(tmp_path):
    p = tmp_path / "prior.csv"
    write_csv(p, [
        {"path": "Photos-19/t/a.xmp", "filename": "a.xmp", "category": "bird",
         "confidence": "0.90", "note": "bird (0.90)"},
        {"path": "Photos-19/t/b.xmp", "filename": "b.xmp", "category": "bird",
         "confidence": "0.40", "note": "bird (0.40)"},
        {"path": "Photos-24/t/c.xmp", "filename": "c.xmp", "category": "animal",
         "confidence": "0.95", "note": "animal (0.95)"},
    ], FIELDS)
    assert bl.load_filter_set(p, 0.6) == {"Photos-19/t/b.xmp", "Photos-24/t/c.xmp"}


def test_filter_csv_falls_back_to_parsing_note(tmp_path):
    """Rows written before the category column still classify correctly."""
    p = tmp_path / "prior.csv"
    write_csv(p, [
        {"path": "Photos-24/t/c.xmp", "filename": "c.xmp", "confidence": "0.95",
         "note": "animal (0.95)"},
    ], ["path", "filename", "confidence", "note"])
    assert bl.load_filter_set(p, 0.6) == {"Photos-24/t/c.xmp"}


def test_filter_csv_tolerates_a_spreadsheet_bom(tmp_path):
    """Editing the worklist in Excel adds a BOM; it must not rename `path`."""
    p = tmp_path / "prior.csv"
    write_csv(p, [
        {"path": "Photos-19/t/b.xmp", "filename": "b.xmp", "category": "bird",
         "confidence": "0.40", "note": "bird (0.40)"},
    ], FIELDS, encoding="utf-8-sig")
    assert bl.load_filter_set(p, 0.6) == {"Photos-19/t/b.xmp"}


def test_filter_csv_keys_on_path_so_repeated_basenames_stay_distinct(tmp_path):
    p = tmp_path / "prior.csv"
    write_csv(p, [
        {"path": "Photos-19/t1/a.xmp", "filename": "a.xmp", "category": "bird",
         "confidence": "0.10", "note": "bird (0.10)"},
        {"path": "Photos-24/t2/a.xmp", "filename": "a.xmp", "category": "bird",
         "confidence": "0.99", "note": "bird (0.99)"},
    ], FIELDS)
    assert bl.load_filter_set(p, 0.6) == {"Photos-19/t1/a.xmp"}


def test_filter_csv_without_a_path_column_is_refused(tmp_path):
    """A pre-mirrored-export CSV is basename-keyed and cannot identify a photo."""
    p = tmp_path / "old.csv"
    write_csv(p, [{"filename": "a.xmp", "confidence": "0.10", "note": "bird (0.10)"}],
              ["filename", "confidence", "note"])
    with pytest.raises(SystemExit):
        bl.load_filter_set(p, 0.6)


def test_no_filter_requested(tmp_path):
    assert bl.load_filter_set(None, 0.6) is None
    assert bl.load_filter_set("", 0.6) is None
