"""Tests for code/lib/export_report.py -- run from within test/: `pytest lib/`."""
import pytest

from code.lib.export_report import NOT_FOUND, ExportReport, parse

REPORT = """This file is a video. (2)
    D:\\Lightroom\\MediaFiles\\Photos\\Photos-20\\2020-10-02 海湾\\clip.mp4
    D:\\Lightroom\\MediaFiles\\Photos\\Photos-25\\2025-06-01 Village\\IMG_1145.mp4
The file could not be found. (3)
    D:\\Lightroom\\MediaFiles\\Photos\\Photos-19\\2019-06-08 山公园\\_D5V0824.nef
    D:\\Lightroom\\MediaFiles\\Photos\\Photos-24\\2024-10-13~19 City\\_D8S2887-Enhanced-NR.dng
    D:\\Lightroom\\MediaFiles\\Photos\\Photos-24\\2024-10-13~19 City\\_D8S2887.nef
"""


@pytest.fixture
def report(tmp_path):
    (tmp_path / "export.report.txt").write_text(REPORT, encoding="utf-8")
    return ExportReport.load(tmp_path)


def test_parses_every_entry_under_its_reason(tmp_path):
    (tmp_path / "r.txt").write_text(REPORT, encoding="utf-8")
    entries = parse(tmp_path / "r.txt")
    assert len(entries) == 5
    assert sum(1 for e in entries if e.reason == NOT_FOUND) == 3
    assert sum(1 for e in entries if e.reason == "This file is a video.") == 2


def test_windows_paths_are_reanchored_at_the_library(report):
    assert report.reason_for("Photos-19/2019-06-08 山公园", "_D5V0824") == NOT_FOUND


def test_render_decorations_are_stripped_to_the_sidecar_stem(report):
    """`X-Enhanced-NR.dng` has no sidecar; its failure explains `X.xmp`."""
    assert report.reason_for("Photos-24/2024-10-13~19 City", "_D8S2887") == NOT_FOUND


def test_unlisted_photo_has_no_reason(report):
    assert report.reason_for("Photos-19/2019-06-08 山公园", "_D5V9999") is None
    assert report.reason_for("Photos-19/other trip", "_D5V0824") is None


def test_missing_report_is_not_an_error(tmp_path):
    empty = ExportReport.load(tmp_path / "nowhere")
    assert empty.entries == []
    assert empty.reason_for("Photos-19/trip", "x") is None


def test_reason_survives_a_count_suffix(report):
    """The header carries a `(N)` tally that is not part of the reason."""
    assert all("(" not in e.reason for e in report.entries)
