"""Tests for the audit's multi-claim section -- run from within test/: `pytest tools/`.

`find_multi_claim()` reports what the labeler's JPEG-driven walk does when several
images reach one sidecar. It is a *report*, not a repair: the resolution is already
correct, and the count is watched as a fingerprint of the export's shape.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from audit_dataset import find_multi_claim  # noqa: E402

from code.lib.jpg_index import JpgIndex  # noqa: E402

TRIP = "Photos-19/2019-01-13 crane"


def build(tmp_path, sidecar_stems, jpg_stems):
    for stem in sidecar_stems:
        p = tmp_path / "xmp" / TRIP / f"{stem}.xmp"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("<x/>")
    for stem in jpg_stems:
        p = tmp_path / "jpg" / TRIP / f"{stem}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\xff\xd8\xff")
    xr, jr = tmp_path / "xmp", tmp_path / "jpg"
    return find_multi_claim(JpgIndex(xr, jr), xr, jr)


def test_a_clean_pairing_reports_nothing(tmp_path):
    assert build(tmp_path, ["a", "b"], ["a", "b"]) == []


def test_master_plus_virtual_copy_is_one_group(tmp_path):
    groups = build(tmp_path, ["a"], ["a", "a-2"])
    assert len(groups) == 1
    g = groups[0]
    assert g["path"] == f"{TRIP}/a.xmp"
    assert g["keeps"] == "a" and g["csv_only"] == "a-2"
    assert g["exact_match_wins"] is True


def test_several_copies_all_land_in_csv_only(tmp_path):
    g = build(tmp_path, ["a"], ["a", "a-2", "a-3", "a-Edit"])[0]
    assert g["keeps"] == "a"
    assert sorted(g["csv_only"].split(";")) == ["a-2", "a-3", "a-Edit"]


def test_a_copy_with_its_own_sidecar_is_not_a_claimant(tmp_path):
    """`a-2.xmp` exists, so `a-2.jpg` is its own capture rather than a's copy."""
    assert build(tmp_path, ["a", "a-2"], ["a", "a-2"]) == []


def test_the_anomaly_is_flagged_when_no_exact_export_exists(tmp_path):
    """Only decorated exports -- the master was never exported, so one of them wins."""
    g = build(tmp_path, ["a"], ["a-Enhanced-NR", "a-Enhanced-NR-2"])[0]
    assert g["exact_match_wins"] is False
    assert g["keeps"] == "a-Enhanced-NR"
    assert g["csv_only"] == "a-Enhanced-NR-2"


def test_the_winner_is_never_chosen_by_filename_order(tmp_path):
    """On filenames `a-2.jpg` sorts before `a.jpg`; on stems it does not."""
    g = build(tmp_path, ["a"], ["a", "a-2"])[0]
    assert g["keeps"] == "a", "sorting on the filename would hand it to the copy"


def test_orphan_jpgs_are_not_groups(tmp_path):
    assert build(tmp_path, ["a"], ["a", "IMG_0001"]) == []
