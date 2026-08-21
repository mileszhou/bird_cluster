"""Tests for tools/audit_report_safety.py -- run from within test/.

The trip-folder pattern is the one worth pinning: it has to catch
`2019-08-27 Greenwich` while ignoring the dated prose every document in this
repository is written in, and the two are the same shape.
"""
from tools.audit_report_safety import TRIP_RE


def test_catches_a_trip_folder():
    assert TRIP_RE.search("Photos-19/2019-08-27 Greenwich & St Paul Church/x.jpg")


def test_catches_the_range_form():
    assert TRIP_RE.search("Photos-21/2021-05-07~08 Sandwich Bay/x.jpg")


def test_catches_a_chinese_trip_name():
    assert TRIP_RE.search("2016-01-16 茅家埠")


def test_ignores_dated_prose():
    """Every document here dates its own paragraphs; matching those made the
    check fire on all of them, which is how a checker stops being read."""
    for line in ("Raised 2026-08-21 by Miles, while waiting.",
                 "Settled 2026-08-12, after the repo was briefly public.",
                 "Explored 2026-08-11 to -14.",
                 "Given up 2026-08-09.",
                 "Added 2026-08-12: ad-hoc run commands."):
        assert not TRIP_RE.search(line), line


def test_ignores_a_bare_date():
    assert not TRIP_RE.search("on 2026-08-21 the run finished")
