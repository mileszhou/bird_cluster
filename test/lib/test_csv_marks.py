"""Tests for code/lib/csv_marks.py -- run from within test/: `pytest lib/`."""
import pytest

from code.lib.csv_marks import (BY_BIT, REGISTRY, apply_mark, column_for,
                                describe, missing, read_state)


def test_no_marker_reads_as_nothing():
    assert read_state(["a", "b"]) == (0, None)
    assert describe(0) == "nothing"


def test_state_is_hex_in_the_column_name():
    assert read_state(["a", "C_3"]) == (3, "C_3")
    assert read_state(["a", "C_1F"]) == (31, "C_1F")
    assert column_for(31) == "C_1F"


def test_a_column_that_merely_looks_similar_is_not_a_marker():
    assert read_state(["C_", "C_zz", "cluster_id"]) == (0, None)


def test_missing_prerequisites_are_named():
    tool = REGISTRY["order_assignments"]
    assert [t.name for t in missing(0, tool)] == ["add_cluster_size"]
    assert missing(REGISTRY["add_cluster_size"].bit, tool) == []


def test_applying_replaces_the_column_and_sets_the_bit():
    rows = [{"a": "1", "C_1": ""}, {"a": "2", "C_1": ""}]
    rows, col = apply_mark(rows, 1, "C_1", REGISTRY["order_assignments"])
    assert col == "C_3"
    assert all("C_1" not in r and r["C_3"] == "" for r in rows)


def test_bits_are_unique():
    """A reused bit would make old files claim a tool ran that never did."""
    bits = [t.bit for t in REGISTRY.values()]
    assert len(bits) == len(set(bits)) == len(BY_BIT)


def test_unknown_bits_are_reported_not_ignored():
    """A file written by a newer checkout should say so."""
    assert "unknown bits" in describe(0xF0)
