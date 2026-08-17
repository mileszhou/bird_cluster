"""Tests for tools/export_seriated.py -- run from within test/: `pytest tools/`.

Covers `plan_dates`, which decides the whole layout: which images share a date,
how many fit on one, and which carry a colour. Everything the reviewer sees in
Lightroom follows from it, and none of it is checkable by looking at one file.
"""
from datetime import datetime

from tools.export_seriated import (SLOTS_PER_DAY, TAIL_COLORS,
                                   TAIL_SLOTS_PER_DAY, plan_dates)

BASE = datetime(2000, 1, 1)


def dates(plan):
    return [w.date() for _, _, w, _ in plan]


def test_a_kept_cluster_owns_its_date():
    plan, day = plan_dates([("a", [0, 1, 2]), ("b", [3, 4])], [], BASE)
    assert day == 2
    assert len(set(dates(plan))) == 2
    assert all(colour == "" for *_, colour in plan)      # never coloured


def test_a_kept_cluster_larger_than_a_day_spills_onto_the_next():
    members = list(range(SLOTS_PER_DAY + 5))
    plan, day = plan_dates([("a", members)], [], BASE)
    assert day == 2
    assert len(set(dates(plan))) == 2


def test_the_tail_is_capped_at_a_hundred_a_date():
    """A kept cluster is filtered to; the tail is scrolled through, and 1,440
    thumbnails on one date is a scroll nobody finishes."""
    tail = [(str(c), list(range(c * 30, c * 30 + 30))) for c in range(10)]  # 300
    plan, day = plan_dates([], tail, BASE)
    counts = {}
    for _, _, when, _ in plan:
        counts[when.date()] = counts.get(when.date(), 0) + 1
    assert max(counts.values()) == TAIL_SLOTS_PER_DAY
    assert day == 3                                      # 300 / 100


def test_pooled_clusters_alternate_colour():
    tail = [(str(c), [c]) for c in range(5)]
    plan, _ = plan_dates([], tail, BASE)
    colours = [colour for *_, colour in plan]
    assert colours == [TAIL_COLORS[i % 2] for i in range(5)]
    assert all(a != b for a, b in zip(colours, colours[1:]))


def test_a_colour_never_leaks_out_of_the_tail():
    """A colour present at all means 'this date holds several clusters'."""
    plan, _ = plan_dates([("big", [0, 1])], [("s1", [2]), ("s2", [3])], BASE)
    by_cluster = {cid: colour for cid, _, _, colour in plan}
    assert by_cluster["big"] == ""
    assert by_cluster["s1"] and by_cluster["s2"]


def test_the_tail_starts_after_the_kept_clusters():
    plan, day = plan_dates([("a", [0]), ("b", [1])], [("s", [2])], BASE)
    kept = [w for cid, _, w, _ in plan if cid in ("a", "b")]
    pooled = [w for cid, _, w, _ in plan if cid == "s"]
    assert max(kept).date() < min(pooled).date()
    assert day == 3


def test_no_tail_means_no_extra_date():
    plan, day = plan_dates([("a", [0])], [], BASE)
    assert day == 1 and len(plan) == 1
