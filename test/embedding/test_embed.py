"""Tests for code/embedding/embed.py -- run from within test/: `pytest embedding/`.

Covers the offline half: the scan/filter/resolve pass and the resumption
checkpoint. The HTTP half needs a live embed server and is not exercised here.
"""
import json

import pytest

from code.embedding.embed import already_done, collect, library_dirs

XMP = """<?xml version="1.0"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/">
   <dc:subject><rdf:Seq>{items}</rdf:Seq></dc:subject>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""


def sidecar(data_dir, library, trip, stem, keywords):
    d = data_dir / "xmp" / library / trip
    d.mkdir(parents=True, exist_ok=True)
    items = "".join(f"<rdf:li>{k}</rdf:li>" for k in keywords)
    (d / f"{stem}.xmp").write_text(XMP.format(items=items), encoding="utf-8")


def jpeg(data_dir, library, trip, stem):
    d = data_dir / "jpg" / library / trip
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff")


@pytest.fixture
def data_dir(tmp_path):
    """One bird with a jpg, one bird without, one non-bird, one other year."""
    d = tmp_path / "data"
    sidecar(d, "Photos-19", "trip a", "bird_ok", ["bird", "hj-黑鹤-black stork(98%)"])
    jpeg(d, "Photos-19", "trip a", "bird_ok")
    sidecar(d, "Photos-19", "trip a", "bird_nojpg", ["bird", "ml-麻雀-house sparrow(95%)"])
    sidecar(d, "Photos-19", "trip a", "scenic", ["scenery", "x-y-A hill(98%)"])
    jpeg(d, "Photos-19", "trip a", "scenic")
    sidecar(d, "Photos-21", "trip b", "bird_2021", ["bird", "hj-黑鹤-black stork(90%)"])
    jpeg(d, "Photos-21", "trip b", "bird_2021")
    return d


def test_library_dirs_filters(data_dir):
    assert [y for y, _ in library_dirs(data_dir, None)] == ["2019", "2021"]
    assert [y for y, _ in library_dirs(data_dir, ["2019"])] == ["2019"]
    assert library_dirs(data_dir, ["1999"]) == []


def test_collect_keeps_only_resolvable_birds(data_dir):
    cands, stats, per_year = collect(data_dir, ["2019"], 0.0)
    assert stats["xmp"] == 3          # scenery is scanned...
    assert stats["bird"] == 2         # ...but not counted as a bird
    assert [c.stem for c in cands] == ["bird_ok"]
    assert stats["unresolved"] == 1
    assert per_year == {"2019": 1}


def test_collect_records_provenance(data_dir):
    cands, _, _ = collect(data_dir, ["2019"], 0.0)
    c = cands[0]
    assert c.key == "Photos-19/trip a/bird_ok.xmp"
    assert (c.year, c.library, c.trip) == ("2019", "Photos-19", "trip a")
    assert c.species == "black stork"
    assert c.confidence == pytest.approx(0.98)
    assert c.jpg.name == "bird_ok.jpg"


def test_collect_year_filter_spans_years(data_dir):
    cands, _, per_year = collect(data_dir, None, 0.0)
    assert per_year == {"2019": 1, "2021": 1}
    # Keys carry the library and trip, so identical stems cannot collide.
    assert len({c.key for c in cands}) == 2


def test_min_confidence_drops_low_labels(data_dir):
    cands, stats, _ = collect(data_dir, None, 0.95)
    assert [c.stem for c in cands] == ["bird_ok"]
    assert stats["below_min_confidence"] == 1


def test_already_done_empty(tmp_path):
    assert already_done(tmp_path / "nope.jsonl") == (set(), set())


def test_already_done_reads_keys_and_models(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text(
        json.dumps({"key": "a", "model": "m1"}) + "\n"
        + json.dumps({"key": "b", "model": "m1"}) + "\n"
    )
    assert already_done(p) == ({"a", "b"}, {"m1"})


def test_already_done_flags_mixed_and_unrecorded_models(tmp_path):
    """A row with no model predates the guard and must not pass as compatible."""
    p = tmp_path / "e.jsonl"
    p.write_text(
        json.dumps({"key": "a", "model": "m1"}) + "\n"
        + json.dumps({"key": "b"}) + "\n"
    )
    keys, models = already_done(p)
    assert keys == {"a", "b"}
    assert models == {"m1", "(unrecorded)"}


def test_already_done_tolerates_partial_last_line(tmp_path):
    """A run killed mid-write leaves a truncated line; resumption must survive it."""
    p = tmp_path / "e.jsonl"
    p.write_text(json.dumps({"key": "a", "model": "m1"}) + "\n" + '{"key": "b", "embed')
    keys, models = already_done(p)
    assert keys == {"a"} and models == {"m1"}
