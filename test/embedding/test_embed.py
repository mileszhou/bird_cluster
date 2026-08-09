"""Tests for code/embedding/embed.py -- run from within test/: `pytest embedding/`.

Covers the offline half: the CSV-driven scan and the resumption checkpoint. The
HTTP half needs a live embed server and is not exercised here.
"""
import csv
import json

import pytest

from code.embedding.embed import (already_done, collect, effective_category,
                                  effective_species)

COLUMNS = ["jpg", "xmp", "filename", "category", "label", "label_cn", "confidence",
           "note", "prior_category", "prior_label", "applied", "run_label",
           "response_json"]


def jpeg(data_dir, rel):
    p = data_dir / "jpg" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\xff\xd8\xff")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})


def row(jpg, category="bird", label="black stork", conf="0.98", **kw):
    base = {"jpg": jpg, "xmp": jpg.replace(".jpg", ".xmp"), "category": category,
            "label": label, "confidence": conf, "applied": "written"}
    base.update(kw)
    return base


@pytest.fixture
def dataset(tmp_path):
    """A bird, a non-bird, a bird in another year, a sidecar-less bird, and a
    row naming an image that is not on disk."""
    d = tmp_path / "data"
    rows = [
        row("Photos-19/trip a/bird_ok.jpg"),
        row("Photos-19/trip a/scenic.jpg", category="scenery", label="a hill"),
        row("Photos-21/trip b/bird_2021.jpg", conf="0.90"),
        # never had a sidecar -- invisible to any sidecar walk
        row("Photos-19/trip a/IMG_0001.jpg", xmp="", applied="csv-only",
            label="house sparrow", conf="0.95"),
        row("Photos-19/trip a/gone.jpg"),
    ]
    for r in rows:
        if r["jpg"] != "Photos-19/trip a/gone.jpg":
            jpeg(d, r["jpg"])
    csv_path = tmp_path / "label" / "bird_identification_output.csv"
    write_csv(csv_path, rows)
    return d, csv_path


# --- the effective category/species rule ------------------------------------

def test_effective_category_is_this_runs_verdict_normally():
    assert effective_category({"category": "Bird", "applied": "written"}) == "bird"


def test_effective_category_defers_to_the_kept_one():
    """`bird` is never demoted, so the CSV's `category` is not what the library holds."""
    r = {"category": "animal", "applied": "kept-existing", "prior_category": "bird"}
    assert effective_category(r) == "bird"


def test_effective_species_follows_the_same_overruling():
    """Otherwise a never-demoted row contributes a scene description as a species."""
    r = {"category": "scenery", "label": "cherry blossoms in full bloom",
         "confidence": "0.98", "applied": "kept-existing", "prior_category": "bird",
         "prior_label": "bqsy-\u767d\u79cb\u6c99\u9e2d-smew(95%)"}
    species, conf = effective_species(r)
    assert species == "smew"
    assert conf == pytest.approx(0.95)   # the discarded call's 0.98 must not leak


def test_effective_species_survives_an_unparseable_prior():
    r = {"category": "scenery", "label": "a hill", "applied": "kept-existing",
         "prior_category": "bird", "prior_label": "bird"}
    assert effective_species(r) == (None, None)


# --- the scan ---------------------------------------------------------------

def test_collect_keeps_the_effective_birds(dataset):
    d, csv_path = dataset
    cands, stats, per_year = collect(d, csv_path, ["2019"], 0.0)
    assert stats["rows"] == 5
    assert stats["in_scope"] == 4            # the 2021 row is out of scope
    assert stats["bird"] == 3                # scenery excluded
    assert sorted(c.stem for c in cands) == ["IMG_0001", "bird_ok"]
    assert per_year == {"2019": 2}


def test_a_sidecarless_image_is_collected(dataset):
    """The whole point of the CSV guide: no sidecar walk could reach this row."""
    d, csv_path = dataset
    cands, _, _ = collect(d, csv_path, ["2019"], 0.0)
    orphan = next(c for c in cands if c.stem == "IMG_0001")
    assert orphan.xmp == ""
    assert orphan.species == "house sparrow"


def test_key_is_the_image_path_not_the_sidecar(dataset):
    d, csv_path = dataset
    cands, _, _ = collect(d, csv_path, ["2019"], 0.0)
    c = next(c for c in cands if c.stem == "bird_ok")
    assert c.key == "Photos-19/trip a/bird_ok.jpg"
    assert (c.year, c.library, c.trip) == ("2019", "Photos-19", "trip a")
    assert c.species == "black stork"
    assert c.confidence == pytest.approx(0.98)
    assert c.jpg.name == "bird_ok.jpg"


def test_a_row_naming_a_missing_image_is_counted_not_embedded(dataset):
    d, csv_path = dataset
    cands, stats, _ = collect(d, csv_path, ["2019"], 0.0)
    assert stats["missing_image"] == 1
    assert all(c.stem != "gone" for c in cands)


def test_year_filter_spans_years(dataset):
    d, csv_path = dataset
    cands, _, per_year = collect(d, csv_path, None, 0.0)
    assert per_year == {"2019": 2, "2021": 1}
    assert len({c.key for c in cands}) == 3


def test_min_confidence_drops_low_labels(dataset):
    d, csv_path = dataset
    cands, stats, _ = collect(d, csv_path, None, 0.95)
    assert stats["below_min_confidence"] == 1        # the 2021 row at 0.90
    assert all(c.stem != "bird_2021" for c in cands)


def test_a_sidecar_keyed_csv_is_refused(tmp_path):
    """A pre-inversion CSV names sidecars, which this step never enumerates."""
    p = tmp_path / "old.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "category", "label"])
        w.writeheader()
        w.writerow({"path": "Photos-19/t/a.xmp", "category": "bird", "label": "x"})
    with pytest.raises(SystemExit):
        collect(tmp_path, p, None, 0.0)


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
