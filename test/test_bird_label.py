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


FIELDS = ["path", "source", "filename", "category", "label", "label_cn", "confidence",
          "note", "prior_category", "prior_label", "applied", "run_label", "response_json"]


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


# --- keyword writing: bird wins, anything else defers -----------------------

XMP = """<?xml version="1.0"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/"
                   xmlns:lr="http://ns.adobe.com/lightroom/1.0/">
   <dc:subject><rdf:{box}>{items}</rdf:{box}></dc:subject>
   <lr:hierarchicalSubject><rdf:Bag>{items}</rdf:Bag></lr:hierarchicalSubject>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""


def sidecar(tmp_path, keywords, box="Bag", name="a.xmp"):
    items = "".join(f"<rdf:li>{k}</rdf:li>" for k in keywords)
    p = tmp_path / name
    p.write_text(XMP.format(items=items, box=box), encoding="utf-8")
    return p


def subjects(path):
    from code.lib.xmp_labels import read_subjects
    return tuple(read_subjects(path))


def test_bird_overwrites_an_existing_bird_label(tmp_path):
    p = sidecar(tmp_path, ["bird", "old-旧-old bird(95%)"])
    action, removed = bl.set_keywords_in_xmp(p, "bird", "new-新-new bird(88%)")
    assert action == bl.APPLIED_WRITTEN
    assert subjects(p) == ("bird", "new-新-new bird(88%)")
    assert removed == ("bird", "old-旧-old bird(95%)")


def test_bird_overwrites_a_non_bird_category(tmp_path):
    """Promotion to bird is the one judgement this pipeline is trusted on."""
    p = sidecar(tmp_path, ["mountain landscape with glacier", "scenery"])
    action, _ = bl.set_keywords_in_xmp(p, "bird", "new-新-new bird(88%)")
    assert action == bl.APPLIED_WRITTEN
    assert subjects(p) == ("bird", "new-新-new bird(88%)")


def test_non_bird_defers_to_an_existing_category(tmp_path):
    """The existing category is finer-grained; a fresh `scenery` loses detail."""
    p = sidecar(tmp_path, ["mountain landscape with glacier", "scenery"])
    action, _ = bl.set_keywords_in_xmp(p, "scenery", "a hillside(70%)")
    assert action == bl.APPLIED_KEPT
    assert subjects(p) == ("mountain landscape with glacier", "scenery")


def test_non_bird_never_demotes_an_existing_bird(tmp_path):
    """Follows from the rule, and guards the clustering set against a false negative."""
    p = sidecar(tmp_path, ["bird", "old-旧-old bird(95%)"])
    action, _ = bl.set_keywords_in_xmp(p, "animal", "a beetle(80%)")
    assert action == bl.APPLIED_KEPT
    assert subjects(p) == ("bird", "old-旧-old bird(95%)")


def test_non_bird_writes_when_nothing_is_there(tmp_path):
    """Otherwise the 24,918 unlabelled sidecars would never get a category."""
    p = sidecar(tmp_path, [])
    action, _ = bl.set_keywords_in_xmp(p, "scenery", "a hillside(70%)")
    assert action == bl.APPLIED_WRITTEN
    assert subjects(p) == ("scenery", "a hillside(70%)")


def test_non_bird_writes_over_user_keywords_without_a_category(tmp_path):
    """A hand-written keyword is not a category, so it does not block the write."""
    p = sidecar(tmp_path, ["xs-小隼-Kestrel"])
    action, _ = bl.set_keywords_in_xmp(p, "scenery", "a hillside(70%)")
    assert action == bl.APPLIED_WRITTEN
    assert subjects(p) == ("xs-小隼-Kestrel", "scenery", "a hillside(70%)")


def test_user_keywords_survive_and_stay_first(tmp_path):
    p = sidecar(tmp_path, ["bhl-山公园", "bird", "old-旧-old bird(95%)"])
    bl.set_keywords_in_xmp(p, "bird", "new-新-new bird(88%)")
    assert subjects(p) == ("bhl-山公园", "bird", "new-新-new bird(88%)")


def test_hierarchical_subject_is_written_in_step(tmp_path):
    """Left stale, Lightroom resurrects the old keywords from it on import."""
    p = sidecar(tmp_path, ["bird", "old-旧-old bird(95%)"])
    bl.set_keywords_in_xmp(p, "bird", "new-新-new bird(88%)")
    text = p.read_text(encoding="utf-8")
    hierarchical = text[text.index("hierarchicalSubject"):]
    assert "new-新-new bird(88%)" in hierarchical
    assert "old-旧-old bird(95%)" not in hierarchical


@pytest.mark.parametrize("box", ["Bag", "Seq"])
def test_existing_container_kind_is_reused(tmp_path, box):
    """Adding a second container makes the keyword list reader-dependent."""
    p = sidecar(tmp_path, ["bird", "old-旧-old bird(95%)"], box=box)
    bl.set_keywords_in_xmp(p, "bird", "new-新-new bird(88%)")
    text = p.read_text(encoding="utf-8")
    body = text[text.index("<dc:subject"):text.index("</dc:subject>")]
    assert body.count("<rdf:Bag") + body.count("<rdf:Seq") == 1
    assert f"<rdf:{box}" in body


def test_writing_is_idempotent(tmp_path):
    """A resumed run re-processes sidecars it already wrote; they must not stack."""
    p = sidecar(tmp_path, ["bhl-山公园"])
    for _ in range(3):
        bl.set_keywords_in_xmp(p, "bird", "new-新-new bird(88%)")
    assert subjects(p) == ("bhl-山公园", "bird", "new-新-new bird(88%)")


def test_unparseable_sidecar_reports_failure(tmp_path):
    p = tmp_path / "bad.xmp"
    p.write_text("<not xml")
    action, removed = bl.set_keywords_in_xmp(p, "bird", "x-y-z(90%)")
    assert action == bl.APPLIED_FAILED and removed == ()


# --- work items: sidecars, plus JPEGs that never had one --------------------

@pytest.fixture
def dataset(tmp_path):
    """A sidecar with a jpg, a sidecar without, and two jpgs with no sidecar."""
    from code.lib.jpg_index import JpgIndex
    trip = "Photos-19/2019-01-13 crane"
    (tmp_path / "xmp" / trip).mkdir(parents=True)
    (tmp_path / "jpg" / trip).mkdir(parents=True)
    sidecar(tmp_path / "xmp" / trip, ["bird", "old-旧-old bird(95%)"], name="paired.xmp")
    sidecar(tmp_path / "xmp" / trip, [], name="unexported.xmp")
    for stem in ("paired", "IMG_0001", "_D5C1940"):
        (tmp_path / "jpg" / trip / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff")
    return tmp_path, JpgIndex(tmp_path / "xmp", tmp_path / "jpg")


def test_orphan_jpgs_are_excluded_by_default(dataset):
    root, index = dataset
    items = bl.build_items(index, root / "xmp", root / "jpg", None, False)
    assert {i.source for i in items} == {bl.SOURCE_XMP}
    assert sorted(i.name for i in items) == ["paired.xmp", "unexported.xmp"]


def test_orphan_jpgs_are_included_on_request(dataset):
    root, index = dataset
    items = bl.build_items(index, root / "xmp", root / "jpg", None, True)
    orphans = [i for i in items if i.source == bl.SOURCE_JPG_ONLY]
    assert sorted(i.name for i in orphans) == ["IMG_0001.jpg", "_D5C1940.jpg"]
    assert all(i.xmp is None and i.jpg is not None for i in orphans)


def test_orphan_key_is_the_jpg_path_and_cannot_collide_with_a_sidecar(dataset):
    """Both trees are keyed the same way; the extension keeps them distinct."""
    root, index = dataset
    items = bl.build_items(index, root / "xmp", root / "jpg", None, True)
    keys = [i.key for i in items]
    assert len(set(keys)) == len(keys)
    assert "Photos-19/2019-01-13 crane/paired.xmp" in keys
    assert "Photos-19/2019-01-13 crane/IMG_0001.jpg" in keys


def test_sidecar_without_an_export_still_becomes_an_item(dataset):
    """It gets a `missing JPEG` CSV row rather than being dropped silently."""
    root, index = dataset
    items = bl.build_items(index, root / "xmp", root / "jpg", None, True)
    unexported = next(i for i in items if i.name == "unexported.xmp")
    assert unexported.jpg is None and unexported.xmp is not None


def test_year_filter_applies_to_orphans_too(dataset):
    root, index = dataset
    assert bl.build_items(index, root / "xmp", root / "jpg", ["2019"], True)
    with pytest.raises(SystemExit):
        bl.build_items(index, root / "xmp", root / "jpg", ["2024"], True)
