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


def test_keys_carry_the_folder_not_just_the_basename(tree):
    """The whole point: one basename, three photos, three distinct keys."""
    keys = {bl.WorkItem(p.relative_to(tree).as_posix(), p.name, p, None, False).key
            for p in tree.rglob("*.xmp")}
    assert keys == {
        "Photos-19/2019-01-13 crane/_D8S0025.xmp",
        "Photos-19/2019-06-08 heron/_D8S0025.xmp",
        "Photos-24/2024-10-13 santiago/_D8S0025.xmp",
    }


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


FIELDS = ["jpg", "xmp", "filename", "category", "label", "label_cn", "confidence",
          "note", "prior_category", "prior_label", "applied", "run_label", "response_json"]


def test_filter_csv_selects_animals_and_low_confidence(tmp_path):
    p = tmp_path / "prior.csv"
    write_csv(p, [
        {"jpg": "Photos-19/t/a.jpg", "filename": "a.jpg", "category": "bird",
         "confidence": "0.90", "note": "bird (0.90)"},
        {"jpg": "Photos-19/t/b.jpg", "filename": "b.jpg", "category": "bird",
         "confidence": "0.40", "note": "bird (0.40)"},
        {"jpg": "Photos-24/t/c.jpg", "filename": "c.jpg", "category": "animal",
         "confidence": "0.95", "note": "animal (0.95)"},
    ], FIELDS)
    assert bl.load_filter_set(p, 0.6) == {"Photos-19/t/b.jpg", "Photos-24/t/c.jpg"}


def test_filter_csv_falls_back_to_parsing_note(tmp_path):
    """Rows written before the category column still classify correctly."""
    p = tmp_path / "prior.csv"
    write_csv(p, [
        {"jpg": "Photos-24/t/c.jpg", "filename": "c.jpg", "confidence": "0.95",
         "note": "animal (0.95)"},
    ], ["jpg", "filename", "confidence", "note"])
    assert bl.load_filter_set(p, 0.6) == {"Photos-24/t/c.jpg"}


def test_filter_csv_tolerates_a_spreadsheet_bom(tmp_path):
    """Editing the worklist in Excel adds a BOM; it must not rename `jpg`."""
    p = tmp_path / "prior.csv"
    write_csv(p, [
        {"jpg": "Photos-19/t/b.jpg", "filename": "b.jpg", "category": "bird",
         "confidence": "0.40", "note": "bird (0.40)"},
    ], FIELDS, encoding="utf-8-sig")
    assert bl.load_filter_set(p, 0.6) == {"Photos-19/t/b.jpg"}


def test_filter_csv_keys_on_jpg_so_repeated_basenames_stay_distinct(tmp_path):
    p = tmp_path / "prior.csv"
    write_csv(p, [
        {"jpg": "Photos-19/t1/a.jpg", "filename": "a.jpg", "category": "bird",
         "confidence": "0.10", "note": "bird (0.10)"},
        {"jpg": "Photos-24/t2/a.jpg", "filename": "a.jpg", "category": "bird",
         "confidence": "0.99", "note": "bird (0.99)"},
    ], FIELDS)
    assert bl.load_filter_set(p, 0.6) == {"Photos-19/t1/a.jpg"}


def test_filter_csv_without_a_jpg_column_is_refused(tmp_path):
    """A sidecar-keyed CSV names rows this JPEG-driven run never enumerates."""
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


def test_a_non_bird_result_now_replaces_an_existing_bird(tmp_path):
    """The guard was given up: it held 554 rows and was wrong more often than right.

    66 were provably a same-stem twin's label from the basename-keyed era, and
    where both runs disagreed on the subject the newer call was the better one on
    inspection. The cost is penguins and the odd owl, accepted deliberately --
    the embedding step is expected to recover penguins as their own cluster.
    """
    p = sidecar(tmp_path, ["bird", "old-旧-old bird(95%)"])
    action, _ = bl.set_keywords_in_xmp(p, "animal", "a beetle(80%)")
    assert action == bl.APPLIED_WRITTEN
    assert subjects(p) == ("animal", "a beetle(80%)")


def test_the_bird_guard_can_be_restored(tmp_path):
    p = sidecar(tmp_path, ["bird", "old-旧-old bird(95%)"])
    action, _ = bl.set_keywords_in_xmp(p, "animal", "a beetle(80%)", protect_bird=True)
    assert action == bl.APPLIED_KEPT
    assert subjects(p) == ("bird", "old-旧-old bird(95%)")


def test_giving_up_the_bird_guard_leaves_other_categories_deferring(tmp_path):
    """Only the bird half was dropped; scenery still defers to the finer early text."""
    p = sidecar(tmp_path, ["mountain landscape with glacier", "scenery"])
    action, _ = bl.set_keywords_in_xmp(p, "animal", "a beetle(80%)")
    assert action == bl.APPLIED_KEPT


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


def test_user_keyword_paths_are_not_flattened(tmp_path):
    """`People|Family|Miles` is one keyword naming a place in Lightroom's tree.

    The previous writer mirrored the flat dc:subject list into
    hierarchicalSubject, so it came back as three unrelated top-level keywords.
    71 sidecars in the library carry such paths.
    """
    p = tmp_path / "a.xmp"
    p.write_text(XMP.format(items="<rdf:li>Miles</rdf:li>", box="Bag")
                 .replace("<lr:hierarchicalSubject><rdf:Bag><rdf:li>Miles</rdf:li>",
                          "<lr:hierarchicalSubject><rdf:Bag><rdf:li>People|Family|Miles</rdf:li>"),
                 encoding="utf-8")
    bl.set_keywords_in_xmp(p, "bird", "new-新-new bird(88%)")
    text = p.read_text(encoding="utf-8")
    hierarchical = text[text.index("hierarchicalSubject"):]
    assert "People|Family|Miles" in hierarchical
    assert "new-新-new bird(88%)" in hierarchical


def test_the_rest_of_the_file_is_left_alone(tmp_path):
    """These sidecars are rsynced into a Lightroom library; the diff must be readable."""
    p = sidecar(tmp_path, ["bhl-山公园"])
    before = p.read_text(encoding="utf-8")
    bl.set_keywords_in_xmp(p, "bird", "new-新-new bird(88%)")
    after = p.read_text(encoding="utf-8")
    for line in before.splitlines():
        if "rdf:li" in line or "subject" in line.lower():
            continue
        assert line in after.splitlines()
    assert "ns0:" not in after and "<x:xmpmeta" in after


def test_unparseable_sidecar_reports_failure(tmp_path):
    p = tmp_path / "bad.xmp"
    p.write_text("<not xml")
    action, removed = bl.set_keywords_in_xmp(p, "bird", "x-y-z(90%)")
    assert action == bl.APPLIED_FAILED and removed == ()


# --- work items: one per JPEG, sidecar as destination -----------------------

TRIP = "Photos-19/2019-01-13 crane"


@pytest.fixture
def dataset(tmp_path):
    """A paired capture, a decorated-only capture, a virtual copy, and orphans."""
    from code.lib.jpg_claim import SidecarClaims
    (tmp_path / "xmp" / TRIP).mkdir(parents=True)
    (tmp_path / "jpg" / TRIP).mkdir(parents=True)
    for stem in ("paired", "denoised", "copied", "unexported"):
        sidecar(tmp_path / "xmp" / TRIP, ["bird"], name=f"{stem}.xmp")
    for stem in ("paired",                  # plain export
                 "denoised-Enhanced-NR",    # only export of its capture
                 "copied", "copied-2",      # master plus a virtual copy
                 "IMG_0001", "_D5C1940"):   # never had a raw
        (tmp_path / "jpg" / TRIP / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff")
    return tmp_path, SidecarClaims(tmp_path / "xmp")


def build(root, claims, years=None):
    """(items, taken) -- the in-scope sidecar set is checked separately."""
    items, taken, _ = bl.build_items(claims, root / "xmp", root / "jpg", years)
    return items, taken


def test_one_item_per_jpg(dataset):
    root, claims = dataset
    items, _ = build(root, claims)
    assert sorted(i.name for i in items) == [
        "IMG_0001.jpg", "_D5C1940.jpg", "copied-2.jpg", "copied.jpg",
        "denoised-Enhanced-NR.jpg", "paired.jpg"]
    assert all(i.jpg is not None for i in items)


def test_key_is_the_jpg_path(dataset):
    root, claims = dataset
    items, _ = build(root, claims)
    keys = [i.key for i in items]
    assert len(set(keys)) == len(keys)
    assert f"{TRIP}/paired.jpg" in keys


def test_a_decorated_only_export_still_reaches_its_sidecar(dataset):
    """Without this, 189 captures whose only export is -Enhanced-NR go unlabelled."""
    root, claims = dataset
    items, _ = build(root, claims)
    it = next(i for i in items if i.name == "denoised-Enhanced-NR.jpg")
    assert it.xmp is not None and it.xmp.name == "denoised.xmp"


def test_the_exact_match_wins_the_sidecar_not_the_virtual_copy(dataset):
    """Stem order puts `copied` before `copied-2`, so first-claimant is correct."""
    root, claims = dataset
    items, _ = build(root, claims)
    master = next(i for i in items if i.name == "copied.jpg")
    copy = next(i for i in items if i.name == "copied-2.jpg")
    assert master.xmp.name == "copied.xmp" and master.owns_xmp
    # The copy still *names* its capture -- that is what lets a post-hoc dedup
    # group every export of one frame -- but it does not write it.
    assert copy.xmp.name == "copied.xmp"
    assert not copy.owns_xmp, "an alternate edit must not overwrite the capture's label"


def test_orphans_are_ordinary_items_with_no_sidecar(dataset):
    root, claims = dataset
    items, _ = build(root, claims)
    orphans = [i for i in items if i.name in ("IMG_0001.jpg", "_D5C1940.jpg")]
    assert len(orphans) == 2
    assert all(i.xmp is None and not i.owns_xmp for i in orphans)


def test_an_orphan_and_an_alternate_edit_are_distinguishable(dataset):
    """Both are csv-only, but only the orphan has no capture at all.

    Collapsing them would lose the link from `copied-2.jpg` back to the frame it
    is an edit of -- 313 alternate edits against 5,229 photos that never had a raw.
    """
    root, claims = dataset
    items, _ = build(root, claims)
    orphan = next(i for i in items if i.name == "IMG_0001.jpg")
    edit = next(i for i in items if i.name == "copied-2.jpg")
    assert not orphan.owns_xmp and not edit.owns_xmp      # both are csv-only
    assert orphan.xmp is None                             # ...but tell them apart
    assert edit.xmp is not None


def test_a_sidecar_no_jpg_reaches_is_reported_not_silently_dropped(dataset):
    """The blind spot of walking the image tree -- it must be countable."""
    root, claims = dataset
    _, taken = build(root, claims)
    assert claims.total() - len(taken) == 1        # unexported.xmp


def test_out_of_scope_sidecars_are_not_counted_as_unreached(dataset):
    """Otherwise every scoped run screams about the years it was told to skip.

    `./run-vllm --include-from` over two libraries warned about 31,612 sidecars
    that were simply not in scope, which trains the reader to ignore the warning
    that matters.
    """
    from code.lib.path_filter import PathFilter
    root, claims = dataset
    only_paired = PathFilter(include=[f"{TRIP}/paired.jpg"])
    _, taken, expected = bl.build_items(claims, root / "xmp", root / "jpg",
                                        None, only_paired)
    assert expected == {(TRIP, "paired")}       # not all four sidecars
    assert expected - taken == set()            # and it was reached


def test_claims_are_stable_across_a_resumed_run(dataset):
    """Assignment happens over the whole tree, before the checkpoint filters."""
    root, claims = dataset
    first, _ = build(root, claims)
    second, _ = build(root, SidecarClaimsAgain(root))
    assert [(i.key, i.xmp) for i in first] == [(i.key, i.xmp) for i in second]


def SidecarClaimsAgain(root):
    from code.lib.jpg_claim import SidecarClaims
    return SidecarClaims(root / "xmp")


def test_exclude_from_narrows_the_walk(dataset):
    """Same manifests work for labelling and embedding -- both key on data/jpg."""
    from code.lib.path_filter import PathFilter
    root, claims = dataset
    items, _, _ = bl.build_items(claims, root / "xmp", root / "jpg", None,
                              PathFilter(exclude=[TRIP]))
    assert items == []
    items, _, _ = bl.build_items(claims, root / "xmp", root / "jpg", None,
                              PathFilter(exclude=[f"{TRIP}/paired.jpg"]))
    assert all(i.name != "paired.jpg" for i in items)
    assert any(i.name == "copied.jpg" for i in items)


def test_include_from_restricts_the_walk(dataset):
    from code.lib.path_filter import PathFilter
    root, claims = dataset
    items, _, _ = bl.build_items(claims, root / "xmp", root / "jpg", None,
                              PathFilter(include=[f"{TRIP}/paired.jpg"]))
    assert [i.name for i in items] == ["paired.jpg"]


def test_year_filter(dataset):
    root, claims = dataset
    assert build(root, claims, ["2019"])[0]
    with pytest.raises(SystemExit):
        build(root, claims, ["2024"])
