"""Tests for code/lib/xmp_write.py -- run from within test/: `pytest lib/`.

The point of this module is that a labelled sidecar stays diffable, so most of
these assert on the *text*, not just on the parsed keywords.
"""
import pytest

from code.lib.xmp_write import (
    HIER_TAG,
    SUBJECT_TAG,
    XmpEditError,
    items_of,
    merge_hierarchical,
    set_subject_keywords,
    verify_only_keywords_changed,
)
import xml.etree.ElementTree as ET

# Lightroom's real layout: one attribute per indented line, one space per level.
# Reserialising this is what turned 348 lines into 120.
LIGHTROOM = """<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 7.0-c000 1.000000, 0000/00/00-00:00:00        ">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:tiff="http://ns.adobe.com/tiff/1.0/"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:lr="http://ns.adobe.com/lightroom/1.0/"
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
   tiff:Make="NIKON CORPORATION"
   tiff:Model="NIKON D610"
   crs:Version="18.4">
{subject}   <crs:Look>
    <rdf:Description
     crs:Name="Adobe Color"/>
   </crs:Look>
{hier}  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""

SUBJECT_BLOCK = """   <dc:subject>
    <rdf:Bag>
     <rdf:li>Family</rdf:li>
     <rdf:li>Miles</rdf:li>
    </rdf:Bag>
   </dc:subject>
"""

HIER_BLOCK = """   <lr:hierarchicalSubject>
    <rdf:Bag>
     <rdf:li>People|Family|Miles</rdf:li>
    </rdf:Bag>
   </lr:hierarchicalSubject>
"""


def sidecar(subject=SUBJECT_BLOCK, hier=HIER_BLOCK):
    return LIGHTROOM.format(subject=subject, hier=hier)


def changed_lines(before, after):
    """The +/- lines `git diff` would show."""
    import difflib
    return [l for l in difflib.unified_diff(before.splitlines(), after.splitlines(), n=0)
            if l[:1] in "+-" and not l.startswith(("+++", "---"))]


# --- the file survives ------------------------------------------------------

def test_everything_outside_the_keywords_is_byte_identical():
    before = sidecar()
    after = set_subject_keywords(before, ["Family", "Miles", "bird"])
    for line in before.splitlines():
        if "rdf:li" in line or "dc:subject" in line:
            continue
        assert line in after.splitlines(), f"lost or reflowed: {line!r}"


def test_lightroom_prefixes_and_attribute_layout_are_untouched():
    """Reserialising renamed x: to ns0: and collapsed the attribute block."""
    after = set_subject_keywords(sidecar(), ["bird"])
    assert "<x:xmpmeta" in after and "ns0:" not in after
    assert '   tiff:Make="NIKON CORPORATION"\n   tiff:Model="NIKON D610"' in after


def test_inserting_a_subject_touches_only_the_inserted_lines():
    """24,389 of 43,728 sidecars have no dc:subject; this is the common path."""
    before = sidecar(subject="")
    after = set_subject_keywords(before, ["bird", "x-鸟-x(90%)"])
    added = changed_lines(before, after)
    assert all(l.startswith("+") for l in added), "an insertion must delete nothing"
    assert len(added) == 6           # open, Bag, two li, /Bag, close
    assert items_of(ET.fromstring(after), SUBJECT_TAG) == ["bird", "x-鸟-x(90%)"]


def test_insertion_anchors_past_nested_descriptions():
    """The first </rdf:Description> usually closes the nested crs:Look one."""
    after = set_subject_keywords(sidecar(subject=""), ["bird"])
    root = ET.fromstring(after)
    outer = root.find(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description")
    assert outer.find(SUBJECT_TAG) is not None
    # the nested Description still carries its own attribute and no keywords
    nested = [d for d in root.iter("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description")
              if d is not outer]
    assert nested and nested[0].find(SUBJECT_TAG) is None


def test_indentation_matches_the_surrounding_file():
    after = set_subject_keywords(sidecar(subject=""), ["bird"])
    assert "   <dc:subject>\n    <rdf:Bag>\n     <rdf:li>bird</rdf:li>" in after


@pytest.mark.parametrize("box", ["Bag", "Seq"])
def test_container_kind_is_reused(box):
    before = sidecar(subject=SUBJECT_BLOCK.replace("Bag", box))
    after = set_subject_keywords(before, ["bird"])
    body = after[after.index("<dc:subject"):after.index("</dc:subject>")]
    assert f"<rdf:{box}" in body
    assert body.count("<rdf:Bag") + body.count("<rdf:Seq") == 1


def test_a_single_line_block_stays_on_one_line():
    before = sidecar(subject="   <dc:subject><rdf:Bag><rdf:li>a</rdf:li></rdf:Bag></dc:subject>\n")
    after = set_subject_keywords(before, ["a", "bird"])
    assert "<dc:subject><rdf:Bag><rdf:li>a</rdf:li><rdf:li>bird</rdf:li></rdf:Bag></dc:subject>" in after


def test_markup_in_a_keyword_is_escaped():
    after = set_subject_keywords(sidecar(), ["Tom & Jerry <b>"])
    assert "Tom &amp; Jerry &lt;b&gt;" in after
    assert items_of(ET.fromstring(after), SUBJECT_TAG) == ["Tom & Jerry <b>"]


def test_dc_is_declared_when_the_sidecar_has_no_binding():
    before = sidecar(subject="", hier="").replace(
        '    xmlns:dc="http://purl.org/dc/elements/1.1/"\n', "")
    after = set_subject_keywords(before, ["bird"])
    assert items_of(ET.fromstring(after), SUBJECT_TAG) == ["bird"]


# --- hierarchical keywords are paths, not a flat mirror ---------------------

def test_user_keyword_paths_survive_a_bird_label():
    """The old writer flattened People|Family|Miles into three loose keywords."""
    hier = merge_hierarchical(["People|Family|Miles"], {"bird"}, ["bird", "x-鸟-x(90%)"])
    assert hier == ["People|Family|Miles", "bird", "x-鸟-x(90%)"]


def test_our_own_flat_entries_are_replaced():
    hier = merge_hierarchical(["Scenes|Flowers", "bird", "old-旧-old(95%)"],
                              {"bird", "old-旧-old(95%)"}, ["bird", "new-新-new(88%)"])
    assert hier == ["Scenes|Flowers", "bird", "new-新-new(88%)"]


def test_a_path_ending_in_one_of_our_keywords_is_ours():
    assert merge_hierarchical(["Wildlife|bird"], {"bird"}, ["bird"]) == ["bird"]


def test_hierarchical_is_rewritten_in_place():
    after = set_subject_keywords(sidecar(), ["Family", "Miles", "bird"],
                                 ["People|Family|Miles", "bird"])
    assert items_of(ET.fromstring(after), HIER_TAG) == ["People|Family|Miles", "bird"]


def test_hierarchical_is_never_created():
    """An absent mirror is not a gap to fill; only a stale one is a problem."""
    after = set_subject_keywords(sidecar(hier=""), ["bird"], ["bird"])
    assert "hierarchicalSubject" not in after


# --- the guard --------------------------------------------------------------

def test_guard_accepts_a_correct_edit():
    before = sidecar()
    after = set_subject_keywords(before, ["Family", "bird"], ["People|Family|Miles", "bird"])
    verify_only_keywords_changed(before, after, ["Family", "bird"],
                                 ["People|Family|Miles", "bird"])


def test_guard_rejects_a_lost_element():
    before = sidecar()
    after = set_subject_keywords(before, ["bird"]).replace(
        '     crs:Name="Adobe Color"/>', "")
    with pytest.raises(XmpEditError):
        verify_only_keywords_changed(before, after, ["bird"])


def test_guard_rejects_a_changed_attribute():
    before = sidecar()
    after = set_subject_keywords(before, ["bird"]).replace("NIKON D610", "NIKON D850")
    with pytest.raises(XmpEditError):
        verify_only_keywords_changed(before, after, ["bird"])


def test_guard_rejects_broken_xml():
    with pytest.raises(XmpEditError):
        verify_only_keywords_changed(sidecar(), "<not xml", ["bird"])


def test_guard_rejects_the_wrong_keywords():
    before = sidecar()
    after = set_subject_keywords(before, ["bird"])
    with pytest.raises(XmpEditError):
        verify_only_keywords_changed(before, after, ["scenery"])
