"""Tests for code/lib/xmp_labels.py -- run from within test/: `pytest lib/`."""
import pytest

from code.lib.xmp_labels import parse_label, read_labels, read_subjects

XMP_TEMPLATE = """<?xml version="1.0"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/">
   <dc:subject><rdf:Seq>{items}</rdf:Seq></dc:subject>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""


def write_xmp(tmp_path, keywords, name="a.xmp"):
    items = "".join(f"<rdf:li>{k}</rdf:li>" for k in keywords)
    path = tmp_path / name
    path.write_text(XMP_TEMPLATE.format(items=items), encoding="utf-8")
    return path


def test_parse_model_label():
    lab = parse_label("hgyl-黑冠夜鹭-black-crowned night heron(95%)")
    assert lab.pinyin == "hgyl"
    assert lab.chinese == "黑冠夜鹭"
    # The English name contains hyphens itself, so only the first two fields split.
    assert lab.english == "black-crowned night heron"
    assert lab.confidence == pytest.approx(0.95)


@pytest.mark.parametrize("keyword", [
    "kq-孔雀-Peacock",       # hand-written species keyword: no confidence
    "bhl-山公园",             # user location tag
    "add=20231014",          # batch marker
    "bird",                  # bare category
    "",
])
def test_non_label_keywords_rejected(keyword):
    """Only the trailing (NN%) makes it a model label."""
    assert parse_label(keyword) is None


def test_english_name_lowercased():
    assert parse_label("x-y-Great Cormorant(98%)").english == "great cormorant"


def test_read_labels_bird(tmp_path):
    p = write_xmp(tmp_path, ["bird", "hgyl-黑冠夜鹭-black-crowned night heron(95%)"])
    got = read_labels(p)
    assert got.is_bird
    assert got.categories == ("bird",)
    assert got.species == "black-crowned night heron"


def test_read_labels_ignores_user_keywords(tmp_path):
    """Pre-existing user tags coexist with the injected ones and must not confuse it."""
    p = write_xmp(tmp_path, ["add=20231014", "bhl-山公园", "bird", "ml-麻雀-house sparrow(98%)"])
    got = read_labels(p)
    assert got.is_bird
    assert got.species == "house sparrow"


def test_read_labels_non_bird(tmp_path):
    p = write_xmp(tmp_path, ["scenery", "x-y-Sunset over Lofoten fjord(98%)"])
    got = read_labels(p)
    assert not got.is_bird
    assert got.categories == ("scenery",)


def test_multiple_categories_preserved(tmp_path):
    p = write_xmp(tmp_path, ["bird", "people", "x-y-z(90%)"])
    assert read_labels(p).categories == ("bird", "people")


def test_unlabelled_sidecar(tmp_path):
    """Well-formed but never labelled: empty subjects, not an error."""
    p = write_xmp(tmp_path, [])
    got = read_labels(p)
    assert got.subjects == () and got.label is None and not got.is_bird


def test_malformed_sidecar_is_none(tmp_path):
    """None (unparseable) must stay distinguishable from [] (unlabelled)."""
    p = tmp_path / "bad.xmp"
    p.write_text("<not xml")
    assert read_subjects(p) is None
    assert read_labels(p) is None


def test_missing_file_is_none(tmp_path):
    assert read_labels(tmp_path / "nope.xmp") is None
