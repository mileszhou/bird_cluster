"""Tests for code/lib/xmp_labels.py -- run from within test/: `pytest lib/`."""
import pytest

from code.lib.xmp_labels import parse_label, read_labels, read_subjects, split_keywords

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
    "sgy-山公园",             # user location tag
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
    p = write_xmp(tmp_path, ["add=20231014", "sgy-山公园", "bird", "ml-麻雀-house sparrow(98%)"])
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


# --- split_keywords: which entries did this pipeline write? -----------------
#
# Three generations of injected labels exist in the dataset, and the earliest
# is indistinguishable from an ordinary keyword by shape alone. The category
# keyword is the discriminator: only this pipeline writes one.

def test_current_generation_is_ours(tmp_path):
    ours, theirs = split_keywords(["bird", "dhbo-大黑背鸥-great black-backed gull(95%)"])
    assert ours == ("bird", "dhbo-大黑背鸥-great black-backed gull(95%)")
    assert theirs == ()


def test_early_free_text_is_ours_when_a_category_is_present(tmp_path):
    """`mountain landscape with glacier` carries no confidence suffix."""
    ours, theirs = split_keywords(["mountain landscape with glacier", "scenery"])
    assert ours == ("mountain landscape with glacier", "scenery")
    assert theirs == ()


def test_hand_written_species_keywords_are_never_touched(tmp_path):
    """The user's own py-cn-en keywords carry no confidence and no category."""
    subjects = ["gycl- Andean Motmot-高原翠鴗", "add=20231014", "Rivertown"]
    ours, theirs = split_keywords(subjects)
    assert ours == ()
    assert theirs == tuple(subjects)


def test_a_stray_label_is_ours_even_without_a_category(tmp_path):
    """The `(NN%)` shape is self-identifying wherever it appears."""
    ours, theirs = split_keywords(["Rivertown", "person playing tennis(98%)", "_nb"])
    assert ours == ("person playing tennis(98%)", "_nb")
    assert theirs == ("Rivertown",)


def test_a_category_claims_descriptive_phrases_only(tmp_path):
    """A category marks the sidecar as ours, but single tokens stay the user's.

    `People` sits beside the user's own `Family`/`Miles` tags in this dataset,
    so the category alone must not be enough to claim them.
    """
    ours, theirs = split_keywords(["Family", "Miles", "a person on a beach", "People"])
    assert theirs == ("Family", "Miles")
    assert ours == ("a person on a beach", "People")


def test_hand_written_names_survive_a_labelled_sidecar(tmp_path):
    """After a full re-label every sidecar has a category; this must still hold."""
    ours, theirs = split_keywords(["pp-昵称", "xs-小隼-Kestrel", "bird", "x-y-z(90%)"])
    assert theirs == ("pp-昵称", "xs-小隼-Kestrel")
    assert ours == ("bird", "x-y-z(90%)")


def test_ordering_is_preserved(tmp_path):
    ours, theirs = split_keywords(["a-b-c", "bird(90%)", "x-y-z", "q-r-s(10%)"])
    assert ours == ("bird(90%)", "q-r-s(10%)")
    assert theirs == ("a-b-c", "x-y-z")


def test_empty_is_empty(tmp_path):
    assert split_keywords([]) == ((), ())
