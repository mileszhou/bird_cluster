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
    "gy-公园",             # user location tag
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
    p = write_xmp(tmp_path, ["add=20231014", "gy-公园", "bird", "ml-麻雀-house sparrow(98%)"])
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


# --- the browsing export's rank keywords ------------------------------------
#
# Added after `--labels-only` was found to double them. `ord:`/`fam:`/`gen:` are
# a fourth generation of pipeline keyword, and a generation split_keywords does
# not recognise is preserved as the user's and then written again, so every
# re-caption grows the list. The failure is silent and only visible in a photo
# manager's keyword panel.

def test_rank_keywords_are_ours():
    ours, theirs = split_keywords(
        ["ord:Passeriformes", "fam:Monarchidae", "gen:Terpsiphone",
         "gen:Terpsiphone paradisi"])
    assert theirs == ()
    assert len(ours) == 4


def test_rank_rule_does_not_claim_the_users_keywords():
    """Narrow on purpose: a prefix, then one capitalised Linnaean name.

    A user keyword that merely contains a colon must survive. These are the
    shapes that would be lost if the pattern were loosened to "has a colon".
    """
    subjects = ["notes: my own thing", "genus: whatever", "my:tag", "Family",
                "fam:", "ord:lowercase", "xs-小隼-Kestrel"]
    ours, theirs = split_keywords(subjects)
    assert ours == ()
    assert set(theirs) == set(subjects)


def test_recaptioning_is_idempotent_for_ranks():
    """What the doubling bug actually looked like, as a partition.

    Re-writing keywords replaces `ours` and keeps `theirs`, so a rank keyword
    landing in `theirs` is one that comes back beside its own replacement.
    """
    written = ["hzww-黑枕王鹟-black-naped monarch(98%)",
               "ord:Passeriformes", "fam:Monarchidae", "gen:Terpsiphone"]
    hand = "sd-寿带-Asian Paradise-flycatcher"
    ours, theirs = split_keywords([hand] + written)
    assert theirs == (hand,), "a hand-written keyword must never be claimed"
    assert set(ours) == set(written), "every pipeline keyword must be replaced"


def test_prediction_keywords_are_ours():
    """`bc:` is BioCLIP's own call, written beside the label-derived ranks.

    Every prefix the pipeline writes must be claimed here. One that is missing
    is preserved as the user's and then written again, so each re-caption
    doubles it -- the bug the rank rule above was added for, which recurs for
    each new prefix unless the rule grows with it.
    """
    written = ["bc-ord:Passeriformes", "bc-fam:Monarchidae", "bc-gen:Terpsiphone",
               "bc:Indian paradise flycatcher", "bc-conf:low"]
    ours, theirs = split_keywords(written)
    assert theirs == ()
    assert len(ours) == len(written)


def test_prediction_rule_spares_lookalike_user_keywords():
    subjects = ["bc:", "bc-conf:whatever", "abc:something", "my:tag"]
    ours, theirs = split_keywords(subjects)
    assert ours == ()
    assert set(theirs) == set(subjects)


def test_species_keywords_from_both_sources_are_ours():
    """`sp:` is the labeller's species, `bc:` BioCLIP's, written side by side.

    The review compares them, so both must be plain filterable keywords -- and
    both must be claimed here, or a re-caption doubles them.
    """
    written = ["sp:black-naped monarch", "sp-sci:Hypothymis azurea",
               "bc:Indian paradise flycatcher", "bc-sci:Terpsiphone paradisi"]
    ours, theirs = split_keywords(written)
    assert theirs == ()
    assert len(ours) == len(written)
    # and a hand-written keyword still survives beside them
    ours, theirs = split_keywords(written + ["sd-寿带-Asian Paradise-flycatcher"])
    assert theirs == ("sd-寿带-Asian Paradise-flycatcher",)


# --- source tags in place of the confidence ---------------------------------

def test_tag_form_labels_are_ours():
    """`(Q)` marks a keyword as this pipeline's, exactly as `(98%)` used to."""
    ours, theirs = split_keywords(
        ["ptcn-普通翠鸟-common kingfisher(Q)", "ptcn-普通翠鸟-common kingfisher(G)",
         "ptcn-普通翠鸟-common kingfisher(99%)", "scenery(Q)"])
    assert theirs == ()
    assert len(ours) == 4


def test_tag_rule_spares_parenthesised_user_keywords():
    """The tag is uppercase and short so ordinary parentheses survive.

    This is the rule that could destroy hand-written work if loosened: a user
    keyword ending in `(juvenile)` or `(nest)` must never be claimed.
    """
    subjects = ["Kestrel (juvenile)", "my note (nest)", "xs-小隼-Kestrel", "(Q)x"]
    ours, theirs = split_keywords(subjects)
    assert ours == ()
    assert set(theirs) == set(subjects)


def test_parse_label_reads_both_forms():
    tagged = parse_label("ptcn-普通翠鸟-common kingfisher(Q)")
    assert tagged.english == "common kingfisher"
    assert tagged.confidence is None, "a tag carries no confidence, and must not fake one"
    old = parse_label("ptcn-普通翠鸟-common kingfisher(99%)")
    assert old.english == "common kingfisher" and old.confidence == 0.99
