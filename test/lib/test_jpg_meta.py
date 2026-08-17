"""Tests for code/lib/jpg_meta.py -- run from within test/: `pytest lib/`.

Covers the colour label, which is the one edit that is not a keyword and so is
not protected by `verify_only_keywords_changed`. The JPEG segment surgery is
exercised end to end by the export itself against real files; what is worth
pinning here is the text substitution, because a regex that matches too much
would quietly corrupt an XMP packet.
"""
import pytest

from code.lib.jpg_meta import set_color_label
from code.lib.xmp_write import XmpEditError

PACKET = ('<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
          ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
          '  <rdf:Description rdf:about=""\n'
          '    xmlns:xmp="http://ns.adobe.com/xap/1.0/"\n'
          '   xmp:Rating="2"\n'
          '   {label}>\n'
          '  </rdf:Description>\n'
          ' </rdf:RDF>\n'
          '</x:xmpmeta>')


def test_replaces_an_existing_attribute():
    """The common case: 516 of one export's 10,982 files already carry one."""
    before = PACKET.format(label='xmp:Label="Safari"')
    after = set_color_label(before, "Red")
    assert 'xmp:Label="Red"' in after
    assert "Safari" not in after
    assert after == before.replace("Safari", "Red")   # nothing else moved


def test_inserts_when_absent():
    before = PACKET.format(label='xmp:CreatorTool="x"')
    after = set_color_label(before, "Blue")
    assert 'xmp:Label="Blue"' in after
    assert 'xmp:Rating="2"' in after                  # untouched
    assert after.replace(' xmp:Label="Blue"', "", 1) == before


def test_handles_the_element_form():
    before = "<rdf:Description><xmp:Label>Yellow</xmp:Label></rdf:Description>"
    assert set_color_label(before, "Red") == \
        "<rdf:Description><xmp:Label>Red</xmp:Label></rdf:Description>"


def test_only_the_first_description_gains_one():
    """Packets nest Descriptions inside crs:Look and friends; one label only."""
    before = ('<rdf:Description rdf:about="a"></rdf:Description>'
              '<rdf:Description rdf:about="b"></rdf:Description>')
    after = set_color_label(before, "Red")
    assert after.count("xmp:Label") == 1


def test_rating_is_not_mistaken_for_a_label():
    """`xmp:Label` must not match `xmp:Labels` or share a prefix with Rating."""
    before = '<rdf:Description xmp:Rating="5" xmp:LabelSet="none"></rdf:Description>'
    after = set_color_label(before, "Red")
    assert 'xmp:Rating="5"' in after and 'xmp:LabelSet="none"' in after
    assert 'xmp:Label="Red"' in after


def test_refuses_a_packet_with_nowhere_to_put_it():
    with pytest.raises(XmpEditError):
        set_color_label("<x:xmpmeta></x:xmpmeta>", "Red")
