"""Tests for code/lib/sidecar_meta.py -- run from within test/: `pytest lib/`."""
import pytest

from code.lib.sidecar_meta import capture_key, normalise_datetime, read_meta
from code.lib.trips import frame_id

XMP = """<?xml version="1.0"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description
    xmlns:exif="http://ns.adobe.com/exif/1.0/"
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
    xmlns:xmpMM="http://ns.adobe.com/xap/1.0/mm/"
    exif:DateTimeOriginal="{dto}"
    xmp:CreateDate="{dto}"
    crs:RawFileName="{raw}"
    xmpMM:DocumentID="{did}"
    xmpMM:OriginalDocumentID="{odid}"/>
 </rdf:RDF>
</x:xmpmeta>
"""


def write(tmp_path, name, dto="2019-05-08T17:37:30", odid="ODID1", did="DID1", raw="x.nef"):
    path = tmp_path / name
    path.write_text(XMP.format(dto=dto, odid=odid, did=did, raw=raw), encoding="utf-8")
    return path


@pytest.mark.parametrize("value,expected", [
    ("2018-03-24T12:30:58", "2018-03-24T12:30:58"),
    ("2018-03-24T12:30:58.82", "2018-03-24T12:30:58"),      # sub-seconds dropped
    ("2019-11-22T17:42:30+08:00", "2019-11-22T17:42:30"),   # zone dropped
    ("2019-05-08 17:37:30", "2019-05-08T17:37:30"),         # space separator
    ("", ""),
])
def test_normalise_datetime(value, expected):
    assert normalise_datetime(value) == expected


def test_subsecond_precision_does_not_split_a_capture(tmp_path):
    """~10% of sidecars record sub-seconds; the same photo must still match."""
    a = read_meta(write(tmp_path, "a.xmp", dto="2018-03-24T12:30:58.82", odid=""))
    b = read_meta(write(tmp_path, "b.xmp", dto="2018-03-24T12:30:58", odid=""))
    assert a.captured == b.captured
    assert capture_key(a, frame_id("_D5V3035")) == capture_key(b, frame_id("_D5V3035"))


def test_original_document_id_is_preferred_over_time(tmp_path):
    """A timezone correction changes the recorded day but not the capture."""
    a = read_meta(write(tmp_path, "a.xmp", dto="2019-11-22T17:42:30+08:00", odid="SAME"))
    b = read_meta(write(tmp_path, "b.xmp", dto="2019-11-23T00:42:30+08:00", odid="SAME"))
    assert a.captured != b.captured
    assert capture_key(a, frame_id("_B230814")) == capture_key(b, frame_id("_B230814"))


def test_falls_back_to_time_and_frame_without_odid(tmp_path):
    meta = read_meta(write(tmp_path, "a.xmp", odid="", did=""))
    key = capture_key(meta, frame_id("_D5D9339"))
    assert key[0] == "time+frame"


def test_different_captures_do_not_collide(tmp_path):
    a = read_meta(write(tmp_path, "a.xmp", dto="2019-05-08T17:37:30", odid="A"))
    b = read_meta(write(tmp_path, "b.xmp", dto="2019-05-08T17:37:31", odid="B"))
    assert capture_key(a, frame_id("_D5D9339")) != capture_key(b, frame_id("_D5D9340"))


def test_virtual_copy_detected_from_stem(tmp_path):
    assert read_meta(write(tmp_path, "20181229-_D8S7789-2.xmp")).is_virtual_copy
    assert not read_meta(write(tmp_path, "20181229-_D8S7789.xmp")).is_virtual_copy


def test_import_renamed_copy_matches_its_twin(tmp_path):
    """A clash on import prefixes one copy only; both are still one capture."""
    a = read_meta(write(tmp_path, "_D5C5710.xmp", odid=""))
    b = read_meta(write(tmp_path, "20190328-_D5C5710.xmp", odid=""))
    assert capture_key(a, frame_id(a and "_D5C5710")) == \
           capture_key(b, frame_id("20190328-_D5C5710"))


def test_fields_are_read(tmp_path):
    meta = read_meta(write(tmp_path, "a.xmp", raw="20190508-_D5D9339.nef", odid="O", did="D"))
    assert meta.raw_file_name == "20190508-_D5D9339.nef"
    assert meta.original_document_id == "O"
    assert meta.document_id == "D"
    assert meta.capture_date == "2019-05-08"


def test_malformed_sidecar_is_none(tmp_path):
    bad = tmp_path / "bad.xmp"
    bad.write_text("<not xml")
    assert read_meta(bad) is None
    assert read_meta(tmp_path / "missing.xmp") is None


def test_capture_key_none_without_any_identity(tmp_path):
    path = tmp_path / "empty.xmp"
    path.write_text('<?xml version="1.0"?><x:xmpmeta xmlns:x="adobe:ns:meta/"/>')
    meta = read_meta(path)
    assert capture_key(meta, None) is None
