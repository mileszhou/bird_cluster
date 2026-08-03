"""Tests for code/lib/jpg_index.py -- run from within test/: `pytest lib/`."""
import pytest

from code.lib.jpg_index import ExtraKind, JpgIndex, Verdict, library_year


def build(tmp_path, xmp: dict, jpg: dict):
    """Write a miniature mirrored dataset: {folder: [stems]} for each tree."""
    for root, spec, ext in (("xmp", xmp, ".xmp"), ("jpg", jpg, ".jpg")):
        for folder, stems in spec.items():
            d = tmp_path / root / folder
            d.mkdir(parents=True, exist_ok=True)
            for stem in stems:
                (d / f"{stem}{ext}").write_bytes(b"x")
    return JpgIndex(tmp_path / "xmp", tmp_path / "jpg")


TRIP = "Photos-19/2019-01-13 trip"


@pytest.fixture
def index(tmp_path):
    """Every resolution outcome in one dataset.

    a  exported under its own name
    b  only a virtual copy was exported          -> derived
    c  only the AI Denoise render was exported   -> derived
    d  in an exported folder, but not exported   -> no_jpg
    e  its whole folder was never exported       -> no_folder
    """
    return build(
        tmp_path,
        xmp={TRIP: ["a", "b", "c", "d"], "Photos-19/2019-02-01 unexported": ["e"]},
        jpg={TRIP: ["a", "b-2", "c-Enhanced-NR", "phone-shot"],
             "Photos-19/2019-03-01 no sidecars": ["z"]},
    )


def resolve(index, folder, stem):
    return index.resolve(index.xmp_dir / folder / f"{stem}.xmp")


def test_exact_stem_in_the_mirrored_folder(index):
    m = resolve(index, TRIP, "a")
    assert m.verdict is Verdict.OK
    assert m.ok and m.path.name == "a.jpg"


def test_virtual_copy_is_accepted_as_derived(index):
    m = resolve(index, TRIP, "b")
    assert m.verdict is Verdict.DERIVED
    assert m.ok and m.path.name == "b-2.jpg"


def test_enhanced_nr_render_is_accepted_as_derived(index):
    """The denoise render is a separate DNG with no sidecar of its own."""
    m = resolve(index, TRIP, "c")
    assert m.verdict is Verdict.DERIVED
    assert m.ok and m.path.name == "c-Enhanced-NR.jpg"


def test_folder_exported_but_photo_was_not(index):
    m = resolve(index, TRIP, "d")
    assert m.verdict is Verdict.NO_JPG
    assert not m.ok and m.path is None


def test_folder_never_exported(index):
    m = resolve(index, "Photos-19/2019-02-01 unexported", "e")
    assert m.verdict is Verdict.NO_FOLDER
    assert not m.ok


def test_unexported_folders(index):
    assert index.unexported_folders() == ["Photos-19/2019-02-01 unexported"]


def test_extras_split_orphans_from_virtual_copies(tmp_path):
    """A decorated sibling of an already-matched sidecar is not an unclaimed file."""
    index = build(tmp_path, xmp={TRIP: ["a"]}, jpg={TRIP: ["a", "a-2", "phone"]})
    extras = {e.path.name: e for e in index.extras()}
    assert extras["a-2.jpg"].kind is ExtraKind.DERIVED
    assert extras["a-2.jpg"].parent_stem == "a"
    assert extras["phone.jpg"].kind is ExtraKind.ORPHAN
    assert extras["phone.jpg"].parent_stem == ""
    assert resolve(index, TRIP, "a").paths[0].name == "a.jpg"


def test_folder_with_no_sidecars_is_all_orphans(index):
    orphans = [e for e in index.extras() if e.folder == "Photos-19/2019-03-01 no sidecars"]
    assert [e.path.name for e in orphans] == ["z.jpg"]
    assert all(e.kind is ExtraKind.ORPHAN for e in orphans)
    assert index.orphan_folders() == ["Photos-19/2019-03-01 no sidecars"]


def test_exact_match_claims_a_jpg_before_a_prefix_match(tmp_path):
    """With sidecars `a` and `a-2`, `a-2.jpg` is a-2's export, not a's copy."""
    index = build(tmp_path, xmp={TRIP: ["a", "a-2"]}, jpg={TRIP: ["a-2", "a-3"]})
    assert resolve(index, TRIP, "a-2").path.name == "a-2.jpg"
    assert resolve(index, TRIP, "a-2").verdict is Verdict.OK
    assert resolve(index, TRIP, "a").path.name == "a-3.jpg"
    assert resolve(index, TRIP, "a").verdict is Verdict.DERIVED


def test_longest_sidecar_stem_wins_a_contested_derived_jpg(tmp_path):
    index = build(tmp_path, xmp={TRIP: ["a", "a-2"]}, jpg={TRIP: ["a", "a-2-3"]})
    assert resolve(index, TRIP, "a-2").path.name == "a-2-3.jpg"
    assert resolve(index, TRIP, "a").path.name == "a.jpg"


def test_several_virtual_copies_all_attach_to_the_sidecar(tmp_path):
    index = build(tmp_path, xmp={TRIP: ["a"]}, jpg={TRIP: ["a-2", "a-3"]})
    m = resolve(index, TRIP, "a")
    assert m.verdict is Verdict.DERIVED
    assert [p.name for p in m.paths] == ["a-2.jpg", "a-3.jpg"]


def test_identical_stems_in_different_trips_do_not_interfere(tmp_path):
    """Counter wraparound is harmless now the folder is part of the key."""
    other = "Photos-24/2024-05-05 elsewhere"
    index = build(tmp_path, xmp={TRIP: ["a"], other: ["a"]},
                  jpg={TRIP: ["a"], other: ["a"]})
    assert resolve(index, TRIP, "a").path.parent.name.endswith("trip")
    assert resolve(index, other, "a").path.parent.name.endswith("elsewhere")


def test_sidecars_lists_every_sidecar(index):
    assert sorted(p.stem for p in index.sidecars()) == ["a", "b", "c", "d", "e"]


@pytest.mark.parametrize("library,expected", [
    ("Photos-19", "2019"), ("Photos-25", "2025"), ("Photos-2019", None), ("jpg", None),
])
def test_library_year(library, expected):
    assert library_year(library) == expected
