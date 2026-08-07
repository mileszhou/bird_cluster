"""Tests for code/lib/jpg_claim.py -- run from within test/: `pytest lib/`."""
import pytest

from code.lib.jpg_claim import SidecarClaims, sort_key

TRIP = "Photos-19/2019-01-13 crane"


@pytest.fixture
def claims(tmp_path):
    d = tmp_path / TRIP
    d.mkdir(parents=True)
    for stem in ("_D8S0025", "_D8S0026", "_D8S0026-2", "IMG-with-dashes"):
        (d / f"{stem}.xmp").write_text("<x/>")
    return SidecarClaims(tmp_path)


def test_exact_stem_wins(claims):
    assert claims.claim(TRIP, "_D8S0025") == "_D8S0025"


def test_a_decorated_export_claims_its_base(claims):
    assert claims.claim(TRIP, "_D8S0025-Enhanced-NR") == "_D8S0025"
    assert claims.claim(TRIP, "_D8S0025-2") == "_D8S0025"
    assert claims.claim(TRIP, "_D8S0025-Pano") == "_D8S0025"


def test_a_sidecar_of_its_own_beats_the_base(claims):
    """`A-2.xmp` exists, so `A-2.jpg` is its own capture, not A's virtual copy."""
    assert claims.claim(TRIP, "_D8S0026-2") == "_D8S0026-2"


def test_longest_base_wins(claims):
    """Otherwise `IMG-with-dashes-2` would claim `IMG` and label the wrong photo."""
    assert claims.claim(TRIP, "IMG-with-dashes-2") == "IMG-with-dashes"


def test_an_unknown_decoration_needs_no_code_change(claims):
    assert claims.claim(TRIP, "_D8S0025-SomethingNew") == "_D8S0025"


def test_no_sidecar_at_all(claims):
    assert claims.claim(TRIP, "IMG_0001") is None
    assert claims.claim(TRIP, "totally-unrelated") is None


def test_an_unmirrored_folder_claims_nothing(claims):
    assert claims.claim("Photos-24/2024-10-13 santiago", "_D8S0025") is None


def test_the_claim_never_looks_at_another_jpg(tmp_path):
    """Locality: the same JPEG resolves identically whatever else was exported.

    An earlier draft asked whether a plain `X.jpg` existed before letting
    `X-Enhanced-NR.jpg` claim `X.xmp`, which made one photo's handling depend on
    another's. First-claimant-in-stem-order replaces that check.
    """
    d = tmp_path / TRIP
    d.mkdir(parents=True)
    (d / "_D8S0025.xmp").write_text("<x/>")
    bare = SidecarClaims(tmp_path).claim(TRIP, "_D8S0025-Enhanced-NR")
    (tmp_path.parent / "irrelevant.jpg").write_bytes(b"\xff\xd8\xff")
    assert SidecarClaims(tmp_path).claim(TRIP, "_D8S0025-Enhanced-NR") == bare == "_D8S0025"


def test_total_counts_every_sidecar(claims):
    assert claims.total() == 4


# --- ordering ---------------------------------------------------------------

def test_stem_order_puts_the_exact_match_first():
    """On filenames `X-2.jpg` sorts before `X.jpg` ('-' is 0x2D, '.' is 0x2E),
    which would hand the sidecar to the virtual copy. On stems it does not."""
    stems = ["_D8S0025-2", "_D8S0025", "_D8S0025-Enhanced-NR"]
    assert sorted(stems, key=lambda s: sort_key(TRIP, s))[0] == "_D8S0025"
    assert sorted(f"{s}.jpg" for s in stems)[0] == "_D8S0025-2.jpg"  # the trap


def test_folder_separates_repeated_stems():
    a = sort_key("Photos-19/t1", "_D8S0025")
    b = sort_key("Photos-19/t2", "_D8S0025")
    assert a != b and sorted([b, a]) == [a, b]
