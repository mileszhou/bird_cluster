"""Tests for code/lib/path_filter.py -- run from within test/: `pytest lib/`."""
import pytest

from code.lib.path_filter import PathFilter, build, read_paths

ZOO = "Photos-16/2016-07-12 City Zoo"
KEY = f"{ZOO}/_D5S1234.jpg"


def test_no_lists_allows_everything():
    f = PathFilter()
    assert not f
    assert f.allows(KEY)


def test_a_folder_line_takes_its_subtree():
    assert not PathFilter(exclude=[ZOO]).allows(KEY)


def test_a_trailing_slash_is_the_same_folder():
    assert not PathFilter(exclude=[ZOO + "/"]).allows(KEY)


def test_an_exact_file_line():
    f = PathFilter(exclude=[KEY])
    assert not f.allows(KEY)
    assert f.allows(f"{ZOO}/_D5S9999.jpg")


def test_matching_is_on_path_segments_not_characters():
    """`Photos-2` must not swallow `Photos-24`, which a prefix test would."""
    f = PathFilter(exclude=["Photos-2"])
    assert f.allows("Photos-24/trip/a.jpg")
    assert f.allows("Photos-21/trip/a.jpg")


def test_a_line_may_be_any_depth():
    """`a/b/c` is a folder line like any other -- data/jpg is two deep today, but
    the source library has nested trip folders, so depth is not assumable."""
    nested = "Photos-24/2024-05-01 Gold Coast/2024-04-19 Melboune Birds"
    key = f"{nested}/_D5D0372.jpg"
    assert not PathFilter(exclude=[nested]).allows(key)
    # ...and a parent line takes the nested child with it
    assert not PathFilter(exclude=["Photos-24/2024-05-01 Gold Coast"]).allows(key)


def test_include_restricts_to_the_listed_subtrees():
    f = PathFilter(include=["Photos-24", "Photos-25/trip b"])
    assert f.allows("Photos-24/anything/a.jpg")
    assert f.allows("Photos-25/trip b/a.jpg")
    assert not f.allows("Photos-25/trip c/a.jpg")
    assert not f.allows("Photos-16/trip/a.jpg")


def test_exclude_wins_over_include():
    """Carving a known-bad subset out of a wanted range is the usual shape."""
    f = PathFilter(include=["Photos-16"], exclude=[ZOO])
    assert f.allows("Photos-16/other trip/a.jpg")
    assert not f.allows(KEY)


# --- the file format --------------------------------------------------------

def test_read_paths_drops_comments_and_blanks(tmp_path):
    p = tmp_path / "list.txt"
    p.write_text("# a comment\n\n"
                 f"{ZOO}   # trailing comment\n"
                 "Photos-24/trip/\n", encoding="utf-8")
    assert read_paths(p) == [ZOO, "Photos-24/trip"]


def test_read_paths_tolerates_a_spreadsheet_bom(tmp_path):
    """These get edited in Excel, and trip folders are Chinese."""
    p = tmp_path / "list.txt"
    p.write_text("Photos-19/2019-01-13 山公园\n", encoding="utf-8-sig")
    assert read_paths(p) == ["Photos-19/2019-01-13 山公园"]


def test_read_paths_normalises_windows_separators(tmp_path):
    p = tmp_path / "list.txt"
    p.write_text("Photos-24\\trip b\n", encoding="utf-8")
    assert read_paths(p) == ["Photos-24/trip b"]


@pytest.fixture
def manifest_dir(tmp_path, monkeypatch):
    """Point the manifest directory at a temp one; callers pass bare names."""
    import code.lib.path_filter as pf
    monkeypatch.setattr(pf, "MANIFEST_DIR", tmp_path)
    return tmp_path


def test_build_from_files(manifest_dir):
    inc, exc = manifest_dir / "i.txt", manifest_dir / "e.txt"
    inc.write_text("Photos-16\n", encoding="utf-8")
    exc.write_text(f"{ZOO}\n", encoding="utf-8")
    f = build("i.txt", "e.txt")
    assert f.allows("Photos-16/other/a.jpg")
    assert not f.allows(KEY)
    d = f.describe()
    assert "include 1 path(s)" in d and "exclude 1 path(s)" in d
    # naming the file matters: a count alone looks the same for the wrong manifest
    assert str(inc) in d and str(exc) in d


def test_build_with_nothing_is_falsy():
    assert not build(None, None)


def test_a_missing_list_is_fatal(tmp_path, manifest_dir):
    """Never "no filter": an include list exists to narrow, so ignoring a typo'd
    one runs the whole library instead of the subset asked for."""
    with pytest.raises(FileNotFoundError):
        read_paths(tmp_path / "typo.t")
    with pytest.raises(FileNotFoundError):
        build("typo.t", None)


def test_an_all_comments_list_is_fatal(tmp_path):
    """Parsed, it is indistinguishable from no list at all."""
    p = tmp_path / "list.txt"
    p.write_text("# everything commented out\n\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_paths(p)


# --- manifests live in manifests/, wherever the caller starts from ----------

def test_a_bare_name_is_manifest_relative():
    from code.lib.path_filter import MANIFEST_DIR, resolve
    assert resolve("no-zoos.txt") == (MANIFEST_DIR / "no-zoos.txt").resolve()


def test_a_prefixed_path_is_repo_relative():
    """The form tab-completion produces, so what you type is what is on disk."""
    from code.lib.path_filter import MANIFEST_DIR, resolve
    assert resolve("manifests/no-zoos.txt") == (MANIFEST_DIR / "no-zoos.txt").resolve()


def test_subdirectories_work_either_way():
    from code.lib.path_filter import MANIFEST_DIR, resolve
    want = (MANIFEST_DIR / "captive" / "zoos.txt").resolve()
    assert resolve("captive/zoos.txt") == want
    assert resolve("manifests/captive/zoos.txt") == want


def test_an_absolute_path_inside_the_dir_is_accepted():
    from code.lib.path_filter import MANIFEST_DIR, resolve
    p = MANIFEST_DIR / "ok.txt"
    assert resolve(str(p)) == p.resolve()


@pytest.mark.parametrize("bad", [
    "manifests/../data/label/args.json",   # the leading segment says manifests...
    "../secret.txt",
    "/tmp/scratch.txt",
])
def test_anything_landing_outside_is_refused(bad):
    """Checked on the *resolved* location, not the first path segment.

    An earlier version compared the leading segment, so `manifests/../` walked
    straight out while looking compliant.
    """
    from code.lib.path_filter import resolve
    with pytest.raises(ValueError):
        resolve(bad)


# --- two manifest roots -----------------------------------------------------
# `manifests/` is versioned and means the same to anyone; `local/manifests/` is
# gitignored and holds lists that name real places. Both are inside the repo,
# which is what keeps a run's recorded scope findable.

def _roots(tmp_path, monkeypatch):
    from code.lib import path_filter as pf
    versioned = tmp_path / "manifests"
    local = tmp_path / "local" / "manifests"
    versioned.mkdir(parents=True)
    local.mkdir(parents=True)
    monkeypatch.setattr(pf, "MANIFEST_DIR", versioned)
    monkeypatch.setattr(pf, "LOCAL_MANIFEST_DIR", local)
    return pf, versioned, local


def test_bare_name_falls_back_to_the_local_root(tmp_path, monkeypatch):
    pf, _, local = _roots(tmp_path, monkeypatch)
    (local / "trips.txt").write_text("Photos-16/x\n", encoding="utf-8")
    assert pf.resolve("trips.txt") == (local / "trips.txt").resolve()


def test_bare_name_prefers_the_versioned_root(tmp_path, monkeypatch):
    pf, versioned, local = _roots(tmp_path, monkeypatch)
    for d in (versioned, local):
        (d / "both.txt").write_text("Photos-16/x\n", encoding="utf-8")
    assert pf.resolve("both.txt") == (versioned / "both.txt").resolve()


def test_repo_relative_form_works_for_both(tmp_path, monkeypatch):
    pf, versioned, local = _roots(tmp_path, monkeypatch)
    (versioned / "a.txt").write_text("x\n", encoding="utf-8")
    (local / "b.txt").write_text("x\n", encoding="utf-8")
    assert pf.resolve("manifests/a.txt") == (versioned / "a.txt").resolve()
    assert pf.resolve("local/manifests/b.txt") == (local / "b.txt").resolve()


@pytest.mark.parametrize("bad", ["/tmp/x.txt", "manifests/../data/x", "local/../../x"])
def test_escapes_are_still_refused(tmp_path, monkeypatch, bad):
    pf, _, _ = _roots(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="manifests/ or local/manifests/"):
        pf.resolve(bad)
