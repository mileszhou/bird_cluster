"""Tests for dataset resolution -- run from within test/: `pytest lib/`.

The behaviour under test is mostly about *not* falling through. Two mechanisms
in this project have now failed by quietly resolving to a smaller dataset and
reporting success -- a stale `--years 2019` default, and the signature check
that said "not the master" about the master once its file went missing. A typo
in `data_dir` is the same hazard in a new place, so a declaration is binding.
"""
import pytest

from code.lib import config


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """A repo root with ./data and ./sample_data, and no config.local.toml."""
    for name in ("data", "sample_data", "elsewhere"):
        (tmp_path / name / "jpg").mkdir(parents=True)
    (tmp_path / "config.toml").write_text("", encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(config, "LOCAL_CONFIG_PATH", tmp_path / "config.local.toml")
    monkeypatch.delenv("BIRD_DATA_DIR", raising=False)
    config.load_config.cache_clear()
    yield tmp_path
    config.load_config.cache_clear()


def write(root, path):
    """Set data_dir in the (ignored) local config."""
    (root / "config.local.toml").write_text(f'data_dir = "{path}"\n', encoding="utf-8")
    config.load_config.cache_clear()


def test_prefers_the_submodule_when_nothing_is_declared(fake_repo):
    assert config.data_dir() == fake_repo / "data"


def test_falls_to_the_sample_when_the_submodule_is_absent(fake_repo):
    (fake_repo / "data" / "jpg").rmdir()
    assert config.data_dir() == fake_repo / "sample_data"


def test_an_empty_submodule_directory_is_not_a_dataset(fake_repo):
    """`git submodule update --init` leaves data/ present and empty on failure."""
    (fake_repo / "data" / "jpg").rmdir()
    assert (fake_repo / "data").is_dir()
    assert config.data_dir() == fake_repo / "sample_data"


def test_data_dir_wins_over_both_defaults(fake_repo):
    write(fake_repo, "./elsewhere")
    assert config.data_dir() == fake_repo / "elsewhere"


def test_data_dir_resolves_relative_to_the_repo_not_the_cwd(fake_repo, monkeypatch, tmp_path):
    sub = tmp_path / "somewhere" / "deep"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)
    write(fake_repo, "./elsewhere")
    assert config.data_dir() == fake_repo / "elsewhere"


def test_a_local_file_without_data_dir_is_not_a_declaration(fake_repo):
    (fake_repo / "config.local.toml").write_text(
        '[servers.vllm]\nhost = "gpu-box"\n', encoding="utf-8")
    config.load_config.cache_clear()
    assert config.data_dir() == fake_repo / "data"


def test_a_bad_declaration_stops_the_run(fake_repo):
    """The whole point: never fall through to a different dataset."""
    write(fake_repo, "./typo")
    with pytest.raises(SystemExit, match="no jpg/ directory"):
        config.data_dir()


def test_a_bad_declaration_stops_even_though_defaults_exist(fake_repo):
    write(fake_repo, "/definitely/not/here")
    assert (fake_repo / "data" / "jpg").is_dir()      # a fallback is available
    with pytest.raises(SystemExit):
        config.data_dir()                             # and must not be taken


def test_override_beats_everything(fake_repo):
    write(fake_repo, "./elsewhere")
    assert config.data_dir("/given/on/the/command/line") == \
        __import__("pathlib").Path("/given/on/the/command/line")


def test_env_beats_the_local_config(fake_repo, monkeypatch):
    write(fake_repo, "./elsewhere")
    monkeypatch.setenv("BIRD_DATA_DIR", "/from/the/environment")
    assert str(config.data_dir()) == "/from/the/environment"
