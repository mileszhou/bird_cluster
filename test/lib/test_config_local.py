"""config.local.toml layered over config.toml -- run from within test/.

The tracked file must hold values that work for a stranger. Host names were the
counter-example: `darwin` and `spark` are the author's own boxes, so a public
clone either failed to resolve them or -- worse on a network where those names
exist -- reached somebody else's machine. And the only fix was editing a tracked
file, which leaves the tree dirty for as long as the setup is right.
"""
import textwrap

import pytest

from code.lib import config


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """Point config at a tmp pair and clear the lru_cache around each test."""
    tracked = tmp_path / "config.toml"
    local = tmp_path / "config.local.toml"
    tracked.write_text(textwrap.dedent("""
        current_working_output = "./output"
        [servers.vllm]
        host = "localhost"
        port = 8000
        [servers.embed]
        host = "localhost"
        port = 9100
    """), encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_PATH", tracked)
    monkeypatch.setattr(config, "LOCAL_CONFIG_PATH", local)
    config.load_config.cache_clear()
    yield local
    config.load_config.cache_clear()


def test_a_stranger_with_no_local_file_gets_localhost(cfg):
    assert config.server_url("vllm") == "http://localhost:8000"
    assert config.server_url("embed") == "http://localhost:9100"


def test_local_file_overrides_the_host(cfg):
    cfg.write_text('[servers.vllm]\nhost = "gpu-box"\n', encoding="utf-8")
    config.load_config.cache_clear()
    assert config.server_url("vllm") == "http://gpu-box:8000"


def test_overriding_host_alone_keeps_the_tracked_port(cfg):
    """The merge is recursive; a local file need not restate a whole table."""
    cfg.write_text('[servers.embed]\nhost = "gpu-box"\n', encoding="utf-8")
    config.load_config.cache_clear()
    assert config.server_url("embed") == "http://gpu-box:9100"


def test_untouched_sections_are_unaffected(cfg):
    cfg.write_text('[servers.vllm]\nhost = "gpu-box"\n', encoding="utf-8")
    config.load_config.cache_clear()
    assert config.server_url("embed") == "http://localhost:9100"


def test_local_file_can_override_a_top_level_key(cfg):
    cfg.write_text('current_working_output = "./output_003"\n', encoding="utf-8")
    config.load_config.cache_clear()
    assert str(config.working_output()) == "output_003"


def test_a_missing_local_file_is_not_an_error(cfg):
    assert not cfg.exists()
    assert config.load_config()["servers"]["vllm"]["host"] == "localhost"


def test_unknown_server_still_raises(cfg):
    with pytest.raises(KeyError):
        config.server_url("nope")
