"""Loads non-secret project configuration from config.toml (repo root).

Secrets (API keys) stay in .env, which is gitignored and sourced by shell scripts.
config.toml is checked in and holds everything safe to commit: which host/port
each backend server runs on, and which run directory the analysis tools work on.
"""

import tomllib
from functools import lru_cache
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.toml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def server_url(name: str, scheme: str = "http", path: str = "") -> str:
    """Build a server URL from the [servers.<name>] entry in config.toml."""
    servers = load_config().get("servers", {})
    if name not in servers:
        raise KeyError(f"No [servers.{name}] entry in {CONFIG_PATH}")
    entry = servers[name]
    return f"{scheme}://{entry['host']}:{entry['port']}{path}"


def working_output(default: str = "./output") -> Path:
    """The run directory the analysis tools default to.

    Runs are archived under new names (`./clean` moves `output/` to
    `output_NNN_<description>/`), so a tool that hardcodes `./output` analyses
    whatever happens to be there rather than what was meant. Setting
    `current_working_output` in config.toml re-points every post-processor at
    once -- including at an archive, to re-analyse a run after moving on.

    A `--output-dir` flag still wins; this is only the default.
    """
    return Path(load_config().get("current_working_output", default))
