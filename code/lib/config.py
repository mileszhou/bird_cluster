"""Loads non-secret project configuration from config.toml (repo root).

Secrets (API keys) stay in .env, which is gitignored and sourced by shell scripts.
config.toml is checked in and holds everything safe to commit — currently just
which host/port each backend server runs on.
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
