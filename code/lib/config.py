"""Loads non-secret project configuration from config.toml (repo root).

Secrets (API keys) stay in .env, which is gitignored and sourced by shell scripts.
config.toml is checked in and holds everything safe to commit: which host/port
each backend server runs on, and which run directory the analysis tools work on.
"""

import os
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


def data_dir(override=None) -> Path:
    """The dataset to work on, resolved in order of specificity.

        1. an explicit --data-dir
        2. $BIRD_DATA_DIR
        3. ./data, the private submodule
        4. config.toml's `data_dir`
        5. ./sample_data, shipped

    **Every candidate is tested the same way: does it hold a `jpg/` tree.**
    There was once a signature check on `./data` -- a hash of a secret living
    only inside the private dataset -- so that a run recording `data_dir: ./data`
    made a claim that could be verified. It is gone, on 2026-08-11, and
    `docs/design/dataset-resolution.md` records why. Briefly: it checked
    *identity* while every failure that actually happens is one of *integrity*
    (a half-finished rsync, a dataset from before the dedup), there is no
    adversary since substituting a dataset is a supported feature, and the
    gitlink already stops a stray `data/` from being committed -- `git add -f`
    refuses to descend into an uninitialised submodule path, which `.gitignore`
    does not.

    It also failed open in the worst direction. With `signature.txt` absent the
    check said "not the master", resolution fell through `config.toml` to
    `./sample_data`, and a 109-image sample shadowed a 49,270-image library
    while reporting success. That is the failure shape this project keeps
    meeting: a stage that does no useful work and cannot tell.

    A `jpg/` tree, not `is_dir()`. `git submodule update --init` against an
    unreachable host **fails but leaves `data/` present and empty**, so an
    existence test would send every fresh clone into an empty directory instead
    of the sample. Found by cloning and trying it.

    Nobody edits a versioned file to get the right answer: on a working checkout
    `./data` wins and the `config.toml` line is inert whatever it says; on a
    clone that line is what points at the user's own library. $BIRD_DATA_DIR
    covers a library outside the repo entirely -- a path, not a secret, so it
    does not belong in .env.
    """
    if override:
        return Path(override)
    env = os.environ.get("BIRD_DATA_DIR")
    if env:
        return Path(env)
    root = CONFIG_PATH.parent
    configured = load_config().get("data_dir")
    for candidate in ["./data"] + ([configured] if configured else []) + ["./sample_data"]:
        p = Path(candidate)
        p = p if p.is_absolute() else root / p
        if (p / "jpg").is_dir():
            return p
    raise SystemExit(
        "error: no dataset found. Expected ./data (the private submodule -- "
        "`git submodule update --init`) or ./sample_data (shipped), or set "
        "BIRD_DATA_DIR / data_dir in config.toml.")
