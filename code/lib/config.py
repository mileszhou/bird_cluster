"""Loads non-secret project configuration from config.toml (repo root).

Secrets (API keys) stay in .env, which is gitignored and sourced by shell scripts.
config.toml is checked in and holds everything safe to commit: which host/port
each backend server runs on, and which run directory the analysis tools work on.
"""

import hashlib
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


# sha256 of the value in `<master>/signature.txt`. Only the hash is public, so a
# directory cannot be made to pass as the canonical dataset by anyone who has not
# cloned it -- deriving the value from this would mean inverting sha256.
MASTER_SIGNATURE_SHA = "c0dac9b799814d96489fe974fb1022f38142f9919774ae4fc27491875465f1d8"


def master_data_available(path=None) -> bool:
    """True when `path` is *the* canonical dataset, not merely a directory named
    like it.

    The point is provenance rather than defence. Nobody here is an adversary --
    substituting your own dataset is an intended feature, via `$BIRD_DATA_DIR`,
    `data_dir`, or `sample_data/`. What this buys is that a run recording
    `data_dir: ./data` made a *checkable* claim: it really was the dataset the
    published figures refer to, not a lookalike assembled later.

    A checked-in expected *value* would be forgeable by anyone reading the repo.
    Checking a hash of a value that lives only inside the private dataset is not:
    the value cannot be recovered from the hash, so only someone who has the
    dataset can produce a directory that passes.
    """
    root = Path(path) if path else CONFIG_PATH.parent / "data"
    sig = root / "signature.txt"
    try:
        value = sig.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return hashlib.sha256(value.encode()).hexdigest() == MASTER_SIGNATURE_SHA


def data_dir(override=None) -> Path:
    """The dataset to work on, resolved in order of specificity.

        1. an explicit --data-dir
        2. $BIRD_DATA_DIR
        3. config.toml's `data_dir`, if that directory exists
        4. ./data, if it exists
        5. ./sample_data

`./data` is accepted only when it passes `master_data_available()` -- the
    signature check -- so an empty directory left behind by an uninitialised
    submodule is skipped rather than used. That case is the reason the test is
    not `is_dir()`: `git submodule update --init` leaves `data/` present and
    empty when the clone fails, which is exactly the situation the fallback
    exists to handle. Other candidates need only a `jpg/` tree, since they are
    the user's own data and have nothing to prove. A working checkout has
    `data/` and lands there. Neither has to edit a committed file to get the
    right answer, which matters because config.toml is versioned and a local
    edit to it shows up as a dirty tree forever.

    $BIRD_DATA_DIR exists for the third case: a library that lives outside the
    repo entirely. It is a path, not a secret, so it does not belong in .env.
    """
    if override:
        return Path(override)
    env = os.environ.get("BIRD_DATA_DIR")
    if env:
        return Path(env)
    root = CONFIG_PATH.parent
    configured = load_config().get("data_dir")
    master = root / "data"
    if master_data_available(master):
        return master
    for candidate in ([configured] if configured else []) + ["./sample_data"]:
        p = Path(candidate)
        p = p if p.is_absolute() else root / p
        if (p / "jpg").is_dir():
            return p
    raise SystemExit(
        "error: no dataset found. Expected ./data (the private submodule -- "
        "`git submodule update --init`) or ./sample_data (shipped), or set "
        "BIRD_DATA_DIR / data_dir in config.toml.")
