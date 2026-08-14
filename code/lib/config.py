"""Loads non-secret project configuration from config.toml (repo root).

Secrets (API keys) stay in .env, which is gitignored and sourced by shell scripts.
config.toml is checked in and holds everything safe to commit: which host/port
each backend server runs on, and which run directory the analysis tools work on.
"""

import os
import tomllib
from functools import lru_cache
from pathlib import Path

def _project_root() -> Path:
    """The repository root: `$PROJECT_ROOT` if set, else inferred from this file.

    One definition, so nothing else counts `parents[n]`. That count differs by
    how deep a file sits -- 1 from `tools/`, 2 from `code/lib/` -- and it is
    silently wrong the moment a file moves, which makes every path in the
    project depend on where its own source happens to live.

    `$PROJECT_ROOT` wins when set, so a caller can point the whole project at a
    checkout other than the one the code was imported from. It is validated
    rather than trusted: a wrong value would otherwise surface far away, as a
    missing config or an empty dataset, which is the failure this project keeps
    trying to stop happening.
    """
    env = os.environ.get("PROJECT_ROOT")
    if not env:
        return Path(__file__).resolve().parents[2]
    root = Path(env).expanduser().resolve()
    if not (root / "config.toml").is_file() or not (root / "code").is_dir():
        raise SystemExit(
            f"error: $PROJECT_ROOT={env!r} does not look like this repository "
            f"(no config.toml, or no code/).")
    return root


PROJECT_ROOT = _project_root()
CONFIG_PATH = PROJECT_ROOT / "config.toml"
LOCAL_CONFIG_PATH = CONFIG_PATH.with_name("config.local.toml")


def _merge(base: dict, over: dict) -> dict:
    """`over` wins, recursively, so a local file may set one key of a table."""
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


@lru_cache(maxsize=1)
def load_config() -> dict:
    """config.toml, with config.local.toml layered over it.

    The tracked file holds values that work for a stranger; the local file holds
    the ones that are true only of your machines. Host names are the obvious
    case: `darwin` and `spark` mean nothing to anyone else, and before this split
    the only way to point the tools at your own servers was to edit a committed
    file -- so every user's tree was dirty for as long as their setup was right.
    This file holds everything machine-specific except secrets: host names and
    `data_dir`. Secrets stay in `.env` because the run scripts `source` it as
    shell, which TOML cannot be, and because a leaked key is a different problem
    from a leaked hostname. Two ignored files, one boundary.

    The merge is recursive so a local file can override one key of a table --
    `[servers.vllm] host = ...` without restating the port.
    """
    with open(CONFIG_PATH, "rb") as f:
        config = tomllib.load(f)
    try:
        with open(LOCAL_CONFIG_PATH, "rb") as f:
            config = _merge(config, tomllib.load(f))
    except FileNotFoundError:
        pass
    return config


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
        3. config.local.toml's `data_dir`
        4. ./data, the private submodule
        5. ./sample_data, shipped

    A fresh clone therefore runs on the sample with no setup at all, and a
    checkout with the submodule finds it without setup either. `data_dir` in
    config.local.toml is for the third case: a library somewhere else entirely.
    $BIRD_DATA_DIR stays for the one-off -- CI, a quick comparison -- where
    writing a file would be heavier than the job.

    This lived in a dedicated `.datapath` file for a day. Folding it into
    config.local.toml leaves **one** ignored file for everything
    machine-specific rather than two, which is this project's own rule about
    second mechanisms. `.env` stays separate for a real reason: the run scripts
    `source` it as shell, which a TOML file cannot be, and a secret is a
    different risk class from a hostname.

    Note the key is back in a *config file* but not in a *tracked* one, which
    was the whole objection: config.toml is committed, config.local.toml is
    ignored.

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

    Nobody edits a versioned file to get the right answer, and now nobody has to
    reason about whether their edit is inert either: the file they touch is
    ignored. `$BIRD_DATA_DIR` is a path, not a secret, so it does not belong in
    .env.
    """
    if override:
        return Path(override)
    env = os.environ.get("BIRD_DATA_DIR")
    if env:
        return Path(env)
    root = CONFIG_PATH.parent
    resolve = lambda c: (Path(c).expanduser() if Path(c).expanduser().is_absolute()
                         else root / Path(c).expanduser())

    # A declaration is binding. If config.local.toml names a data_dir, that is
    # the dataset or the run stops -- it must never quietly fall through to
    # ./data or the sample, which is how a typo becomes a run against the wrong
    # population that reports success. Same reason the stages exit on an empty
    # selection, and the same failure the signature check produced on its way out.
    declared = load_config().get("data_dir")
    if declared:
        p = resolve(declared)
        if not (p / "jpg").is_dir():
            raise SystemExit(
                f"error: config.local.toml sets data_dir = {declared!r}, which has no "
                f"jpg/ directory (looked in {p}).\n"
                f"       Fix the path, or remove the line to fall back to ./data "
                f"or ./sample_data.")
        return p

    for candidate in ("./data", "./sample_data"):
        p = resolve(candidate)
        if (p / "jpg").is_dir():
            return p
    raise SystemExit(
        "error: no dataset found. Expected ./sample_data (shipped -- see the README "
        "for how to fetch it), ./data (the private submodule -- "
        "`git submodule update --init`), or data_dir in config.local.toml "
        "(copy _config.local.toml).")


def display_path(path) -> Path:
    """`path` relative to the repo when it is inside it, else unchanged.

    data_dir() returns an absolute path, and a tracked report that prints one
    carries somebody's home directory into git -- so it diffs on every machine
    and stops being a record of what moved in the *dataset*. Keeping the reports
    diffable is the whole reason their filenames never change.
    """
    try:
        return Path(path).relative_to(CONFIG_PATH.parent)
    except ValueError:
        return Path(path)
