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


DATAPATH = CONFIG_PATH.parent / ".datapath"


def declared_data_dir() -> str | None:
    """The path written in `.datapath`, or None.

    `.datapath` is the user's own copy of the tracked `_datapath` template, and
    is gitignored -- the whole point. `config.toml` is committed, so configuring
    the dataset there means a working tree that is dirty for as long as you keep
    your setting, and a `git diff` whose first hunk is always yours. Splitting
    the local part into an ignored file is the same shape as `_env` -> `.env`,
    which this project already uses for secrets.

    Format: the first non-blank, non-comment line. Relative paths resolve from
    the repository root rather than the caller's cwd, so a run started from a
    subdirectory picks the same dataset as one started from the top.
    """
    try:
        text = DATAPATH.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            return line
    return None


def data_dir(override=None) -> Path:
    """The dataset to work on, resolved in order of specificity.

        1. an explicit --data-dir
        2. $BIRD_DATA_DIR
        3. .datapath, your local copy of the _datapath template
        4. ./data, the private submodule
        5. ./sample_data, shipped

    A fresh clone therefore runs on the sample with no setup at all, and a
    checkout with the submodule finds it without setup either. `.datapath` is
    for the third case: a library somewhere else entirely. $BIRD_DATA_DIR stays
    for the one-off -- CI, a quick comparison -- where writing a file would be
    heavier than the job.

    There is no `data_dir` key in config.toml any more. It did this same job
    through a second mechanism, and one way to point at a dataset means one
    place for it to be wrong -- the same argument that removed `--years`.

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

    # A declaration is binding. If `.datapath` names a path, that is the dataset
    # or the run stops -- it must never quietly fall through to ./data or the
    # sample, which is how a typo becomes a run against the wrong population
    # that reports success. Same reason the stages exit on an empty selection,
    # and the same failure the signature check produced on its way out.
    declared = declared_data_dir()
    if declared:
        p = resolve(declared)
        if not (p / "jpg").is_dir():
            raise SystemExit(
                f"error: .datapath names {declared!r}, which has no jpg/ directory "
                f"(looked in {p}).\n"
                f"       Fix the path, or delete .datapath to fall back to ./data "
                f"or ./sample_data.")
        return p

    for candidate in ("./data", "./sample_data"):
        p = resolve(candidate)
        if (p / "jpg").is_dir():
            return p
    raise SystemExit(
        "error: no dataset found. Expected ./sample_data (shipped -- see the README "
        "for how to fetch it), ./data (the private submodule -- "
        "`git submodule update --init`), or a path in .datapath (copy _datapath).")


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
