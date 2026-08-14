"""Include/exclude lists over keys, in the spirit of rsync's --include-from.

The scope of a run is worth recording rather than arranging. The alternative --
moving folders out of `data/` so a run cannot see them -- mutates the dataset for
every consumer, leaves no trace of what was excluded, cannot express two
different scopes at once, and is a submodule commit each time. A list is
non-destructive, reads in a diff, and can be copied into the run's own metadata
so a result carries the scope that produced it.

Deliberately small. This is not the manifest described in CLAUDE.md -- no
predicates, no set algebra, no query language. Those want designing against real
clustering requirements, which do not exist yet. What exists is the need to say
"not the zoo trips" or "just these three folders", and that is a list of paths.

**A key is a path relative to an agreed-upon root** -- `data/jpg` for everything
downstream of labelling, which is what the CSV's `jpg` column, the checkpoint
and the JSONL key all hold. The root is a convention, not a lookup: the key
identifies the work item, and nothing here opens a file.

Not `data/xmp`. A sidecar is no longer an identity, it is an *acceptor* -- a
place a run's label is deposited as a by-product, alongside the real output in
the curated `data/label/`. 5,229 images have no acceptor at all, so keying on one
cannot name them.

**Lines are paths, not patterns.** No globs, no regex, no character matching. A
line is either a key, or a folder that stands for every key beneath it:

    # comments and blank lines are ignored
    Photos-16/2016-07-12 City Zoo        # a folder: the whole subtree
    Photos-24/2024-09-05 Birds/_D5D0372.jpg   # one image

Any depth is allowed -- `a/b/c` is a folder line like any other, and a parent
line takes nested children with it. Comparison is on whole path segments, so
`Photos-2` does not take `Photos-24`, which a string prefix would.
"""

from pathlib import Path

from code.lib.config import PROJECT_ROOT

# Scope lists live here and nowhere else. They are inputs a run records by
# *path*, so the file has to be in the repo for the scope to be reconstructible
# later -- a manifest under /tmp makes a run.json a dangling reference, and the
# population a result was computed over becomes unrecoverable. Naming the
# directory in the code rather than in every command line is what enforces it.
MANIFEST_DIR = PROJECT_ROOT / "manifests"


class PathFilter:
    """Decides whether an image key is in scope. Empty include list = everything."""

    def __init__(self, include=(), exclude=(), include_src=None, exclude_src=None):
        self.include = tuple(include)
        self.exclude = tuple(exclude)
        # Where the lines came from, so a run can say which manifest it used
        # rather than only how many lines it held. A count alone looks identical
        # whether you passed the right file or the wrong one.
        self.include_src = include_src
        self.exclude_src = exclude_src

    def __bool__(self):
        return bool(self.include or self.exclude)

    @staticmethod
    def _matches(key: str, paths) -> bool:
        """True if any line is the key itself or a folder containing it.

        The trailing separator is what keeps this on segment boundaries: a bare
        `startswith` would let `Photos-2` take `Photos-24`.
        """
        for p in paths:
            if key == p or key.startswith(p.rstrip("/") + "/"):
                return True
        return False

    def allows(self, key: str) -> bool:
        """Exclude wins: a key named by both lists is out.

        That is the safer precedence for the thing this is usually used for --
        carving a known-bad subset out of an otherwise wanted range -- and it
        matches what rsync users expect from an --exclude.
        """
        if self.include and not self._matches(key, self.include):
            return False
        return not self._matches(key, self.exclude)

    def describe(self) -> str:
        if not self:
            return "no path filter"
        bits = []
        if self.include:
            bits.append(f"include {len(self.include)} path(s)"
                        + (f" from {self.include_src}" if self.include_src else ""))
        if self.exclude:
            bits.append(f"exclude {len(self.exclude)} path(s)"
                        + (f" from {self.exclude_src}" if self.exclude_src else ""))
        return "; ".join(bits)


def read_paths(path) -> list[str]:
    """One path per line; `#` comments and blank lines dropped.

    utf-8-sig because these lists get edited in a spreadsheet as often as an
    editor, and trip folders are named in Chinese -- the same reason the audit
    worklists carry a BOM.

    A missing file is fatal, with the name it looked for. A typo'd manifest must
    never read as "no filter": an include list exists to narrow, so ignoring one
    runs the *whole* library instead of the subset asked for -- the loudest
    possible way to be wrong about scope, and silent.

    An empty file is fatal for the same reason, one step subtler: a list whose
    every line was commented out is indistinguishable from no list at all once
    it has been parsed.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"path list not found: {p}\n"
            f"       Scope lists live in manifests/. Ignoring a missing include "
            f"list would silently run everything.")
    out = []
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line.replace("\\", "/").strip("/"))
    if not out:
        raise ValueError(
            f"path list is empty: {p}\n"
            f"       Every line is blank or commented out, which would read as "
            f"no filter at all.")
    return out


LOCAL_MANIFEST_DIR = MANIFEST_DIR.parent / "local" / "manifests"


def resolve(name) -> Path:
    """A manifest argument -> its path, which must land in one of two directories.

    One rule: it is a path, and after resolving it, it has to be under
    `manifests/` or under `local/manifests/`. Everything else follows.

    - `exclude-captive.txt` and `captive/zoos.txt` are read as manifest-relative,
      falling back to `local/manifests/` when not found in `manifests/`.
    - `manifests/inclusion-all.txt` and `local/manifests/exclude-captive.txt` are
      read as repo-relative, so what you type is what is on disk -- the form tab
      completion produces, and the one worth typing.
    - `manifests/../data/x` and `/tmp/x` are refused. Checking the *resolved*
      location rather than the leading path segment is what makes that true; an
      earlier version compared the segment and let `manifests/../` walk out.

    **Two directories, because a manifest can be one of two things.** A list of
    libraries (`Photos-16`) means the same to anyone and is versioned in
    `manifests/`. A list of real trips names places somebody went, so it lives
    in the gitignored `local/manifests/` alongside the other local working files
    -- see the `local/` note in CLAUDE.md.

    Both are still inside the repository, which is the part worth keeping: a
    list read from `/tmp` makes a run's recorded scope a dangling reference and
    the population behind a result unrecoverable. A gitignored manifest weakens
    that for the one person who has it; a manifest outside the tree loses it for
    everyone, including them.
    """
    p = Path(str(name))
    roots = (MANIFEST_DIR, LOCAL_MANIFEST_DIR)
    if p.is_absolute():
        candidates = [p]
    elif p.parts and p.parts[0] in ("manifests", "local"):
        candidates = [MANIFEST_DIR.parent / p]
    else:
        candidates = [r / p for r in roots]

    resolved = next((c.resolve() for c in candidates if c.exists()),
                    candidates[0].resolve())
    if not any(r.resolve() in resolved.parents for r in roots):
        raise ValueError(
            f"manifest must live in manifests/ or local/manifests/, but {name!r} "
            f"resolves to {resolved}\n"
            f"       Keeping scope lists in the repository is what lets a run's "
            f"recorded scope be found again.")
    return resolved


def build(include_from=None, exclude_from=None) -> PathFilter:
    return PathFilter(
        include=read_paths(resolve(include_from)) if include_from else (),
        exclude=read_paths(resolve(exclude_from)) if exclude_from else (),
        include_src=str(resolve(include_from)) if include_from else None,
        exclude_src=str(resolve(exclude_from)) if exclude_from else None,
    )
