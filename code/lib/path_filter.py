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


class PathFilter:
    """Decides whether an image key is in scope. Empty include list = everything."""

    def __init__(self, include=(), exclude=()):
        self.include = tuple(include)
        self.exclude = tuple(exclude)

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
            bits.append(f"{len(self.include)} include")
        if self.exclude:
            bits.append(f"{len(self.exclude)} exclude")
        return ", ".join(bits) + " path(s)"


def read_paths(path) -> list[str]:
    """One path per line; `#` comments and blank lines dropped.

    utf-8-sig because these lists get edited in a spreadsheet as often as an
    editor, and trip folders are named in Chinese -- the same reason the audit
    worklists carry a BOM.
    """
    out = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line.replace("\\", "/").strip("/"))
    return out


def build(include_from=None, exclude_from=None) -> PathFilter:
    return PathFilter(
        include=read_paths(include_from) if include_from else (),
        exclude=read_paths(exclude_from) if exclude_from else (),
    )
