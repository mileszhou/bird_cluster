"""Which sidecar a JPEG carries its label into -- the reverse of `JpgIndex`.

`JpgIndex` answers "given this sidecar, which JPEG shows it?", which is the right
question when the sidecar tree drives the work. The labeler now walks the JPEG
tree instead: the JPEG is what the model actually sees, and 5,229 exports have no
sidecar at all (phone shots, in-camera JPEGs, raws lost to a filing mistake) yet
are ordinary members of the population -- 607 of them come from
interchangeable-lens bodies and cluster in birding trips.

**The claim is local.** It looks only at the JPEG's own name and the sidecar tree,
never at another JPEG. That keeps per-photo processing independent, which is what
makes the walk parallelisable and the code simple to reason about. An earlier
draft asked "does a plain `X.jpg` exist?" before letting `X-Enhanced-NR.jpg` claim
`X.xmp` -- correct, but it made one photo's handling depend on another's.

Locality turns out to be free. Two facts make it work:

- Stripping decorations to the first existing base sidecar covers **all** 43,728
  sidecars. Nothing is orphaned by dropping the cross-JPEG check.
- 307 sidecars end up claimed by more than one JPEG (`X.jpg` and `X-2.jpg` both
  reach `X.xmp`). Processed in stem-sorted order the **exact match always sorts
  first** -- `X` is a proper prefix of `X-2`, so it compares smaller -- and this
  holds for every one of the 307 groups in the library. First claimant wins is
  therefore not a tie-break heuristic; it is exactly the right answer, reached
  without comparing the candidates.

The 313 later claimants keep their CSV row and write no sidecar (`csv-only`).
They are `-2`/`-3` virtual copies, `-Edit`, and `-Enhanced-NR` denoise renders --
deliberate alternate edits of a capture whose label already reached its sidecar,
not errors to be cleaned up.
"""

from pathlib import Path


class SidecarClaims:
    """The sidecar tree, indexed for local reverse lookup."""

    def __init__(self, xmp_root: Path):
        self.xmp_root = Path(xmp_root)
        self.by_folder: dict[str, set[str]] = {}
        for path in self.xmp_root.rglob("*.xmp"):
            folder = path.parent.relative_to(self.xmp_root).as_posix()
            self.by_folder.setdefault(folder, set()).add(path.stem)

    def total(self) -> int:
        return sum(len(v) for v in self.by_folder.values())

    def claim(self, folder: str, stem: str) -> str | None:
        """The sidecar stem this JPEG's label belongs to, or None.

        Exact stem first; otherwise strip trailing `-<decoration>` segments,
        longest base first, and take the first sidecar that exists. Prefix-based,
        so a decoration nobody has seen yet needs no code change.
        """
        stems = self.by_folder.get(folder)
        if not stems:
            return None
        if stem in stems:
            return stem
        parts = stem.split("-")
        for i in range(len(parts) - 1, 0, -1):
            base = "-".join(parts[:i])
            if base in stems:
                return base
        return None

    def path_for(self, folder: str, base: str) -> Path:
        return self.xmp_root / folder / f"{base}.xmp"


def sort_key(folder: str, stem: str) -> tuple[str, str]:
    """Order that puts a capture's exact export ahead of its decorated ones.

    Sorting must be on the *stem*, not the filename: `X-2.jpg` sorts before
    `X.jpg` as a filename, because '-' (0x2D) precedes '.' (0x2E), which would
    hand the sidecar to the virtual copy. On stems, `X` < `X-2` as a prefix.
    """
    return (folder, stem)
