# Manifests

Path lists that define the **scope** of a run — what a stage should look at, or
skip. Consumed by `--include-from` / `--exclude-from`
(`code/lib/path_filter.py`).

## How they are named

Pass the **file name**, not a path — `--exclude-from exclude-captive.txt`. The directory is
resolved by `code/lib/path_filter.py`, so it cannot be pointed elsewhere: a list read from
outside the repo makes a run's recorded scope a dangling reference. A redundant `manifests/`
prefix is accepted and stripped; anything else is refused.

## What they are

One key per line, relative to the agreed-upon root (`data/jpg` downstream of
labelling). `#` starts a comment, blank lines are ignored. A line is either a key
or a folder standing for every key beneath it, at any depth. **Paths, not
patterns** — no globs, no regex. Comparison is on whole path segments, so
`Photos-2` does not take `Photos-24`.

## Why they are versioned

Unlike `project/reports/*.csv`, which are gitignored because a run regenerates
them, a manifest is an **input**. A run records the manifest's *path* in its
`run.json`, so the scope is only reproducible if the file at that path is still
the file that was used. An untracked manifest makes a run's metadata a dangling
reference.

Keep them small and commented. A manifest that says *why* a folder is out is
worth several times one that just lists it — the reasoning is what someone needs
in a year, and it is exactly what gets lost.

## Why not just move the folders

Moving folders out of `data/` so a run cannot see them mutates the dataset for
every consumer, leaves no record of what was excluded, cannot express two scopes
at once, and is a submodule commit each time. A list is non-destructive, reads in
a diff, and can be swapped per experiment.

## Current

- **`exclude-captive.txt`** — zoos and aviaries, 1,090 bird photos across 12
  trips. The species mix in a collection is an artefact of the collection rather
  than a place or season, and the enclosure is a background the model can learn
  instead of the bird.

  Note what it deliberately does *not* exclude: the Safari "Safari" trips.
  Safari is a waterhole in a national park and those birds are wild — a
  keyword match on `safari` drops 143 genuinely wild wild records. Worth
  remembering before generating one of these by pattern.
