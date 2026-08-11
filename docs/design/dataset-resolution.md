# Which dataset a run uses

> **Status: settled 2026-08-11. No signature.**
>
> Every candidate is accepted on the same test — does it hold a `jpg/` tree —
> and `./data` has nothing to prove that the others do not. The signature check
> was implemented, set aside as the leading candidate against it, and removed
> once it produced a live wrong answer. The reasoning is kept below rather than
> deleted: a road looked at and rejected is worth as much as the ones taken.

## The problem

The repository is going to GitHub. The dataset is not: `data/` is a 6.4 GB
submodule on a private host, and one file in it (`embed/embeddings.jsonl`,
452 MB) exceeds GitHub's 100 MB per-file hard limit on its own.

So a clone has no data, and three audiences need different things from the same
checkout:

- **a fresh clone** should run out of the box, on a small shipped `sample_data/`
- **someone with their own photo library** should point the tools at it without
  editing a versioned file
- **this project** should use the canonical `data/`, and should be able to say
  afterwards that it did

The third is the interesting one. A run records `data_dir` in its `args.json`,
and that recording is part of a result's provenance. "This figure came from the
canonical dataset" should be a checkable claim, not an assumption.

## Resolution order

Most specific first. Implemented in `code/lib/config.py:data_dir()`.

| | source | notes |
|---|---|---|
| 1 | `--data-dir` | an instruction for this invocation |
| 2 | `$BIRD_DATA_DIR` | a path, not a secret — deliberately not `.env` |
| 3 | **`.datapath`** | your copy of the tracked `_datapath` template; gitignored |
| 4 | `./data` | the private submodule |
| 5 | `./sample_data` | the shipped sample, last resort |

Every candidate is accepted on one test: does it hold a `jpg/` tree.

**A path in `.datapath` is binding.** If it does not resolve, the run stops — it
must never fall through to `./data` or the sample, because that turns a typo
into a run against a different population that reports success. This is the
failure the signature check produced on its way out, and the same reason both
stages exit on an empty selection.

Two properties this buys:

**No one edits a versioned file to get the right answer.** The earlier design
put the user's path in `config.toml` and relied on it being *inert* on a working
checkout — correct, but it still meant a tracked file carrying a local value and
a `git diff` whose first hunk was always yours. `.datapath` is gitignored and
`_datapath` is the tracked template, which is the `_env` → `.env` pattern this
project already uses for secrets. `config.toml` has no `data_dir` key any more:
it did this job through a second mechanism, and one way to point at a dataset
means one place for it to be wrong — the argument that removed `--years`.

**A fresh clone simply runs.** It falls to `sample_data/` with no setup.

### The empty-directory trap

Testing `Path("data").is_dir()` is wrong, and wrong in exactly the case that
matters. `git submodule update --init` against an unreachable host **fails but
leaves `data/` present and empty**, so an existence test sends every GitHub clone
into an empty directory instead of the sample. Found by cloning the repo and
trying it, not by reading the code.

Every candidate is therefore tested for a `jpg/` tree, `./data` included. The
signature happened to subsume this — an empty directory has no signature either
— which is why the trap stayed hidden while the check was in place.

### Experimenting on another dataset

`--data-dir`, `$BIRD_DATA_DIR`, or the `config.toml` line with `./data` moved
aside. All three outrank or bypass it, and all three are recorded in the run's
`args.json`, so the substitution is visible in the result rather than inferred.

(Under the signature the recipe was to rename `signature.txt`. That worked, and
was also how the check came to fail silently once the file was never committed.)

## What the gitlink already does

Most of what the signature was reaching for is already true, for free, because
`data` is a submodule path rather than an ordinary directory. Measured on a
fresh clone with a user's photos dropped into `data/`:

    $ git status --short                        (nothing — clean)
    $ git add -A --dry-run                      (nothing — not swept in)
    $ git add "data/jpg/.../IMG_1.jpg"
    fatal: Pathspec 'data/jpg/.../IMG_1.jpg' is in submodule 'data'
    $ git add -f "data/jpg/.../IMG_1.jpg"
    fatal: Pathspec ... is in submodule 'data'   ← -f does not override
    $ git check-ignore -v data/                 (not ignored at all)

Git refuses to descend into an uninitialised submodule path, and refuses *with
an error* rather than silently skipping. That is stronger than `.gitignore`,
which `-f` defeats. So `./data` does not need reserving to stay safe: a user's
library there is inert to git, cannot be committed by accident, and the code can
simply use it.

That removes the practical argument for the signature and leaves only the
provenance one — which is the argument with the defect.

### One exception: a git repo inside `data/`

A *plain* directory at `data/` is invisible. A directory containing its own
`.git` is not:

     M data
            modified:   data (new commits)

The parent then sees a submodule whose HEAD differs from the pin, and says so
permanently. A user who wants their dataset under version control should keep it
elsewhere and point `config.toml` at it, rather than making `./data` a
repository. Worth a line in the README, since it is the one arrangement that
leaves a clone's tree dirty.

## The signature check — removed 2026-08-11

*Kept as a record of the reasoning, not as a description of the code. Nothing
below is implemented any more; `master_data_available()` and
`MASTER_SIGNATURE_SHA` are gone from `code/lib/config.py`.*

`data/signature.txt` held a 32-byte random value. `code/lib/config.py` holds
only its SHA-256. `master_data_available()` reads the file, hashes it, compares.

The property: **only someone who has cloned the private dataset can produce a
directory that passes**, because recovering the value from the hash means
inverting SHA-256. Storing an expected *value* instead would be forgeable by
anyone reading the repo — a password published next to the lock.

### Reservations

Enough to hold the whole mechanism open.

**It checks identity, not integrity — and integrity is the failure that actually
happens.** A half-finished rsync, a dataset from before the dedup, a submodule at
the wrong commit: every one of those passes the signature, because
`signature.txt` copied faithfully alongside 30% of the images is still the right
value. The check is precise about the question nobody was going to get wrong.

**There is no adversary.** Anyone who can create `./data` already has write
access to the working tree. Substituting a dataset is an intended feature, via
`$BIRD_DATA_DIR` and `config.toml`. Cryptographic machinery in a place with no
attacker invites the reader to look for a threat model that does not exist.

**It couples two repositories.** The value lives in the private dataset, the
hash in the public code; rotating means changing both together, and any
published `args.json` from before a rotation refers to a signature that no longer
validates.

**It leaks permanently and silently if it leaks at all.** One paste of
`signature.txt` into a report, a log, or a message and the guarantee is void with
no way to notice.

**It is opaque.** A reader meets a magic hex constant and cannot tell what it
protects or why. That is the opposite of how the rest of this project documents
itself.

## Alternatives considered

| | catches | cost | forgeable |
|---|---|---|---|
| nothing — test for `jpg/` | absent dataset | free | trivially |
| marker file with a known value | absent, wrong-name | free | trivially |
| **signature (hash of a secret)** | **absent, lookalike** | free | no |
| submodule pin: `git -C data rev-parse HEAD` vs the gitlink | absent, **wrong version** | free | yes |
| manifest: file count + total bytes | absent, **partial sync**, wrong version | 0.11 s | yes |
| content hash of every file | all of the above, **corruption** | ~25 s | no |

Measured on the current dataset: 49,224 files, 7.53 GB.

The pin check is notable for costing nothing and catching the version error the
signature cannot — the parent already records the exact dataset commit, so no new
file is needed and it cannot drift. The manifest catches partial sync for 0.11 s.
Neither resists forgery, which may not matter.

**The decision is no signature at all**: accept `./data` on a `jpg/` tree like
any other location. If a version check is wanted later, the submodule pin gives
it for nothing; if an integrity check is wanted, the manifest costs 0.11 s. Both
can be added when a real failure motivates them.

## What actually settled it

Not the argument — the argument had been made and the mechanism was left in
place anyway. What settled it was the check failing, in the direction that costs
most.

`data/signature.txt` was written but never committed to the submodule, so on
this machine it did not exist. `master_data_available()` therefore reported
"not the canonical dataset" about the canonical dataset, resolution fell through
`config.toml` to `./sample_data`, and:

    data_dir() -> /home/miles/projects/bird_cluster/sample_data

A 109-image sample shadowing a 49,270-image library, with nothing to indicate
it. Nothing consumed `data_dir()` yet — all six run scripts still pass
`--data-dir ./data` — so no run was misdirected, and adding `sample_data/` was
what armed it, since before that the fallback had nothing to find.

The lesson is not that the hash was wrong but that **the check failed open into
a smaller dataset**, which is the same shape as the `--years 2019` default that
embedded a seventh of the library and reported success. A mechanism guarding
provenance that can silently substitute the population is worse than no
mechanism, because the claim it exists to protect is the one it breaks.

## Open questions

1. ~~Does the signature survive?~~ **Resolved 2026-08-11: no, removed.**
2. Provenance without it: is the submodule pin recorded in `args.json` enough to
   say which dataset a run used? It is free and exact, but only meaningful when
   `./data` really is the submodule.
3. What goes in `sample_data/` — **partly answered.** It exists: 109 JPEGs,
   103 sidecars, 33 MB, its own git repo inside a gitignored directory (which
   is why a repo there is invisible to the parent, unlike one at `data/`).
   Labels stripped with `tools/strip_labels.py`, since a sample should ship the
   pipeline's input rather than its output. It covers sidecar-less images (6),
   `-2` virtual copies (3), an `&` in a trip name, sidecars with no
   `dc:subject` (54), and excludes `people`. Still missing: a **non-ASCII trip
   name** (nothing exercises the `utf-8-sig` worklist path that exists because
   trip folders are Chinese) and a **`Seq` `dc:subject`** (all 49 are `Bag`,
   against ~1,748 `Seq` sidecars in the library). No `label/` either, so the
   embedding step cannot run standalone from a fresh clone.
4. Should the clustering sample ship as **vectors rather than images**? 2,000
   real embeddings are 5.9 MB as float32, against ~31 MB for 200 photos — and
   clustering needs vectors, not pixels, so a reader could see real structure
   without publishing photographs.

## Notes

`CLAUDE.md` reserves `docs/` for product documentation — "output meant for
whoever uses the result rather than notes about building it" — and this document
is half product (how to point the tools at your own data) and half design
rationale. Either the wording wants widening or this belongs under `project/`.
Filed here as directed; worth settling alongside the rest.
