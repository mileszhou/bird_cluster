# Which dataset a run uses — TENTATIVE

> **Status: tentative, with a leading candidate.**
>
> **No signature.** Accept `./data` when it holds a `jpg/` tree, like any other
> candidate. The signature check is implemented but set aside — its reservations
> stand, and the argument that finally settled it is below under *What the
> gitlink already does*. Nothing should depend on the signature mechanism.

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
| 3 | **valid `./data`** | the canonical dataset, signature-checked |
| 4 | `config.toml`'s `data_dir` | ignored while 3 is valid |
| 5 | `./sample_data` | shipped, last resort |

Two properties this buys:

**No one edits a versioned file to get the right answer.** `config.toml` is
checked in, so a local edit to it would show as a dirty tree indefinitely. On a
working checkout the `data_dir` line is inert regardless of what it says; on a
clone it is what points at the user's library.

**A fresh clone simply runs.** It falls to `sample_data/` with no setup.

### The empty-directory trap

Testing `Path("data").is_dir()` is wrong, and wrong in exactly the case that
matters. `git submodule update --init` against an unreachable host **fails but
leaves `data/` present and empty**, so an existence test sends every GitHub clone
into an empty directory instead of the sample. Found by cloning the repo and
trying it, not by reading the code.

Candidates other than `./data` are therefore tested for a `jpg/` tree. `./data`
is tested by signature — which subsumes the problem, since an empty directory
has no signature either.

### Experimenting on another dataset

Rename `data/signature.txt` aside. The master stops validating, the resolver
falls through to `config.toml`, and the rename shows up in `git -C data status`
as a reminder to put it back.

This is also the argument for making a valid master **absolute** — outranking
even `--data-dir` — so that an `args.json` claiming `./data` can never be wrong.
Currently it does not; flags 1 and 2 still win. Unresolved.

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

## The signature check — implemented, set aside

`data/signature.txt` holds a 32-byte random value. `code/lib/config.py` holds
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

**The leading candidate is no signature at all**: accept `./data` on a `jpg/`
tree like any other location. If a version check is wanted later, the submodule
pin gives it for nothing; if an integrity check is wanted, the manifest costs
0.11 s. Both can be added when a real failure motivates them, and neither has to
be decided now.

## Open questions

1. ~~Does the signature survive?~~ Leading candidate: no. See above.
2. Provenance without it: is the submodule pin recorded in `args.json` enough to
   say which dataset a run used? It is free and exact, but only meaningful when
   `./data` really is the submodule.
3. What goes in `sample_data/` — it should be chosen to exercise the edge cases
   (sidecar-less image, derived export, duplicate capture, Chinese and
   comma-bearing trip names, Bag *and* Seq sidecars), and must exclude the
   `people` category since a public sample is permanent.
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
