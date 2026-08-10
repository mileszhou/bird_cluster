# Which dataset a run uses — TENTATIVE

> **Status: not settled.** The resolution order is agreed and implemented. The
> signature check is implemented but under review — see *Reservations*, which
> are substantial enough that it may not survive. Nothing downstream should
> depend on the signature mechanism until this note says otherwise.

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

## The signature check

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

A plausible resolution is **manifest for integrity + pin for version + no
signature at all**, accepting that a determined person could fake a canonical
dataset and observing that no one has any reason to.

## Open questions

1. Does the signature survive, or is provenance better served by the pin and a
   manifest?
2. If it survives: should a valid master outrank `--data-dir`?
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
