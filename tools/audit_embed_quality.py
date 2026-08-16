#!/usr/bin/env python3
"""Compare embedding runs against the labels, so a change can be judged.

The question this exists to answer is "did that change make the vectors
better?" -- raised concretely by the input resolution, which turned out to have
been 224x224 for the whole first run while the exports were 1024px. Arguing
about whether more pixels help is cheaper to settle by measuring.

**It measures the vectors, not the clustering.** The primary number is
leave-one-out **1-NN accuracy**: for each image, is its nearest neighbour in
vector space the same species? That is a direct property of the embedding, with
no `min_cluster_size` in it. Comparing clusterings instead would confound the
embedding with HDBSCAN's parameters -- a resolution that looked worse at mcs15
might look better at mcs5, and neither would be telling you about the vectors.
AMI against a clustering is still worth having and can be added; it answers a
different question.

**Same population, or the numbers are not comparable.** Runs are intersected on
`key` before anything is computed, so two runs that embedded slightly different
sets are still judged on the images they share. The intersection size is
reported; a run that shrinks it is the interesting kind of problem.

**Three numbers, because one hides things.** Micro 1-NN is over every image and
is therefore dominated by the common species. The other two are restricted to
species with at least `MIN_SPECIES` images -- the species discovery report's
cutoff -- one micro and one macro. Macro gives a rare bird the same weight as a
mallard, so a change that lifts micro while dropping macro has helped the easy
cases at the tail's expense, which is worth knowing before adopting it.

The restriction is not tidiness: a species with a single image scores zero by
construction, since its nearest neighbour is necessarily something else, and
1,246 of the 2,820 species here have exactly one image. An all-species macro
would mostly measure how long that tail is, which does not change between runs
and buries what does.

Labels here are the pipeline's own `species` field, already resolved through the
never-demote rule by `embed.py`. They are **not ground truth** -- roughly 1.4% of
the set is known to carry an older pipeline's over-calls -- so the absolute value
of any of these numbers means less than the difference between two runs measured
the same way. That is the whole design: this is a comparison instrument.

**It writes into `local/`, because this repository is public.** The report
carries no filenames, trips, dates or people by construction, but it is still
computed from a private collection, and a species list is not as harmless as it
looks: a black-necked crane and an American white pelican in the same collection
say a good deal about where the photographer has been. `local/` is gitignored
wholesale, which is a stronger guarantee than a per-file rule in `.gitignore` --
the failure mode being guarded against is a new report landing in a tracked
directory and nobody noticing.

`--anonymise` replaces each species with a stable digest of its name
(`sp-3f2a`), keeping the tables diffable and the counts meaningful while naming
nothing. That is for a copy meant to leave this machine; the default names the
birds, because the default location does not leave it.

Output goes to a **fixed path** so it can be diffed between runs in an editor,
and there are no timestamps in the body for the same reason -- the only lines
that move are the ones that mean something. `--snapshot` additionally files a
numbered copy under `local/reports/archive/`, the same way `./tool-audit`
does.

    python3 -m tools.audit_embed_quality
    python3 -m tools.audit_embed_quality --run data/embed --run output/embed
    python3 -m tools.audit_embed_quality --snapshot image-size-1024
    python3 -m tools.audit_embed_quality --anonymise        # safe to paste
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("PROJECT_ROOT")
                or str(Path(__file__).resolve().parents[1]))

from code.lib.config import PROJECT_ROOT  # noqa: E402

# local/ is gitignored wholesale; this repository is public and the report is
# computed from a private collection. See the module docstring.
REPORT = PROJECT_ROOT / "local" / "reports" / "embed_quality_report.md"
ARCHIVE = PROJECT_ROOT / "local" / "reports" / "archive"
MIN_SPECIES = 20            # the species discovery report's cutoff
CHUNK = 1024                # rows per similarity block; 1024x27k floats is ~110 MB


def discover() -> list[Path]:
    """The curated baseline and the live run -- the two that are always meant.

    Deliberately *not* every `output_*/embed` on disk. Sweeping those in picked
    up old partial runs of 85 and 7,488 vectors whose keys do not overlap, and
    since the comparison intersects on `key`, one of them empties the population
    and the whole report with it. An archive is compared by naming it, which is
    also what makes a `run.json` record of the comparison meaningful.
    """
    return [path for path in (PROJECT_ROOT / "data" / "embed",
                              PROJECT_ROOT / "output" / "embed")
            if (path / "embeddings.jsonl").is_file()]


def load(run_dir: Path):
    """(keys, species, vectors, provenance) for one run."""
    keys, species, vectors = [], [], []
    with open(run_dir / "embeddings.jsonl") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                vec = row["embedding"]
            except (json.JSONDecodeError, KeyError):
                continue          # a truncated last line from an interrupted run
            keys.append(row["key"])
            species.append((row.get("species") or "").strip().lower())
            vectors.append(vec)

    info = {"model": "(unrecorded)", "image_size": "(unrecorded)"}
    run_json = run_dir / "run.json"
    if run_json.is_file():
        try:
            meta = json.loads(run_json.read_text())
            info["model"] = meta.get("model") or info["model"]
            info["image_size"] = (meta.get("image_size")
                                  or (meta.get("server") or {}).get("image_size")
                                  or info["image_size"])
        except json.JSONDecodeError:
            pass
    return keys, np.array(species), np.asarray(vectors, dtype=np.float32), info


def one_nn_correct(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Boolean per row: is the nearest *other* vector the same species?

    Vectors arrive L2-normalised from the server, so a dot product is cosine
    similarity and the largest one is the nearest neighbour. Renormalised here
    anyway, because a run that failed to normalise would otherwise be silently
    compared on a different metric.

    Done in blocks: the full 27k x 27k similarity matrix is 2.9 GB and never
    needed all at once, only its per-row argmax.
    """
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    out = np.empty(len(X), dtype=bool)
    for start in range(0, len(X), CHUNK):
        stop = min(start + CHUNK, len(X))
        sim = X[start:stop] @ X.T
        # Exclude self, which is always the nearest at similarity 1.
        sim[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        out[start:stop] = labels[sim.argmax(axis=1)] == labels[start:stop]
    return out


def label_of(run_dir: Path) -> str:
    """A name that distinguishes runs. `.name` does not -- it is `embed` for all
    of them, since the run directory is `<archive>/embed`."""
    if run_dir.is_relative_to(PROJECT_ROOT):
        return str(run_dir.relative_to(PROJECT_ROOT))
    return str(run_dir)


def score(correct: np.ndarray, labels: np.ndarray) -> dict:
    """Micro over everything, macro and micro over the species big enough to mean something.

    Macro is taken over species with >=MIN_SPECIES images rather than all of
    them. A species with a single image scores 0 by construction -- its nearest
    neighbour is necessarily a different species -- and there are thousands of
    those, so an all-species macro mostly measures how long the tail is, which
    does not change between runs and buries what does.
    """
    counts = {s: int((labels == s).sum()) for s in set(labels.tolist())}
    per_species = {s: float(correct[labels == s].mean()) for s in counts}
    big = sorted(s for s, n in counts.items() if n >= MIN_SPECIES)
    big_mask = np.isin(labels, big)
    return {
        "micro": float(correct.mean()),
        "macro": float(np.mean([per_species[s] for s in big])) if big else float("nan"),
        "singletons": sum(1 for n in counts.values() if n == 1),
        "big_micro": float(correct[big_mask].mean()) if big_mask.any() else float("nan"),
        "big_species": len(big),
        "species": len(counts),
        "per_species": per_species,
        "counts": counts,
    }


def name_of(species: str, anonymise: bool) -> str:
    """The species, or a stable digest of it.

    Stable across runs and machines so the anonymised report still diffs
    line-for-line; short because collisions here cost nothing but a shared row.
    """
    if not anonymise:
        return species
    return "sp-" + hashlib.sha256(species.encode("utf-8")).hexdigest()[:4]


def table(rows, headers, aligns=None) -> list[str]:
    aligns = aligns or ["---"] * len(headers)
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(aligns) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def next_index() -> int:
    highest = 0
    for path in ARCHIVE.glob("*.md"):
        m = re.match(r"(\d+)", path.name)
        if m:
            highest = max(highest, int(m.group(1)))
    return highest + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, action="append", default=None,
                    help="an embedding run directory holding embeddings.jsonl; "
                         "repeat to compare. Default: every run found on disk")
    ap.add_argument("--report", type=Path, default=REPORT,
                    help=f"fixed output path (default {REPORT.relative_to(PROJECT_ROOT)})")
    ap.add_argument("--snapshot", nargs="?", const="", default=None, metavar="LABEL",
                    help="also file a numbered copy under project/reports/archive/")
    ap.add_argument("--anonymise", action="store_true",
                    help="replace species names with a stable digest, for a copy "
                         "that is going to leave this machine")
    ap.add_argument("--movers", type=int, default=15,
                    help="how many per-species changes to list (default 15)")
    args = ap.parse_args()

    runs = args.run or discover()
    if not runs:
        sys.exit("error: no embeddings.jsonl found. Pass --run, or embed something first.")

    loaded = []
    for run_dir in runs:
        if not (run_dir / "embeddings.jsonl").is_file():
            sys.exit(f"error: no embeddings.jsonl in {run_dir}")
        keys, species, X, info = load(run_dir)
        print(f"  loaded {len(keys):,} vectors from {run_dir} "
              f"({info['model']} @ {info['image_size']})", flush=True)
        loaded.append((run_dir, keys, species, X, info))

    shared = set(loaded[0][1])
    for _, keys, *_ in loaded[1:]:
        shared &= set(keys)
    if not shared:
        sizes = "\n".join(
            f"       {r}: {len(k):,} keys, {len(set(k) & set(loaded[0][1])):,} "
            f"shared with {loaded[0][0].name}" for r, k, *_ in loaded)
        sys.exit("error: these runs share no keys; there is nothing to compare.\n"
                 + sizes + "\n       A partial or sample run cannot be compared "
                 "against a full one -- name the runs you mean with --run.")
    order = sorted(shared)
    print(f"  comparing on {len(order):,} images shared by all {len(loaded)} run(s)",
          flush=True)

    results = []
    for run_dir, keys, species, X, info in loaded:
        index = {k: i for i, k in enumerate(keys)}
        take = np.array([index[k] for k in order])
        labels = species[take]
        stats = score(one_nn_correct(X[take], labels), labels)
        results.append((run_dir, info, stats))
        print(f"  {run_dir}: 1-NN {stats['micro']:.4f}", flush=True)

    lines = [
        "# Embedding quality", "",
        "Leave-one-out 1-NN accuracy against the pipeline's own species labels:",
        "for each image, is its nearest neighbour in vector space the same species?",
        "A property of the vectors alone -- no clustering parameters are involved.",
        "",
        f"Population: **{len(order):,} images** shared by all {len(results)} run(s), "
        f"**{results[0][2]['species']:,} species** -- "
        f"**{results[0][2]['big_species']:,}** with >={MIN_SPECIES} images, "
        f"**{results[0][2]['singletons']:,}** with only one.",
        "",
        f"`1-NN` is over every image; `macro` and `>={MIN_SPECIES}` are over the "
        f"{results[0][2]['big_species']:,} species with at least {MIN_SPECIES} images, "
        "since a single-image species scores zero however good the vectors are.",
        "",
        "The labels are not ground truth -- about 1.4% of the set carries an older",
        "pipeline's over-calls -- so read the *difference* between runs, not the",
        "absolute value.",
        "",
    ]
    if args.anonymise:
        lines += ["Species are shown as a stable digest of the name, not the name.", ""]
    lines += [
        "## Runs", "",
    ]
    lines += table(
        [[f"`{label_of(r)}`", i["image_size"], i["model"].split("/")[-1],
          f"{s['micro']:.4f}", f"{s['macro']:.4f}", f"{s['big_micro']:.4f}"]
         for r, i, s in results],
        ["run", "image size", "model", "1-NN", f"macro>={MIN_SPECIES}", f">={MIN_SPECIES}"],
        ["---", "---", "---", "---:", "---:", "---:"])

    base_dir, _, base = results[0]
    if len(results) > 1:
        lines += ["", f"Baseline for the deltas below: `{label_of(base_dir)}`.", ""]
    for run_dir, info, stats in results[1:]:
        d_micro = stats["micro"] - base["micro"]
        d_macro = stats["macro"] - base["macro"]
        lines += [
            f"## `{label_of(run_dir)}` vs baseline", "",
            f"micro **{d_micro:+.4f}**, macro **{d_macro:+.4f}**, "
            f">={MIN_SPECIES} **{stats['big_micro'] - base['big_micro']:+.4f}**", "",
        ]
        moved = sorted(
            ((s, stats["per_species"][s] - base["per_species"][s], base["counts"][s])
             for s in base["per_species"] if base["counts"][s] >= MIN_SPECIES),
            key=lambda t: t[1])
        worse = [m for m in moved if m[1] < 0][:args.movers]
        better = [m for m in reversed(moved) if m[1] > 0][:args.movers]
        for title, rows in (("Most improved", better), ("Most degraded", worse)):
            if not rows:
                continue
            lines += [f"### {title}", ""]
            lines += table([[name_of(s, args.anonymise), n,
                             f"{base['per_species'][s]:.3f}",
                             f"{stats['per_species'][s]:.3f}", f"{d:+.3f}"]
                            for s, d, n in rows],
                           ["species", "images", "before", "after", "change"],
                           ["---", "---:", "---:", "---:", "---:"])
            lines += [""]

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         text=True, cwd=PROJECT_ROOT).strip()
    except Exception:
        commit = "unknown"
    lines += ["", "---", "",
              f"Generated by `tools/audit_embed_quality.py` at commit `{commit}`.",
              "No timestamp: this file is meant to be diffed between runs.", ""]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> {args.report}")

    if args.snapshot is not None:
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        label = f"-{args.snapshot}" if args.snapshot else ""
        dest = ARCHIVE / f"{next_index():03d}_{args.report.stem}{label}.md"
        dest.write_text("\n".join(lines), encoding="utf-8")
        print(f"  -> {dest}")


if __name__ == "__main__":
    main()
