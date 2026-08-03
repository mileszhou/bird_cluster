"""Step 1 of the semantic-clustering pipeline: embed the bird photos.

Walks the sidecar tree, keeps the ones bird_label categorised as `bird`,
resolves each to its exported JPEG, and POSTs batches to the embed server
(`code/embedding/embed_server.py`), appending one JSON object per image to
`embeddings.jsonl`.

    ./run-embed                                     # whatever --years defaults to
    python3 -m code.embedding.embed --years 2019    # one year
    python3 -m code.embedding.embed --years 2019 --dry-run

Dataset layout -- the export mirrors the photo library, so nothing is hardcoded
per year:

    <data-dir>/xmp/<library>/<trip>/*.xmp     # Photos-19/2019-01-13 山公园/_D8S0025.xmp
    <data-dir>/jpg/<library>/<trip>/*.jpg     # same folder, same stem

`--years 2019` selects by the library's year (`Photos-19`).

Categories come from each sidecar's own keywords, not from
`bird_identification_output.csv` -- see `code/lib/xmp_labels.py` for why the CSV
is not safe to key by basename.

Where a capture was exported more than once -- a master plus its Lightroom
virtual copies -- only the first export is embedded. The copies are alternate
edits of one frame, so embedding each would pile near-duplicate vectors into a
density-based clustering step for no extra information.

Output is JSONL rather than one array so the run is append-friendly and
resumable: the checkpoint is just the set of keys already in the file, matching
the append-mode-CSV resumption `bird_label.py` uses. Re-running skips them.
"""

import argparse
import base64
import json
import logging
import signal
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from code.lib.config import server_url
from code.lib.jpg_index import JpgIndex, Verdict, library_year
from code.lib.xmp_labels import read_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("embed")

# Set by SIGINT so the in-flight batch finishes and gets written before exit,
# rather than tearing down mid-write. Same deferred-handler approach as
# bird_label.py's batch loop.
_INTERRUPTED = False


def _on_sigint(signum, frame):
    global _INTERRUPTED
    _INTERRUPTED = True
    logger.warning("interrupt received -- finishing current batch, then stopping")


class Candidate:
    """One bird sidecar that resolved to a JPEG."""

    __slots__ = ("key", "year", "library", "trip", "xmp", "jpg", "species",
                 "confidence", "verdict", "copies")

    def __init__(self, key, year, library, trip, xmp, jpg, species, confidence,
                 verdict, copies=1):
        self.key, self.year, self.library, self.trip = key, year, library, trip
        self.xmp, self.jpg = xmp, jpg
        self.species, self.confidence, self.verdict = species, confidence, verdict
        self.copies = copies  # exported files for this capture; only jpg is embedded

    @property
    def stem(self):
        """Camera filename stem, shared by the sidecar and its JPEG."""
        return self.xmp.stem


def library_dirs(data_dir: Path, years: list[str] | None):
    """(year, library folder) pairs under `xmp/`, filtered by `--years`."""
    out = []
    for d in sorted((data_dir / "xmp").iterdir()):
        if not d.is_dir():
            continue
        year = library_year(d.name)
        if year is None or (years and year not in years):
            continue
        out.append((year, d))
    return out


def collect(data_dir: Path, years, min_confidence: float):
    """Find every bird sidecar with a resolvable JPEG. Returns (candidates, counters)."""
    index = JpgIndex(data_dir / "xmp", data_dir / "jpg")
    stats = Counter()
    per_year = Counter()
    candidates = []
    wanted = {d.name for _, d in library_dirs(data_dir, years)}

    for xmp in index.sidecars():
        rel = xmp.relative_to(data_dir / "xmp")
        library = rel.parts[0]
        if library not in wanted:
            continue
        stats["xmp"] += 1
        labels = read_labels(xmp)
        if labels is None:
            stats["unparseable_xmp"] += 1
            continue
        if not labels.is_bird:
            continue
        stats["bird"] += 1

        if labels.label is None:
            stats["bird_without_label"] += 1
        elif labels.label.confidence < min_confidence:
            stats["below_min_confidence"] += 1
            continue

        match = index.resolve(xmp)
        stats[f"verdict_{match.verdict.value}"] += 1
        if not match.ok:
            stats["unresolved"] += 1
            continue
        if len(match.paths) > 1:
            stats["multiple_copies"] += 1

        year = library_year(library)
        candidates.append(Candidate(
            key=rel.as_posix(),
            year=year,
            library=library,
            trip=str(Path(*rel.parts[1:-1])) if len(rel.parts) > 2 else "",
            xmp=xmp, jpg=match.path,
            species=labels.species,
            confidence=labels.label.confidence if labels.label else None,
            verdict=match.verdict.value,
            copies=len(match.paths),
        ))
        per_year[year] += 1
        stats["resolved"] += 1

    return candidates, stats, per_year


def already_done(path: Path) -> tuple[set[str], set[str]]:
    """Keys already embedded, and the set of models that produced them.

    The models matter: vectors from two different backbones live in unrelated
    spaces, so appending one to a file of the other would silently poison every
    distance the clustering step computes.
    """
    if not path.is_file():
        return set(), set()
    done, models = set(), set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done.add(row["key"])
            except (json.JSONDecodeError, KeyError):
                continue
            models.add(row.get("model") or "(unrecorded)")
    return done, models


def encode(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def probe(embed_url: str):
    try:
        r = requests.get(f"{embed_url}/health", timeout=30)
        r.raise_for_status()
        info = r.json()
        logger.info(f"embed server: model={info.get('model')} device={info.get('device')} "
                    f"dim={info.get('dim')}")
        return info
    except requests.RequestException as exc:
        sys.exit(f"error: embed server at {embed_url} unreachable ({exc})\n"
                 f"       start it with ./run-server-embed on the GPU host")


def embed_batch(embed_url: str, batch, workers: int, retries: int = 3):
    """POST one batch, returning a list of vectors aligned with `batch`."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        payload = list(pool.map(lambda c: encode(c.jpg), batch))

    for attempt in range(1, retries + 1):
        try:
            r = requests.post(f"{embed_url}/embed", json={"images": payload}, timeout=600)
            r.raise_for_status()
            vectors = r.json()["embeddings"]
            if len(vectors) != len(batch):
                raise ValueError(f"server returned {len(vectors)} vectors for {len(batch)} images")
            return vectors
        except (requests.RequestException, ValueError, KeyError) as exc:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            logger.warning(f"batch failed ({exc}); retrying in {wait}s "
                           f"[{attempt}/{retries - 1}]")
            time.sleep(wait)


def report(stats: Counter, per_year: Counter, candidates, title):
    logger.info(f"--- {title} ---")
    logger.info(f"  sidecars scanned      : {stats['xmp']}")
    logger.info(f"  bird sidecars         : {stats['bird']}")
    if stats["below_min_confidence"]:
        logger.info(f"  dropped, low conf     : {stats['below_min_confidence']}")
    if stats["bird_without_label"]:
        logger.info(f"  bird, no parsed label : {stats['bird_without_label']}")
    logger.info(f"  resolved to a JPEG    : {stats['resolved']}")
    if stats["multiple_copies"]:
        logger.info(f"      exported >1 time  : {stats['multiple_copies']} "
                    f"(virtual copies; first export embedded)")
    logger.info(f"  unresolved            : {stats['unresolved']}")
    for verdict in Verdict:
        n = stats.get(f"verdict_{verdict.value}", 0)
        if n:
            logger.info(f"      {verdict.value:<24}: {n}")
    if per_year:
        logger.info("  by year: " + ", ".join(f"{y}={per_year[y]}" for y in sorted(per_year)))
    species = {c.species for c in candidates if c.species}
    logger.info(f"  distinct species      : {len(species)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("./data"))
    ap.add_argument("--output-dir", type=Path, default=Path("./output/embed"))
    ap.add_argument("--years", default="2019",
                    help="comma-separated years to include, or 'all' (default: 2019). "
                         "Start with one year; widen once the run looks right.")
    ap.add_argument("--embed-url", default=None,
                    help="default: config.toml [servers.embed]")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="stop after N images (0 = no limit)")
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="skip bird sidecars whose label confidence is below this")
    ap.add_argument("--encode-workers", type=int, default=8,
                    help="threads used to read and base64 the JPEGs of a batch")
    ap.add_argument("--dry-run", action="store_true",
                    help="scan and resolve only; no network calls, no output written")
    args = ap.parse_args()

    years = None if args.years.strip().lower() == "all" else [
        y.strip() for y in args.years.split(",") if y.strip()
    ]

    if not (args.data_dir / "xmp").is_dir() or not (args.data_dir / "jpg").is_dir():
        sys.exit(f"error: expected {args.data_dir}/xmp and {args.data_dir}/jpg to exist")

    logger.info(f"scanning {args.data_dir} for years="
                f"{'all' if years is None else ','.join(years)}")
    candidates, stats, per_year = collect(args.data_dir, years, args.min_confidence)
    report(stats, per_year, candidates, "scan")

    if args.dry_run:
        logger.info("dry run: nothing embedded, nothing written")
        return

    if not candidates:
        sys.exit("error: nothing to embed -- check --years and --data-dir")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "embeddings.jsonl"
    done, prior_models = already_done(out_path)
    todo = [c for c in candidates if c.key not in done]
    if done:
        logger.info(f"resuming: {len(done)} already in {out_path}, {len(todo)} to go")

    embed_url = args.embed_url or server_url("embed")
    info = probe(embed_url)
    model = info.get("model") or "(unknown)"

    # Never append vectors from one backbone to a file written by another --
    # they are not in the same space, and nothing downstream could detect it.
    foreign = prior_models - {model}
    if foreign:
        sys.exit(
            f"error: {out_path} already holds embeddings from {sorted(foreign)}, but the "
            f"server is serving {model!r}.\n"
            f"       Mixing backbones would corrupt every distance downstream. Either point "
            f"--output-dir at a new directory, or delete {out_path} to re-embed from scratch."
        )

    if args.limit:
        todo = todo[:args.limit]
        logger.info(f"--limit {args.limit}: embedding {len(todo)} this run")
    if not todo:
        logger.info("nothing left to do")
        return

    (args.output_dir / "run.json").write_text(json.dumps({
        "years": years,
        "batch_size": args.batch_size, "min_confidence": args.min_confidence,
        "embed_url": embed_url, "server": info, "model": model,
        "data_dir": str(args.data_dir), "candidates": len(candidates),
    }, indent=2))

    signal.signal(signal.SIGINT, _on_sigint)
    written = failed = 0
    t0 = time.time()
    with open(out_path, "a") as fh:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start:start + args.batch_size]
            try:
                vectors = embed_batch(embed_url, batch, args.encode_workers)
            except Exception as exc:
                failed += len(batch)
                logger.error(f"batch at offset {start} failed permanently: {exc}")
                if _INTERRUPTED:
                    break
                continue

            for cand, vector in zip(batch, vectors):
                fh.write(json.dumps({
                    "key": cand.key, "year": cand.year, "library": cand.library,
                    "trip": cand.trip, "stem": cand.stem,
                    "jpg_path": str(cand.jpg), "xmp_path": str(cand.xmp),
                    "species": cand.species, "confidence": cand.confidence,
                    "match": cand.verdict, "model": model, "embedding": vector,
                }, ensure_ascii=False) + "\n")
            fh.flush()
            written += len(batch)

            rate = written / max(time.time() - t0, 1e-6)
            logger.info(f"{written}/{len(todo)} embedded ({rate:.1f} img/s)")

            if _INTERRUPTED:
                logger.warning("stopping after completed batch")
                break

    logger.info(f"wrote {written} embeddings to {out_path}"
                + (f" ({failed} failed)" if failed else ""))


if __name__ == "__main__":
    main()
