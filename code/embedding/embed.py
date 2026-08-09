"""Step 1 of the semantic-clustering pipeline: embed the bird photos.

Reads a curated labelling run, keeps the birds, and POSTs batches to the embed
server (`code/embedding/embed_server.py`), appending one JSON object per image
to `embeddings.jsonl`.

    ./run-embed                                     # whatever --years defaults to
    python3 -m code.embedding.embed --years 2019    # one year
    python3 -m code.embedding.embed --years 2019 --dry-run

**The image is the unit, and the CSV is the guide.** The label set comes from
`<label-dir>/bird_identification_output.csv` -- one row per exported image, the
`jpg` column naming it relative to `<data-dir>/jpg`. Nothing here reads a
sidecar. That is not a shortcut, it is the design:

- **5,229 exported images have no sidecar at all**, and 569 of them are birds.
  A sidecar walk cannot enumerate them, so they were invisible to every earlier
  version of this step -- 2% of the clustering set, concentrated in birding
  trips.
- The labelling run already resolved which file backs each capture, including
  the 189 whose only export is decorated (`-Enhanced-NR`, `-2`). Re-deriving it
  here with a second resolver is drift waiting to happen.
- The key has to be the image path for the same reason: a sidecar path cannot
  name an image that has no sidecar.

The `jpg` column is a path, so a non-JPEG export would ride through unchanged;
only the column's name assumes the format.

Dataset layout -- the export mirrors the photo library, so nothing is hardcoded
per year:

    <data-dir>/jpg/<library>/<trip>/*.jpg          # Photos-19/2019-01-13 山公园/_D8S0025.jpg
    <label-dir>/bird_identification_output.csv     # one row per image

`--years 2019` selects by the library's year (`Photos-19`).

Output is JSONL rather than one array so the run is append-friendly and
resumable: the checkpoint is just the set of keys already in the file, matching
the append-mode-CSV resumption `bird_label.py` uses. Re-running skips them.
"""

import argparse
import base64
import csv
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
from code.lib import path_filter
from code.lib.jpg_index import library_year
from code.lib.xmp_labels import parse_label   # parses a CSV string; reads no file

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
    """One bird image to embed. `key` is its path relative to `<data-dir>/jpg`."""

    __slots__ = ("key", "year", "library", "trip", "stem", "xmp", "jpg",
                 "species", "confidence", "applied")

    def __init__(self, key, year, library, trip, stem, xmp, jpg, species,
                 confidence, applied):
        self.key, self.year, self.library, self.trip = key, year, library, trip
        self.stem = stem
        # `xmp` is the capture's sidecar as a string, "" when the image never
        # had one. Carried through so alternate edits of one frame stay
        # groupable downstream; nothing here reads the file.
        self.xmp, self.jpg = xmp, jpg
        self.species, self.confidence = species, confidence
        self.applied = applied


def _overruled(row) -> bool:
    """True when the sidecar kept its prior label and this run's was discarded."""
    return row.get("applied") == "kept-existing" and bool(row.get("prior_category"))


def effective_category(row) -> str:
    """The category the library actually holds, lowercased.

    The CSV's `category` is *this run's verdict*. Where `applied` is
    `kept-existing` the run's answer was overruled and the sidecar kept
    `prior_category` -- `bird` is never demoted by a later non-bird result,
    which is what guards the clustering set against a false negative.

    Resolving it here, once, is deliberate: 554 photos differ between the two
    readings, and a consumer that filters on `category == 'bird'` drops exactly
    the set the never-demote rule exists to protect.
    """
    if _overruled(row):
        return row["prior_category"].strip().lower()
    return (row.get("category") or "").strip().lower()


def effective_species(row) -> tuple[str | None, float | None]:
    """(species, confidence) matching the category `effective_category` returns.

    The same overruling applies to the label, and forgetting it is worse than
    forgetting it for the category, because the result still *looks* like a
    species. On a never-demoted row `label` holds this run's non-bird answer --
    "cherry blossoms in full bloom at w..." -- while the bird is in
    `prior_label` as `bqsy-白秋沙鸭-smew(95%)`. Taking `label` there adds 294
    scene descriptions to the species vocabulary and mislabels 554 vectors.

    `prior_label` is `;`-joined and in the `py-cn-en(NN%)` shape, so the
    confidence comes back with it; the CSV's own `confidence` column belongs to
    the discarded non-bird call.
    """
    if _overruled(row):
        for part in (row.get("prior_label") or "").split(";"):
            label = parse_label(part.strip())
            if label:
                return label.english, label.confidence
        return None, None
    species = (row.get("label") or "").strip().lower() or None
    try:
        return species, float(row.get("confidence") or 0.0)
    except ValueError:
        return species, 0.0


def collect(data_dir: Path, label_csv: Path, years, min_confidence: float,
            categories=frozenset({"bird"}), paths=None):
    """Images the labelling CSV names in `categories`. Returns (candidates, counters).

    `categories=None` takes everything. Embedding the whole library rather than
    the birds is the shape of a real experiment, not a convenience: selecting the
    set *by* the label means label errors in the excluded classes can never be
    found, since they are excluded by construction. See
    `project/ideas/01-rediscover-mislabels-by-clustering.md`.
    """
    stats = Counter()
    per_year = Counter()
    candidates = []
    jpg_root = data_dir / "jpg"

    with open(label_csv, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if "jpg" not in (reader.fieldnames or []):
            sys.exit(f"error: {label_csv} has no 'jpg' column -- it predates the "
                     f"image-keyed schema and names sidecars rather than images. "
                     f"Point --label-dir at a current labelling run.")
        for row in reader:
            stats["rows"] += 1
            rel = (row.get("jpg") or "").strip()
            if not rel:
                stats["no_image_path"] += 1
                continue

            parts = Path(rel).parts
            library = parts[0]
            year = library_year(library)
            if year is None or (years and year not in years):
                continue
            if paths is not None and not paths.allows(rel):
                stats["filtered_out"] += 1
                continue
            stats["in_scope"] += 1

            cat = effective_category(row)
            if categories is not None and cat not in categories:
                continue
            stats["selected"] += 1
            stats[f"cat_{cat or 'none'}"] += 1

            species, conf = effective_species(row)
            if conf is not None and conf < min_confidence:
                stats["below_min_confidence"] += 1
                continue

            jpg = jpg_root / rel
            if not jpg.is_file():
                # The CSV names an image the tree no longer has -- a re-export
                # dropped it, or the CSV predates the current data/.
                stats["missing_image"] += 1
                continue

            candidates.append(Candidate(
                key=rel,
                year=year,
                library=library,
                trip=str(Path(*parts[1:-1])) if len(parts) > 2 else "",
                stem=Path(rel).stem,
                xmp=(row.get("xmp") or "").strip(),
                jpg=jpg,
                species=species,
                confidence=conf,
                applied=(row.get("applied") or "").strip(),
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
    logger.info(f"  CSV rows              : {stats['rows']}")
    logger.info(f"  in the selected years : {stats['in_scope']}")
    logger.info(f"  selected (effective)  : {stats['selected']}")
    for key in sorted(k for k in stats if k.startswith("cat_")):
        logger.info(f"      {key[4:]:<18}: {stats[key]}")
    if stats["below_min_confidence"]:
        logger.info(f"  dropped, low conf     : {stats['below_min_confidence']}")
    logger.info(f"  to embed              : {stats['resolved']}")
    if stats["missing_image"]:
        logger.warning(f"  NAMED BUT NOT ON DISK : {stats['missing_image']} -- the CSV "
                       f"and data/jpg disagree; re-run ./run-audit")
    if stats["filtered_out"]:
        logger.info(f"  excluded by path list : {stats['filtered_out']}")
    if stats["no_image_path"]:
        logger.warning(f"  rows with no jpg path : {stats['no_image_path']}")
    if per_year:
        logger.info("  by year: " + ", ".join(f"{y}={per_year[y]}" for y in sorted(per_year)))
    species = {c.species for c in candidates if c.species}
    logger.info(f"  distinct species      : {len(species)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("./data"))
    ap.add_argument("--label-dir", type=Path, default=Path("./data/label"),
                    help="a curated labelling run; its bird_identification_output.csv "
                         "is the guide (default: ./data/label)")
    ap.add_argument("--output-dir", type=Path, default=Path("./output/embed"))
    ap.add_argument("--years", default="2019",
                    help="comma-separated years to include, or 'all' (default: 2019). "
                         "Start with one year; widen once the run looks right.")
    ap.add_argument("--embed-url", default=None,
                    help="default: config.toml [servers.embed]")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="stop after N images (0 = no limit)")
    ap.add_argument("--include-from", type=Path, default=None,
                    help="file of paths relative to data/jpg; only these are embedded. "
                         "A folder line takes its whole subtree. See code/lib/path_filter.py")
    ap.add_argument("--exclude-from", type=Path, default=None,
                    help="file of paths relative to data/jpg to skip; exclude wins over "
                         "include")
    ap.add_argument("--categories", default="bird",
                    help="comma-separated effective categories to embed, or 'all' "
                         "(default: bird). `all` is what makes label errors in the "
                         "non-bird classes discoverable -- filtering by label means "
                         "its mistakes there are excluded by construction")
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="skip birds whose label confidence is below this. Note the "
                         "model's confidence is not calibrated -- 99%% of rows sit at "
                         "0.95 or 0.98 -- so this filters far less than it appears to")
    ap.add_argument("--encode-workers", type=int, default=8,
                    help="threads used to read and base64 the JPEGs of a batch")
    ap.add_argument("--dry-run", action="store_true",
                    help="scan and resolve only; no network calls, no output written")
    args = ap.parse_args()

    years = None if args.years.strip().lower() == "all" else [
        y.strip() for y in args.years.split(",") if y.strip()
    ]

    if not (args.data_dir / "jpg").is_dir():
        sys.exit(f"error: expected {args.data_dir}/jpg to exist")
    label_csv = args.label_dir / "bird_identification_output.csv"
    if not label_csv.is_file():
        sys.exit(f"error: {label_csv} not found. --label-dir must point at a curated "
                 f"labelling run; see 'a person curates into data/' in CLAUDE.md.")

    logger.info(f"reading {label_csv} for years="
                f"{'all' if years is None else ','.join(years)}")
    categories = None if args.categories.strip().lower() == "all" else frozenset(
        c.strip().lower() for c in args.categories.split(",") if c.strip())
    logger.info(f"categories: {'all' if categories is None else ','.join(sorted(categories))}")
    paths = path_filter.build(args.include_from, args.exclude_from)
    if paths:
        logger.info(f"path filter: {paths.describe()}")
    candidates, stats, per_year = collect(args.data_dir, label_csv, years,
                                          args.min_confidence, categories, paths)
    report(stats, per_year, candidates, "scan")

    if args.dry_run:
        logger.info("dry run: nothing embedded, nothing written")
        return

    if not candidates:
        sys.exit("error: nothing to embed -- check --years, --label-dir and --data-dir")

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
        "categories": args.categories,
        "include_from": str(args.include_from) if args.include_from else None,
        "exclude_from": str(args.exclude_from) if args.exclude_from else None,
        "embed_url": embed_url, "server": info, "model": model,
        "data_dir": str(args.data_dir), "label_csv": str(label_csv),
        "candidates": len(candidates),
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
                    "jpg_path": str(cand.jpg), "xmp": cand.xmp,
                    "species": cand.species, "confidence": cand.confidence,
                    "applied": cand.applied, "model": model, "embedding": vector,
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
