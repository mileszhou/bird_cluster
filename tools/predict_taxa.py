#!/usr/bin/env python3
"""Zero-shot Linnaean taxonomy for an embedding run, from BioCLIP's text tower.

**No pass over pixels.** A CLIP-style backbone puts images and label text in one
space, so a classification is `image_vector @ text_vector`. The image vectors are
already on disk -- `/embed` returns exactly `encode_image` output -- so the whole
library is classified against a new checklist for the cost of one matmul. That is
the argument for having embedded with BioCLIP at all: re-running the checklist is
minutes, not GPU-hours, and a *different* checklist is a different question asked
of the same vectors.

**This is a labelling, and it is kept as one.** The prediction does not go into
`embeddings.jsonl`. A clustering artifact that carried a frozen `species` column
is exactly the bug that put one bird in a JPEG and another in the index beside it
across 70% of an export; the fix was one owner per fact. So this writes its own
CSV keyed by the `jpg` path, joinable to the label CSV and to any clustering, and
owned by nobody else.

**It is a third opinion, not an oracle.** BioCLIP is supervised on Linnaean names
and so is a genuinely decorrelated reading against two VLMs -- which is the
property `project/findings/03` says makes a second labelling useful for bounding
error. It is not ground truth: the checklist is a vocabulary, not a verdict, and
a confident wrong answer looks identical to a right one.

    python3 -m tools.predict_taxa                        # the live run
    python3 -m tools.predict_taxa --run output/embed-bioclip --top-k 5
    python3 -m tools.predict_taxa --taxon-class Mammalia # a different checklist

The checklist is TreeOfLife-200M's own taxon list -- the vocabulary the model was
trained against, which is the right label space for zero-shot: a name the text
tower never saw is a name it cannot rank fairly. Restricting to a class is the
single largest accuracy lever available here, and costs nothing: 867,455 taxa
down to 11,131 for Aves, which is 98% of the label space removed before the
model has to discriminate anything.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, os.environ.get("PROJECT_ROOT")
                or str(Path(__file__).resolve().parents[1]))

from code.lib.config import PROJECT_ROOT, server_url  # noqa: E402

TOL_REPO = "imageomics/TreeOfLife-200M"
TOL_FILE = "embeddings/txt_emb_species.json"
RANKS = ("kingdom", "phylum", "class", "order", "family", "genus", "species")

# Both knobs were measured on this collection rather than assumed, and the
# finding is that neither matters. Over a random 6,000-image sample the four
# text forms agree with the existing labelling to within 0.002 of each other
# (0.2922 to 0.2942); prompt ensembling over the 80 OpenAI ImageNet templates
# that lift generic CLIP moved nothing either, which is unsurprising for a model
# fine-tuned on taxonomic strings -- the paraphrases blur the text distribution
# it actually learned. So one template, chosen because it is 80x cheaper and
# nothing measured argues for the alternative.
#
# Measure on a *random* sample if this is ever revisited. The JSONL is written
# in path order, so its first rows are one or two years and a third of the
# species; scored on the head of the file every one of these numbers reads about
# 0.45, and the ranking between the forms changes too.
TEMPLATE = "a photo of {}."
TEXT_FORM = "taxa+common"


def taxon_text(sci: list[str], common: str, form: str = TEXT_FORM) -> str:
    """The string handed to the text tower for one taxon.

    Empty ranks are dropped rather than padded: a fossil genus with no order or
    family would otherwise get two blank tokens in the middle of its name.
    """
    ranks = " ".join(x for x in sci if x)
    binom = " ".join(x for x in sci[-2:] if x)
    return {"taxa+common": f"{ranks} {common}".strip(),
            "taxa": ranks,
            "common": common or binom,
            "binom+common": f"{binom} {common}".strip()}[form]


def load_checklist(taxon_class: str | None):
    """(rows, texts) for the label space, from the model's own training taxonomy."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(TOL_REPO, TOL_FILE, repo_type="dataset")
    names = json.loads(Path(path).read_text(encoding="utf-8"))
    if taxon_class:
        names = [n for n in names if n[0][2] == taxon_class]
    if not names:
        sys.exit(f"error: no taxa in class {taxon_class!r}. "
                 f"It is a Linnaean class name, e.g. Aves, Mammalia, Insecta.")
    rows = [{**dict(zip(RANKS, n[0])), "common_name": n[1]} for n in names]
    return rows, [taxon_text(n[0], n[1]) for n in names]


def load_run(run_dir: Path):
    """(keys, vectors, run.json) for an embedding run."""
    jsonl = run_dir / "embeddings.jsonl"
    if not jsonl.is_file():
        sys.exit(f"error: no embeddings.jsonl in {run_dir}")
    keys, vecs = [], []
    with open(jsonl) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            keys.append(row["key"])
            vecs.append(row["embedding"])
    meta = {}
    if (run_dir / "run.json").is_file():
        meta = json.loads((run_dir / "run.json").read_text())
    return keys, np.asarray(vecs, dtype=np.float32), meta


def embed_texts(url: str, texts: list[str], batch: int = 256) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch):
        chunk = [TEMPLATE.format(t) for t in texts[i:i + batch]]
        resp = requests.post(f"{url}/embed_text", json={"texts": chunk}, timeout=600)
        if resp.status_code == 501:
            sys.exit(f"error: {resp.json().get('detail')}\n"
                     f"       Serve a CLIP-style backbone: "
                     f"MODEL=hf-hub:imageomics/bioclip-2.5-vith14 ./server-embed")
        resp.raise_for_status()
        out.extend(resp.json()["embeddings"])
        print(f"\r  encoded {min(i + batch, len(texts)):,}/{len(texts):,} taxa",
              end="", flush=True)
    print()
    return np.asarray(out, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=Path("./output/embed-bioclip"),
                    help="embedding run holding the image vectors to classify")
    ap.add_argument("--output-dir", type=Path, default=Path("./output/taxa"))
    ap.add_argument("--embed-url", default=None,
                    help="default: config.toml [servers.embed]")
    ap.add_argument("--taxon-class", default="Aves",
                    help="restrict the checklist to one Linnaean class -- the "
                         "single largest accuracy lever here. Pass '' for all "
                         "867,455 taxa (default: Aves)")
    ap.add_argument("--top-k", type=int, default=3,
                    help="how many candidates to record per image (default 3). "
                         "The runner-up is what makes the margin readable")
    ap.add_argument("--force", action="store_true",
                    help="classify even if the vectors came from a different "
                         "model than the server serves. Almost always wrong")
    args = ap.parse_args()

    url = (args.embed_url or server_url("embed")).rstrip("/")
    health = requests.get(f"{url}/health", timeout=30).json()
    keys, X, meta = load_run(args.run)
    print(f"  {len(keys):,} image vectors from {args.run} "
          f"({meta.get('model')} @ {meta.get('image_size')})")

    # The one guard that matters. Image and text vectors are only in the same
    # space if one model produced both -- classify DINOv3 vectors against
    # BioCLIP's text tower and every row comes back confidently meaningless,
    # with nothing downstream able to notice. This is the same check `embed.py`
    # makes before appending to a JSONL, for the same reason.
    if meta.get("model") != health.get("model") and not args.force:
        sys.exit(f"error: these vectors are from {meta.get('model')!r} but the "
                 f"server serves {health.get('model')!r}.\n"
                 f"       Image and text vectors from two models do not share a "
                 f"space; the result would look fine and mean nothing.\n"
                 f"       Serve the matching model, or pass --force if you are "
                 f"certain.")

    rows, texts = load_checklist(args.taxon_class or None)
    print(f"  checklist: {len(rows):,} taxa"
          + (f" in class {args.taxon_class}" if args.taxon_class else "")
          + f", {len({r['family'] for r in rows})} families, "
            f"{len({r['order'] for r in rows})} orders")
    T = embed_texts(url, texts)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    k = min(args.top_k, len(rows))
    fields = (["jpg"] + list(RANKS) + ["common_name", "score", "margin"]
              + [f"alt{i}_common" for i in range(2, k + 1)]
              + [f"alt{i}_score" for i in range(2, k + 1)])

    csv_path = out_dir / "taxa_predictions.csv"
    # utf-8-sig: every CSV this project writes carries a BOM so Excel reads the
    # Chinese trip names in the key rather than the system codepage.
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i in range(0, len(X), 1024):
            sim = X[i:i + 1024] @ T.T
            idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            order = np.take_along_axis(sim, idx, 1).argsort(axis=1)[:, ::-1]
            idx = np.take_along_axis(idx, order, 1)
            top = np.take_along_axis(sim, idx, 1)
            for j in range(idx.shape[0]):
                best = rows[idx[j, 0]]
                rec = {"jpg": keys[i + j], **{r: best[r] for r in RANKS},
                       "common_name": best["common_name"],
                       "score": round(float(top[j, 0]), 4),
                       "margin": round(float(top[j, 0] - top[j, 1]), 4)}
                for a in range(1, k):
                    rec[f"alt{a + 1}_common"] = rows[idx[j, a]]["common_name"]
                    rec[f"alt{a + 1}_score"] = round(float(top[j, a]), 4)
                writer.writerow(rec)
            print(f"\r  classified {min(i + 1024, len(X)):,}/{len(X):,}",
                  end="", flush=True)
    print()

    (out_dir / "run.json").write_text(json.dumps({
        "source": str((args.run / "embeddings.jsonl").resolve()),
        "images": len(keys), "checklist_repo": TOL_REPO, "checklist_file": TOL_FILE,
        "taxon_class": args.taxon_class, "taxa": len(rows),
        "text_form": TEXT_FORM, "template": TEMPLATE, "top_k": k,
        "embed_url": url, "server": health,
    }, indent=2))
    print(f"  -> {csv_path}")


if __name__ == "__main__":
    main()
