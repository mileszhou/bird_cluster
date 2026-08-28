#!/usr/bin/env python3
"""Give the existing labelling a taxonomy, by mapping its vocabulary to a checklist.

The labeller writes English common names -- `common raven`, `mallard duckling` --
which carry no rank structure at all. So nothing downstream can ask whether two
species in one cluster are *related*, only whether their strings differ. That is
the gap `project/findings/01` ran into: every target available to it was
appearance-tinged folk taxonomy, and "these look alike" and "these are kin"
predict the same thing.

This resolves each of the ~2,820 vocabulary entries to a Linnaean taxon, so every
labelled image gains a genus, family and order **without changing its label**.
The species column keeps its owner; this is a join table beside it.

**Two independent routes, and their disagreement is the useful output.**

1. *String* -- normalise and match the common name against the checklist. High
   precision, and blind: it cannot match `mallard duckling`, it cannot tell that
   an over-call is one, and a name absent from the checklist is simply missing.
2. *Vote* -- the modal taxon BioCLIP predicted across the images actually
   carrying that label. Blind in the opposite direction: it sees the pixels and
   not the word, so it lands somewhere for every entry including the junk ones.

Neither is ground truth. Where both fire and agree, the mapping is about as good
as it gets here. Where they disagree, something is wrong and the row says so
rather than picking a winner quietly -- typically the label is an over-call
(`silver gull` over a husky), a life stage the checklist does not carry, or a
genuinely hard bird BioCLIP missed.

**Rank matters more than species here.** A vote over 40 images is far steadier at
family than at species, which is the rank the clustering question needs anyway.
The per-rank vote share is recorded so a consumer can set its own bar.

    python3 -m tools.map_label_taxa
    python3 -m tools.map_label_taxa --min-vote 0.5

Outputs, both keyed for joining and both gitignored with the rest of `output/`:
    vocabulary_taxa.csv   one row per label -- the mapping, its evidence, its doubts
    label_taxonomy.csv    one row per image -- jpg, label, and the ranks it inherits
"""

import argparse
import collections
import csv
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, os.environ.get("PROJECT_ROOT")
                or str(Path(__file__).resolve().parents[1]))

from tools.predict_taxa import RANKS, load_checklist  # noqa: E402

# Stripped only on a second pass, after the whole name has failed: these are
# real distinctions a photographer makes and the checklist does not carry, so
# `mallard duckling` should reach Anas platyrhynchos rather than nothing -- but
# stripping first would silently merge `little blue heron` into `little heron`.
QUALIFIERS = ("duckling", "chick", "juvenile", "immature", "fledgling", "nestling",
              "adult", "male", "female", "pair", "breeding", "nonbreeding",
              "in flight", "flying", "swimming", "perched")


def normalise(name: str) -> str:
    """A form two spellings of one bird can meet in.

    Deliberately narrow. Case, accents, punctuation and hyphenation vary freely
    between checklists and labellers and mean nothing; `grey`/`gray` is the one
    lexical variant frequent enough to be worth hard-coding. Anything cleverer
    belongs in the vote, which does not need to guess about words at all.
    """
    s = unicodedata.normalize("NFKD", name.strip().lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("-", " ").replace("'", "").replace("’", "")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\bgray\b", "grey", s)
    return re.sub(r"\s+", " ", s).strip()


def build_tail_index(by_common: dict[str, dict]) -> dict[str, list[tuple[str, dict]]]:
    """Checklist names grouped by their last word.

    A bird's common name almost always ends in the group noun -- gull, grebe,
    warbler -- and an alias differs at the front (`black-headed gull` against the
    checklist's `Common Black-headed Gull`). Bucketing on the tail makes
    containment matching both fast and much less reckless: `grebe` is only ever
    compared against grebes.
    """
    index: dict[str, list[tuple[str, dict]]] = collections.defaultdict(list)
    for name, row in by_common.items():
        if name:
            index[name.rsplit(" ", 1)[-1]].append((name, row))
    return index


def contained_match(norm: str, index, want_family: str = "") -> tuple[dict | None, int]:
    """Checklist entries whose name contains the label, or vice versa.

    An alias is usually the checklist name with a qualifier added or dropped, so
    whole-word containment in either direction catches most of them. It is also
    exactly where a careless match does damage -- `black-billed gull` contains
    neither more nor less of `black-headed gull` than chance -- so containment
    only ever *proposes*: with several candidates the family vote decides, and
    with no vote to decide it, nothing is returned.
    """
    if not norm:
        return None, 0
    cands = []
    for name, row in index.get(norm.rsplit(" ", 1)[-1], ()):
        if re.search(rf"(?:^|\s){re.escape(norm)}(?:\s|$)", name) or \
           re.search(rf"(?:^|\s){re.escape(name)}(?:\s|$)", norm):
            cands.append(row)
    if len(cands) == 1:
        return cands[0], 1
    if len(cands) > 1 and want_family:
        agree = [c for c in cands if c["family"] == want_family]
        if len(agree) == 1:
            return agree[0], len(cands)
    return None, len(cands)


def strip_qualifier(name: str) -> str:
    """`mallard duckling` -> `mallard`. Returns the name unchanged if nothing goes."""
    s = name
    for q in QUALIFIERS:
        s = re.sub(rf"\b{re.escape(q)}\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings", type=Path,
                    default=Path("./output/embed-bioclip/embeddings.jsonl"),
                    help="the population and its labels. Defaults to the BioCLIP "
                         "run because that is the set being clustered; any run "
                         "carrying `key` and `species` works")
    ap.add_argument("--predictions", type=Path,
                    default=Path("./output/taxa/taxa_predictions.csv"),
                    help="per-image taxa from tools.predict_taxa, for the vote")
    ap.add_argument("--output-dir", type=Path, default=Path("./output/taxa"))
    ap.add_argument("--taxon-class", default="Aves")
    ap.add_argument("--min-vote", type=float, default=0.34,
                    help="vote share below which a rank is left blank rather than "
                         "guessed (default 0.34 -- a plurality of a three-way split)")
    args = ap.parse_args()

    # --- the population, and the vocabulary it uses ---
    labels: dict[str, str] = {}
    for line in open(args.embeddings):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        labels[row["key"]] = (row.get("species") or "").strip().lower()
    vocab = collections.Counter(labels.values())
    vocab.pop("", None)
    print(f"  {len(labels):,} images, {len(vocab):,} distinct labels")

    # --- the checklist ---
    rows, _ = load_checklist(args.taxon_class)
    by_common: dict[str, dict] = {}
    for r in rows:
        for form in (r["common_name"], f"{r['genus']} {r['species']}"):
            key = normalise(form or "")
            if key and key not in by_common:      # first wins; the list is stable
                by_common[key] = r
    tail_index = build_tail_index(by_common)
    print(f"  checklist: {len(rows):,} taxa, {len(by_common):,} matchable names")

    # --- route 2: what BioCLIP called the images carrying each label ---
    votes: dict[str, dict[str, collections.Counter]] = collections.defaultdict(
        lambda: {rank: collections.Counter() for rank in RANKS})
    with open(args.predictions, encoding="utf-8-sig", newline="") as fh:
        for pred in csv.DictReader(fh):
            lab = labels.get(pred["jpg"])
            if not lab:
                continue
            for rank in RANKS:
                if pred[rank]:
                    votes[lab][rank][pred[rank]] += 1

    def voted(lab, rank):
        c = votes[lab][rank]
        if not c:
            return "", 0.0
        value, n = c.most_common(1)[0]
        return value, n / sum(c.values())

    # --- resolve, and record how ---
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, dict] = {}
    vocab_method: dict[str, str] = {}
    stats = collections.Counter()

    vocab_path = out_dir / "vocabulary_taxa.csv"
    fields = (["label", "images", "method", "matched_name"] + list(RANKS)
              + ["vote_family", "vote_family_share", "vote_species_share", "agrees"])
    with open(vocab_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for lab, n in vocab.most_common():
            norm = normalise(lab)
            hit = by_common.get(norm)
            method = "string"
            if hit is None:
                stripped = normalise(strip_qualifier(norm))
                hit = by_common.get(stripped) if stripped and stripped != norm else None
                method = "string-qualifier" if hit else None
            if hit is None:
                # Containment, with the family vote as the tie-breaker.
                vf_probe, vfs_probe = voted(lab, "family")
                hit, _ = contained_match(
                    norm, tail_index, vf_probe if vfs_probe >= args.min_vote else "")
                if hit is None:
                    stripped = normalise(strip_qualifier(norm))
                    if stripped and stripped != norm:
                        hit, _ = contained_match(
                            stripped, tail_index,
                            vf_probe if vfs_probe >= args.min_vote else "")
                method = "string-contained" if hit else None

            vf, vfs = voted(lab, "family")
            vsp, vsps = voted(lab, "species")
            agrees = ""
            if hit is not None:
                agrees = "yes" if (vf and vf == hit["family"]) else ("no" if vf else "")
            else:
                # Fall back to the vote, rank by rank, so a label can gain a
                # family without having to win a species it may not have.
                hit = {r: "" for r in RANKS}
                hit["common_name"] = ""
                for rank in RANKS:
                    value, share = voted(lab, rank)
                    if share >= args.min_vote:
                        hit[rank] = value
                method = "vote" if any(hit[r] for r in RANKS) else "none"

            stats[method] += 1
            vocab_method[lab] = method
            if method and method.startswith("string") and agrees == "no":
                stats["string-but-vote-disagrees"] += 1
            mapping[lab] = hit
            writer.writerow({"label": lab, "images": n, "method": method,
                             "matched_name": hit.get("common_name", ""),
                             **{r: hit.get(r, "") for r in RANKS},
                             "vote_family": vf,
                             "vote_family_share": round(vfs, 3),
                             "vote_species_share": round(vsps, 3),
                             "agrees": agrees})

    # --- the per-image join, which is what a cluster review actually reads ---
    img_path = out_dir / "label_taxonomy.csv"
    with open(img_path, "w", newline="", encoding="utf-8-sig") as fh:
        # `method` rides along because a consumer has to know which route a row
        # took. A binomial reached by the vote came from BioCLIP's reading of the
        # pixels, not from the label -- so presenting it beside the label as
        # "what the labeller meant, scientifically" would be two columns showing
        # one opinion. Only the string routes can honestly claim that.
        writer = csv.DictWriter(fh, fieldnames=["jpg", "label", "method"] + list(RANKS))
        writer.writeheader()
        for key, lab in labels.items():
            hit = mapping.get(lab) or {}
            writer.writerow({"jpg": key, "label": lab,
                             "method": (vocab_method.get(lab) or ""),
                             **{r: hit.get(r, "") for r in RANKS}})

    covered = sum(n for lab, n in vocab.items() if mapping.get(lab, {}).get("family"))
    print(f"\n  by method: " + ", ".join(f"{k} {v:,}" for k, v in stats.most_common()))
    print(f"  images with a family: {covered:,} / {len(labels):,} "
          f"({covered / len(labels):.1%})")
    print(f"  -> {vocab_path}\n  -> {img_path}")


if __name__ == "__main__":
    main()
