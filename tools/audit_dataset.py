#!/usr/bin/env python3
"""
Audit a bird_label dataset: XMP sidecars <-> JPEG exports <-> result CSVs.

Reads the dataset layout produced by bird_label runs:

    <data-dir>/xmp/result-<YYYY>/
        bird_identification_output.csv      # per-year results (bare basenames)
        raw/<half-year>/<trip>/*.xmp        # sidecars, keywords injected
    <data-dir>/jpg/<half-year>/*.jpg        # flat export, no trip nesting

and reports, per year and per half-year folder, how many bird-categorised
sidecars resolve to a JPEG, plus the ones that do not and why. Nothing is
written back into the dataset -- this is read-only.

Category and species come from each sidecar's own dc:subject keywords via
`code/lib/xmp_labels.py`, not from the CSV; JPEG resolution uses the same
`code/lib/jpg_index.py` the embedding pipeline uses, so the "usable" count here
is exactly what a run with `--match-policy expected` would see.

Usage:
    python3 tools/audit_dataset.py                       # report to stdout
    python3 tools/audit_dataset.py --report docs/reports/01-data_preview_report.md \
                                  --issues-dir output/audit
"""
import argparse
import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.lib.jpg_index import JpgIndex, MatchPolicy, Verdict  # noqa: E402
from code.lib.xmp_labels import CATEGORIES, read_labels  # noqa: E402

# JpgIndex splits off-folder matches into same-year (export-boundary offset,
# same shoot) and cross-year (counter wraparound, wrong photo). The report keeps
# that split because they mean opposite things.
OK = Verdict.OK.value
OFF_SAME = Verdict.OFF_FOLDER_SAME_YEAR.value
OFF_CROSS = Verdict.OFF_FOLDER_CROSS_YEAR.value
AMBIG = Verdict.AMBIGUOUS.value
NO_JPG = Verdict.NO_JPG.value
BAD = (OFF_SAME, OFF_CROSS, AMBIG, NO_JPG)


def csv_missing_jpeg_stems(year_dir):
    """Basenames the original run recorded as 'missing JPEG'.

    Basename-keyed, so this is only a hint for a follow-up re-run, never used to
    decide a category.
    """
    path = year_dir / "bird_identification_output.csv"
    if not path.is_file():
        return set()
    out = set()
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if "missing jpeg" in (row.get("note") or "").lower():
                out.add(Path(row["filename"]).stem)
    return out


def audit(data_dir):
    xmp_root, jpg_root = data_dir / "xmp", data_dir / "jpg"
    if not xmp_root.is_dir() or not jpg_root.is_dir():
        sys.exit(f"error: expected {xmp_root} and {jpg_root} to exist")

    # SAME_YEAR so find() reports the finest verdict; the report classifies on
    # the verdict itself rather than on whether this policy accepted it.
    index = JpgIndex(jpg_root, policy=MatchPolicy.SAME_YEAR)
    jpg_folders = sorted(p.name for p in jpg_root.iterdir() if p.is_dir())

    result = {
        "jpg_total": sum(len(v) for v in index.stems.values()),
        "jpg_unique_stems": len(index.stems),
        "jpg_folders": jpg_folders,
        "colliding_stems": {s: sorted({p.parent.name for p in paths})
                            for s, paths in index.colliding_stems().items()},
        "years": {},
        "issues": [],
        "species": collections.Counter(),
        "unparsable": [],
    }

    for year_dir in sorted(xmp_root.glob("result-*")):
        year = year_dir.name[len("result-"):]
        raw = year_dir / "raw"
        if not raw.is_dir():
            continue
        missing_csv = csv_missing_jpeg_stems(year_dir)

        stats = collections.Counter()
        by_folder = collections.defaultdict(collections.Counter)
        species = collections.Counter()
        recoverable = 0

        for xmp in sorted(raw.rglob("*.xmp")):
            stats["xmp"] += 1
            labels = read_labels(xmp)
            if labels is None:
                result["unparsable"].append(str(xmp.relative_to(data_dir)))
                continue
            if not labels.subjects:
                stats["no_keywords"] += 1
            for cat in labels.categories:
                stats[f"cat_{cat}"] += 1

            match = index.find(xmp, raw)
            if xmp.stem in missing_csv and match.verdict is Verdict.OK:
                recoverable += 1

            if not labels.is_bird:
                continue
            stats["bird"] += 1

            rel = xmp.relative_to(raw)
            folder = rel.parts[0] if len(rel.parts) > 1 else "(root)"
            by_folder[folder]["bird"] += 1

            if labels.label:
                stats["labelled"] += 1
                species[labels.species] += 1
                result["species"][labels.species] += 1

            verdict = match.verdict.value
            stats[verdict] += 1
            by_folder[folder][verdict] += 1
            if verdict != OK:
                result["issues"].append({
                    "year": year, "half_year": folder,
                    "trip": rel.parts[1] if len(rel.parts) > 2 else "(none)",
                    "stem": xmp.stem, "verdict": verdict,
                    "expected_folder": match.expected_folder or "",
                    "found_folders": ";".join(sorted({p.parent.name for p in match.candidates})),
                    "species": labels.species or "",
                    "in_csv_as_missing_jpeg": xmp.stem in missing_csv,
                })

        result["years"][year] = {
            "stats": stats, "by_folder": dict(by_folder), "species": species,
            "missing_jpeg_rows": len(missing_csv), "recoverable": recoverable,
        }
    return result


def pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "-"


def render(result, data_dir):
    L = []
    w = L.append
    years = result["years"]
    tot = collections.Counter()
    for y in years.values():
        tot.update(y["stats"])

    w("# Data preview / audit report")
    w("")
    w(f"Generated by `tools/audit_dataset.py` over `{data_dir}`. Read-only; the dataset is not")
    w("modified. Re-run any time to refresh -- it takes ~15s over 35k sidecars.")
    w("")
    w("## Layout assumed")
    w("")
    w("```")
    w("<data-dir>/xmp/result-<YYYY>/bird_identification_output.csv")
    w("<data-dir>/xmp/result-<YYYY>/raw/<half-year>/<trip>/*.xmp")
    w("<data-dir>/jpg/<half-year>/*.jpg          # flat, no trip nesting")
    w("```")
    w("")
    w("A sidecar's JPEG is looked up in the jpg folder whose name equals the sidecar's")
    w("half-year folder -- `raw/2023.1/<trip>/_X.xmp` -> `jpg/2023.1/_X.jpg`. Category and")
    w("species come from the sidecar's own `dc:subject` keywords (`code/lib/xmp_labels.py`),")
    w("never from the CSV: the CSV's `filename` column is a bare basename with no folder, and")
    w("basenames repeat within a year, so a basename-keyed lookup silently mixes up rows.")
    w("Measured duplicate basenames per year's CSV: 2018=180, 2019=76, 2021=429, 2023=2,")
    w("2024=347, 2025=484 -- and of those, 89 / 4 / 81 / 0 / 153 / 119 respectively have")
    w("duplicates that *disagree* on category.")
    w("")

    w("## Totals")
    w("")
    w(f"- JPEGs indexed: **{result['jpg_total']}** in {len(result['jpg_folders'])} folders, "
      f"**{result['jpg_unique_stems']}** unique stems")
    w(f"- Stems present in more than one jpg folder: **{len(result['colliding_stems'])}** "
      f"({pct(len(result['colliding_stems']), result['jpg_unique_stems'])} of unique stems) "
      f"-- camera counter wraparound")
    w(f"- Sidecars scanned: **{tot['xmp']}**")
    w(f"- Bird-categorised sidecars: **{tot['bird']}**, of which **{tot['labelled']}** carry a "
      f"parseable `py-cn-en(NN%)` label")
    w(f"- **Resolved in the expected folder: {tot[OK]}** ({pct(tot[OK], tot['bird'])} of bird "
      f"sidecars) -- the usable set under `--match-policy expected`")
    w(f"- Plus **{tot[OFF_SAME]}** same-year off-folder matches, accepted under the default "
      f"`--match-policy same_year`, giving **{tot[OK] + tot[OFF_SAME]}** usable")
    w(f"- Rejected: **{tot[OFF_CROSS]}** cross-year (wraparound -- wrong photo), "
      f"**{tot[AMBIG]}** ambiguous, **{tot[NO_JPG]}** no JPEG anywhere")
    w(f"- Distinct English species names: **{len(result['species'])}**")
    if result["unparsable"]:
        w(f"- Sidecars that would not parse: {len(result['unparsable'])}")
    w("")

    w("## Per year")
    w("")
    w("`usable` = expected-folder hits. `+same-yr` = additionally accepted by the default")
    w("policy. `cross-yr` = silently wrong if a whole-tree stem fallback is used.")
    w("")
    w("| year | xmp | bird | labelled | usable | +same-yr | cross-yr | ambig | no jpg | species |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for year in sorted(years):
        s = years[year]["stats"]
        w(f"| {year} | {s['xmp']} | {s['bird']} | {s['labelled']} | **{s[OK]}** | {s[OFF_SAME]} | "
          f"{s[OFF_CROSS]} | {s[AMBIG]} | {s[NO_JPG]} | {len(years[year]['species'])} |")
    w(f"| **total** | **{tot['xmp']}** | **{tot['bird']}** | **{tot['labelled']}** | "
      f"**{tot[OK]}** | **{tot[OFF_SAME]}** | **{tot[OFF_CROSS]}** | **{tot[AMBIG]}** | "
      f"**{tot[NO_JPG]}** | **{len(result['species'])}** |")
    w("")

    w("### Category mix per year")
    w("")
    w("Categories are not exclusive -- a sidecar can carry more than one. `no keywords` is a")
    w("well-formed sidecar that was never labelled.")
    w("")
    w("| year | " + " | ".join(CATEGORIES) + " | no keywords |")
    w("|---|" + "---|" * (len(CATEGORIES) + 1))
    for year in sorted(years):
        s = years[year]["stats"]
        w(f"| {year} | " + " | ".join(str(s[f'cat_{c}']) for c in CATEGORIES)
          + f" | {s['no_keywords']} |")
    w("")

    w("## Unresolved bird sidecars, by half-year folder")
    w("")
    w("Grouped by the folder that *should* have held the JPEG; only folders with at least one")
    w("unresolved sidecar are listed. Two very different failures live here:")
    w("")
    w("- **`cross-yr`** -- exactly one JPEG with that stem exists, in a *different year*. This")
    w("  is camera counter wraparound: the same frame number reused years apart. A matcher that")
    w("  falls back to a whole-tree stem lookup returns **a photo from an unrelated shoot**,")
    w("  with no error. This is the class that has to be refused.")
    w("- **`+same-yr`** -- the only JPEG is in an adjacent folder *within the same year*, i.e.")
    w("  the export split fell in a different place than the sidecar split. Same shoot, right")
    w("  photo; safe to accept.")
    w("")
    w("| year | half-year folder | bird | usable | +same-yr | cross-yr | ambig | no jpg | jpg folder exists |")
    w("|---|---|---|---|---|---|---|---|---|")
    jpg_set = set(result["jpg_folders"])
    for year in sorted(years):
        for folder in sorted(years[year]["by_folder"]):
            c = years[year]["by_folder"][folder]
            if not any(c[v] for v in BAD):
                continue
            w(f"| {year} | `{folder}` | {c['bird']} | {c[OK]} | {c[OFF_SAME]} | {c[OFF_CROSS]} | "
              f"{c[AMBIG]} | {c[NO_JPG]} | {'yes' if folder in jpg_set else '**NO**'} |")
    w("")
    raw_folders = {f for y in years.values() for f in y["by_folder"]}
    orphans = sorted(raw_folders - jpg_set)
    w("Half-year folders present in `raw/` with **no matching jpg folder at all** (nothing in")
    w("them can ever resolve correctly -- exclude these years or export the JPEGs):")
    w("")
    w("".join(f"- `{f}`\n" for f in orphans).rstrip() if orphans else "- (none)")
    w("")

    w("### Where the off-folder JPEGs actually live")
    w("")
    w("Expected -> found folder pairs, worst first. Same-year pairs are export-boundary")
    w("offsets; cross-year pairs are wraparound.")
    w("")
    pairs = collections.Counter(
        (i["year"], i["expected_folder"], i["found_folders"], i["verdict"])
        for i in result["issues"] if i["verdict"] in (OFF_SAME, OFF_CROSS, AMBIG)
    )
    w("| count | year | expected | found in | verdict |")
    w("|---|---|---|---|---|")
    for (year, exp, found, verdict), n in pairs.most_common(40):
        w(f"| {n} | {year} | `{exp}` | `{found}` | {verdict} |")
    w("")
    w("Full per-sidecar detail, including trip folder and species, is in")
    w("`output/audit/unresolved_bird_sidecars.csv` (regenerate with `--issues-dir`). Filter by")
    w("`verdict` to get a specific class, e.g. every wraparound casualty:")
    w("")
    w("```bash")
    w("python3 tools/audit_dataset.py --issues-dir output/audit")
    w("awk -F, '$5==\"off_folder_cross_year\"' output/audit/unresolved_bird_sidecars.csv")
    w("```")
    w("")
    w("`output/audit/colliding_jpg_stems.csv` lists every stem that appears in more than one")
    w("jpg folder, with the folders -- the raw wraparound evidence, independent of any sidecar.")
    w("")

    w("## Species distribution")
    w("")
    sp = result["species"]
    counts = sorted(sp.values(), reverse=True)
    w(f"- {len(sp)} distinct English names over {sum(counts)} labelled bird photos")
    for threshold in (5, 20, 50, 100):
        w(f"- names with >= {threshold} photos: {sum(1 for c in counts if c >= threshold)}")
    w(f"- singletons (exactly 1 photo): {sum(1 for c in counts if c == 1)}")
    w("")
    w("The tail is long -- most names are near-singletons, and these are the VLM's per-photo")
    w("guesses, so the tail is partly genuine rarity and partly noise. A density-based")
    w("clustering step should be expected to leave a large unclustered/noise fraction; that is")
    w("the shape of the data, not a failure of the method.")
    w("")
    w("Top 30 by photo count:")
    w("")
    w("| photos | species |")
    w("|---|---|")
    for name, n in sp.most_common(30):
        w(f"| {n} | {name} |")
    w("")

    w("## 'missing JPEG' rows recoverable now")
    w("")
    w("Rows the original run wrote as `missing JPEG` (label `unknown`, confidence 0.00) because")
    w("the export was not mounted at the time, whose JPEG resolves now. Re-running the labeler")
    w("over these would grow the bird set. Counted by basename against the CSV, so treat as an")
    w("estimate rather than an exact list.")
    w("")
    w("| year | missing-JPEG rows | resolvable now |")
    w("|---|---|---|")
    trm = trr = 0
    for year in sorted(years):
        m, r = years[year]["missing_jpeg_rows"], years[year]["recoverable"]
        trm += m
        trr += r
        if m:
            w(f"| {year} | {m} | {r} |")
    w(f"| **total** | **{trm}** | **{trr}** |")
    w("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("./data"))
    ap.add_argument("--report", type=Path, default=None,
                    help="write the markdown report here (default: stdout)")
    ap.add_argument("--issues-dir", type=Path, default=None,
                    help="also write per-sidecar issue CSVs into this directory")
    args = ap.parse_args()

    result = audit(args.data_dir)
    report = render(result, args.data_dir)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report)
        print(f"wrote {args.report}")
    else:
        sys.stdout.write(report)

    if args.issues_dir:
        args.issues_dir.mkdir(parents=True, exist_ok=True)
        path = args.issues_dir / "unresolved_bird_sidecars.csv"
        fields = ["year", "half_year", "trip", "stem", "verdict", "expected_folder",
                  "found_folders", "species", "in_csv_as_missing_jpeg"]
        with open(path, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=fields)
            wr.writeheader()
            wr.writerows(sorted(result["issues"],
                                key=lambda r: (r["verdict"], r["year"], r["half_year"],
                                               r["trip"], r["stem"])))
        print(f"wrote {path} ({len(result['issues'])} rows)")

        path = args.issues_dir / "colliding_jpg_stems.csv"
        with open(path, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["stem", "folders"])
            for stem, folders in sorted(result["colliding_stems"].items()):
                wr.writerow([stem, ";".join(folders)])
        print(f"wrote {path} ({len(result['colliding_stems'])} rows)")


if __name__ == "__main__":
    main()
