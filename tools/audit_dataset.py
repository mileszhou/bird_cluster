#!/usr/bin/env python3
"""
Audit the dataset: XMP sidecars <-> exported JPEGs.

The export mirrors the photo library one-for-one, so the layout is flat in
structure and direct in mapping:

    <data-dir>/xmp/<library>/<trip>/*.xmp        # Photos-19/2019-01-13 山公园/
    <data-dir>/jpg/<library>/<trip>/*.jpg        # same folder, same stem
    <data-dir>/jpg/export.report.txt             # what the exporter skipped, and why

There are exactly two ways the two trees can disagree, and they matter to very
different degrees:

* **A sidecar with no JPEG** -- a photo the pipeline cannot see. Every one is
  listed with its trip folder and sidecar path, split by whether the exporter
  said why (`export.report.txt` reporting the raw as missing means the sidecar
  is dangling) or stayed silent (a technical failure worth chasing).
* **A JPEG with no sidecar** -- a photo that was never raw, so it never had a
  sidecar to begin with. Harmless and reported as a count only.

Category and species come from each sidecar's own `dc:subject` keywords via
`code/lib/xmp_labels.py`. Nothing is written back into the dataset; this is
read-only.

Worklists are written alongside the report (default ./project/reports):

    missing_jpg.csv          sidecars with no exported JPEG -- the manual worklist
    extra_jpg.csv            JPEGs no sidecar claims
    unprocessed_sidecars.csv never labelled, JPEG available now -> run bird_label

All three carry a `path` column -- the sidecar's path relative to `data/xmp` --
which is the same key `bird_label.py`'s CSV and checkpoint and `embed.py`'s JSONL
use, so a worklist joins directly against any run's output. They are written
UTF-8 with a BOM so Excel does not mangle the Chinese trip names; read them back
with `encoding="utf-8-sig"`.

The report keeps a fixed filename so `git diff` between runs shows exactly what
moved in the dataset; the worklist CSVs are gitignored, being fully regenerated
each run. `--snapshot` additionally files a numbered copy under archive/.

Usage:
    ./run-audit                                  # report + worklists -> project/reports
    ./run-audit --snapshot after-2024-recovery   # ...and archive/NN-...-after-2024-recovery.md
    python3 tools/audit_dataset.py --stdout      # report to stdout, write nothing
"""
import argparse
import collections
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.lib.export_report import NOT_FOUND, ExportReport  # noqa: E402
from code.lib.jpg_claim import SidecarClaims, sort_key  # noqa: E402
from code.lib.jpg_index import ExtraKind, JpgIndex, Verdict, library_year  # noqa: E402
from code.lib.xmp_labels import CATEGORIES, read_labels  # noqa: E402

OK = Verdict.OK.value
DERIVED = Verdict.DERIVED.value
NO_JPG = Verdict.NO_JPG.value
NO_FOLDER = Verdict.NO_FOLDER.value
MISSING = (NO_JPG, NO_FOLDER)

# Said of a sidecar whose JPEG is absent and whose raw the exporter never
# mentioned -- so the raw was there and the export still produced nothing.
UNREPORTED = "(not mentioned by the exporter)"


def audit(data_dir):
    xmp_root, jpg_root = data_dir / "xmp", data_dir / "jpg"
    if not xmp_root.is_dir() or not jpg_root.is_dir():
        sys.exit(f"error: expected {xmp_root} and {jpg_root} to exist")

    index = JpgIndex(xmp_root, jpg_root)
    export = ExportReport.load(jpg_root)

    result = {
        "export_report": export,
        "unexported_folders": index.unexported_folders(),
        "orphan_folders": index.orphan_folders(),
        "libraries": {},
        "species": collections.Counter(),
        "unparsable": [],
        "sidecars": [],
        "extras": [],
        "derived_suffixes": collections.Counter(),
    }

    for folder in sorted(index.xmp_folders):
        library = folder.split("/")[0]
        lib = result["libraries"].setdefault(library, {
            "stats": collections.Counter(), "species": collections.Counter(),
        })
        stats = lib["stats"]

        for stem in sorted(index.xmp_folders[folder]):
            xmp = index.xmp_folders[folder][stem]
            stats["xmp"] += 1
            labels = read_labels(xmp)
            if labels is None:
                result["unparsable"].append(str(xmp.relative_to(data_dir)))
                continue
            if not labels.subjects:
                stats["no_keywords"] += 1
            for cat in labels.categories:
                stats[f"cat_{cat}"] += 1

            match = index.resolve(xmp)
            stats[match.verdict.value] += 1
            for jpg in match.paths:
                result["derived_suffixes"][jpg.stem[len(stem):]] += 1

            if labels.is_bird:
                stats["bird"] += 1
                stats[f"bird_{match.verdict.value}"] += 1
                if labels.label:
                    stats["labelled"] += 1
                    lib["species"][labels.species] += 1
                    result["species"][labels.species] += 1

            entry = export.entry_for(folder, stem)
            result["sidecars"].append({
                # `path` is the key every other tool uses -- bird_label's
                # checkpoint and CSV, embed.py's JSONL -- so a worklist can be
                # joined against a run's output without reconstructing it.
                "path": xmp.relative_to(xmp_root).as_posix(),
                "year": library_year(library) or "",
                "library": library, "folder": folder, "trip": folder.split("/", 1)[-1],
                "stem": stem, "xmp": xmp,
                "categories": labels.categories,
                "is_bird": labels.is_bird,
                "species": labels.species or "",
                "labelled": labels.label is not None,
                "has_keywords": bool(labels.subjects),
                "verdict": match.verdict.value,
                "jpgs": match.paths,
                "expected_jpg": jpg_root / folder / f"{stem}.jpg",
                "reason": entry.reason if entry else "",
                "raw_name": entry.name if entry else "",
            })

    for extra in index.extras():
        result["extras"].append(extra)
    result["jpg_total"] = sum(len(paths)
                              for folder in index.jpg_folders.values()
                              for paths in folder.values())
    result["multi_claim"] = find_multi_claim(index, xmp_root, jpg_root)
    return result


def find_multi_claim(index, xmp_root: Path, jpg_root: Path):
    """Sidecars that more than one JPEG would claim, under the labeler's rule.

    `bird_label.py` walks the JPEG tree and each image claims a sidecar locally
    (`code/lib/jpg_claim.py`), so a capture exported more than once -- master
    plus a `-2` virtual copy, say -- produces several JPEGs reaching one sidecar.
    The first in stem-sorted order keeps it, which is always the exact-stem
    match; the rest are labelled to the CSV alone.

    That resolution is correct and needs no repair, so this is reported rather
    than flagged as an error. It is worth watching all the same: the count is a
    fingerprint of the export's shape, and a jump in it after a re-export means
    something changed about how virtual copies were emitted.
    """
    claims = SidecarClaims(xmp_root)
    by_sidecar: dict[tuple[str, str], list[str]] = {}
    for folder in sorted(index.jpg_folders):
        for stem in sorted(index.jpg_folders[folder]):
            base = claims.claim(folder, stem)
            if base is not None:
                by_sidecar.setdefault((folder, base), []).append(stem)
    groups = []
    for (folder, base), stems in sorted(by_sidecar.items()):
        if len(stems) < 2:
            continue
        winner = min(stems, key=lambda s: sort_key(folder, s))
        groups.append({
            "path": f"{folder}/{base}.xmp",
            "folder": folder, "library": folder.split("/")[0],
            "trip": folder.split("/", 1)[-1], "stem": base,
            "keeps": winner,
            "csv_only": ";".join(s for s in stems if s != winner),
            "exact_match_wins": winner == base,
        })
    return groups


MISSING_FIELDS = ["reason", "verdict", "year", "library", "trip", "stem", "category",
                  "species", "raw_name", "expected_jpg", "folder", "path", "xmp_path"]


def find_missing(result):
    """Sidecars with no exported JPEG, annotated with the exporter's reason.

    Sorted with the unexplained ones first -- those are the technical gaps a
    re-export might clear, as against a raw that is simply gone.
    """
    out = []
    for rec in result["sidecars"]:
        if rec["verdict"] not in MISSING:
            continue
        out.append({
            "reason": rec["reason"] or UNREPORTED,
            "verdict": rec["verdict"],
            "year": rec["year"], "library": rec["library"], "trip": rec["trip"],
            "stem": rec["stem"],
            "category": ";".join(rec["categories"]) or "(none)",
            "species": rec["species"],
            # The raw as the exporter named it: what to go looking for on disk.
            "raw_name": rec["raw_name"],
            # Where the JPEG would have landed, so a re-export can be checked.
            "expected_jpg": str(rec["expected_jpg"]),
            "folder": rec["folder"],
            "path": rec["path"],
            "xmp_path": str(rec["xmp"]),
        })
    out.sort(key=lambda r: (r["reason"] != UNREPORTED, r["library"], r["trip"], r["stem"]))
    return out


UNPROCESSED_FIELDS = ["jpg_available", "reason", "year", "library", "trip", "stem",
                      "verdict", "copies", "jpg_path", "path", "xmp_path"]


def find_unprocessed(result):
    """Sidecars carrying no label whose JPEG is available -- ready for bird_label."""
    out = []
    for rec in result["sidecars"]:
        if rec["has_keywords"] and rec["categories"]:
            continue
        out.append({
            "jpg_available": bool(rec["jpgs"]),
            "reason": "no keywords" if not rec["has_keywords"] else "keywords but no category",
            "year": rec["year"], "library": rec["library"], "trip": rec["trip"],
            "stem": rec["stem"],
            "verdict": rec["verdict"],
            "copies": len(rec["jpgs"]),
            "jpg_path": str(rec["jpgs"][0]) if rec["jpgs"] else "",
            "path": rec["path"],
            "xmp_path": str(rec["xmp"]),
        })
    out.sort(key=lambda r: (not r["jpg_available"], r["library"], r["trip"], r["stem"]))
    return out


EXTRA_FIELDS = ["kind", "year", "library", "folder", "jpg", "decorates_sidecar", "jpg_path"]

# `path` is the sidecar, relative to data/xmp -- the audit tools' key. `keeps`
# and `csv_only` are JPEG stems within the same folder.
MULTI_CLAIM_FIELDS = ["path", "library", "trip", "stem", "keeps", "csv_only",
                      "exact_match_wins", "folder"]

# Trip folders are named in Chinese as often as not, and these worklists get
# opened in Excel on the Windows box the library lives on. Excel reads a
# BOM-less UTF-8 CSV as the system codepage and mangles every such name; the BOM
# makes it read UTF-8. Python's csv reads it back cleanly with `utf-8-sig`.
CSV_ENCODING = "utf-8-sig"


def write_csv(path: Path, fields: list[str], rows) -> None:
    with open(path, "w", newline="", encoding=CSV_ENCODING) as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def next_snapshot_path(archive_dir: Path, stem: str, label: str = "") -> Path:
    """Next free `NN-<stem>[-<label>].md` in the archive directory.

    Snapshots are numbered so the sequence of runs stays readable, while the
    live report keeps a fixed filename -- that stable name is what lets `git
    diff` show what changed between two runs.
    """
    highest = 0
    if archive_dir.is_dir():
        for existing in archive_dir.glob("*.md"):
            m = re.match(r"^(\d+)-", existing.name)
            if m:
                highest = max(highest, int(m.group(1)))
    suffix = f"-{label}" if label else ""
    return archive_dir / f"{highest + 1:02d}-{stem}{suffix}.md"


def pct(n, d):
    return f"{100.0 * n / d:.2f}%" if d else "-"


def render(result, data_dir, missing, unprocessed):
    L = []
    w = L.append
    libs = result["libraries"]
    tot = collections.Counter()
    for lib in libs.values():
        tot.update(lib["stats"])
    extras = result["extras"]
    orphans = [e for e in extras if e.kind is ExtraKind.ORPHAN]
    dup_exports = [e for e in extras if e.kind is ExtraKind.DERIVED]

    w("# Data preview / audit report")
    w("")
    w(f"Generated by `tools/audit_dataset.py` over `{data_dir}`. Read-only; the dataset is not")
    w("modified. Re-run with `./run-audit`.")
    w("")
    w("## Layout")
    w("")
    w("```")
    w("<data-dir>/xmp/<library>/<trip>/*.xmp     # Photos-19/2019-01-13 山公园/_D8S0025.xmp")
    w("<data-dir>/jpg/<library>/<trip>/*.jpg     # same folder, same stem")
    w("<data-dir>/jpg/export.report.txt          # what the exporter skipped, and why")
    w("```")
    w("")
    w("The export mirrors the library one-for-one, so a sidecar's JPEG is a lookup inside its")
    w("own trip folder. Stems repeat across the library (camera counter wraparound) but that no")
    w("longer matters, the folder being part of the key -- there is nothing left to be ambiguous")
    w("about. Category and species come from each sidecar's own `dc:subject` keywords")
    w("(`code/lib/xmp_labels.py`).")
    w("")

    w("## Totals")
    w("")
    resolved = tot[OK] + tot[DERIVED]
    w(f"- Sidecars scanned: **{tot['xmp']}**")
    w(f"- Matched to an exported JPEG: **{resolved}** ({pct(resolved, tot['xmp'])}) — "
      f"{tot[OK]} under their own name, {tot[DERIVED]} via a derived export")
    w(f"- **Sidecars with no JPEG: {len(missing)}** ({pct(len(missing), tot['xmp'])}) — worklist below")
    w(f"- JPEGs exported: **{result['jpg_total']}**, of which **{len(orphans)}** have no sidecar "
      f"(harmless — never a raw file, so never had one)")
    w(f"- JPEGs claiming a sidecar another JPEG also claims: **{sum(len(g['csv_only'].split(';')) for g in result['multi_claim'])}** "
      f"across {len(result['multi_claim'])} sidecars — resolved, see below")
    w(f"- Bird-categorised sidecars: **{tot['bird']}**, of which **{tot['labelled']}** carry a "
      f"parseable `py-cn-en(NN%)` label")
    w(f"- Bird sidecars with a JPEG: **{tot['bird_' + OK] + tot['bird_' + DERIVED]}** — the "
      f"usable clustering set")
    w(f"- Distinct English species names: **{len(result['species'])}**")
    if result["unparsable"]:
        w(f"- Sidecars that would not parse: {len(result['unparsable'])}")
    w("")

    # ---------------------------------------------------------------- missing
    w("## Discrepancy 1 — sidecars with no exported JPEG")
    w("")
    w("A photo the pipeline cannot see. The exporter's own log")
    w("(`data/jpg/export.report.txt`) separates the two causes: where it reports the raw as")
    w("missing, the sidecar is dangling — the raw is gone from disk, so nothing could be")
    w("exported, and recovering it means going back to the original raw. Where the log says")
    w("nothing, the raw was there and the export still produced no file: a technical failure,")
    w("and the rows worth chasing first.")
    w("")
    by_reason = collections.Counter(r["reason"] for r in missing)
    w("| sidecars | exporter said |")
    w("|---|---|")
    for reason, n in by_reason.most_common():
        w(f"| {n} | {reason} |")
    w(f"| **total** | **{len(missing)}** |")
    w("")

    unexplained = [r for r in missing if r["reason"] == UNREPORTED]
    if unexplained:
        w(f"### The {len(unexplained)} the exporter never mentioned")
        w("")
        w("These are the technical gaps: the raw was present and no JPEG came out. Worth a look")
        w("by hand — a re-export of just these folders may clear them.")
        w("")
        w("| library | trip | sidecar | category |")
        w("|---|---|---|---|")
        for r in unexplained:
            w(f"| {r['library']} | `{r['trip']}` | `{r['stem']}.xmp` | {r['category']} |")
        w("")

    by_trip = collections.Counter(f"{r['folder']}" for r in missing)
    w("### By trip folder")
    w("")
    w("Fixing whole folders clears the most at once. Folders with **no exported folder at all**")
    w("are marked — those are worth checking as a unit before anything else.")
    w("")
    unexported = set(result["unexported_folders"])
    w("| sidecars | trip folder | exported folder |")
    w("|---|---|---|")
    for folder, n in by_trip.most_common():
        w(f"| {n} | `{folder}` | {'**MISSING**' if folder in unexported else 'yes'} |")
    w("")
    w("Every row is in `project/reports/missing_jpg.csv`, with the sidecar path, the exporter's")
    w("reason, `raw_name` (the raw as the exporter named it — what to go looking for on disk;")
    w("`X-Enhanced-NR.dng` and `X.nef` are different files even though one sidecar covers both)")
    w("and `expected_jpg` (where the JPEG would have landed, so a re-export can be checked).")
    w("")

    # ------------------------------------------------------------------ extra
    w("## Discrepancy 2 — exported JPEGs with no sidecar")
    w("")
    w("Harmless, and reported as a count only: the source photo was never a raw file, so there")
    w("is no sidecar for it to have. These are phone and compact-camera shots, very unlikely to")
    w("be bird photos, and the pipeline simply never looks at them.")
    w("")
    w(f"- JPEGs no sidecar claims: **{len(orphans)}**")
    w(f"- of those, in a trip folder with no sidecars at all: "
      f"**{sum(1 for e in orphans if e.folder in set(result['orphan_folders']))}** "
      f"across {len(result['orphan_folders'])} folders")
    w(f"- Additional decorated exports of a sidecar that already matched (extra virtual copies): "
      f"**{len(dup_exports)}**")
    w("")
    by_lib = collections.Counter(e.folder.split("/")[0] for e in orphans)
    w("| library | JPEGs with no sidecar |")
    w("|---|---|")
    for library in sorted(by_lib):
        w(f"| {library} | {by_lib[library]} |")
    w("")
    w("Listed in `project/reports/extra_jpg.csv` if a spot-check is ever wanted.")
    w("")

    # ---------------------------------------------------------------- derived
    w("## Derived exports")
    w("")
    w(f"**{tot[DERIVED]}** sidecars have no JPEG under their own name but do have a decorated")
    w("one in the same folder, and that decorated file is the right photo:")
    w("")
    w("- `-2`, `-3`, … — a Lightroom **virtual copy**, exported under its copy name. When the")
    w("  master itself was not in the export set, this is the capture's only export.")
    w("- `-Enhanced-NR` — the **AI Denoise** render, a separate DNG that carries its metadata")
    w("  internally and so has no sidecar of its own; the raw's sidecar covers it.")
    w("")
    w("Matching is done a whole folder at a time, exact hits claimed first, so where sidecars")
    w("`A` and `A-2` both exist `A-2.jpg` goes to `A-2` rather than being read as `A`'s virtual")
    w("copy. Counting these as missing would have inflated discrepancy 1 from "
      f"{len(missing)} to {len(missing) + tot[DERIVED]}.")
    w("")
    w("| exports | suffix |")
    w("|---|---|")
    for suffix, n in result["derived_suffixes"].most_common(15):
        if suffix:
            w(f"| {n} | `{suffix}` |")
    w("")

    # ------------------------------------------------------------ multi-claim
    groups = result["multi_claim"]
    extra_claimants = sum(len(g["csv_only"].split(";")) for g in groups)
    w("## Sidecars claimed by more than one JPEG")
    w("")
    w("The labeler walks the JPEG tree, so a capture exported more than once sends several")
    w("images at one sidecar. **This is resolved, not a defect** — the first claimant in")
    w("stem-sorted order keeps the sidecar and the rest are labelled to the CSV alone. That")
    w("winner is always the exact-stem match, because `A` is a proper prefix of `A-2` and so")
    w("sorts first.")
    w("")
    w(f"- Sidecars with several claimants: **{len(groups)}**")
    w(f"- JPEGs downgraded to CSV-only as a result: **{extra_claimants}**")
    anomalies = [g for g in groups if not g["exact_match_wins"]]
    if anomalies:
        w(f"- **{len(anomalies)} where the winner is not the exact-stem export** — every one of")
        w("  these is a capture whose master was never exported, so a decorated file is the only")
        w("  candidate. Listed below; check them if the count moved unexpectedly.")
    else:
        w("- The exact-stem export wins in every group.")
    w("")
    w("Worth watching rather than fixing: the count is a fingerprint of the export's shape, so a")
    w("jump after a re-export means something changed about how virtual copies were emitted.")
    w("Full listing in `multi_claim.csv`.")
    w("")
    if groups:
        by_lib = collections.Counter(g["library"] for g in groups)
        w("| library | sidecars with several claimants |")
        w("|---|---|")
        for lib, n in sorted(by_lib.items()):
            w(f"| {lib} | {n} |")
        w("")
    for g in anomalies[:15]:
        w(f"- `{g['path']}` → keeps `{g['keeps']}.jpg`, CSV-only: `{g['csv_only']}`")
    if anomalies:
        w("")

    # ------------------------------------------------------------ per library
    w("## Per library")
    w("")
    w("`ok` = JPEG under the sidecar's own name; `derived` = a decorated export of it;")
    w("`no jpg` / `no folder` together are discrepancy 1.")
    w("")
    w("| library | year | xmp | ok | derived | no jpg | no folder | bird | labelled | species |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for library in sorted(libs):
        s = libs[library]["stats"]
        w(f"| {library} | {library_year(library) or '?'} | {s['xmp']} | {s[OK]} | {s[DERIVED]} | "
          f"{s[NO_JPG]} | {s[NO_FOLDER]} | {s['bird']} | {s['labelled']} | "
          f"{len(libs[library]['species'])} |")
    w(f"| **total** | | **{tot['xmp']}** | **{tot[OK]}** | **{tot[DERIVED]}** | "
      f"**{tot[NO_JPG]}** | **{tot[NO_FOLDER]}** | **{tot['bird']}** | **{tot['labelled']}** | "
      f"**{len(result['species'])}** |")
    w("")

    w("### Category mix per library")
    w("")
    w("Categories are not exclusive — a sidecar can carry more than one. `no keywords` is a")
    w("well-formed sidecar that was never labelled.")
    w("")
    w("| library | " + " | ".join(CATEGORIES) + " | no keywords |")
    w("|---|" + "---|" * (len(CATEGORIES) + 1))
    for library in sorted(libs):
        s = libs[library]["stats"]
        w(f"| {library} | " + " | ".join(str(s[f'cat_{c}']) for c in CATEGORIES)
          + f" | {s['no_keywords']} |")
    w("")

    # ----------------------------------------------------------- unprocessed
    w("## Sidecars still to label")
    w("")
    w("Sidecars carrying no category keyword at all — the labeler never reached them. Those")
    w("whose JPEG is present are ready to run now.")
    w("")
    ready = [r for r in unprocessed if r["jpg_available"]]
    w(f"- unlabelled sidecars: **{len(unprocessed)}**")
    w(f"- of those, JPEG available — **ready to run: {len(ready)}**")
    w(f"- blocked on a missing JPEG: {len(unprocessed) - len(ready)}")
    w("")
    by_lib = collections.Counter(r["library"] for r in ready)
    if by_lib:
        w("| library | ready to run |")
        w("|---|---|")
        for library in sorted(by_lib):
            w(f"| {library} | {by_lib[library]} |")
        w("")
    w("Full list: `project/reports/unprocessed_sidecars.csv` (ready ones first).")
    w("")

    # --------------------------------------------------------------- species
    w("## Species distribution")
    w("")
    sp = result["species"]
    counts = sorted(sp.values(), reverse=True)
    w(f"- {len(sp)} distinct English names over {sum(counts)} labelled bird photos")
    for threshold in (5, 20, 50, 100):
        w(f"- names with >= {threshold} photos: {sum(1 for c in counts if c >= threshold)}")
    w(f"- singletons (exactly 1 photo): {sum(1 for c in counts if c == 1)}")
    w("")
    w("The tail is long — most names are near-singletons, and these are the VLM's per-photo")
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
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("./data"))
    ap.add_argument("--report", type=Path, default=Path("./project/reports/data_preview_report.md"),
                    help="write the markdown report here")
    ap.add_argument("--issues-dir", type=Path, default=Path("./project/reports"),
                    help="write the worklist CSVs into this directory")
    ap.add_argument("--stdout", action="store_true",
                    help="print the report instead of writing any files")
    ap.add_argument("--snapshot", nargs="?", const="", default=None, metavar="LABEL",
                    help="also keep a numbered copy of this report under "
                         "<issues-dir>/archive/NN-<name>[-LABEL].md. The live report keeps "
                         "its fixed filename either way, so `git diff` between runs stays "
                         "readable; the snapshot is for comparing against a specific past run")
    args = ap.parse_args()

    result = audit(args.data_dir)
    missing = find_missing(result)
    unprocessed = find_unprocessed(result)
    report = render(result, args.data_dir, missing, unprocessed)

    if args.stdout:
        sys.stdout.write(report)
        return

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report)
        print(f"wrote {args.report}")

        if args.snapshot is not None:
            snap = next_snapshot_path(args.report.parent / "archive",
                                      args.report.stem, args.snapshot)
            snap.parent.mkdir(parents=True, exist_ok=True)
            snap.write_text(report)
            print(f"wrote {snap}")
    else:
        sys.stdout.write(report)

    if not args.issues_dir:
        return
    args.issues_dir.mkdir(parents=True, exist_ok=True)

    path = args.issues_dir / "missing_jpg.csv"
    write_csv(path, MISSING_FIELDS, missing)
    unexplained = sum(1 for r in missing if r["reason"] == UNREPORTED)
    print(f"wrote {path} ({len(missing)} rows, {unexplained} the exporter never mentioned)")

    path = args.issues_dir / "extra_jpg.csv"
    write_csv(path, EXTRA_FIELDS, [
        {"kind": e.kind.value, "year": library_year(e.folder.split("/")[0]) or "",
         "library": e.folder.split("/")[0], "folder": e.folder,
         "jpg": e.path.name, "decorates_sidecar": e.parent_stem,
         "jpg_path": str(e.path)}
        for e in sorted(result["extras"], key=lambda e: (e.kind.value, str(e.path)))
    ])
    orphans = sum(1 for e in result["extras"] if e.kind is ExtraKind.ORPHAN)
    print(f"wrote {path} ({len(result['extras'])} rows, {orphans} with no sidecar at all)")

    path = args.issues_dir / "unprocessed_sidecars.csv"
    write_csv(path, UNPROCESSED_FIELDS, unprocessed)
    ready = sum(1 for r in unprocessed if r["jpg_available"])
    print(f"wrote {path} ({len(unprocessed)} rows, {ready} ready to run)")

    path = args.issues_dir / "multi_claim.csv"
    write_csv(path, MULTI_CLAIM_FIELDS, result["multi_claim"])
    odd = sum(1 for g in result["multi_claim"] if not g["exact_match_wins"])
    print(f"wrote {path} ({len(result['multi_claim'])} rows, "
          f"{odd} where the exact-stem export is not the winner)")


if __name__ == "__main__":
    main()
