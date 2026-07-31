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

With --issues-dir it also writes four worklists for manual repair, alongside the
report (default ./project/reports -- progress/analysis output lives under
./project; ./docs is for product documentation):

    duplicate_frames.csv         same frame in neighbouring trips -> a duplicate
    photos_in_doubt.csv          every sidecar whose JPEG did not resolve cleanly
    unprocessed_sidecars.csv     never labelled / "missing JPEG" -> re-run bird_label
    colliding_jpg_stems.csv      raw stem-collision evidence, independent of sidecars

The report keeps a fixed filename so `git diff` between runs is readable; the
worklist CSVs are gitignored, being large and fully regenerated each run.
`--snapshot` additionally files a numbered copy under archive/ to compare against.

Usage:
    ./run-audit                                  # report + worklists -> project/reports
    ./run-audit --snapshot after-2019-fixes      # ...and archive/NN-...-after-2019-fixes.md
    python3 tools/audit_dataset.py --stdout      # report to stdout, write nothing
    python3 tools/audit_dataset.py --no-verify-pixels
"""
import argparse
import collections
import csv
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.lib.jpg_index import JpgIndex, MatchPolicy, Verdict  # noqa: E402
from code.lib.trips import Timeline, frame_id  # noqa: E402
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
        "sidecars": [],      # one record per sidecar, for the worklists below
        "timeline": Timeline.from_dataset(xmp_root),
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

            rel = xmp.relative_to(raw)
            folder = rel.parts[0] if len(rel.parts) > 1 else "(root)"
            trip = rel.parts[1] if len(rel.parts) > 2 else "(none)"

            # Every sidecar, labelled or not, bird or not -- the worklists need
            # the ones the per-year bird stats deliberately skip.
            result["sidecars"].append({
                "year": year, "half_year": folder, "trip": trip,
                "stem": xmp.stem, "xmp": xmp,
                "frame": frame_id(xmp.stem),
                "categories": labels.categories,
                "is_bird": labels.is_bird,
                "species": labels.species or "",
                "labelled": labels.label is not None,
                "has_keywords": bool(labels.subjects),
                "verdict": match.verdict.value,
                "jpg": match.path,
                "candidates": match.candidates,
                "expected_folder": match.expected_folder or "",
                "in_csv_as_missing_jpeg": xmp.stem in missing_csv,
            })

            if not labels.is_bird:
                continue
            stats["bird"] += 1
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
                    "year": year, "half_year": folder, "trip": trip,
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


def pixel_hash(path):
    """Hash of the decoded pixels, or None if the image will not open.

    File bytes are useless for spotting a duplicated export -- re-exporting the
    same frame rewrites metadata, so byte-identical copies are the exception
    even when the pixels match exactly. Decoding is the only reliable test.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            return f"{im.size[0]}x{im.size[1]}:" + hashlib.blake2b(
                im.tobytes(), digest_size=16).hexdigest()
    except Exception:
        return None


# What to do about a cross-trip frame collision. The timeline verdict alone only
# narrows the field: neighbouring trips make a duplicate *likely*, but measuring
# the pixels is what settles it, and the two disagree often enough to matter.
ACT_REMOVE = "remove one copy (pixels identical)"
ACT_KEEP = "keep both (different photos)"
ACT_CHECK = "check by hand (jpg missing or unreadable)"
ACT_PARTIAL = "check by hand (some copies match, some do not)"
ACT_NONE = "no action (wraparound: different photos, far apart in time)"
ACT_UNDATED = "check by hand (trip folder has no date, cannot place on timeline)"


def duplicate_action(verdict, pixels):
    if verdict == "wraparound":
        return ACT_NONE
    if verdict == "undecidable":
        return ACT_UNDATED
    if pixels == "identical":
        return ACT_REMOVE
    if pixels == "different":
        return ACT_KEEP
    if pixels == "partial":
        return ACT_PARTIAL
    if pixels is None:
        return ACT_CHECK if verdict == "duplicate" else ACT_PARTIAL
    return ACT_CHECK


def find_duplicate_frames(result, window=1, verify_pixels=False):
    """Frames appearing in more than one trip, split by timeline distance.

    A camera counter only repeats after wrapping through its whole range, which
    takes far longer than a trip or two. So the same frame number in
    neighbouring trips cannot be two different photos -- it is one photo filed
    twice. Far apart on the timeline, the same collision is ordinary wraparound.
    """
    timeline = result["timeline"]
    by_frame = collections.defaultdict(list)
    for rec in result["sidecars"]:
        if rec["frame"] is not None:
            by_frame[rec["frame"]].append(rec)

    groups = []
    for frame, recs in by_frame.items():
        trips = {}
        for rec in recs:
            key = f"{rec['year']}/{rec['half_year']}/{rec['trip']}"
            trips.setdefault(key, []).append(rec)
        if len(trips) < 2:
            continue

        members = []
        for key, rs in trips.items():
            trip = timeline.by_key.get(key)
            members.append({"trip_key": key, "trip": trip, "recs": rs})
        members.sort(key=lambda m: (m["trip"].rank if m["trip"] and m["trip"].rank >= 0
                                    else 10**9, m["trip_key"]))

        pairs = []
        for a, b in zip(members, members[1:]):
            ta, tb = a["trip"], b["trip"]
            if ta is None or tb is None or ta.rank < 0 or tb.rank < 0:
                verdict = "undecidable"
            elif timeline.neighbours(ta, tb, window):
                verdict = "duplicate"
            else:
                verdict = "wraparound"
            pairs.append({
                "a": a, "b": b, "verdict": verdict,
                "rank_gap": (abs(ta.rank - tb.rank)
                             if ta and tb and ta.rank >= 0 and tb.rank >= 0 else None),
                "day_gap": timeline.day_gap(ta, tb) if ta and tb else None,
            })

        verdicts = {p["verdict"] for p in pairs}
        group_verdict = ("duplicate" if verdicts == {"duplicate"}
                         else "wraparound" if verdicts == {"wraparound"}
                         else "undecidable" if verdicts == {"undecidable"}
                         else "mixed")

        pixels = None
        if verify_pixels and group_verdict in ("duplicate", "mixed"):
            hashes = {}
            for m in members:
                for r in m["recs"]:
                    if r["jpg"] is not None:
                        hashes.setdefault(pixel_hash(r["jpg"]), []).append(r)
            known = {h for h in hashes if h is not None}
            pixels = ("identical" if len(known) == 1 and len(hashes) == 1
                      else "different" if len(known) == len(hashes) and len(known) > 1
                      else "unreadable" if not known else "partial")

        groups.append({
            "frame": frame, "members": members, "pairs": pairs,
            "verdict": group_verdict, "pixels": pixels,
            "action": duplicate_action(group_verdict, pixels),
        })

    groups.sort(key=lambda g: (g["verdict"], g["frame"].code, g["frame"].num))
    return groups


def find_unprocessed(result):
    """Sidecars the labeler never got a result for, that could be run now.

    Two ways a sidecar ends up unlabelled: it carries no keywords at all (the
    run never reached it), or the run reached it but recorded "missing JPEG"
    because the export was not mounted. Either is worth re-running now if the
    JPEG resolves today.
    """
    out = []
    for rec in result["sidecars"]:
        never_labelled = not rec["has_keywords"] or not rec["categories"]
        missing_jpeg = rec["in_csv_as_missing_jpeg"]
        if not (never_labelled or missing_jpeg):
            continue
        out.append({
            "year": rec["year"], "half_year": rec["half_year"], "trip": rec["trip"],
            "stem": rec["stem"],
            "reason": ("no keywords" if never_labelled and not missing_jpeg
                       else "missing JPEG in CSV" if missing_jpeg and not never_labelled
                       else "no keywords + missing JPEG in CSV"),
            "jpg_resolves_now": rec["jpg"] is not None,
            "verdict": rec["verdict"],
            "jpg_path": str(rec["jpg"]) if rec["jpg"] else "",
            "xmp_path": str(rec["xmp"]),
        })
    out.sort(key=lambda r: (not r["jpg_resolves_now"], r["year"], r["half_year"],
                            r["trip"], r["stem"]))
    return out


def find_photos_in_doubt(result, duplicate_groups):
    """Every sidecar whose JPEG did not resolve cleanly, plus duplicate members.

    One row per sidecar, with the trip folder, so each can be tracked down by
    hand. Sidecars of every category are included, not just birds.
    """
    dup_member = {}
    for g in duplicate_groups:
        if g["verdict"] not in ("duplicate", "mixed"):
            continue
        for m in g["members"]:
            for r in m["recs"]:
                dup_member[id(r)] = g

    out = []
    for rec in result["sidecars"]:
        group = dup_member.get(id(rec))
        clean = rec["verdict"] == OK
        if clean and group is None:
            continue
        if group is not None:
            others = [f"{m['trip_key']}" for m in group["members"]
                      if not any(id(r) == id(rec) for r in m["recs"])]
            issue = f"duplicate frame ({group['verdict']})"
            detail = "also in: " + " | ".join(others)
        else:
            issue = rec["verdict"]
            detail = ("found in: "
                      + ";".join(sorted({p.parent.name for p in rec["candidates"]})
                                 ) if rec["candidates"] else "no jpg anywhere")
        out.append({
            "year": rec["year"], "half_year": rec["half_year"], "trip": rec["trip"],
            "stem": rec["stem"],
            "issue": issue,
            "category": ";".join(rec["categories"]) or "(none)",
            "species": rec["species"],
            "expected_folder": rec["expected_folder"],
            "detail": detail,
            "xmp_path": str(rec["xmp"]),
        })
    out.sort(key=lambda r: (r["issue"], r["year"], r["half_year"], r["trip"], r["stem"]))
    return out


def pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "-"


def render_worklists(result, dup_groups, doubt, unprocessed):
    """The manual-repair sections: duplicates, doubt list, re-run list."""
    L = []
    w = L.append
    timeline = result["timeline"]
    dated = [t for t in timeline.trips if t.rank >= 0]

    w("## Worklist 1 -- duplicate frames (neighbouring trips)")
    w("")
    w(f"Trip folders sorted by their `YYYY-MM-DD` name prefix give a timeline of {len(dated)} "
      f"dated trips ({len(timeline.trips) - len(dated)} undated). A camera's frame counter only")
    w("repeats after wrapping through its entire range -- thousands of frames, months or years")
    w("of shooting -- so the same frame number **cannot** be two different photos within one")
    w("trip or between neighbouring trips. When it shows up in neighbouring trips it is one")
    w("photo filed twice. Far apart on the timeline, the same collision is ordinary wraparound.")
    w("")
    w("Frame identity ignores the filename decorations Lightroom and file sync add, so")
    w("`20190113-_D8S0025-2` and `_D8S0025` count as the same frame.")
    w("")
    counts = collections.Counter(g["verdict"] for g in dup_groups)
    w("| timeline verdict | frames | meaning |")
    w("|---|---|---|")
    w(f"| duplicate | {counts['duplicate']} | neighbouring trips -- candidate for one photo filed twice |")
    w(f"| wraparound | {counts['wraparound']} | far apart -- genuinely different photos, leave alone |")
    w(f"| mixed | {counts['mixed']} | more than two trips, some neighbouring and some not |")
    w(f"| undecidable | {counts['undecidable']} | a trip folder has no date prefix, so no timeline position |")
    w("")

    pixel_counts = collections.Counter(g["pixels"] for g in dup_groups if g["pixels"])
    if pixel_counts:
        w("**The timeline narrows the field; the pixels decide.** Every candidate above is")
        w("checked by decoding both JPEGs and hashing the result, and the two signals disagree")
        w("often enough that the timeline alone should not be acted on:")
        w("")
        for k, v in pixel_counts.most_common():
            w(f"- pixels {k}: {v} frames")
        w("")
        disagree = sum(1 for g in dup_groups
                       if g["verdict"] == "duplicate" and g["pixels"] == "different")
        if disagree:
            w(f"In particular **{disagree} frames sit in neighbouring trips yet are demonstrably")
            w("different photos** -- some only a day apart, occasionally the same day. So a frame")
            w("counter here does repeat far faster than a full wrap-through would explain (a card")
            w("format or a second body with the same code will do it). Treat neighbouring-trip")
            w("collisions as *candidates*, not as proof.")
            w("")
    else:
        w("Pixel verification was skipped (`--no-verify-pixels`), so these are timeline")
        w("inferences only. File bytes cannot substitute -- re-exporting rewrites metadata, so")
        w("copies are byte-distinct even when the pixels are identical.")
        w("")

    w("Recommended action per frame, which is the `action` column of the CSV:")
    w("")
    act = collections.Counter(g["action"] for g in dup_groups)
    w("| frames | action |")
    w("|---|---|")
    for a, n in act.most_common():
        w(f"| {n} | {a} |")
    w("")
    w(f"So **{act[ACT_REMOVE]} frames are confirmed duplicates** safe to collapse to one copy,")
    w(f"and {act[ACT_CHECK] + act[ACT_PARTIAL] + act[ACT_UNDATED]} need a look by hand.")
    w("")

    actionable = [g for g in dup_groups if g["verdict"] in ("duplicate", "mixed")]
    if actionable:
        w("Trips involved in a confirmed or suspected duplicate, worst first -- fixing these")
        w("folders clears the most at once:")
        w("")
        trip_hits = collections.Counter()
        for g in actionable:
            for m in g["members"]:
                trip_hits[m["trip_key"]] += 1
        w("| frames | trip |")
        w("|---|---|")
        for key, n in trip_hits.most_common(25):
            w(f"| {n} | `{key}` |")
        w("")
    w("Full list with every filename: `project/reports/duplicate_frames.csv`.")
    w("")

    w("## Worklist 2 -- photos in doubt")
    w("")
    w("Every sidecar whose JPEG did not resolve cleanly, plus every member of a duplicate")
    w("group. One row per sidecar with its trip folder, so each can be tracked down by hand.")
    w("All categories are included, not only birds.")
    w("")
    by_issue = collections.Counter(r["issue"] for r in doubt)
    w("| issue | sidecars |")
    w("|---|---|")
    for issue, n in by_issue.most_common():
        w(f"| {issue} | {n} |")
    w(f"| **total** | **{len(doubt)}** |")
    w("")
    trips_hit = collections.Counter(f"{r['year']}/{r['half_year']}/{r['trip']}" for r in doubt)
    if trips_hit:
        w("Most affected trips:")
        w("")
        w("| sidecars | trip |")
        w("|---|---|")
        for key, n in trips_hit.most_common(25):
            w(f"| {n} | `{key}` |")
        w("")
    w("Full list: `project/reports/photos_in_doubt.csv`.")
    w("")

    w("## Worklist 3 -- sidecars to re-run through bird_label")
    w("")
    w("Sidecars the labeler never produced a result for: either they carry no keywords at all")
    w("(the run never reached them) or the run recorded `missing JPEG` because the export was")
    w("not mounted. Those whose JPEG resolves today are ready to re-run now.")
    w("")
    ready = [r for r in unprocessed if r["jpg_resolves_now"]]
    w(f"- unprocessed sidecars: **{len(unprocessed)}**")
    w(f"- of those, JPEG resolves now -- **ready to re-run: {len(ready)}**")
    w(f"- still unresolvable: {len(unprocessed) - len(ready)}")
    w("")
    by_year = collections.Counter(r["year"] for r in ready)
    if by_year:
        w("| year | ready to re-run |")
        w("|---|---|")
        for y in sorted(by_year):
            w(f"| {y} | {by_year[y]} |")
        w("")
    by_reason = collections.Counter(r["reason"] for r in unprocessed)
    w("| reason | sidecars |")
    w("|---|---|")
    for reason, n in by_reason.most_common():
        w(f"| {reason} | {n} |")
    w("")
    w("Full list: `project/reports/unprocessed_sidecars.csv` (sorted with the ready ones first).")
    w("")
    return "\n".join(L)


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
    w("`project/reports/unresolved_bird_sidecars.csv` (regenerate with `./run-audit`). Filter by")
    w("`verdict` to get a specific class, e.g. every wraparound casualty:")
    w("")
    w("```bash")
    w("./run-audit")
    w("awk -F, '$5==\"off_folder_cross_year\"' project/reports/unresolved_bird_sidecars.csv")
    w("```")
    w("")
    w("`project/reports/colliding_jpg_stems.csv` lists every stem that appears in more than one")
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
    ap.add_argument("--report", type=Path, default=Path("./project/reports/data_preview_report.md"),
                    help="write the markdown report here")
    ap.add_argument("--issues-dir", type=Path, default=Path("./project/reports"),
                    help="write the per-sidecar worklist CSVs into this directory")
    ap.add_argument("--stdout", action="store_true",
                    help="print the report instead of writing any files")
    ap.add_argument("--snapshot", nargs="?", const="", default=None, metavar="LABEL",
                    help="also keep a numbered copy of this report under "
                         "<issues-dir>/archive/NN-<name>[-LABEL].md. The live report keeps "
                         "its fixed filename either way, so `git diff` between runs stays "
                         "readable; the snapshot is for comparing against a specific past run")
    ap.add_argument("--neighbour-window", type=int, default=1,
                    help="how many timeline positions apart two trips may be and still count "
                         "as neighbours, i.e. a frame collision between them is a duplicate "
                         "rather than wraparound (default 1 = strictly adjacent trips)")
    ap.add_argument("--no-verify-pixels", action="store_true",
                    help="skip decoding the duplicate candidates' JPEGs. Verification is on by "
                         "default and only touches candidate groups, so it costs a couple of "
                         "seconds -- and the timeline alone gets a meaningful number of them "
                         "wrong, so skipping it is rarely worth it")
    args = ap.parse_args()

    result = audit(args.data_dir)
    dup_groups = find_duplicate_frames(result, window=args.neighbour_window,
                                       verify_pixels=not args.no_verify_pixels)
    doubt = find_photos_in_doubt(result, dup_groups)
    unprocessed = find_unprocessed(result)

    report = render(result, args.data_dir)
    report += "\n" + render_worklists(result, dup_groups, doubt, unprocessed)

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

        # Worklist 1: one row per sidecar involved in a cross-trip frame collision,
        # grouped by frame so the copies of one photo sit together.
        path = args.issues_dir / "duplicate_frames.csv"
        fields = ["action", "verdict", "pixels", "frame", "year", "half_year", "trip", "trip_date",
                  "stem", "category", "species", "min_rank_gap", "min_day_gap",
                  "other_trips", "jpg_path", "xmp_path"]
        rows = 0
        with open(path, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=fields)
            wr.writeheader()
            for g in sorted(dup_groups, key=lambda x: (x["action"] != ACT_REMOVE, x["action"])):
                gaps = [p["rank_gap"] for p in g["pairs"] if p["rank_gap"] is not None]
                days = [p["day_gap"] for p in g["pairs"] if p["day_gap"] is not None]
                for m in g["members"]:
                    others = [x["trip_key"] for x in g["members"] if x is not m]
                    for r in m["recs"]:
                        wr.writerow({
                            "action": g["action"],
                            "verdict": g["verdict"], "pixels": g["pixels"] or "",
                            "frame": f"{g['frame'].code}{g['frame'].num}",
                            "year": r["year"], "half_year": r["half_year"], "trip": r["trip"],
                            "trip_date": m["trip"].when.isoformat() if m["trip"] and m["trip"].when else "",
                            "stem": r["stem"],
                            "category": ";".join(r["categories"]) or "(none)",
                            "species": r["species"],
                            "min_rank_gap": min(gaps) if gaps else "",
                            "min_day_gap": min(days) if days else "",
                            "other_trips": " | ".join(others),
                            "jpg_path": str(r["jpg"]) if r["jpg"] else "",
                            "xmp_path": str(r["xmp"]),
                        })
                        rows += 1
        dup_n = sum(1 for g in dup_groups if g["verdict"] in ("duplicate", "mixed"))
        print(f"wrote {path} ({rows} rows, {len(dup_groups)} frames, "
              f"{dup_n} duplicate/mixed)")

        # Worklist 2
        path = args.issues_dir / "photos_in_doubt.csv"
        fields = ["issue", "year", "half_year", "trip", "stem", "category", "species",
                  "expected_folder", "detail", "xmp_path"]
        with open(path, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=fields)
            wr.writeheader()
            wr.writerows(doubt)
        print(f"wrote {path} ({len(doubt)} rows)")

        # Worklist 3
        path = args.issues_dir / "unprocessed_sidecars.csv"
        fields = ["jpg_resolves_now", "reason", "year", "half_year", "trip", "stem",
                  "verdict", "jpg_path", "xmp_path"]
        with open(path, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=fields)
            wr.writeheader()
            wr.writerows(unprocessed)
        ready = sum(1 for r in unprocessed if r["jpg_resolves_now"])
        print(f"wrote {path} ({len(unprocessed)} rows, {ready} ready to re-run)")


if __name__ == "__main__":
    main()
