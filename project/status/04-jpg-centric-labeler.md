# 04 — JPEG-centric labeler, ready for the full run

Branch `cluster`. Supersedes `03-data-reconciled.md`, which recorded the dataset reaching a
clean audit. This one records the labeler being rebuilt around that dataset, and is the
**baseline commit taken immediately before one more round of Lightroom deduplication**.

Everything below is verified against the real tree (43,728 sidecars, 49,270 JPEGs), not
inferred. 169 tests pass.

## What changed

### 1. The walk inverted: JPEG-driven, not sidecar-driven

The labeler used to iterate sidecars and resolve each one's JPEG, with sidecar-less images an
opt-in extra (`--include-orphan-jpg`, off by default — so 5,229 photos were never labelled).
It now walks `data/jpg`: **one row per JPEG**, and the sidecar is a destination for the label
rather than the thing enumerated. The flag is gone; those photos are ordinary members of the
population.

```
items (one per jpg):       49270
  write to a sidecar:      43728        (each sidecar exactly once)
  csv-only:                 5542        = 5229 never had a raw + 313 alternate edits
sidecars reached:          43728 of 43728     unreached: 0
```

**The claim rule is local** (`code/lib/jpg_claim.py`) — it reads the JPEG's own name and the
sidecar tree, never another JPEG, so per-photo processing stays independent. Exact stem first,
else strip trailing `-<decoration>` segments longest-base-first.

307 sidecars are reached by more than one JPEG. First claimant in **stem-sorted** order wins,
and that is always the exact-stem match — `A` is a proper prefix of `A-2`, so it sorts first;
verified for all 307 groups. Not a tie-break heuristic: the right answer, reached without
comparing candidates. Sort on the stem, never the filename (`A-2.jpg` < `A.jpg`, since `-` is
0x2D and `.` is 0x2E, which would hand the sidecar to the virtual copy).

Claims are assigned over the whole sorted tree **before** the checkpoint filters anything, so
`build_items()` is a pure function of the two trees and a resumed run reproduces the identical
assignment. Tracking claims within one process would hand a sidecar to a virtual copy after a
Ctrl-C.

**The blind spot this introduces:** a sidecar no JPEG reaches is not skipped with a warning, it
is never enumerated. 0 today, but that is a property of the current export, so the labeler
counts unreached sidecars and warns, and the audit reports the same number.

### 2. Sidecars are edited as text, not reserialised

`code/lib/xmp_write.py`. The old writer parsed with ElementTree and re-wrote the file, which
renamed every namespace prefix it had not registered (`x:xmpmeta` → `ns0:xmpmeta`) and
collapsed Lightroom's one-attribute-per-line layout. A 348-line sidecar came back as 120 and
`diff` showed the whole file changed — unreviewable, for files being rsynced into a photo
library. Registering the source's own prefixes fixes the names but not the reflow; the loss is
inherent to parse → object → serialise.

Now the parser only *reads*; the change is applied as a text substitution. **Diffs went from
268 lines to 4–6.** `verify_only_keywords_changed()` re-parses every edit and asserts nothing
outside the keyword block moved, so the shortcut fails loudly instead of corrupting a file.

Verified over the whole library: 43,728 sidecars edited in a dry run, 0 guard failures, max
diff 8 lines. On the 549 written in a live run: `xmpMM:History` and every `crs:` develop
setting unchanged, no attribute or element lost, no `ns0:` prefixes.

### 3. `lr:hierarchicalSubject` no longer flattened

It holds keyword *paths* — `People|Family|Miles` is one keyword naming a position in
Lightroom's keyword tree. The old writer mirrored the flat `dc:subject` list into it, which
would have turned that into three unrelated top-level keywords on import. 71 sidecars carry
paths; 12 would have been stripped unconditionally on a full run. `merge_hierarchical()` now
replaces only entries whose *leaf* is one of ours.

### 4. CSV schema, and a guard on it

`jpg, xmp, filename, category, label, label_cn, confidence, note, prior_category, prior_label,
applied, run_label, response_json`

`jpg` is the key (relative to `data/jpg`); `xmp` is the sidecar written (relative to
`data/xmp`, empty for `csv-only`). `source` retired — the two columns say it directly. The
`jpg` column matters because for 189 captures the only export is decorated, so the image that
produced a label cannot be derived from the sidecar name.

`check_csv_schema()` refuses to resume against a mismatched header rather than appending rows
one field left of their labels. `csvfile.flush()` now precedes the checkpoint write, so a hard
kill cannot mark a photo done with no row (Ctrl-C was never affected — it exits through the
`with`).

### 5. Audit extended

`multi_claim.csv` plus a report section: sidecars several JPEGs reach, who keeps it, who goes
CSV-only. Reported rather than repaired — the resolution is deterministic — but the count is a
fingerprint of the export's shape.

## Corrections to earlier documents

`CLAUDE.md` carried stale figures from before the reconciliation. Now: **0** sidecars with no
JPEG (was 105), **189** resolving via a derived export (was 5,990), **5,229** JPEGs with no
sidecar (was 4,857), **17,590** Bags / **1,748** Seqs.

`./run-dedup` reports **45 cross-trip duplicates**, not the 0 CLAUDE.md claimed. The first pass
cleared them on 2026-08-02; the reconciliation that followed reintroduced some. That is what
the next round addresses.

`JpgIndex.extras()` is *not* buggy — an earlier census here counted `match.path` (the first)
where a `DERIVED` sidecar legitimately carries several in `match.paths`. Counting all of them
reconciles exactly: 49,270 = 49,270.

## State at this commit

- `data/` submodule at `a1bb91b`, clean. This is the pre-dedup snapshot of every sidecar.
- `project/reports/data_preview_report.md` and `sidecar_dedup_report.md` committed as the
  **before** side of the dedup diff. Their filenames never change, so `git diff` after the next
  run shows exactly what moved.
- `output/` holds a 192-row smoke run on the new schema, not a real pass. Discard it.
- The full run has **not** been started — it waits on the dedup and re-export.

## Next

1. Lightroom dedup + re-export (see `05-post-dedup-checklist.md`).
2. `./clean`, then the full run: 49,270 images, ~11–18 hours at the 0.75–1.33 img/s observed.
3. Archive the finished CSV outside `output*/` before any later `./clean`.
4. Then embedding — `embed.py` still walks sidecars and needs pointing at the CSV.

## Open, not blocking

- **Confidence is not calibrated.** The model emits two values, 0.95 and 0.98, so
  `--conf-threshold` and `--no-bird` can never fire and `--filter-csv`'s low-confidence branch
  is dead code. The prompt asks for "a float between 0.0 and 1.0" with no rubric.
- **The server directory is misnamed** `/data/models/Qwen3-VL-132B-Instruct` while serving a
  32B model. `args.json` records the directory, so it will misattribute the whole run.
- **`embed.py` is still sidecar-driven** and structurally cannot see the 5,229 sidecar-less
  JPEGs. Repointing it at the CSV is the next architectural step; the filter must read
  `'bird' in (category, prior_category)`, not `category == 'bird'`, or it drops every photo the
  never-demote rule protects.
