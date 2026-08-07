# 05 — Post-dedup completeness checklist

Run after the Lightroom deduplication and re-export, **before** starting the full labelling
run. The baseline to compare against is the commit carrying `04-jpg-centric-labeler.md`, which
was taken deliberately just before the dedup.

Every number below has a recorded "before". The point of the checklist is that each check has
an expected direction — a number moving the *wrong* way is the signal, not the number itself.

---

## 0. Before you start

- [ ] `git status` in `data/` is clean and the submodule pointer is committed. The re-export
      changes both trees; without a commit there is no "after" to diff against.
- [ ] Note the new `data/` commit hash here: `________________`

---

## 1. The one check that gates everything else

**Sidecars no JPEG reaches.** This is the blind spot of the JPEG-driven walk: such a photo is
not skipped with a warning, it is never enumerated. It will be labelled by nobody and noticed
by no one except this check.

- [ ] `./run-audit`
- [ ] **`missing_jpg.csv` has 0 rows.** Before: 0.

If it is non-zero, do not start the run. A partial re-export is the usual cause — the report
splits them by the exporter's own reason. Re-export those folders and repeat.

The labeler warns on the same condition independently, so a non-zero count will also appear at
the top of the run as `⚠️ N sidecars have no JPEG and will not be labelled`.

---

## 2. Did the dedup do what was intended?

- [ ] `./run-dedup`
- [ ] `git diff project/reports/sidecar_dedup_report.md` — this is the direct before/after.

| line | before | expect after |
|---|---|---|
| cross-trip duplicates | **45** | **0**, or a small remainder you chose to keep |
| within-trip duplicates | 0 | 0 |
| virtual copies | 5 | 5 — *deliberate alternate edits, never delete* |
| redundant sidecars to remove | 45 | matches whatever you actually removed |

- [ ] If a cross-trip duplicate survived, it is because you decided to keep it — confirm that
      rather than assuming the tool missed it.
- [ ] Virtual copies must **not** have dropped. They share an `OriginalDocumentID` but have
      their own `DocumentID`; losing one means an edit was deleted.

---

## 3. Did the counts move consistently?

`git diff project/reports/data_preview_report.md`. Sidecars removed should reduce the JPEG
count by the same amount — otherwise the export and the library disagree.

| figure | before | expect after |
|---|---|---|
| sidecars scanned | 43,728 | 43,728 − (sidecars you removed) |
| matched to a JPEG | 43,728 (100%) | **100%** — this must not fall |
| under their own name | 43,539 | ↓ by roughly the same |
| via a derived export | 189 | ~unchanged |
| JPEGs exported | 49,270 | ↓ by the removed captures' exports |
| JPEGs with no sidecar | 5,229 | **~unchanged** — a jump means a re-export orphaned files |
| bird-categorised sidecars | 6,505 | ↓ only by removed duplicates |

- [ ] **Matched is still 100%.** Everything else is informational; this one is a gate.
- [ ] "JPEGs with no sidecar" did not jump. A large increase means the re-export wrote files
      into folders the sidecar tree does not mirror — check `orphan_folders` in the report.

---

## 4. Export shape

- [ ] `multi_claim.csv`: **307 rows** before, 313 JPEGs downgraded to `csv-only`.

A modest fall is expected (removed duplicates take their exports with them). A *jump* means
something changed about how Lightroom emitted virtual copies — worth understanding before
committing 11+ hours of labelling to it.

- [ ] `exact_match_wins = false` count: **1** before. These are captures whose master was never
      exported, so a decorated file is the only candidate. If this grows, masters are being
      dropped from the export set.

---

## 5. Nothing was labelled with stale keys

- [ ] `./clean` — the current `output/` references the pre-dedup trees. Its checkpoint keys are
      JPEG paths, so anything renamed or removed would resume onto the wrong photo.
- [ ] Confirm `output/` is empty apart from `raw/` being re-copied on the next run.

---

## 6. Sanity check before committing to the long run

- [ ] Start `./run-vllm` and read the first three lines:
      - `⚙️  N JPEGs to label; M write to a sidecar, K to the CSV only` — N, M, K should match
        the audit's figures exactly.
      - **No** `⚠️ … sidecars have no JPEG` line.
      - Server model line — confirm it is the model you mean. The directory is currently
        misnamed 132B while serving 32B; `args.json` records the directory.
- [ ] Let ~200 images run, Ctrl-C, then check:
      - `wc -l output/processed.txt` equals the CSV's data rows.
      - Header is the 14-column schema starting `jpg,xmp,filename`.
      - A written sidecar's diff against `data/xmp` is 4–6 lines, all `<rdf:li>`.
      - At least one `applied=csv-only` row appears (sidecar-less JPEGs are in scope now).
- [ ] `./clean` again, then start the real run.

---

## 7. After the full run completes

- [ ] Archive the CSV **outside** `output*/` before any later `./clean` — `output*/` is
      gitignored and `./clean` is `rm -rf`. The `results/<label>/` pattern from git history is
      the precedent.
- [ ] `rsync` `output/raw/` into the library, then *Metadata → Read Metadata from File* in
      Lightroom. The writer does not bump `xmp:MetadataDate`, so Lightroom will not flag the
      change on its own.
- [ ] Spot-check in Lightroom that develop history and edits survived on a handful of photos
      that carried heavy `crs:` settings.
- [ ] Only then consider `data/xmp` for re-population from the library.

---

## Deliberately not on this list

**Duplicate captures producing two rows.** With cross-trip duplicates cleared this mostly
disappears, but 5 virtual copies remain by design and they are two rows for one capture. That
is correct and expected — dedupe on the CSV if and when a study needs it, per principle 3.
Do not chase it in Lightroom.
