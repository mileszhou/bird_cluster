# 08 — A second labeller, and the two-level export

Session of 2026-08-24, on `main`. Two threads: a second VLM was run over the whole
library and compared against the first, and the second clustering level was built and
rendered for browsing. Everything described here is committed except where it says
otherwise.

Commits, oldest first: `bc4b75d` `72cc3b2` `f8362fa` `3aad920` `9abedfe` `8578a7c`
`a7c5f29` `ce4dbb7` `eaad8cc` `925ead3`. (`e444983`, `PROJECT_ROOT` after sourcing the
venv, is Miles's and landed mid-session.)

## Where things are

    output/label/        gemma-4-31B-it, 49,224 rows, complete
    output/embed/        DINOv3 512x512, 27,194 vectors  (copied from the 512 run)
    output/cluster/      the mcs 3/5/8/15/40 sweep       (same origin)
    output/cluster2/mcs3 layout.csv + run.json, 70 branches   -- NEW this session
    output/lightroom/jpg mcs3 and mcs8 flat exports, re-rendered after the species fix

    local/output_gemma/  the gemma run, archived, with PROVENANCE.md and FINDINGS.md
    local/output_qwen/   the Qwen3-VL-32B run + its embed/cluster, archived

`local/` is a separate git repository and is Miles's responsibility. **512 is the default
resolution now** and is no longer named in directories.

## Thread 1 — the second labelling

`google/gemma-4-31B-it` relabelled all 49,224 images against `data/label/`'s
`Qwen3-VL-32B-Instruct`. It **lost on every parameter-free measure** and that is not what
the run was for.

**What it bought is a bound on the winner's error**, which no single run can produce. Two
routes sharing no assumption put the better labelling at ~33.6% and ~36.6% species error,
against a `confidence` column implying 3.2%. Method in `project/findings/03`, numbers in
`local/output_gemma/FINDINGS.md`, per-species detail in
`local/reports/label_vs_cluster_report.md` (regenerate: `local/label_vs_cluster_report.py`,
`local/calibration_finding.py`).

Three things worth carrying forward:

1. **Match the vocabularies before measuring disagreement.** Exact string agreement is
   0.288; optimal one-to-one assignment (Hungarian on the contingency table) lifts it to
   0.380. Skipping that step overstates any error bound by nine points.
2. **The gap is self-consistency, not capability.** Same vocabulary size, same label
   length, but on images whose nearest neighbour is a cosine of 0.99 away the old run
   repeats its own name 0.94 of the time and the new one 0.72.
3. **A second labeller is worth running even when it loses**, and for calibration the
   useful one is the most *decorrelated*, not the best — shared errors inflate agreement
   and bound less.

**Do not promote `output/label` into `data/label`.** It is the second opinion; the bound
needs both readings.

**`output/label/raw/` was deleted deliberately** when the run was archived. It held this
labelling's keywords for 43,683 sidecars, and rsyncing it into Lightroom on autopilot would
install the labelling that lost.

## Thread 2 — level 2, and the export

`code/cluster/cluster2.py` (`./run-cluster2`) groups level-1 **leaves** into **branches**:
Ward over the leaf medoids, cut adaptively until no branch holds more than `--max-leaves`
(27). On mcs3 that gives **70 branches over 1,194 leaves**, median 17, max 27, nothing
pooled.

The vocabulary is settled: **leaf** = level-1 HDBSCAN cluster, **branch** = level-2 Ward
group. Level 2 has **no `mcs`** — Ward has no such parameter, and the `mcs3` in the path
names the level-1 run being grouped.

**Two levels of time carry the two levels** into Lightroom: month per branch, date per
leaf, minute per image, second left free. Verified a strict bijection — 70 months, 1,194
dates, none shared, no leaf split.

**`layout.csv` is structure only; the render owns the caption.** This was the session's
main architectural correction and it came from Miles: clustering does not read a labelling
because it does not depend on one, so a species column in a clustering artifact is a second
copy of a fact owned elsewhere. A guard was written first and then deleted — a guard exists
because two artifacts *can* disagree, and removing the second copy is the better fix.

    ./run-cluster2 --run output/cluster/mcs3
    python3 -m tools.export_seriated --layout output/cluster2/mcs3/layout.csv \
        --label-dir output/label
    # or local/exp-jpg2, which runs the pair

**Re-computation happens in `output/`.** A run archived under `local/output_*` is copied
back first; the wrappers point at the standard locations deliberately.

## What was learned from looking at it

`project/findings/01` is updated with this, and it closes two of its own open questions.

Miles reviewed the two-level export. Branches **do** pull near-identical leaves together
and sometimes reunite a species level 1 had split, but they **mostly do not read as a rank
above species** — what gathers leaves into a branch is that they look alike, which is not
the relation of shared descent he had hoped for.

The numbers half-agree, and the two halves are complementary rather than competing:

- Only the numbers could say the branching is *real* (leaves share their branch's dominant
  head noun 0.42 of the time against a shuffled null of 0.11 — 70 sigma), that it is a
  coarser **rank** rather than a blur (leaves score higher on species than head noun,
  branches the reverse), and that it is **not place** (branch-vs-trip 0.47 against
  head-noun 0.63). None of that is visible in a grid.
- Only looking could say what the rank is made of, because every target available to the
  metrics is appearance-tinged — the head noun is folk taxonomy, so "branches agree with
  head nouns" is what *both* hypotheses predict.

**Both verdicts are provisional and wait on the same missing artifact.** Miles is not a
bird taxonomist: what he saw is *different species in one branch*, and whether those species
are close relatives is exactly what an expert supplies. A branch of four congeners looks
like a mixture and is a genus.

## Bugs found and fixed, worth not re-introducing

1. **`index.csv`'s species came from `assignments.csv`**, which freezes whatever labelling
   was current when the *clustering* ran, while the JPEG keyword came from `--label-dir`.
   They agree until you export a second labeller, then disagree on 70% of rows in silence.
   Fixed by reading the label CSV, then structurally by removing the duplicate fact.
2. **Do not recover a species by parsing the composed keyword back.** `LABEL_RE` splits on
   `-` and `label_cn` is not free of hyphens or Latin text, so a hyphenated name parses one
   segment short and the English field swallows part of the Chinese one — 184 rows of one run
   come back doubled, `<name>-<name>`. Use `effective_english(row)`: the row is the source,
   the keyword is a rendering of it.
3. **The composer is the real defect, and is not fixed.** `bird_label.py:499` and
   `jpg_meta.py:265` both join three fields with `-` where two of them can contain `-`. The
   fix is to emit the bare `english(NN%)` form when `label_cn` is not clean Chinese — the
   code already does exactly that for scenery and people. ~1,655 bird rows are affected.
   **Open.**
4. **`utf-8-sig` when reading a CSV this project wrote.** They carry a BOM so Excel can show
   Chinese trip names; a plain read glues it to the first field name.

## Open, in rough priority order

1. **A taxonomy checklist** (IOC / eBird / Clements) dropped into `local/`. It is the single
   artifact two separate findings are blocked on: it turns `findings/03`'s consistency bound
   into an accuracy measure, and it settles `findings/01` by letting us look only where
   appearance and descent disagree — convergent look-alikes in different families.
2. **Different clustering methods, as a stability study rather than a bake-off.** Designed
   this session, not started. The point is not which method wins — scoring against labels
   that are ~35% wrong would reward agreeing with the noise — but which groups survive
   *every* criterion, which is what makes a batch safe to review as a unit. Four criteria
   that genuinely differ in what they assume a cluster is: HDBSCAN (density mode), k-means
   (convex, variance-balanced), Ward (least added variance), Louvain on a kNN graph
   (community). Everything needed is installed; `networkx` has Louvain, no `igraph`. Ward at
   27k needs the kNN connectivity graph or ~6 GB, and Louvain needs the same graph, so build
   it once. Compare only over points HDBSCAN assigns, so nobody is scored on the noise class.
   Deliverable is a co-association count per image pair, not a leaderboard.
3. **The `label_cn` composer fix** (bug 3 above).
4. **A wrapper for the render.** `run-cluster2`'s header has to end with a `python3 -m` line
   because the render has no wrapper, and it is the only stage shaped that way. If the
   two-level browse proves itself, that is what earns promotion — `local/exp-jpg2` is the
   candidate to extract from.
5. **`stats.py` is still unwritten**, and has been since the original plan.

## Smaller things

- `project/plans/` and `project/status/` keep the name `discover.py` deliberately: they are
  dated records, not live documentation.
- `docs/glossary.md` is new — abbreviations, house vocabulary, and the measures, with
  borrowed terms marked so a reader knows which ones have a literature behind them.
- CLAUDE.md's claim that `data/label` carries 3,693 `kept-existing` rows is **wrong**; it
  has 466. The 3,693 is the pre-merge first pass, as `local/output_qwen/label/PROVENANCE.md`
  records. Not fixed.
- `git check-ignore -v --no-index <path>` is the diagnostic for "the ignore rule looks right
  but the file still shows": plain `check-ignore` reports a *tracked* file as not ignored,
  which looks identical to a broken pattern.
