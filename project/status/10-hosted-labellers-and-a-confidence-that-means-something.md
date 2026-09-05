# 10 — Hosted labellers, and a confidence that means something

Sessions of 2026-09-04 and 09-05, on `main`. 22 commits, `e25161b` through
`c7be7ce`. Everything below is committed; nothing is pushed.

Two threads, and the second one only became visible because the first was done.

## Thread 1 — the cloud backend, which had never worked

`--approach chatgpt` raised `NameError` on the first image of every run, and had
done since the module split: the call was unqualified while the function lived in
another module. The run still printed **"Run complete"**, which is why nobody
noticed.

Behind it sat a second copy of the vLLM transport that had drifted while
unexercised: a prompt from before `bird` meant class Aves, `max_tokens` too small
for the JSON now asked for, no timeout, no Chinese-in-`label` post-processing,
and `json.JSONDecode_decodeError` — an attribute that does not exist, so every
code-fenced reply silently became `scenery/unknown`.

**Fixed by deletion.** vLLM, llama.cpp and OpenAI all speak one protocol, so
there is now one transport and one prompt, differing by a URL and an
`Authorization` header. A second copy of a thing nobody runs is a thing that
rots.

Renamed to **`--approach openai`**: `$OPENAI_BASE_URL` points it at any endpoint
speaking the protocol — Azure, OpenRouter, a gateway, a stub — so naming it for
one company's product named something it may never touch. No alias kept, which
cost nothing since no working script could have used the old name.

### What a hosted endpoint needs that a local server does not

Each of these was found by a real run failing, in this order:

1. **Parameters are not universal.** GPT-5 and the o-series reject `max_tokens`
   for `max_completion_tokens` and refuse a non-default `temperature`. A
   hardcoded table of which model wants which spelling goes stale; the endpoint
   is asked instead, its 400 read, the payload corrected and cached per endpoint
   and model.
2. **A reasoning model spends the budget before it answers.** At the inherited
   200 the whole allowance went on thinking and the reply was empty. Both halves
   are needed — room (`REASONING_BUDGET`) *and* less thinking
   (`reasoning_effort=minimal`) — because reasoning tokens are billed and a
   generous budget across 26k images is money spent on deliberation nobody reads.
3. **Not every endpoint announces the problem.** OpenRouter accepts
   `max_tokens: 200` and then starves on it, so the 400-driven adaptation never
   fired. An empty answer that stopped at the limit is now a trigger too.
4. **A 429 is a queue, not a refusal.** Retried with doubling waits honouring
   `Retry-After`; a 4xx other than 429 is not retried, since a retried refusal is
   a paid call wasted. A long `Retry-After` is a verdict on the tier: at 60s a
   request, 26,104 images is eighteen days.
5. **Concurrency was gated to vLLM by accident.** The batch path is a thread pool
   of ordinary HTTP requests, not a vLLM feature. Ungating it took a serial
   sixteen hours to about two.

### And the reporting that hid it

A run that labelled nothing said **"Run complete"** and exited 0, because every
per-image failure is caught, logged, and breaks the loop. Worse: a transport
failure returned the predictor's `scenery/unknown/0.00` defaults, which are a
plausible label for a real photograph — written, checkpointed, never retried. A
dead server or one timeout became a permanent wrong label.

Now: an unanswered photo gets no row, no sidecar edit and **no checkpoint entry**,
so a re-run retries exactly those; a run that labelled nothing exits non-zero; and
a run with gaps says so.

## Thread 2 — confidence, and a second field worth asking for

### The confidence column was useless, and is not always

The labeller's `confidence` has been dismissed here for good reason: it averages
0.968 against a measured ~35% species error, and `findings/03` says plainly it is
not an error estimate. **That is a property of the models tried, not of the
column.** Over 200 images one hosted model produced this:

| confidence | images | agrees with the curated labelling |
|---|---:|---:|
| below 0.70 | 20 | 10.0% |
| 0.70–0.89 | 37 | 35.1% |
| 0.90+ | 143 | 96.5% |

Monotonic, and a **21× ratio** in disagreement rate between the top band and the
rest. 26 distinct values over 200 images, against a model that returns 0.968 for
almost everything.

**What it buys is triage.** Reviewing only below 0.90 is 28.5% of the images and
catches 89.4% of the disagreements — about 7,400 to review instead of 26,104,
with ~650 disagreements accepted unreviewed.

**The threshold is not fittable yet.** Youden's J picks 0.88 on this sample, but
bootstrapping puts the optimum anywhere between 0.88 and 0.98, and 0.88 and 0.90
give identical results here. 0.90 is inside the interval and is a number you can
state. The principled choice needs a cost ratio — how many images you would
review to prevent one accepted error — which is a judgement about purpose rather
than a statistic.

**And "agreement" is not "correct".** The comparison is against a labelling that
is itself ~35% wrong at species, so a 0.90+ row means two independent models
concur, which is evidence and not a verdict.

### Most disagreement is naming, not structure

Worth recording because it settles how much the disagreements matter. Over the
same 200 images: exact string agreement 76.5%, but **ARI 0.96** — for almost any
two photos, both labellings agree on whether they show the same bird. The gap is
the naming-versus-structure distinction, and for a project whose interest is
structure, a consistent wrong name is nearly free.

The exception is worth knowing: a **merge** (one name covering several real
species) destroys a distinction, where a **split** only relabels. 10 merges and
12 splits in 200 images, and the merges concentrate in the low-confidence band —
so the triage already separates the harmful case from the harmless one.

### `label_sci` — the checklist's strong key

`status/09` left 19.6% of bird images taking their taxonomy from BioCLIP's pixel
guess. The cause was not odd labels: the unmatched ones are ordinary birding
vocabulary, and every one of them **is** in the checklist under its binomial.
TreeOfLife carries 11,131 Aves taxa and 10,788 distinct common names, and the gap
is exactly the names people use.

So the prompt asks for the binomial, the CSV carries `label_sci`, and
`map_label_taxa` matches on it before the common name. Over 200 images from one
hosted model: **200/200 present, 200/200 valid checklist binomials**, where the
common name alone matched 170/200. That closes the gap rather than narrowing it.

The asymmetry that justified a schema change: a wrong binomial is cheap — it
matches nothing and falls back to the common name, exactly as a run without the
column does. A wrong common name has no backstop.

## Smaller things, each with a reason

- **`--categories`** filters at run time against a previous labelling, so a paid
  run can skip what is already known not to be a bird: 26,104 of 47,908, 45% of
  the cost, spending the reliable half of a labelling (two labellers agree on
  category 0.9626 and on species 0.288). Resolved at run time rather than through
  a generated manifest — a generated list would live in gitignored `local/`, so a
  `run.json` naming one is a dangling reference for everybody else.
- **Prior labels come from the CSV**, not the sidecars. A sidecar exists only for
  a photo that had a raw, so the old route covered 9,918 of 49,224 rows — the
  paired-verdict comparison was a comparison of nothing over a fifth of the
  library.
- **`--limit`** for trying a model in five calls rather than 47,908, and
  **`--sample-seed`** because walk order is not a sample: 2,000 images in path
  order is 43 folders and 326 species, against 415 and 778 drawn at random.
  Anything fitted on the first describes those trips.
- **`label_cn` is normalised to simplified**, since one bird under two scripts is
  two keywords. It fixes 7 of the 980 names carrying several Chinese forms; the
  other 973 differ by genuine synonymy and remain open.
- **`tools/map_cn_names.py`** builds a canonical Chinese name per taxon from the
  most self-consistent labelling, keyed on the binomial where one is known.
  Reports; does not rewrite.

## Bugs worth not re-introducing

1. **A failure that reports success is worse than a crash.** Two instances this
   session: the run summary, and a transport error returning plausible defaults.
   Both hid real breakage for months.
2. **Test the concurrent path, not only the serial one.** `write_row` gained a
   parameter; the single-image call was updated and the batch one was not, and no
   test drove the batch path — which is the one any real run uses. A 2,000-image
   run died on its first batch.
3. **Walk order is not a sample.** The trap already recorded for the embedding
   JSONL, one stage earlier, where it would have cost a paid run rather than a
   re-measurement.
4. **An adaptation that keys on one signal misses the same failure arriving by
   another.** Adapting on a 400 left the silent-starvation case exactly as broken,
   with a better error message.

## Open, in rough priority order

1. **The review verdict** for `findings/01` — still the item nothing here can
   supply, and now cheaper to produce: with a calibrated confidence, a review can
   start with the images where it is low.
2. **An expert**, for whether species inside a branch are close relatives.
3. **Phase II: ~2,000 images, sampled, batched.** Confirms whether the
   calibration and the binomial validity hold beyond one sample, and gives a real
   cost per image. Fit the threshold on half and check on the other; fitting and
   evaluating on the same rows overstates it.
4. **Re-run the taxonomy against a labelling that has `label_sci`.** The 19.6%
   gap should close, which would let `audit_rank_alignment --source checklist`
   run over the whole population instead of 80% — removing the one caveat on
   `findings/01`'s rank result.
5. **`bird_label.py:39`** still imports torch through a dead backend.
6. The clustering-methods stability study, the `label_cn` composer, `stats.py`.

## A note on the confidence column

If a labelling with calibrated confidence is ever promoted, CLAUDE.md's flat
statement that the confidence "is not an error estimate" becomes false for that
run. It is true of the models used so far and should be qualified rather than
deleted — the point it makes is that the column cannot be trusted *by default*,
which remains right.
