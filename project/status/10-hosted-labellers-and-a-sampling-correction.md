# 10 — Hosted labellers, and what a proper sample did to the findings

Sessions of 2026-09-04 and 09-05, on `main`. 24 commits, `e25161b` through `eeaf7e4`. Everything below is committed; nothing is pushed.

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

### The confidence column is weakly calibrated, and the first reading of it was wrong

**Read the second table, not the first.** A 200-image run suggested this column
had finally become useful; a 2,000-image *sampled* run showed most of that was
the sample. Both are recorded here because the mistake is the more instructive
half.

The labeller's `confidence` has been dismissed here for good reason: it averages
0.968 against a measured ~35% species error, and `findings/03` says plainly it is
not an error estimate. One hosted model returns a genuine spread instead — 49
distinct values over 2,022 rows — so the column is not inherently useless. What
that spread is *worth* is the question, and it is worth much less than it first
appeared.

**The biased first look**, 200 images taken in walk order — 37 distinct labels,
a handful of trips, one species dominating:

| confidence | images | disagrees |
|---|---:|---:|
| below 0.70 | 20 | 90.0% |
| 0.70–0.89 | 37 | 64.9% |
| 0.90+ | 143 | **3.5%** |

That is a 21× ratio, and it supported a workflow: review below 0.90, which is
28.5% of images and catches 89% of disagreements.

**The same model over 2,022 images drawn at random** — 759 labels across 424
folders:

| confidence | images | disagrees |
|---|---:|---:|
| below 0.50 | 97 | 92.8% |
| 0.50–0.69 | 258 | 89.9% |
| 0.70–0.79 | 407 | 88.5% |
| 0.80–0.89 | 361 | 78.7% |
| 0.90–0.94 | 279 | 71.0% |
| 0.95+ | 620 | **37.9%** |

Still monotonic across all six bands, so the calibration is real. But the spread
is **2.4×**, not 21×, and the top band still disagrees on 38%. Reviewing below
0.90 is now **56% of images for 69% of the disagreements** — against 28% for 89%.
There is no threshold that licenses skipping review.

Overall species agreement fell from 76.5% to **30.8%**, which sits alongside the
28.8% measured between the two earlier labellers over the whole library. This
model is an ordinary labeller, not an exceptional one.

**Why the first reading was wrong, and it is not subtle.** `--limit` takes images
in walk order, which is path order: 2,000 images that way is 43 folders and 326
species, against 415 and 778 drawn at random. A sample dominated by one common,
easy species makes any labeller look calibrated and accurate at once. The fix
(`--sample-seed`) landed *between* the two runs, which is the only reason the
error was caught rather than carried into a 26,104-image decision.

**And "agreement" is not "correct" either.** The comparison is against a
labelling itself ~35% wrong at species, so 30.8% agreement is consistent with
both being substantially wrong — which is what `findings/03` already bounded.

### Disagreement is *not* mostly naming, either

The same correction applies. On the 200 biased images, exact agreement was 76.5%
against **ARI 0.96**, which said the two labellings differed about names while
agreeing about the partition — and for a project whose interest is structure,
that would have made the disagreements nearly free.

On the 2,022 sampled images, exact agreement is 30.8% and **ARI is 0.41**, with
284 merges and 281 splits. The two labellings genuinely disagree about which
photos show the same bird. The naming-versus-structure distinction is still the
right frame, and a **merge** (one name covering several real species) still costs
more than a **split** — but the comforting version of the finding was an
artifact.

### `label_sci` — the checklist's strong key

`status/09` left 19.6% of bird images taking their taxonomy from BioCLIP's pixel
guess. The cause was not odd labels: the unmatched ones are ordinary birding
vocabulary, and every one of them **is** in the checklist under its binomial.
TreeOfLife carries 11,131 Aves taxa and 10,788 distinct common names, and the gap
is exactly the names people use.

So the prompt asks for the binomial, the CSV carries `label_sci`, and
`map_label_taxa` matches on it before the common name. Over 2,022 sampled images
from one hosted model: **99.7% present, 94.3% of those valid checklist taxa**,
where the common name alone would have matched 82.1%. That is **242 images, 12%
of the sample, gaining a taxonomy they would otherwise have lost** to a pixel
guess — most of the 19.6% gap, on a representative sample.

**This is the finding that survived the sampling correction**, and it is a
property of asking for the binomial rather than of the model being good: a
binomial is standardised where a common name is not, and a wrong one falls back
harmlessly.

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
   supply. The confidence gives a review order (lowest first) but not a stopping
   point: at 0.95+ the disagreement rate is still 37.9%.
2. **An expert**, for whether species inside a branch are close relatives.
3. **Phase II is done** (2,000 sampled, batched, 44 minutes). It overturned the
   confidence finding and confirmed the binomial one. What it did *not* produce
   is a cost per image, because token usage is not recorded — see below.
4. **Re-run the taxonomy against a labelling that has `label_sci`.** The 19.6%
   gap should close, which would let `audit_rank_alignment --source checklist`
   run over the whole population instead of 80% — removing the one caveat on
   `findings/01`'s rank result.
5. **`bird_label.py:39`** still imports torch through a dead backend.
6. The clustering-methods stability study, the `label_cn` composer, `stats.py`.

## What a hosted pass actually costs, and what that changes

Measured on the sampled run, from the provider's dashboard: **2,000 images, 3.47M
tokens, $1.40** — 1,735 tokens and $0.0007 per image. Roughly 250 tokens out and
1,485 in, so **the image dominates**: the `label_sci` field costs essentially
nothing, and the export resolution is what you are paying for.

| scope | cost | time at `--batch-size 8` |
|---|---:|---:|
| the bird set (`--categories bird`) | **$18** | 9.6 h |
| the whole library | $34 | 18.1 h |

**This inverts the assumption the flags were built under.** `--categories bird`
was added to save 45% of a paid run; at these prices it saves $15 and is no
longer a reason for anything. Cost is not the constraint — *time* is, and the run
saw only 7 rate limits across 2,000 images, so `--batch-size` has clear headroom
above 8.

The figure is for one deliberately cheap model and should not be generalised;
another may be an order of magnitude more. What generalises is the shape: input
tokens dominate, so cost scales with image resolution and the number of images,
not with how much the prompt asks for.

**And it changes which question matters.** At ~$18 a pass, a second and third
labelling over the bird set is ~$37 — so the choice stops being "can we afford
another opinion" and becomes "which opinion is most decorrelated", which is what
`findings/03` says makes a second labelling useful in the first place. A model
agreeing with the curated labelling only 30.8% of the time is a candidate for
that role even though it is unfit for promotion.

**Token usage is still not recorded in the CSV.** The response carries a `usage`
block and only `reasoning_tokens` is read from it, for an error message, so this
number had to come from a dashboard rather than from the artifact. Worth adding:
a run that can price itself needs no external bookkeeping, and the per-image cost
is a property of the run worth keeping beside its parameters.

## A note on the confidence column

CLAUDE.md says flatly that the confidence "is not an error estimate". After the
sampled run that statement stands, and should not be softened. One model returns
a genuine spread and the spread is monotonic against agreement — so the column is
not *meaningless* — but at 0.95+ it still disagrees with the other labelling 38%
of the time, which is not an error estimate by any useful definition.

The wider lesson is the one worth carrying: **a measurement on a convenience
sample can invert.** 21× became 2.4×, ARI 0.96 became 0.41, and 76.5% agreement
became 30.8%, from the same model and the same code. Nothing was wrong with the
first measurement except which images it was taken over.
