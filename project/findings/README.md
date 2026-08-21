# Findings

Things that were **measured**, and what the numbers mean.

Kept separate from `ideas/` and `plans/` because it is a different kind of
document, and the three were collapsing into each other. An idea is a thought
with no evidence yet; a plan is a commitment to an approach; a finding is
neither — it is a result, and it does not oblige anyone to do anything.

`ideas/04` was the entry that made the gap obvious. It arrived as an idea, grew
four tables of measurements over an afternoon, and stopped being an idea while
still sitting in a directory whose README says an entry "does not need a design,
and it should not pretend to have one".

An entry should say enough to survive without the conversation around it: what
was measured, on what, what the numbers were, and what they support. **Generate
the numbers from the artifacts rather than transcribing them**, for the same
reason a run's `FINDINGS.md` does — a hand-copied figure will eventually
disagree with the file beside it. State the caveats as plainly as the result;
a finding whose limits are not written down is one that will be over-claimed
later, by us.

Numbered in the order they were found, like `status/` and `ideas/`. A finding
may point at an idea or a plan, and often should, but it stands on its own if
neither ever happens.

## Two things that belong elsewhere

**A finding about one run** goes in that run's own `FINDINGS.md`, at the root of
its `output_NNN_<description>/`, so it travels with the artifacts it describes.
This directory is for what outlives a single run — a property of the method, not
of the pass.

**A finding about the collection** is data research, not software, and does not
belong in this repository at all. The line is what the document is *about*: how
clustering behaves at a coarse level is method; which of your species get
confused with which is a private photo library. That goes to `local/`, or the
private repository. Run `python3 -m tools.audit_report_safety` before keeping
anything here.
