# `_local/` — the template for your `local/`

`local/` is gitignored and holds everything specific to you: the commands you
actually ran this week, scope lists naming real places, working copies of
generated reports. Nothing in it is published and nothing in the repository
reads it.

This directory is the starting point for that, tracked so there is something to
start from:

    cp -r _local local

Same convention as `_env` → `.env` and `_config.local.toml` →
`config.local.toml`: the underscore copy is tracked and generic, the copy you
make is yours and ignored.

`runbook` is a pipeline pass written as one script and meant to be walked a line
at a time — it is deliberately not executable, because every step has something
worth reading before the next one starts. Its defaults are the project's; change
them, and your copy becomes the record of what you actually ran.

Why the underscore copy exists at all: a command list is the one artifact that
says what a run *was*, and reconstructing one from shell history six months later
does not work. `docs/running-a-study.md` explains why each step is there; this
is the step list itself.
