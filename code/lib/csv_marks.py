"""Which post-processors have been applied to a CSV, recorded in the file itself.

The record is a **column whose name is the state and whose values are empty** --
`C_3` means bits 0 and 1 are set. Every row carries an empty field for it, which
costs one character per line and which every CSV reader ignores.

Why a bitfield in the header rather than something more obvious:

**It travels with the file.** A sidecar `post.json` gets separated from its CSV
by exactly the copy, rename or archive this project does constantly.

**It works for tools that leave no trace.** A tool that only *reorders* rows adds
no column, so "look at the columns to see what ran" cannot detect it. Sorting is
precisely the kind of post-processing expected here, so the record has to be
explicit rather than inferred.

**Dependencies become one expression.** `(state & REQUIRES) == REQUIRES`, with
the registry a constant table instead of logic spread across tools.

Bits are permanent. A tool's bit is its identity, so an obsolete one is retired
by leaving the bit unused, never by reassigning it -- a reused bit would make old
files claim a tool ran that never did. That is the failure mode a registry has to
be disciplined about, and the only one.

The registry lives here so a tool declares a bit and a requirement, nothing more.
"""

import re
from dataclasses import dataclass

MARK_RE = re.compile(r"^C_([0-9A-Fa-f]+)$")


@dataclass(frozen=True)
class Tool:
    bit: int
    name: str
    module: str          # importable path, for resolving a missing prerequisite
    requires: int = 0    # mask of bits that must already be set


# Bit assignments are permanent. Add to the end; never reuse a retired bit.
REGISTRY = {
    "add_cluster_size": Tool(0x01, "add_cluster_size", "tools.add_cluster_size"),
    "order_assignments": Tool(0x02, "order_assignments", "tools.order_assignments",
                              requires=0x01),
}
BY_BIT = {t.bit: t for t in REGISTRY.values()}


def read_state(fieldnames) -> tuple[int, str | None]:
    """(state, the column holding it) from a header row."""
    for name in fieldnames or ():
        m = MARK_RE.match(name or "")
        if m:
            return int(m.group(1), 16), name
    return 0, None


def column_for(state: int) -> str:
    return f"C_{state:X}"


def missing(state: int, tool: Tool) -> list[Tool]:
    """Prerequisite tools whose bits are absent, in registry order."""
    return [BY_BIT[b] for b in sorted(BY_BIT)
            if (tool.requires & b) and not (state & b)]


def describe(state: int) -> str:
    applied = [BY_BIT[b].name for b in sorted(BY_BIT) if state & b]
    unknown = state & ~sum(BY_BIT)
    text = ", ".join(applied) if applied else "nothing"
    if unknown:
        # A file written by a newer checkout than this one. Say so rather than
        # silently ignoring bits we cannot name.
        text += f" (+ unknown bits 0x{unknown:X})"
    return text


def apply_mark(rows, state: int, old_column: str | None, tool: Tool):
    """Set `tool`'s bit on every row, renaming the marker column."""
    new_state = state | tool.bit
    new_column = column_for(new_state)
    for r in rows:
        if old_column and old_column in r:
            del r[old_column]
        r[new_column] = ""
    return rows, new_column
