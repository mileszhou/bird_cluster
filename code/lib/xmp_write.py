"""Set a sidecar's keywords by editing the file text, not by reserialising it.

The obvious implementation -- parse with ElementTree, mutate, `tree.write()` --
is correct but rewrites the whole file. ElementTree's object model has no place
to keep formatting: element text and tails survive, but the whitespace *inside*
an opening tag does not, and Lightroom writes one attribute per indented line:

    <rdf:Description rdf:about=""
       tiff:Make="NIKON CORPORATION"
       tiff:Model="NIKON D610"

On serialisation those collapse onto one line, all `xmlns:` declarations get
hoisted to the root and sorted, and any prefix not passed to
`ET.register_namespace` is renamed to `ns0:`, `ns2:`... -- so `x:xmpmeta`
becomes `ns0:xmpmeta`. A 348-line sidecar comes back as 120 lines and `diff`
reports the entire file changed. Registering the source's own prefixes fixes the
names but not the reflow; the loss is inherent to parse -> object -> serialise.

That matters because these files are rsynced back into a Lightroom library.
A change you cannot read in `git diff` is a change you cannot check before
pushing it across 43k photos.

So: read with the parser (deciding which keywords are the user's needs real
parsing), but apply the change as a text substitution, leaving every byte
outside the keyword block alone. A `dc:subject` insertion becomes a 7-line diff
instead of 268. `verify_only_keywords_changed()` re-parses the result and
asserts nothing but the keywords moved, so the shortcut cannot silently corrupt
a file.
"""

import re
import xml.etree.ElementTree as ET

DC_URI = "http://purl.org/dc/elements/1.1/"
RDF_URI = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
LR_URI = "http://ns.adobe.com/lightroom/1.0/"

SUBJECT_TAG = f"{{{DC_URI}}}subject"
HIER_TAG = f"{{{LR_URI}}}hierarchicalSubject"

# Matched with a backreference so <dc:subject> is closed by </dc:subject> and
# never by some other prefix. The negative lookbehind keeps the self-closing
# form <dc:subject/> out of this pattern -- it is handled separately.
def _block_re(tag: str) -> re.Pattern:
    return re.compile(
        r"(?P<indent>[ \t]*)<(?P<prefix>[\w.-]+):%s\b[^>]*(?<!/)>"
        r"(?P<body>.*?)</(?P=prefix):%s>" % (tag, tag),
        re.S,
    )


SUBJECT_RE = _block_re("subject")
HIER_RE = _block_re("hierarchicalSubject")
EMPTY_SUBJECT_RE = re.compile(r"(?P<indent>[ \t]*)<(?P<prefix>[\w.-]+):subject\b[^>]*/>")
CONTAINER_RE = re.compile(r"<rdf:(Bag|Seq|Alt)\b")


class XmpEditError(Exception):
    """The sidecar is not shaped the way the edit requires."""


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _layout(indent: str, body: str):
    """(container_indent, li_indent), or None when the block is on one line.

    Derived from the file rather than assumed, so Lightroom's one-space-per-level
    and a compact single-line block both round-trip unchanged.
    """
    if "\n" not in body:
        return None
    m = re.search(r"\n([ \t]*)<rdf:(?:Bag|Seq|Alt)\b", body)
    container = m.group(1) if m else indent + " "
    m = re.search(r"\n([ \t]*)<rdf:li\b", body)
    return container, (m.group(1) if m else container + " ")


def _render(indent, prefix, tag, container, items, layout):
    open_tag, close_tag = f"<{prefix}:{tag}>", f"</{prefix}:{tag}>"
    lis = [f"<rdf:li>{_esc(i)}</rdf:li>" for i in items]
    if layout is None:
        return f"{indent}{open_tag}<rdf:{container}>{''.join(lis)}</rdf:{container}>{close_tag}"
    ci, li = layout
    lines = [f"{indent}{open_tag}", f"{ci}<rdf:{container}>"]
    lines += [f"{li}{x}" for x in lis]
    lines += [f"{ci}</rdf:{container}>", f"{indent}{close_tag}"]
    return "\n".join(lines)


def _end_of_open_tag(text: str, start: int) -> int:
    """Index just past the '>' that closes the tag beginning at `start`.

    Quote-aware, because attribute values contain '>' -- the XMP toolkit version
    string in x:xmptk is one.
    """
    quote = None
    for i in range(start, len(text)):
        c = text[i]
        if quote:
            if c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c == ">":
            return i + 1
    raise XmpEditError("unterminated opening tag")


def _dc_prefix(text: str) -> str | None:
    m = re.search(r'xmlns:([\w.-]+)="%s"' % re.escape(DC_URI), text)
    return m.group(1) if m else None


def set_subject_keywords(text: str, subjects, hierarchical=None) -> str:
    """Return `text` with dc:subject set to `subjects`.

    `lr:hierarchicalSubject` is rewritten to `hierarchical` only when the sidecar
    already has one; it is never created. Lightroom mirrors dc:subject there and
    resurrects keywords from it on import, so a stale one is worse than none --
    but an absent one is not a gap to fill.
    """
    subjects = list(subjects)
    match = SUBJECT_RE.search(text) or EMPTY_SUBJECT_RE.search(text)
    if match:
        indent, prefix = match.group("indent"), match.group("prefix")
        body = match.groupdict().get("body") or ""
        found = CONTAINER_RE.search(body)
        container = found.group(1) if found else "Bag"
        block = _render(indent, prefix, "subject", container, subjects, _layout(indent, body))
        text = text[: match.start()] + block + text[match.end():]
    else:
        text = _insert_subject(text, subjects)

    if hierarchical is not None:
        match = HIER_RE.search(text)
        if match:
            indent, prefix, body = match.group("indent"), match.group("prefix"), match.group("body")
            found = CONTAINER_RE.search(body)
            block = _render(indent, prefix, "hierarchicalSubject",
                            found.group(1) if found else "Bag",
                            hierarchical, _layout(indent, body))
            text = text[: match.start()] + block + text[match.end():]
    return text


def _insert_subject(text: str, subjects) -> str:
    """Add a dc:subject to a sidecar that has none (24,389 of 43,728 today).

    Anchored immediately *after* the first `<rdf:Description ...>` opening tag,
    not before the first `</rdf:Description>`: 92% of these sidecars nest further
    Descriptions inside crs:Look and friends, so the first closing tag usually
    belongs to a nested one. Property order inside a Description carries no
    meaning in RDF, so the head of the element is a safe, unambiguous anchor.
    """
    start = text.find("<rdf:Description")
    if start < 0:
        raise XmpEditError("no rdf:Description to attach dc:subject to")
    end = _end_of_open_tag(text, start)

    prefix = _dc_prefix(text)
    if prefix is None:
        # No dc binding to reuse; declare one on the Description we are editing.
        prefix = "dc"
        text = text[: end - 1] + f'\n   xmlns:dc="{DC_URI}"' + text[end - 1:]
        end = _end_of_open_tag(text, start)

    line_start = text.rfind("\n", 0, start) + 1
    indent = text[line_start:start] + " "
    block = _render(indent, prefix, "subject", "Bag", subjects,
                    (indent + " ", indent + "  "))
    return text[:end] + "\n" + block + text[end:]


# --- the guard --------------------------------------------------------------

def items_of(root, tag):
    out = []
    for node in root.iter(tag):
        for container in node:
            out += [(li.text or "") for li in container]
    return out


def _skeleton(root):
    """Everything about the document except the two keyword subtrees.

    Compared before and after an edit, this is what proves the text surgery
    touched nothing else -- attributes, values, element count and element text.
    Whitespace is deliberately not compared: reflowing the keyword block shifts
    the surrounding tails, and that is the one change we are making on purpose.
    """
    facts = []

    def walk(el, depth):
        # Prune both keyword subtrees entirely, node included: inserting a
        # dc:subject into one of the 24,389 sidecars that lack one legitimately
        # adds that node, and their *contents* are checked separately against
        # exactly what was asked for.
        if el.tag in (SUBJECT_TAG, HIER_TAG):
            return
        facts.append((el.tag, depth, tuple(sorted(el.attrib.items())),
                      (el.text or "").strip()))
        for child in el:
            walk(child, depth + 1)

    walk(root, 0)
    return facts


def verify_only_keywords_changed(before: str, after: str, subjects, hierarchical=None):
    """Raise XmpEditError unless `after` is `before` with only the keywords changed.

    Text surgery is faster and infinitely more readable than reserialising, but
    it is regex on XML, so it does not get to be trusted on its own. Re-parsing
    costs one parse per written file and turns a silent corruption into a
    per-file failure the CSV records as `failed`.
    """
    try:
        old = ET.fromstring(before)
        new = ET.fromstring(after)
    except ET.ParseError as exc:
        raise XmpEditError(f"edit produced unparseable XML: {exc}") from exc

    if items_of(new, SUBJECT_TAG) != list(subjects):
        raise XmpEditError("dc:subject is not what was asked for")
    if hierarchical is not None and items_of(old, HIER_TAG):
        if items_of(new, HIER_TAG) != list(hierarchical):
            raise XmpEditError("lr:hierarchicalSubject is not what was asked for")

    old_skeleton, new_skeleton = _skeleton(old), _skeleton(new)
    if old_skeleton != new_skeleton:
        lost = [f for f in old_skeleton if f not in new_skeleton]
        raise XmpEditError(f"edit changed {len(lost)} element(s) outside the keywords: {lost[:2]}")


# --- hierarchical keywords --------------------------------------------------

def merge_hierarchical(existing, ours: set, new_keywords):
    """Keep the user's keyword paths; replace only this pipeline's entries.

    `lr:hierarchicalSubject` holds *paths* -- `People|Family|Miles` is one
    keyword naming a position in Lightroom's keyword tree. The previous
    implementation mirrored the flattened dc:subject list into it, which turned
    that single keyword into three unrelated top-level ones on the next import.
    71 sidecars in the library carry such paths.

    An entry belongs to this pipeline when its *leaf* is one of our keywords, so
    a flat `bird` written by an earlier run is replaced while `Scenes|Flowers`
    survives intact.
    """
    kept = [e for e in existing if (e or "").split("|")[-1] not in ours]
    return kept + [k for k in new_keywords if k not in kept]
