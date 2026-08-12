"""Read bird_label's own output back out of the XMP sidecars it wrote.

bird_label injects two keywords into `dc:subject`: a bare category
(`bird`/`animal`/`people`/`scenery`) and a formatted label from
`label_generator.py` -- `{pinyin}-{chinese}-{english}({confidence}%)`.

Prefer this over the per-year `bird_identification_output.csv` when you need a
sidecar's category or species. The CSV's `filename` column is a bare basename
with no folder, and basenames repeat within a single year (camera counter
wraparound), so several hundred rows per year share a key -- and those
duplicates often disagree on category. A basename-keyed CSV lookup therefore
silently attaches the wrong row. The sidecar is per-file and unambiguous.
"""

import re
import xml.etree.ElementTree as ET
from typing import NamedTuple, Optional

# Categories bird_label writes as a bare keyword. Any other short keyword in
# dc:subject is a pre-existing user tag (locations like "sgy-山公园", batch
# markers like "add=20231014") and is not a category.
CATEGORIES = ("bird", "animal", "people", "scenery")

# A model-written label. The trailing "(NN%)" is load-bearing: hand-written
# species keywords use the same py-cn-en shape but carry no confidence, and
# must not be mistaken for a model label.
LABEL_RE = re.compile(r"^(?P<pinyin>[^-]+)-(?P<chinese>[^-]+)-(?P<english>.+)\((?P<confidence>\d+)%\)$")

# Any injected label, not just a species one. `label_generator` only produces
# the `py-cn-en` shape for bird/animal; scenery and people get a bare
# description plus the confidence -- "person playing tennis(98%)" -- so LABEL_RE
# does not match them and the trailing "(NN%)" is the only common marker.
CONFIDENCE_SUFFIX_RE = re.compile(r"^_?.+\(\d{1,3}%\)$")

# The earliest GPT-4o runs wrote a capitalised category and no confidence
# suffix, so their output is not recognisable by shape alone.
EARLY_CATEGORIES = ("People", "Unknown")

# A hand-written species keyword: the same `py-cn-en` shape as a model label
# but with no confidence. "xs-小隼-Kestrel", "sgy-山公园". These are the user's
# own work and are never claimed, even in a sidecar this pipeline has labelled.
HAND_WRITTEN_RE = re.compile(r"^[A-Za-z]+-\s*[^-]*[一-鿿]")
# mk_label() prefixes a low-confidence result with an underscore; "_nb" is the
# no-bird marker an early run wrote.
NO_BIRD_MARKER = "_nb"


class Label(NamedTuple):
    pinyin: str
    chinese: str
    english: str       # lowercased, stripped -- the species key
    confidence: float  # 0.0-1.0
    raw: str


def parse_label(keyword: str) -> Optional[Label]:
    """Parse one dc:subject keyword as a model label, or None if it is not one."""
    m = LABEL_RE.match(keyword.strip())
    if not m:
        return None
    return Label(
        pinyin=m.group("pinyin").strip(),
        chinese=m.group("chinese").strip(),
        english=m.group("english").strip().lower(),
        confidence=int(m.group("confidence")) / 100.0,
        raw=keyword.strip(),
    )


def read_subjects(xmp_path) -> Optional[list[str]]:
    """All dc:subject keywords in a sidecar. None if the file will not parse.

    None and [] mean different things: None is a malformed sidecar, [] is a
    well-formed one that was never labelled.
    """
    try:
        tree = ET.parse(xmp_path)
    except (ET.ParseError, OSError):
        return None
    out = []
    for node in tree.getroot().iter():
        if node.tag.endswith("}subject"):
            out += [li.text or "" for li in node.iter() if li.tag.endswith("}li")]
    return out


class SidecarLabels(NamedTuple):
    categories: tuple[str, ...]
    label: Optional[Label]
    subjects: tuple[str, ...]

    @property
    def is_bird(self) -> bool:
        return "bird" in self.categories

    @property
    def species(self) -> Optional[str]:
        return self.label.english if self.label else None


def split_keywords(subjects) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Partition a sidecar's keywords into (ours, theirs).

    "Ours" is anything a bird_label run injected, across all three generations
    it has written: a bare category keyword, anything carrying a `(NN%)`
    confidence suffix (species labels *and* the bare scenery/people
    descriptions, which have no `py-cn-en` shape at all), the `_nb` marker, and
    the early GPT-4o output that carried neither a suffix nor a lowercase
    category.

    That last generation is not recognisable by shape -- `mountain landscape
    with glacier` looks like an ordinary keyword. Two signals together identify
    it. The sidecar: a category keyword is only ever written by this pipeline.
    And the text: that generation always wrote a *descriptive phrase*, while
    the user's own keywords are single tokens (`Family`, `Rivertown`, `jx`) or
    the `HAND_WRITTEN_RE` species shape (`xs-小隼-Kestrel`, `pp-昵称`).

    Measured over the dataset, the two signals together claim 2,241 free-text
    entries -- every one a scene description -- while sparing 45 single tokens,
    of which 44 are plainly the user's. The one arguable loss is `waterfall`
    (8 photos), left behind as a stale keyword.

    That asymmetry is deliberate. Leaving a stale keyword is a cosmetic problem
    the user can fix; deleting a keyword they wrote by hand destroys work that
    exists nowhere else. The `HAND_WRITTEN_RE` exemption matters most *after* a
    full re-label, when every sidecar carries a category and the rule would
    otherwise claim any species keyword added afterwards.
    """
    subjects = tuple(subjects)
    has_category = any(s in CATEGORIES or s in EARLY_CATEGORIES for s in subjects)
    ours, theirs = [], []
    for keyword in subjects:
        text = keyword.strip()
        descriptive = " " in text and not HAND_WRITTEN_RE.match(text)
        mine = (text in CATEGORIES
                or text in EARLY_CATEGORIES
                or text == NO_BIRD_MARKER
                or CONFIDENCE_SUFFIX_RE.match(text) is not None
                or (has_category and descriptive))
        (ours if mine else theirs).append(keyword)
    return tuple(ours), tuple(theirs)


def read_labels(xmp_path) -> Optional[SidecarLabels]:
    """Category keywords + model label for one sidecar. None if unparseable."""
    subjects = read_subjects(xmp_path)
    if subjects is None:
        return None
    label = next((lab for lab in (parse_label(s) for s in subjects) if lab), None)
    return SidecarLabels(
        categories=tuple(c for c in CATEGORIES if c in subjects),
        label=label,
        subjects=tuple(subjects),
    )
