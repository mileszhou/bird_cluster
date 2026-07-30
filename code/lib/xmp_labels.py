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
# dc:subject is a pre-existing user tag (locations like "bhl-山公园", batch
# markers like "add=20231014") and is not a category.
CATEGORIES = ("bird", "animal", "people", "scenery")

# A model-written label. The trailing "(NN%)" is load-bearing: hand-written
# species keywords use the same py-cn-en shape but carry no confidence, and
# must not be mistaken for a model label.
LABEL_RE = re.compile(r"^(?P<pinyin>[^-]+)-(?P<chinese>[^-]+)-(?P<english>.+)\((?P<confidence>\d+)%\)$")


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
