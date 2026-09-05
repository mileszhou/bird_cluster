#!/usr/bin/env python3
"""bird_label.py – Bird identification with English and Chinese names.

- Uses the OpenAI Vision API to extract **English** and **Chinese** bird names, plus confidence.
- Output CSV includes: `filename`, `label` (English), `label_cn` (Chinese), `confidence`, `note`, `response_json`.
- No logits are available from the OpenAI API (as noted).

Run with the same CLI flags as before, for example:
```
python3 -m code.bird_label --conf-threshold 0.6 --no-bird 0.2
```
"""

import argparse
import base64
import json
import logging
import os
import shutil
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, Optional
import collections
import csv
import urllib.error
import urllib.request
import re
from PIL import Image
import time

from code.lib.label_generator import pinyin_initials
from code.lib.config import data_dir, model_name, server_url
from code.lib.jpg_index import library_year
from code.lib.jpg_claim import SidecarClaims, sort_key
from code.lib import path_filter
from code.lib.xmp_labels import CATEGORIES, EARLY_CATEGORIES, read_labels, split_keywords
from code.lib import xmp_write
import code.lib.run_hf_bird_model_llamacpp

logger = logging.getLogger("bird_label")


def setup_logging(log_path: Path) -> None:
    """Mirror stdout's status lines into an INFO-level log file at log_path,
    appending so a resumed run's log stays contiguous with prior runs."""
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

# ------------------------------------------------------------
# Helper: read an image and encode as base64 for the OpenAI API.
# ------------------------------------------------------------

def read_image_base64(image_path: Path) -> str:
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

# ------------------------------------------------------------
# XMP keyword handling (unchanged from original).
# ------------------------------------------------------------

# Lightroom writes dc:subject as an rdf:Bag; earlier runs of this script wrote
# an rdf:Seq. Both appear in the dataset (17,590 Bags against 1,748 Seqs), so
# reading must accept either and writing must reuse whichever is already there
# -- adding a second container under one dc:subject yields a sidecar whose
# keyword list depends on which container the reader happens to pick. The
# reuse, and the file-preserving edit itself, live in `code/lib/xmp_write.py`.


# What set_keywords_in_xmp() did, recorded in the CSV's `applied` column.
APPLIED_WRITTEN = "written"        # the new label replaced whatever was there
# No longer produced: the labeller writes its verdict unconditionally. Kept
# because data/label and every archived run still carry rows marked this way,
# and readers must go on understanding them -- notably embed.py's
# effective_category(), which is a no-op on new CSVs and load-bearing on old.
APPLIED_KEPT = "kept-existing"     # a non-bird result deferred to the prior label
APPLIED_CSV_ONLY = "csv-only"      # no sidecar exists; the CSV is the only record
APPLIED_FAILED = "failed"


def existing_category(keywords) -> str:
    """The category keyword already on a sidecar, across both generations."""
    return next((k for k in keywords
                 if k in CATEGORIES or k in EARLY_CATEGORIES), "")


def set_keywords_in_xmp(xmp_path: Path, category: str, label: str):
    """Write this run's label into a sidecar, preserving the user's keywords.

    **This run's verdict is always written. The labeller records; it does not
    arbitrate.** Whatever was there is captured in the CSV's `prior_category` /
    `prior_label` and then replaced. No result is ever discarded on the grounds
    that an older one looked better -- deciding that is the job of whatever
    consumes the output, which has the whole population in front of it and a
    question to answer, neither of which this function has.

    That policy replaced two successive guards, and the reason both failed is
    the same. `bird` was protected so a fresh false negative could not drop a
    photo from the clustering set; sampling the 554 rows it held showed 66 were
    provably a same-stem twin's label from the basename-keyed era -- a wombat
    carrying `common myna`. `scenery` was protected because the early GPT-4o
    pass wrote specific descriptions where a fresh run wrote a bare `scenery`;
    by the time it was measured the current prompt asked for the landmark by
    name, so the guard was preserving *generic* text over identifying text --
    "historic building with columns and flag" kept over "brisbane city hall
    facade", across 2,401 rows. A guard calcifies around the labeller it was
    written for and then quietly inverts.

    It also makes the pipeline a model-comparison instrument. Run a second model
    over a tree the first one labelled and every row carries both verdicts side
    by side -- `category`/`label` against `prior_category`/`prior_label` -- which
    is a paired comparison over the whole library, for free, with no arbitration
    baked in.

    Returns (action, previous keywords). The previous labels are *not* written
    back into the sidecar anywhere: these files go back into Lightroom, and an
    archive property would either pollute the catalog's keyword list or be
    dropped on a round-trip. They are kept in the result CSV instead.

    The file is read with the XML parser but edited as text (see
    `code/lib/xmp_write.py`): reserialising rewrites every line of a sidecar,
    and these files are rsynced back into a Lightroom library, so the diff has
    to stay readable. Every edit is re-parsed and checked before it is written.

    Catches permission errors (library-managed files are often read-only) and
    retries once after a chmod, so one bad file does not abort the run.
    """
    import xml.etree.ElementTree as ET

    try:
        before = xmp_path.read_text(encoding='utf-8')
        root = ET.fromstring(before)

        existing = xmp_write.items_of(root, xmp_write.SUBJECT_TAG)
        ours, theirs = split_keywords(existing)

        # The user's own keywords first, in their original order, then ours.
        keywords = [category, label]
        merged = list(theirs) + [k for k in keywords if k not in theirs]

        # hierarchicalSubject holds keyword *paths*; only our own entries are
        # replaced, so "People|Family|Miles" survives a run that says `bird`.
        prior_hier = xmp_write.items_of(root, xmp_write.HIER_TAG)
        hier_ours, _ = split_keywords([(h or "").split("|")[-1] for h in prior_hier])
        hierarchical = xmp_write.merge_hierarchical(
            prior_hier, set(ours) | set(hier_ours), keywords)

        after = xmp_write.set_subject_keywords(before, merged, hierarchical)
        xmp_write.verify_only_keywords_changed(before, after, merged, hierarchical)

        try:
            xmp_path.write_text(after, encoding='utf-8')
        except PermissionError:
            try:
                xmp_path.chmod(0o666)
                xmp_path.write_text(after, encoding='utf-8')
                logger.info(f"⚠️  Fixed permissions and updated {xmp_path.name}")
            except Exception as e2:
                logger.info(f"⚠️  Could not write XMP file {xmp_path.name}: {e2}. Skipping.")
                return APPLIED_FAILED, ours
        return APPLIED_WRITTEN, ours
    except Exception as e:
        logger.info(f"⚠️  Failed to process XMP file {xmp_path.name}: {e}. Skipping.")
        return APPLIED_FAILED, ()


def load_prior_labels(label_dir: Path) -> dict:
    """{jpg key: (category, label)} from a previous run's CSV.

    **The CSV, not the sidecars.** A sidecar only exists for a photo that had a
    raw file, so reading priors from `data/xmp` covers 20% of this library:
    9,918 of 49,224 rows, and never the 5,229 images that never had a raw at
    all. The CSV is the output of labelling and has a row for every image, which
    is the same reason `embed.py` takes its guide from there.

    That matters because these columns are the entire paired-verdict comparison
    -- run a second model over a tree the first one labelled and every row
    carries both readings. Over a fifth of the rows, that comparison was a
    comparison of nothing.
    """
    csv_path = Path(label_dir) / "bird_identification_output.csv"
    if not csv_path.is_file():
        return {}
    out = {}
    with open(csv_path, newline='', encoding='utf-8-sig') as fh:
        reader = csv.DictReader(fh)
        if 'jpg' not in (reader.fieldnames or []):
            sys.exit(f"error: {csv_path} has no 'jpg' column, so its rows cannot "
                     f"be matched to the images this run walks.")
        for row in reader:
            out[row['jpg']] = _effective(row)
    return out


def _effective(row) -> tuple[str, str]:
    """(category, label) the library actually holds for a row.

    Resolves the never-demote rule, the way `embed.effective_category` does:
    where `applied` is `kept-existing` this run's verdict was overruled and the
    sidecar kept `prior_category`, with the species then in `prior_label`.
    Reading `category` alone describes a decision that was not adopted, and
    silently drops the rows that rule exists to protect.
    """
    if (row.get('applied') or '').strip() == 'kept-existing':
        prior_cat = (row.get('prior_category') or '').strip()
        if prior_cat:
            return prior_cat.lower(), (row.get('prior_label') or '').strip()
    return ((row.get('category') or '').strip().lower(),
            (row.get('label') or '').strip())


def prior_labels(key: str, jpg_key: str = "") -> tuple[str, str]:
    """(category, label) a previous run left on this photo, for the CSV.

    Prefers the prior run's CSV, which has a row per image. Falls back to the
    pristine `data/xmp` when no such CSV was given -- and to *that* tree rather
    than the working copy, so the answer is the same whether or not `output/raw`
    already holds a partial re-label. Reading the working copy would make a
    resumed run record its own fresh labels as the prior ones, quietly
    destroying the very comparison this column exists for.

    An empty key means the JPEG writes to no sidecar; with the CSV route that no
    longer costs the comparison, since the CSV is keyed by the image.
    """
    if PRIOR_LABELS:
        return PRIOR_LABELS.get(jpg_key, ("", ""))
    if not key:
        return "", ""
    labels = read_labels(PRISTINE_XMP_DIR / key)
    if labels is None or not labels.subjects:
        return "", ""
    ours, _ = split_keywords(labels.subjects)
    category = next((s for s in ours if s in CATEGORIES or s in EARLY_CATEGORIES), "")
    return category, ";".join(s for s in ours if s != category)


class WorkItem(NamedTuple):
    """One exported JPEG to label.

    `key` is the JPEG's path relative to `data/jpg` -- the checkpoint entry and
    the CSV's `jpg` column.

    `xmp` is the sidecar this image *belongs to*, and `owns_xmp` says whether
    this image is the one that writes it. The two are separate on purpose. An
    alternate edit (`X-2.jpg`, whose capture's sidecar went to `X.jpg`) still
    names its capture, so grouping the CSV by `xmp` recovers every export of one
    frame -- which is what a post-hoc dedup needs. Collapsing the two would make
    it indistinguishable from a photo that never had a raw at all, and there are
    5,229 of those against 313 alternate edits.

    So `xmp` is None only for a genuine orphan, and `applied` says what happened:
    `csv-only` covers both, `xmp` tells them apart.
    """
    key: str                # checkpoint entry and the CSV's `jpg`
    name: str               # the CSV's `filename`, a bare basename for reading
    jpg: Path               # image sent to the model
    xmp: Optional[Path]     # the capture's sidecar; None only if it never had one
    owns_xmp: bool          # whether this image is the one that writes it

    def xmp_key(self, xmp_root: Path) -> str:
        """The CSV's `xmp` column, relative to `data/xmp`. Empty for an orphan."""
        return "" if self.xmp is None else self.xmp.relative_to(xmp_root).as_posix()


def build_items(claims: SidecarClaims, xmp_root: Path, jpg_root: Path, paths=None):
    """One work item per exported JPEG -- the unit the model actually sees.

    The JPEG is the source: it is what gets embedded downstream, and 5,229 of
    them have no sidecar at all yet are ordinary members of the population. So
    the walk is over `data/jpg`, and a sidecar is a *destination* for the label
    rather than the thing being enumerated.

    Each JPEG claims its sidecar locally (see `code/lib/jpg_claim.py`), and the
    first claimant in stem-sorted order keeps it -- which is always the exact
    stem match. Later claimants are the capture's alternate edits; they keep
    their CSV row and write no sidecar.

    Claims are assigned over the **whole** sorted tree before the checkpoint
    filters anything, so a resumed run reproduces the same assignment: the
    function is a pure function of the two trees. Tracking claims only across a
    single process would hand the sidecar to a virtual copy after a Ctrl-C.
    """
    jpgs = sorted(
        ((p.parent.relative_to(jpg_root).as_posix(), p.stem, p)
         for p in jpg_root.rglob("*.jpg")),
        key=lambda t: sort_key(t[0], t[1]),
    )
    # Sidecars the run is *responsible* for: in a selected library, and not
    # filtered out. The unreached count below is only meaningful against this --
    # comparing against the whole tree makes every scoped run scream about the
    # years it was told to skip.
    expected = set()
    for folder, stems in claims.by_folder.items():
        for base in stems:
            if paths is None or paths.allows(f"{folder}/{base}.jpg"):
                expected.add((folder, base))

    items, taken = [], set()
    for folder, stem, path in jpgs:
        key = path.relative_to(jpg_root).as_posix()
        if paths is not None and not paths.allows(key):
            continue
        base = claims.claim(folder, stem)
        xmp = None if base is None else claims.path_for(folder, base)
        owns = base is not None and (folder, base) not in taken
        if owns:
            taken.add((folder, base))
        items.append(WorkItem(key, path.name, path, xmp, owns))
    return items, taken, expected


CSV_COLUMNS = ['jpg', 'xmp', 'filename',
               'category', 'label', 'label_cn', 'label_sci',
               'confidence', 'note', 'prior_category', 'prior_label',
               'applied', 'run_label', 'response_json']


def check_csv_schema(csv_path: Path) -> None:
    """Refuse to append rows of one shape under a header of another.

    A resumed run opens the CSV in append mode, so a column added since the
    first run would silently shift every later row one field left of its header
    -- and the checkpoint would mark those photos done, so the damage would not
    be retried. Cheaper to stop and ask for a fresh output dir.
    """
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        header = next(csv.reader(f), [])
    if header and header != CSV_COLUMNS:
        missing = [c for c in CSV_COLUMNS if c not in header]
        sys.exit(f"error: {csv_path} was written with a different column set "
                 f"(missing {missing or 'nothing; order differs'}). Appending would "
                 f"misalign every new row. Start a fresh --output-dir, or ./clean "
                 f"and re-run from the beginning.")


def write_row(csv_writer, item, category, label, label_cn, label_sci, conf, note,
              applied, raw_json, args):
    # item.xmp points into the working copy; the relative part is identical in
    # data/xmp, which is where prior_labels() reads from.
    xmp_rel = item.xmp_key(RAW_OUT)
    prior_category, prior_label = prior_labels(xmp_rel, item.key)
    csv_writer.writerow([
        item.key, xmp_rel, item.name,
        category, label, label_cn, label_sci,
        f"{conf:.2f}", note, prior_category, prior_label, applied,
        args.run_label, raw_json,
    ])
    return prior_category

# ------------------------------------------------------------
# vLLM query (via OpenAI-compatible vLLM server, e.g. http://localhost:8000/v1).
# ------------------------------------------------------------
VLLM_SYSTEM_PROMPT = (
    "You are an expert bird, wild animal, and scene identification system. "
    "For the given image, output a JSON object with the following fields: "
    "`category` \u2013 one of: 'bird' (only class Aves \u2014 actual birds), 'animal' (all other animals: insects, butterflies, mammals, reptiles, etc.), 'people', or 'scenery'. "
    "`label` \u2013 for 'bird'/'animal', the English common name using Latin characters only (e.g. 'Grey Wagtail'); "
    "for 'scenery', a concise description naming the specific subject of the scene \u2014 the landmark, landscape feature, or activity in view (e.g. 'Sunset over Lofoten fjord', 'Sahara sand dunes', 'Gothic cathedral facade', 'Hikers on mountain trail') rather than a generic word like 'landscape' or 'outdoor scene'; "
    "for 'people', a brief description of who/what they're doing. "
    "`label_cn` \u2013 the standard Chinese (Mandarin) name or translation of `label` (e.g. '\u7070\u9d3a\u9dcc'). "
    "`label_sci` \u2013 for 'bird'/'animal' only, the scientific binomial (genus and species, e.g. 'Motacilla cinerea'); "
    "empty string for 'people' and 'scenery'. Give the accepted binomial, not a subspecies or an author citation. "
    "`confidence` \u2013 a float between 0.0 and 1.0. "
    "Output ONLY a JSON object, no other text."
)


def get_actual_vllm_model_name(vllm_url: str, requested_model: str) -> str:
    """The model the server actually serves. **Fatal if it cannot be determined.**

    The probe is not a convenience for fixing up `--model`; it is how the run
    learns what it is talking to, and that answer becomes the recorded
    provenance of every row it writes. Carrying on with the *requested* name
    after a failed probe asserts something nobody checked -- which is how a whole
    run came to be attributed to a 132B model that was never loaded.

    Two ways it used to pass silently, both closed here: an unreachable server
    was a warning, and an empty `data` list did not even warn, because the
    mismatch branch was guarded on the list being non-empty.

    A wrong name is not merely mislabelled metadata. vLLM rejects a request for a
    model it does not hold, so the run would fail on every batch anyway -- just
    after copying 43k sidecars and writing an args.json that is now a lie.
    """
    base_url = vllm_url.rstrip('/').replace('/v1', '')
    try:
        probe_request = urllib.request.Request(f'{base_url}/v1/models')
        with urllib.request.urlopen(probe_request, timeout=10) as probe_resp:
            models_data = json.loads(probe_resp.read().decode('utf-8'))
    except Exception as e:
        sys.exit(f"error: could not probe models at {base_url}/v1 ({e}).\n"
                 f"       The run would proceed without knowing which model serves it, and "
                 f"record '{requested_model}' as the provenance of every row.\n"
                 f"       If the server is running, this is asking the wrong host: the\n"
                 f"       host comes from config.toml [servers.vllm], overridden by the\n"
                 f"       gitignored config.local.toml, so a fresh clone or a container\n"
                 f"       defaults to localhost -- which inside a container is the\n"
                 f"       container, not the box with the GPU.\n"
                 f"           cp _config.local.toml config.local.toml   # then set the host\n"
                 f"       Or for one run:  --vllm-url http://<host>:8000\n"
                 f"       To start it:     ./server-vllm")

    available_models = [m['id'] for m in models_data.get('data', [])]
    if not available_models:
        sys.exit(f"error: {base_url}/v1/models returned no models, so the serving model "
                 f"cannot be identified.\n       Refusing to record '{requested_model}' "
                 f"unverified.")
    if not requested_model:
        # No hint given, which is the normal case: the server is the authority
        # on what it serves, so there is nothing to reconcile.
        logger.info(f"\u2139\ufe0f  Server is serving '{available_models[0]}'")
        return available_models[0]
    if requested_model not in available_models:
        actual = available_models[0]
        logger.info(f"\u2139\ufe0f  Server mismatch: requested '{requested_model}', "
                    f"but using '{actual}'")
        return actual
    return requested_model


# Overridable so an OpenAI-compatible proxy (Azure, OpenRouter, a local gateway)
# can be used without a code change -- and so this backend can be exercised end
# to end against a stub, which is how the breakage below was found. The default
# is the real API.
OPENAI_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')

# Parameters a given endpoint+model has already rejected, so the correction is
# made once per run rather than per image. Keyed by both, because two models on
# one endpoint need not agree.
_PARAM_FIXES: dict = {}


# What a reasoning model needs instead of `max_tokens: 200`.
#
# The budget covers *reasoning plus output*, and reasoning comes first: at 200 a
# GPT-5 run spends the whole allowance thinking and returns an empty string with
# `finish_reason: "length"`, which arrives here as "No JSON found" over a blank
# response. Raising it alone is the wrong fix, because reasoning tokens are
# billed -- a generous budget on 49,000 images is real money spent on deliberation
# nobody reads. So ask for less thinking *and* leave room to answer.
#
# `reasoning_effort` is sent only to a model that has already asked for
# `max_completion_tokens`, i.e. one that reasons. If it is rejected anyway the
# same adaptation loop drops it.
def _apply_fixes(payload: dict, fixes: dict) -> None:
    """Apply {parameter: value} to a payload; None removes the parameter."""
    for key, value in fixes.items():
        if value is None:
            payload.pop(key, None)
        else:
            payload[key] = value


def _describe_fix(fix: dict) -> str:
    dropped = [k for k, v in fix.items() if v is None]
    added = [f"{k}={v}" for k, v in fix.items() if v is not None]
    parts = []
    if dropped:
        parts.append("dropping " + ", ".join(f"'{k}'" for k in dropped))
    if added:
        parts.append("using " + ", ".join(added))
    return "; ".join(parts)


REASONING_BUDGET = 2000
REASONING_EFFORT = "minimal"


def _param_fix(detail: str, payload: dict):
    """Read a 400 and work out how to change the payload.

    OpenAI's newer families moved the goalposts: GPT-5 and the o-series reject
    `max_tokens` in favour of `max_completion_tokens`, and reject a `temperature`
    other than the default. A hardcoded list of which models want which spelling
    is a list that goes stale -- new models arrive, old ones retire, and the
    failure is a 400 nobody sees until a paid run dies on its first image.

    So the server is asked instead, the same way the vLLM backend probes rather
    than assuming what is loaded. The endpoint says exactly what it will not
    accept; this reads that and adapts once.

    Returns {parameter: value}, where None means remove it.
    """
    lower = detail.lower()
    if 'max_tokens' in lower and ('max_completion_tokens' in lower
                                  or 'unsupported' in lower):
        # A model that wants this spelling is a reasoning model, so it needs the
        # budget and the effort setting that go with being one.
        return {'max_tokens': None,
                'max_completion_tokens': REASONING_BUDGET,
                'reasoning_effort': REASONING_EFFORT}
    if 'reasoning_effort' in lower:
        return {'reasoning_effort': None}
    if 'temperature' in lower and ('unsupported' in lower or 'not support' in lower):
        return {'temperature': None}
    return None


# The key a predictor puts in `response_json` when it never got an answer, and
# the marker the loop counts. In the CSV it is the record of what went wrong.
PREDICTION_FAILED = "_prediction_failed"

# {jpg key: (category, label)} from the previous labelling, filled at startup.
# Empty means fall back to reading sidecars -- see prior_labels().
PRIOR_LABELS: dict = {}


def to_simplified(text: str) -> str:
    """Chinese names in one script, so a species is one string.

    A model asked for "the standard Chinese name" answers in whichever script it
    feels like, and 普通翠鳥 against 普通翠鸟 is one bird under two keywords: a
    photo manager lists it twice, and any join on the label splits it. Same
    fragmentation the confidence used to cause in a keyword, arriving by a
    different route, so it gets the same answer -- normalise at the boundary,
    once, rather than teaching every consumer about it.

    Simplified because the library and the existing labelling are overwhelmingly
    simplified: 1,509 of 49,224 rows are not, so converting is the smaller move.

    **It is not the main cause of unstable names, and should not be mistaken for
    a fix for that.** Of the 980 English names carrying more than one Chinese
    form, simplification collapses 7. The other 973 differ by genuine synonymy --
    two real Chinese names for one bird -- which no script conversion touches.
    """
    if not text or not any('\u4e00' <= c <= '\u9fff' for c in text):
        return text
    try:
        import zhconv
    except ImportError:  # a client without it still labels, just unnormalised
        return text
    return zhconv.convert(text, 'zh-cn')


def normalise_binomial(text: str) -> str:
    """`Genus species`, or nothing.

    A binomial is standardised where a common name is not, which is the whole
    reason to ask for one: 5,336 bird images fall back to a pixel guess today
    because the checklist has no entry for the *common* name the labeller used,
    while carrying every one of those birds under its binomial. Matching on the
    strong key is what closes that.

    This checks *shape*, not existence: two Latin-script words of at least three
    letters, genus capitalised, species lower-case. A subspecies trinomial keeps
    its first two words; an author citation, a bare genus, Chinese text, or a
    short English phrase like "a bird" is rejected.

    Existence is checked where the checklist is, in `tools.map_label_taxa`. That
    layering is deliberate and it is what makes a wrong binomial cheap: a name
    the checklist does not carry simply fails to match and falls back to the
    common name, exactly as a run with no `label_sci` at all does. A *common*
    name that is wrong has no such backstop, which is the asymmetry that makes
    this field worth asking for.
    """
    parts = (text or "").strip().replace("_", " ").split()
    if len(parts) < 2:
        return ""
    genus, species = parts[0], parts[1]
    if not (genus.isascii() and species.isascii()
            and genus.isalpha() and species.isalpha()
            and len(genus) >= 3 and len(species) >= 3
            and species.islower()):
        return ""
    return f"{genus.capitalize()} {species}"


def prediction_failed(raw_json: str) -> str | None:
    """The error a prediction reported, or None if it produced an answer."""
    try:
        return json.loads(raw_json).get(PREDICTION_FAILED)
    except (json.JSONDecodeError, AttributeError, TypeError):
        return None


def openai_key() -> str:
    """The API key, or a refusal that says what to do about it.

    Checked at the point of use rather than at startup because the other two
    backends need no key at all, and a run against a local server should not
    fail for want of a credential it will never send.
    """
    key = os.getenv('OPENAI_API_KEY')
    if not key:
        sys.exit("error: OPENAI_API_KEY is not set, and the openai backend calls a "
                 "paid API.\n       Put it in .env (cp _env .env, then add the "
                 "line) or export it for this run.")
    return key


def _vllm_chat_completion(messages, model_name: str, vllm_url: str, timeout: int = 120,
                          api_key: str | None = None):
    """POST to any OpenAI-protocol chat endpoint: vLLM, llama.cpp, or OpenAI.

    One function for all three because it is one protocol, and keeping a second
    copy for the cloud is what broke the `openai` backend. That copy drifted
    until it shared nothing but a shape: it carried a prompt from before `bird`
    meant class Aves, a `max_tokens` too small for the JSON now asked for, no
    timeout, and `json.JSONDecode_decodeError` -- an attribute that does not
    exist, so the fenced-JSON fallback raised `AttributeError` and every
    response wrapped in a code fence silently became `scenery/unknown`.

    The key is a parameter rather than something inferred from the URL. Matching
    on `api.openai.com` would send nothing to any other authenticated endpoint
    and cannot be tested without contacting the real one -- so the caller that
    knows it needs a key supplies it, and a local server simply does not.
    """
    base_url = vllm_url.rstrip('/').replace('/v1', '')
    payload = {
        'model': model_name,
        'messages': messages,
        'max_tokens': 200,
        'temperature': 0.0,
    }
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    _apply_fixes(payload, _PARAM_FIXES.get((base_url, model_name), {}))

    def _post():
        request = urllib.request.Request(
            url=f'{base_url}/v1/chat/completions',
            data=json.dumps(payload).encode('utf-8'),
            headers=headers,
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))

    # Loop rather than retry once: a model may object to several parameters, and
    # it reports them one at a time. Bounded by the number of parameters that
    # can be corrected, so a server rejecting for some other reason still stops.
    for _ in range(len(payload) + 1):
        try:
            return _post()
        except urllib.error.HTTPError as exc:
            if exc.code != 400:
                raise
            detail = exc.read().decode('utf-8', 'replace')
            fix = _param_fix(detail, payload)
            if not fix:
                raise RuntimeError(f"{base_url} rejected the request: {detail}") from exc
            _PARAM_FIXES.setdefault((base_url, model_name), {}).update(fix)
            logger.info(f"ℹ️  {model_name}: " + _describe_fix(fix)
                        + " for the rest of this run.")
            _apply_fixes(payload, fix)
    raise RuntimeError(f"{base_url} kept rejecting the request for {model_name}")


def predict_with_vllm(image_path: Path, vllm_url: str, model_name: str,
                      conf_threshold: float, no_bird_conf: float,
                      api_key: str | None = None):
    """Returns (category, label, label_cn, label_sci, confidence, raw_json).
    The model is asked to return a JSON object with the following fields:
    - `category` \u2013 'bird', 'animal', 'people', or 'scenery'
    - `label` \u2013 English name or description
    - `label_cn` \u2013 Chinese name
    - `confidence` \u2013 float 0.0\u20111.0
    """
    category = "scenery"
    label = "unknown"
    label_cn = ""
    label_sci = ""
    confidence = 0.0
    raw_json = "{}"
    content = None
    try:
        img_b64 = read_image_base64(image_path)
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": VLLM_SYSTEM_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ]}
        ]
        response = _vllm_chat_completion(messages, model_name, vllm_url,
                                         api_key=api_key)
        choice = response['choices'][0]
        msg = choice['message']
        content = msg.get('content') or msg.get('reasoning_content', '')
        if not (content or '').strip():
            # Say what actually happened. A reasoning model that spends its whole
            # budget thinking returns an empty string with finish_reason
            # "length", and reporting that as "No JSON found" over a blank
            # response sends the reader hunting for a parsing bug that is not
            # there. The usage block names the cause exactly.
            reason = choice.get('finish_reason')
            used = (response.get('usage') or {}).get(
                'completion_tokens_details', {}).get('reasoning_tokens')
            if reason == 'length':
                raise ValueError(
                    f"the model returned nothing and stopped at the token limit"
                    + (f" after {used} reasoning tokens" if used else "")
                    + f"; raise REASONING_BUDGET (currently {REASONING_BUDGET}) "
                      f"or lower reasoning_effort")
            raise ValueError(f"the model returned an empty response "
                             f"(finish_reason={reason!r})")
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Extract JSON block if there is surrounding text
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if not match:
                raise ValueError('No JSON found in vLLM response')
            data = json.loads(match.group())
        raw_json = json.dumps(data, ensure_ascii=False)
        label = data.get('label', '').lower()
        label_cn = to_simplified(data.get('label_cn', ''))
        label_sci = normalise_binomial(data.get('label_sci', ''))
        confidence = float(data.get('confidence', 0.0))
        category = data.get('category', 'scenery')
        # If model put Chinese in label field, move it to label_cn
        if any('\u4e00' <= c <= '\u9fff' for c in label):
            if not label_cn:
                label_cn = label
            label = ''
        label = label or 'unknown'
    except Exception as e:
        logger.info(f"\u26a0\ufe0f  vLLM request failed for {image_path.name}: {e}")
        logger.info(f"    raw response: {repr(content)}")
        # Say the request failed, rather than returning the defaults and letting
        # them read as an answer. `scenery/unknown/0.00` is a plausible label for
        # a real photograph, so a dead server, a timeout or a truncated reply
        # used to become a *wrong label* on a real bird -- written, checkpointed,
        # and never retried. PREDICTION_FAILED is what the caller looks for.
        raw_json = json.dumps({PREDICTION_FAILED: str(e)}, ensure_ascii=False)
    # Return the collected values (defaults may be unchanged if an error occurred)
    return category, label, label_cn, label_sci, confidence, raw_json


def predict_with_vllm_batch(image_paths: list, vllm_url: str, model_name: str, conf_threshold: float, no_bird_conf: float):
    """Batch vLLM inference against a vLLM server. Fires requests concurrently so the
    server's continuous batching handles them together. Returns a list of
    (category, label, label_cn, label_sci, confidence, raw_json) per image, in order."""
    # Six fields, matching predict_with_vllm. A default of the wrong arity would
    # only fail on the batch path and only when a request errored, which is the
    # least-exercised corner there is.
    results = [("scenery", "unknown", "", "", 0.0, "{}") for _ in image_paths]
    if not image_paths:
        return results

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(image_paths)) as pool:
        futures = {
            pool.submit(predict_with_vllm, path, vllm_url, model_name, conf_threshold, no_bird_conf): i
            for i, path in enumerate(image_paths)
        }
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.info(f"\u26a0\ufe0f  vLLM request failed for {image_paths[idx].name}: {e}")
                # Same marker the single path uses, so the loop skips the row and
                # leaves it uncheckpointed rather than writing the defaults as a
                # verdict.
                results[idx] = ("scenery", "unknown", "", "", 0.0,
                                json.dumps({PREDICTION_FAILED: str(e)}))

    return results

# ------------------------------------------------------------
# Helper to build a compact label string for animal entries.
# ------------------------------------------------------------
def mk_label(category: str, label_en: str, label_cn: str, conf: float) -> str:
    """Construct a label like 'zs-zhangsan (95%)'.
    `label_en` – English name, `label_cn` – Chinese name, `conf` – confidence (0‑1).
    """
    ret = ""
    if category in ("bird", "animal"):
        c_init = pinyin_initials(label_cn)
        ret = f"{c_init}-{label_cn}-{label_en}({conf * 100:.0f}%)"
    else:
        ret = f"{label_en}({conf * 100:.0f}%)"
    if conf<=0.2:
        ret = f"_{ret}"  # low‑confidence marker for non‑bird
    return ret

# ------------------------------------------------------------
# Sidecar lookup for a JPEG.
#
# The export mirrors the photo library, so a JPEG's sidecar sits in the xmp tree
# at the same library/trip folder under the same stem:
#   jpg/Photos-19/2019-01-13 Hill Park/_D8S0025.jpg
#   xmp/Photos-19/2019-01-13 Hill Park/_D8S0025.xmp
# Camera counters wrap, so stems repeat across the library -- keys therefore
# always carry the folder, never a bare basename. The decorated-export cases
# (`-2` virtual copies, `-Enhanced-NR` denoise renders) and the reasoning behind
# the local claim rule live in code/lib/jpg_claim.py.
#
# The claim index is built over RAW_OUT, the working copy of the sidecar tree,
# which mirrors data/xmp exactly -- so the relative keys line up either way.
# `JpgIndex` still resolves the other direction for the audit and the embedding
# step; this module no longer needs it.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Process a single XMP file (extracted for notebook testing)
# ------------------------------------------------------------

def process_single_item(item: "WorkItem", csv_writer, args) -> bool:
    """Process one photo. True if it was labelled, False if the model never answered.

    - Calls the selected model on its JPEG.
    - Updates the sidecar with keyword tags, if it has one.
    - Writes a row to the provided CSV writer.
    - Prints a concise status line.

    **A photo the model could not answer for is left alone entirely** -- no row,
    no sidecar edit, and the caller does not checkpoint it, so a re-run tries it
    again. The alternative is what this used to do: write the predictor's
    `scenery/unknown/0.00` defaults, which are a plausible label for a real
    photograph, and checkpoint them. A dead server or one timeout then became a
    permanent wrong label on a real bird, indistinguishable from a real verdict.
    """
    switch = args.approach
    if switch == "llama.cpp":
        category, label, label_cn, label_sci, conf, raw_json = code.lib.run_hf_bird_model_llamacpp.predict_with_llamacpp(item.jpg, args.model, args.conf_threshold, args.no_bird, args.llama_url)
    elif switch == "openai":
        category, label, label_cn, label_sci, conf, raw_json = predict_with_vllm(
            item.jpg, OPENAI_URL, args.model, args.conf_threshold, args.no_bird,
            api_key=openai_key())
    elif switch == "vllm":
        category, label, label_cn, label_sci, conf, raw_json = predict_with_vllm(item.jpg, args.vllm_url, args.model, args.conf_threshold, args.no_bird)
    else:  # should not happen due to argparse choices, but handle gracefully:
        logger.info(f"⚠️  Unknown approach '{switch}' for {item.name}, skipping.")
        category, label, label_cn, label_sci, conf, raw_json = "scenery", "unknown", "未知", "", 0.0, "{}"

    why = prediction_failed(raw_json)
    if why:
        logger.info(f"⚠️  {item.name}: no answer from the model ({why}); "
                    f"left for a later run")
        return False

    spec = mk_label(category, label, label_cn, conf)
    keywords = [category, spec]
    note = f"{category} ({conf:.2f})"

    if not item.owns_xmp:
        # Nothing for this image to write: either it never had a raw, or its
        # capture's sidecar already went to the exact-stem export. `item.xmp`
        # still names the capture in the latter case.
        applied = APPLIED_CSV_ONLY
    else:
        # This run's verdict, unconditionally; the old one goes to the CSV.
        applied, _ = set_keywords_in_xmp(item.xmp, category, spec)

    write_row(csv_writer, item, category, label, label_cn, label_sci, conf,
              note, applied, raw_json, args)
    if applied == APPLIED_CSV_ONLY:
        logger.info(f"📄 {item.name} → {', '.join(keywords)} (conf={conf:.2f}); CSV only")
    else:
        logger.info(f"✅ {item.name} → {', '.join(keywords)} (conf={conf:.2f})")
    return True


# ------------------------------------------------------------
# Process all side‑car XMP files, generate CSV, update XMP keywords.
# ------------------------------------------------------------

def load_filter_set(filter_csv: Path, conf_threshold: float) -> set | None:
    """Return a set of JPEG keys to process, based on a prior run's CSV.
    Includes images where category == 'animal' or confidence < conf_threshold.
    Returns None if no filter is requested.

    Keyed by the CSV's `jpg` column -- the JPEG's path relative to `data/jpg`,
    which is what this run enumerates. The `filename` column is a bare basename
    and repeats across trips, so it cannot identify a photo.

    Reads the `category` column directly; older rows only encoded it inside
    `note` ("bird (0.90)"), which is parsed as a fallback."""
    if not filter_csv:
        return None
    keys = set()
    # utf-8-sig: tolerates the BOM a spreadsheet adds when the CSV is edited by
    # hand, which would otherwise turn the first header name into "﻿jpg".
    with open(filter_csv, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if 'jpg' not in (reader.fieldnames or []):
            sys.exit(f"error: {filter_csv} has no 'jpg' column -- it predates the "
                     f"JPEG-driven walk, so its keys name sidecars rather than the "
                     f"images this run enumerates. Re-run without --filter-csv, or "
                     f"filter against a CSV from a current run.")
        for row in reader:
            note = row.get('note', '')
            category = (row.get('category')
                        or (note.split('(')[0].strip() if '(' in note else '')).strip()
            try:
                conf = float(row.get('confidence', 1.0))
            except ValueError:
                conf = 1.0
            if category == 'animal' or conf < conf_threshold:
                keys.add(row['jpg'])
    return keys


def process_folder(xmp_root: Path, csv_path: Path, args) -> dict:
    claims = SidecarClaims(xmp_root)
    try:
        paths = path_filter.build(getattr(args, 'include_from', None),
                                  getattr(args, 'exclude_from', None))
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"error: {exc}")
    if paths:
        logger.info(f"⚙️  Path filter: {paths.describe()}")
    items, taken, expected = build_items(claims, xmp_root, JPG_DIR, paths)

    if not items:
        # An empty scope used to be impossible: --years refused a year matching
        # no library. A manifest cannot, so the check lives here instead -- a
        # run that labels nothing and reports success is the same failure the
        # stale --years default produced, one step further along.
        sys.exit("error: nothing to label. An empty selection is treated as a mistake "
                 "rather than a finished run -- check --include-from and --data-dir")

    csv_only = sum(1 for it in items if not it.owns_xmp)
    orphans = sum(1 for it in items if it.xmp is None)
    logger.info(f"⚙️  {len(items)} JPEGs to label; {len(items) - csv_only} write to a "
                f"sidecar, {csv_only} to the CSV only "
                f"({orphans} never had one, {csv_only - orphans} are alternate edits).")

    # The blind spot of walking the JPEG tree: a sidecar no JPEG reaches is not
    # skipped with a warning, it is never enumerated at all. Zero today, and the
    # audit is what keeps it that way -- but a partial re-export would otherwise
    # drop those photos in silence.
    unreached = len(expected - taken)
    if unreached:
        logger.info(f"⚠️  {unreached} of {len(expected)} in-scope sidecars have no JPEG "
                    f"and will not be labelled. Run ./run-audit to list them.")

    filter_set = load_filter_set(getattr(args, 'filter_csv', None), args.conf_threshold)
    if filter_set is not None:
        logger.info(f"⚙️  Filter active: {len(filter_set)} images selected from prior CSV.")
    # Checkpoint keyed by the JPEG's path relative to data/jpg. Never a bare
    # basename: stems repeat across trips as camera counters wrap.
    checkpoint_path = csv_path.parent / "processed.txt"
    processed: set = set()
    if checkpoint_path.is_file():
        processed = set(line.strip() for line in checkpoint_path.read_text().splitlines())

    pending = [
        item for item in items
        if item.key not in processed and (filter_set is None or item.key in filter_set)
    ]

    # After the scope and the checkpoint, so `--limit 5` means five *more*
    # images rather than five from the top of a resumed run. Matches embed.py.
    #
    # It is not a scoping mechanism and does not want to become one: which
    # images you get depends on walk order, so a run narrowed this way cannot
    # be reproduced or described. `--include-from` is how a population is named,
    # and it is the only way -- `--years` was removed for doing the same job
    # through a second mechanism. This is for spending five API calls instead of
    # 47,908 to find out whether a model works at all.
    # Category filter, resolved at run time against the previous labelling
    # rather than through a generated manifest. A manifest would have to live in
    # local/, which is gitignored, so a run.json naming one is a dangling
    # reference for everybody else -- the precise failure the manifest rule
    # exists to prevent. `--prior-labels data/label --categories bird` names a
    # versioned submodule and a word, and reproduces anywhere.
    wanted = {c.strip().lower() for c in (args.categories or "").split(",") if c.strip()}
    if wanted:
        if not PRIOR_LABELS:
            logger.info(f"⚙️  --categories {sorted(wanted)}: no prior labelling to "
                        f"filter on, so labelling everything.")
        else:
            before = len(pending)
            pending = [it for it in pending
                       if PRIOR_LABELS.get(it.key, ("", ""))[0] in wanted]
            logger.info(f"⚙️  --categories {sorted(wanted)}: {len(pending):,} of "
                        f"{before:,} were placed there by the prior labelling.")

    limit = getattr(args, 'limit', 0)
    if limit:
        pending = pending[:limit]
        logger.info(f"⚙️  --limit {limit}: labelling {len(pending)} this run")

    if getattr(args, 'dry_run', False):
        by_lib = collections.Counter(it.key.split("/")[0] for it in items)
        logger.info("  by library: " + ", ".join(f"{k}={by_lib[k]}" for k in sorted(by_lib)))
        if processed:
            logger.info(f"  {len(processed)} already in the checkpoint; "
                        f"{len(pending)} would be sent to the model")
        else:
            logger.info(f"  {len(pending)} would be sent to the model")
        logger.info("🔍 Dry run: nothing labelled, nothing written.")
        return {"pending": len(pending), "labelled": 0, "unanswered": 0,
                "aborted": None, "interrupted": False, "dry_run": True}

    batch_size = getattr(args, 'batch_size', 1)
    use_batch = args.approach == "vllm" and batch_size > 1
    if use_batch:
        logger.info(f"⚙️  vLLM batch mode: batch_size={batch_size}, {len(pending)} images to process.")

    resuming = bool(processed)
    csv_mode = 'a' if resuming else 'w'
    if resuming:
        logger.info(f"⚙️  Resuming: {len(processed)} already processed, {len(pending)} remaining.")
        check_csv_schema(csv_path)
    with open(csv_path, csv_mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not resuming:
            writer.writerow(CSV_COLUMNS)

        # What actually happened, so the caller can say so. A run that labelled
        # nothing used to end with "Run complete": every per-image failure is
        # caught, logged and `break`s the loop, and the caller printed success
        # regardless. That is how the openai backend stayed broken for months
        # -- it raised NameError on the first image of every run anyone tried.
        labelled = 0
        unanswered = 0
        unanswered_keys: set = set()
        aborted = None
        interrupted = False
        def _sigint_handler(sig, frame):
            nonlocal interrupted
            if not interrupted:
                logger.info("\nInterrupt received — finishing current batch before stopping...")
                interrupted = True
        original_sigint = signal.signal(signal.SIGINT, _sigint_handler)

        step = batch_size if use_batch else 1
        try:
            for i in range(0, len(pending), step):
                if use_batch:
                    # Every item is a JPEG that exists -- the walk enumerates the
                    # image tree, so there is no missing-image case to handle.
                    batch = pending[i:i + batch_size]
                    if batch:
                        try:
                            batch_results = predict_with_vllm_batch(
                                [it.jpg for it in batch], args.vllm_url, args.model, args.conf_threshold, args.no_bird
                            )
                            for item, (category, label, label_cn, label_sci, conf, raw_json) in zip(batch, batch_results):
                                why = prediction_failed(raw_json)
                                if why:
                                    logger.info(f"⚠️  {item.name}: no answer from the "
                                                f"model ({why}); left for a later run")
                                    unanswered += 1
                                    unanswered_keys.add(item.key)
                                    continue
                                spec = mk_label(category, label, label_cn, conf)
                                keywords = [category, spec]
                                note = f"{category} ({conf:.2f})"
                                if not item.owns_xmp:
                                    applied = APPLIED_CSV_ONLY
                                else:
                                    applied, _ = set_keywords_in_xmp(
                                        item.xmp, category, spec)
                                write_row(writer, item, category, label, label_cn,
                                                      conf, note, applied, raw_json, args)
                                labelled += 1
                                if applied == APPLIED_CSV_ONLY:
                                    logger.info(f"📄 {item.name} → {', '.join(keywords)} "
                                                f"(conf={conf:.2f}); CSV only")
                                else:
                                    logger.info(f"✅ {item.name} → {', '.join(keywords)} "
                                                f"(conf={conf:.2f})")
                        except Exception as e:
                            logger.info(f"⚠️  Batch error at index {i}: {e}. Stopping to preserve checkpoint.")
                            aborted = f"batch at index {i}: {e}"
                            break
                    # Checkpoint entire batch (including skipped missing-JPEG files).
                    # Flush the CSV first: the checkpoint is closed (and so
                    # durable) every batch while the CSV stays in Python's
                    # buffer, so a hard kill between the two would mark photos
                    # done that have no row. Ctrl-C is unaffected -- it exits
                    # through the `with`, which flushes -- but a SIGKILL or a
                    # power cut is exactly what a checkpoint is for.
                    csvfile.flush()
                    with open(checkpoint_path, 'a', encoding='utf-8') as cp:
                        for item in batch:
                            if item.key not in unanswered_keys:
                                cp.write(f"{item.key}\n")
                else:
                    item = pending[i]
                    try:
                        if process_single_item(item, writer, args):
                            labelled += 1
                            csvfile.flush()  # see the batch path: row before checkpoint
                            with open(checkpoint_path, 'a', encoding='utf-8') as cp:
                                cp.write(f"{item.key}\n")
                        else:
                            # No answer: no row, and deliberately no checkpoint,
                            # so a re-run retries rather than accepting a guess.
                            unanswered += 1
                    except Exception as e:
                        logger.info(f"⚠️  Error processing {item.name}: {e}. Stopping batch to preserve checkpoint.")
                        aborted = f"{item.name}: {e}"
                        break
                if interrupted:
                    logger.info("Stopped cleanly. Re-run to resume.")
                    break
        finally:
            signal.signal(signal.SIGINT, original_sigint)

    return {"pending": len(pending), "labelled": labelled,
            "unanswered": unanswered, "aborted": aborted,
            "interrupted": interrupted}

# ------------------------------------------------------------
# Main script execution
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bird identification with English and Chinese names. Backends: "
                    "vllm (default), llama.cpp, openai.")
    parser.add_argument("--run-label", default="", help="Label for this run (e.g., 'first successful run')")
    parser.add_argument("--model", default="",
                        help="Model name. For vllm/llama.cpp this is only a hint -- the "
                             "server is probed and whatever it actually serves wins, so "
                             "leaving it empty is the honest default. Required in "
                             "practice only for openai, which has nothing to probe "
                             "-- and that one comes from config.toml [models] "
                             "rather than being hardcoded here")
    parser.add_argument("--conf-threshold", type=float, default=0.6, help="Low‑confidence threshold for special keyword (default 0.6)")
    parser.add_argument("--no-bird", type=float, default=0.2, help="Confidence below which we label as 'no bird' (default 0.2)")
    parser.add_argument("--output-dir", default="./output/label",
                        help="Directory for run outputs (default: ./output/label, "
                             "beside ./output/embed and ./output/cluster)")
    parser.add_argument("--data-dir", default=None,
                        help="Root data directory (contains jpg/). Default: resolved by "
                             "code.lib.config.data_dir() -- ./data, else config.toml's "
                             "data_dir, else ./sample_data")
    # `openai` rather than `chatgpt`: the backend speaks the OpenAI *protocol*
    # and $OPENAI_BASE_URL points it at any endpoint that does -- Azure, a
    # gateway, a proxy -- so naming it for one company's product named something
    # it may never touch. Renamed 2026-09-04 with no alias kept, which cost
    # nothing: the backend had raised NameError on the first image of every run
    # since the module split, so no working script could have used the old name.
    parser.add_argument("--approach", choices=["openai", "llama.cpp", "vllm"],
                        default="vllm",
                        help="Inference backend. vllm is the only one fast enough for a "
                             "full run -- the others are kept for comparison against a "
                             "prior model, not for labelling 49k images")
    parser.add_argument("--llama-url", default="", help="URL for LLaMA.cpp API (llama.cpp approach only; default: from config.toml [servers.llama_cpp])")
    parser.add_argument("--vllm-url", default="", help="URL for the vLLM OpenAI-compatible server (vllm approach only; default: from config.toml [servers.vllm])")
    parser.add_argument("--filter-csv", default="", help="Path to a prior run's CSV; only reprocess 'animal' category or low-confidence rows")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of images per vLLM batch (default 1, vllm only)")
    parser.add_argument("--prior-labels", type=Path, default=None,
                        help="directory holding a previous run's "
                             "bird_identification_output.csv, whose verdicts are "
                             "recorded as prior_category/prior_label for the paired "
                             "comparison (default: data/label if present). The CSV "
                             "has a row per image; sidecars cover only the 20%% that "
                             "had a raw file. Pass a nonexistent path to fall back "
                             "to reading sidecars")
    parser.add_argument("--categories", default="",
                        help="comma-separated categories to label, from the "
                             "previous labelling (--prior-labels). Empty, the "
                             "default, labels everything. `--categories bird` "
                             "skips what a prior run already placed elsewhere, "
                             "which is the cheap half of a paid run: two "
                             "labellers agree on category 0.9626 of the time and "
                             "on species 0.288, so this spends the reliable part. "
                             "No effect when there is no prior labelling")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after N images (0 = no limit). For trying a "
                             "model cheaply, not for scoping a run -- which "
                             "images you get depends on walk order. Use a "
                             "manifest when the population matters")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve the scope and report it, then stop. No model probe, "
                             "no sidecar copy, no writes -- use it to check what a "
                             "--include-from selection actually covers")
    parser.add_argument("--include-from", type=str, default="",
                        help="a manifest name -- a file in manifests/, given without the "
                             "directory. Paths, not patterns, relative to data/jpg; a "
                             "folder line takes its whole subtree at any depth")
    parser.add_argument("--exclude-from", type=str, default="",
                        help="a manifest name to skip; exclude wins over include")
    # --include-orphan-jpg is gone: the walk is over data/jpg, so a JPEG with no
    # sidecar is an ordinary member of the population rather than an opt-in extra.
    args = parser.parse_args()
    # openai has no server to probe, so it is the one backend that needs a name
    # up front. vllm and llama.cpp resolve theirs below, from the server itself.
    # It comes from config.toml `[models] openai`, not from a literal here: the
    # right answer changes as models are retired, and a name buried in code is
    # one nobody edits until a run fails.
    if args.approach == "openai" and not args.model:
        args.model = model_name("openai")
        if not args.model:
            sys.exit("error: no model for the openai backend. Set it in "
                     "config.toml under [models], or pass --model.")


    if args.approach == "vllm" and not args.vllm_url:
        args.vllm_url = server_url("vllm", path="/v1")
    if args.approach == "llama.cpp" and not args.llama_url:
        args.llama_url = server_url("llama_cpp", path="/v1")

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(OUTPUT_DIR / "bird_label.log")

    # Both server backends resolve the serving model before anything else runs, and
    # both treat a failed probe as fatal -- see get_actual_vllm_model_name().
    # A dry run skips it: scope is a question about the dataset, not the model,
    # and needing a live server to find out what a manifest selects would defeat
    # the point of being able to check quickly.
    if args.dry_run:
        logger.info("🔍 Dry run: resolving scope only. No model, no writes.")
    elif args.approach == "llama.cpp" and args.llama_url:
        args.model = get_actual_vllm_model_name(args.llama_url, args.model)
    elif args.approach == "vllm" and args.vllm_url:
        args.model = get_actual_vllm_model_name(args.vllm_url, args.model)

    # Paths. `data/` is treated as read-only: the sidecar tree is copied into
    # output/raw and keywords are injected there, never back into the submodule.
    DATA_DIR = data_dir(args.data_dir)
    JPG_DIR = DATA_DIR / "jpg"
    RAW_DIR = DATA_DIR / "xmp"
    # Prior labels are always read from here, never from the working copy --
    # see prior_labels().
    PRISTINE_XMP_DIR = RAW_DIR
    # The previous labelling, for prior_category/prior_label. data/label by
    # default: it is the curated run, and the one a second model is normally
    # being compared against.
    prior_dir = args.prior_labels if args.prior_labels is not None else (DATA_DIR / "label")
    PRIOR_LABELS = load_prior_labels(prior_dir)
    if PRIOR_LABELS:
        logger.info(f"⚙️  Prior labels: {len(PRIOR_LABELS):,} rows from {prior_dir}")
    else:
        logger.info(f"⚙️  Prior labels: no CSV at {prior_dir}; "
                    f"reading them from sidecars instead (covers only images "
                    f"that have one)")
    if not RAW_DIR.is_dir() or not JPG_DIR.is_dir():
        sys.exit(f"error: expected {RAW_DIR} and {JPG_DIR} to exist")

    # Use a single output folder (no per‑run subdirectory)
    RUN_DIR = OUTPUT_DIR
    RAW_OUT = RUN_DIR / "raw"
    CSV_PATH = RUN_DIR / "bird_identification_output.csv"

    # Ensure output directories exist; copy raw data only if not already present
    # The sidecar tree is copied here and keywords are injected into the copy;
    # data/ is never written to. Prior labels are no longer stripped up front:
    # set_keywords_in_xmp() replaces this pipeline's own keywords per sidecar as
    # it goes, which makes a re-run idempotent whether or not the copy already
    # holds labels, and prior_labels() reads the originals from data/ regardless.
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        # Nothing is written, so the 739 MB copy would be pure cost. The claim
        # index reads the same sidecar names either way.
        RAW_OUT = RAW_DIR
    elif not RAW_OUT.exists():
        logger.info(f"📄 Copying sidecar tree {RAW_DIR} → {RAW_OUT} …")
        shutil.copytree(RAW_DIR, RAW_OUT)
    else:
        logger.info(f"⚙️  Raw output folder {RAW_OUT} already exists; reusing existing files.")
    # Save command‑line args for reproducibility (overwrites previous args.json)
    import subprocess
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_hash = "unknown"
    if not args.dry_run:
        with open(RUN_DIR / "args.json", "w", encoding="utf-8") as f:
            json.dump({**vars(args), "git_commit": git_hash}, f, indent=2)

    start_time = time.perf_counter()

    if args.approach == "vllm":
        logger.info(f"🔧 Using vLLM server at {args.vllm_url} with model '{args.model}'.")

    init_time = time.perf_counter()
    init_elapsed = init_time - start_time
    logger.info(f"⏱️  Initialization complete in {init_elapsed:.1f} seconds.")

    # Process and generate CSV
    logger.info("\n🔧 Labelling exported JPEGs…")
    outcome = process_folder(RAW_OUT, CSV_PATH, args)

    end_time = time.perf_counter()
    processing_elapsed = end_time - init_time
    logger.info(f"⏱️  Initialization complete in {init_elapsed:.1f} seconds.")
    logger.info(f"⏱️  Processing time: {processing_elapsed:.1f} seconds.")

    # Say what happened, and exit non-zero when it was not a success. A run that
    # laboured over nothing used to print "Run complete" exactly like one that
    # laboured over 49,000 images: every per-image failure is caught, logged and
    # breaks the loop, and this line did not look. That is precisely how the
    # openai backend stayed broken -- it raised on the first image of every run,
    # said "Run complete", and left a CSV with a header and no rows.
    if outcome.get("dry_run"):
        logger.info(f"\n✅ Dry run complete. Nothing written.")
    elif outcome["aborted"]:
        logger.info(f"\n❌ Run FAILED after {outcome['labelled']:,} of "
                    f"{outcome['pending']:,} images: {outcome['aborted']}")
        logger.info(f"   Partial output in: {RUN_DIR}")
        logger.info("   The checkpoint holds only what was written, so re-running "
                    "resumes rather than repeats.")
        sys.exit(1)
    elif outcome["interrupted"]:
        logger.info(f"\n⏸️  Interrupted after {outcome['labelled']:,} of "
                    f"{outcome['pending']:,} images. Re-run to resume.")
        logger.info(f"   Output so far in: {RUN_DIR}")
    elif outcome["pending"] and not outcome["labelled"]:
        logger.info(f"\n❌ Run labelled nothing, though {outcome['pending']:,} "
                    f"images were selected"
                    + (f"; the model answered for none of them."
                       if outcome["unanswered"] else "."))
        logger.info("   Nothing was checkpointed, so re-running starts over "
                    "rather than skipping them.")
        sys.exit(1)
    elif outcome["unanswered"]:
        logger.info(f"\n⚠️  Run complete with gaps: {outcome['labelled']:,} labelled, "
                    f"{outcome['unanswered']:,} got no answer from the model.")
        logger.info("   The unanswered are not checkpointed — re-run to retry just those.")
        logger.info(f"   Output stored in: {RUN_DIR}")
    else:
        logger.info(f"\n✅ Run complete: {outcome['labelled']:,} images labelled. "
                    f"Output stored in: {RUN_DIR}")

