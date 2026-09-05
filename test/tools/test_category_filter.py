"""Labelling only what a previous run placed in a category.

Resolved at run time against `--prior-labels`, not through a generated manifest.
A manifest would have to live in local/, which is gitignored, so a run.json
naming one is a dangling reference for everyone else -- the exact failure the
manifest rule exists to prevent. `--prior-labels data/label --categories bird`
names a versioned submodule and a word, and reproduces anywhere.

The category is the part of a labelling worth acting on: two independent
labellers agree on it 0.9626 of the time and on species 0.288.
"""
import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "bin" / "python"
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PROJECT_ROOT", str(ROOT))

from code.bird_label import _effective  # noqa: E402

COLUMNS = ["jpg", "xmp", "filename", "category", "label", "label_cn", "confidence",
           "note", "prior_category", "prior_label", "applied", "run_label",
           "response_json"]


def test_never_demote_is_resolved_like_embed_does():
    """`category` is this run's verdict; the library may have kept the prior one.

    Reading `category` alone would drop exactly the rows that rule protects.
    """
    assert _effective({"category": "animal", "applied": "kept-existing",
                       "prior_category": "bird", "prior_label": "x-y-kestrel"}) \
        == ("bird", "x-y-kestrel")
    assert _effective({"category": "Bird", "label": "wagtail",
                       "applied": "written"}) == ("bird", "wagtail")


@pytest.fixture
def tree(tmp_path):
    """A tiny data dir plus a prior labelling that disagrees with itself."""
    from PIL import Image
    d = tmp_path / "data"
    trip = d / "jpg" / "Photos-16" / "trip"
    trip.mkdir(parents=True)
    (d / "xmp").mkdir()
    for n in ("bird1", "bird2", "person1"):
        Image.new("RGB", (8, 8)).save(trip / f"{n}.jpg")

    label = tmp_path / "label"
    label.mkdir()
    rows = [
        {"jpg": "Photos-16/trip/bird1.jpg", "category": "bird", "applied": "written"},
        {"jpg": "Photos-16/trip/person1.jpg", "category": "people", "applied": "written"},
        # overruled: this run said animal, the library kept bird
        {"jpg": "Photos-16/trip/bird2.jpg", "category": "animal",
         "applied": "kept-existing", "prior_category": "bird"},
    ]
    with open(label / "bird_identification_output.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})
    return d, label


def _dry(data, label, *args):
    """A dry run in an isolated output directory.

    `--output-dir` is not optional here even though a dry run writes nothing: it
    still *reads* the checkpoint, and the default is the live `output/label`. A
    test that omits it reports whatever a real run happens to have processed --
    and if one is running, two invocations disagree, which is exactly how this
    was found.
    """
    out = Path(str(data)).parent / "out"
    return subprocess.run(
        [str(PY), "-m", "code.bird_label", "--approach", "openai", "--dry-run",
         "--data-dir", str(data), "--prior-labels", str(label),
         "--output-dir", str(out), *args],
        cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "PROJECT_ROOT": str(ROOT)})


def test_filters_to_the_wanted_category(tree):
    data, label = tree
    done = _dry(data, label, "--categories", "bird")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "2 of 3" in done.stdout, done.stdout
    assert "2 would be sent to the model" in done.stdout, done.stdout


def test_no_filter_by_default(tree):
    data, label = tree
    done = _dry(data, label)
    assert "3 would be sent to the model" in done.stdout, done.stdout


def test_without_a_prior_labelling_it_runs_everything(tree, tmp_path):
    """"If label exists, filter; or the full run." Never a silent empty run."""
    data, _ = tree
    done = _dry(data, tmp_path / "absent", "--categories", "bird")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "no prior labelling to filter on" in done.stdout, done.stdout
    assert "3 would be sent to the model" in done.stdout, done.stdout


def test_a_seeded_sample_is_spread_and_reproducible(tree, tmp_path):
    """Walk order is path order, so `--limit` alone takes consecutive trips.

    Over the real library that is 43 folders and 326 species in 2,000 images,
    against 415 and 778 drawn at random -- and a threshold fitted on the first
    would not generalise. The seed goes into args.json, so the draw is
    reproducible.
    """
    from PIL import Image
    data, label = tree
    # three folders, so a draw can differ from the head of the walk
    rows = []
    for folder in ("a", "b", "c"):
        d = data / "jpg" / "Photos-16" / folder
        d.mkdir(parents=True, exist_ok=True)
        for n in range(4):
            Image.new("RGB", (8, 8)).save(d / f"{n}.jpg")
            rows.append(f"Photos-16/{folder}/{n}.jpg")

    import csv as _csv
    with open(label / "bird_identification_output.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = _csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for key in rows:
            w.writerow({c: "" for c in COLUMNS} | {"jpg": key, "category": "bird",
                                                   "applied": "written"})

    def draw(*extra):
        done = _dry(data, label, "--categories", "bird", "--limit", "6", *extra)
        assert done.returncode == 0, done.stdout + done.stderr
        return done.stdout

    assert "walk order" in draw(), "unseeded runs should say they are in walk order"
    first = draw("--sample-seed", "5")
    assert "pending shuffled" in first, first
    assert draw("--sample-seed", "5") == first, "same seed must give the same draw"


def test_the_seed_shuffles_without_a_limit(tree, tmp_path):
    """A full run needs the shuffle as much as a capped one.

    Tying it to `--limit` was wrong: a full run in walk order is fine only if it
    finishes. Interrupt it, or look at the results before the end, and what you
    have is a path-ordered prefix -- the exact bias the seed exists to remove.
    Shuffled, every prefix is a random sample.
    """
    data, label = tree
    done = _dry(data, label, "--categories", "bird", "--sample-seed", "3")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "pending shuffled" in done.stdout, done.stdout
    assert "any prefix is a random sample" in done.stdout, done.stdout
