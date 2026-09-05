"""Selecting by a previous run's category, as a manifest rather than a flag.

Scope here is a manifest and nothing else -- `--years` was removed for being a
second mechanism -- so narrowing a run to "what the last labelling called a bird"
generates a list rather than adding a predicate. The list is then an ordinary
manifest: recorded in run.json, diffable, and reproducible.
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

from tools.manifest_from_labels import effective_category  # noqa: E402

COLUMNS = ["jpg", "xmp", "filename", "category", "label", "label_cn", "confidence",
           "note", "prior_category", "prior_label", "applied", "run_label",
           "response_json"]


def _labelling(tmp_path, rows):
    d = tmp_path / "label"
    d.mkdir()
    with open(d / "bird_identification_output.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})
    return d


def _run(label_dir, out, *args):
    return subprocess.run(
        [str(PY), "-m", "tools.manifest_from_labels", "--label-dir", str(label_dir),
         "--out", str(out), *args],
        cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "PROJECT_ROOT": str(ROOT)})


def test_never_demote_is_resolved_like_embed_does():
    """`category` is this run's verdict; the library may have kept the prior one.

    Reading `category` alone silently drops exactly the rows the never-demote
    rule was written to protect.
    """
    overruled = {"category": "animal", "applied": "kept-existing",
                 "prior_category": "bird"}
    assert effective_category(overruled) == "bird"
    assert effective_category({"category": "Bird", "applied": "written"}) == "bird"


def test_selects_by_effective_category(tmp_path):
    label_dir = _labelling(tmp_path, [
        {"jpg": "a/1.jpg", "category": "bird", "confidence": "0.9", "applied": "written"},
        {"jpg": "a/2.jpg", "category": "people", "confidence": "0.9", "applied": "written"},
        {"jpg": "a/3.jpg", "category": "animal", "confidence": "0.9",
         "applied": "kept-existing", "prior_category": "bird"},
    ])
    out = ROOT / "local" / "manifests" / "pytest-tmp.txt"
    try:
        done = _run(label_dir, out, "--category", "bird")
        assert done.returncode == 0, done.stdout + done.stderr
        keys = [l for l in out.read_text().splitlines() if l and not l.startswith("#")]
        assert keys == ["a/1.jpg", "a/3.jpg"], keys
    finally:
        out.unlink(missing_ok=True)


def test_a_manifest_outside_the_repo_is_refused(tmp_path):
    """path_filter enforces this when reading; refuse it when writing too, or a
    run.json records a path nothing can resolve later."""
    label_dir = _labelling(tmp_path, [
        {"jpg": "a/1.jpg", "category": "bird", "confidence": "0.9", "applied": "written"}])
    done = _run(label_dir, tmp_path / "loose.txt", "--category", "bird")
    assert done.returncode != 0
    assert "outside manifests/" in done.stdout + done.stderr


def test_selecting_nothing_is_an_error(tmp_path):
    """A manifest matching no row makes a run that does nothing look like a run
    with nothing to do -- the failure both stages already exit on."""
    label_dir = _labelling(tmp_path, [
        {"jpg": "a/1.jpg", "category": "people", "confidence": "0.9", "applied": "written"}])
    out = ROOT / "local" / "manifests" / "pytest-empty.txt"
    try:
        done = _run(label_dir, out, "--category", "bird")
        assert done.returncode != 0
        assert "match" in done.stdout + done.stderr
    finally:
        out.unlink(missing_ok=True)
