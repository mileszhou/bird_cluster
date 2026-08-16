"""Tests for code/lib/csv_post.py -- run from within test/: `pytest lib/`.

Covers `embeddings_for`, which decides *which vectors* an analysis tool reads.
Getting that wrong produces a plausible-looking description of a clustering
computed from somebody else's vectors, so the fallbacks are worth pinning down.
"""
import json

import pytest

from code.lib.csv_post import embeddings_for


def make_run(root, source=None, sibling=True):
    run_dir = root / "cluster" / "mcs15"
    run_dir.mkdir(parents=True)
    if sibling:
        (root / "embed").mkdir()
        (root / "embed" / "embeddings.jsonl").write_text("{}\n")
    if source is not None:
        (run_dir / "run.json").write_text(json.dumps({"source": str(source)}))
    return run_dir


def test_explicit_wins_over_everything(tmp_path):
    run_dir = make_run(tmp_path / "output_001")
    chosen = tmp_path / "elsewhere.jsonl"
    assert embeddings_for(run_dir, chosen) == chosen


def test_the_run_says_what_it_was_built_from(tmp_path):
    """run.json records the source; that is the whole point of recording it."""
    curated = tmp_path / "curated.jsonl"
    curated.write_text("{}\n")
    run_dir = make_run(tmp_path / "output_001", source=curated)
    assert embeddings_for(run_dir) == curated


def test_a_moved_archive_falls_back_to_its_own_sibling(tmp_path):
    """`./clean` moves a whole run root, stranding the absolute path in run.json.

    The sibling embed/ under the same root is that same file in its new home,
    so it must be preferred over any global default -- otherwise archiving a run
    silently re-points its analysis at the live embedding set.
    """
    root = tmp_path / "output_099_moved"
    run_dir = make_run(root, source="/gone/output/embed/embeddings.jsonl")
    assert embeddings_for(run_dir) == root / "embed" / "embeddings.jsonl"


def test_it_refuses_rather_than_guess(tmp_path, monkeypatch):
    """With nothing to go on it must not quietly pick something."""
    import code.lib.csv_post as csv_post
    monkeypatch.setattr(csv_post, "PROJECT_ROOT", tmp_path / "no-such-root")
    run_dir = make_run(tmp_path / "output_001", sibling=False)
    with pytest.raises(SystemExit):
        embeddings_for(run_dir)
