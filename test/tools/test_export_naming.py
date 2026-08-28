"""The export's folder name must come from what the clustering recorded.

`csv_post.embeddings_for` falls back to the live `output/embed` when a run's
recorded source no longer resolves. That is right for *reading* vectors -- you
need numbers from somewhere -- and wrong for *naming*: a BioCLIP clustering
whose path had gone stale would be handed whatever sits in `output/embed` and
called `dinov3-512`, a folder asserting in its name a fact it never recorded.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PROJECT_ROOT", str(ROOT))

from tools.export_seriated import (  # noqa: E402
    clustering_source, embedding_identity, embedding_slug)

BIOCLIP = {"model": "hf-hub:imageomics/bioclip-2.5-vith14",
           "image_size": "224x224/crop"}


def _archive(tmp_path, embed_meta=BIOCLIP, source="/gone/embeddings.jsonl"):
    """A run root shaped like one ./clean archived: embed/ beside cluster2/."""
    root = tmp_path / "output_099"
    run = root / "cluster2" / "mcs3"
    run.mkdir(parents=True)
    (root / "embed").mkdir()
    (root / "embed" / "embeddings.jsonl").write_text("")
    (root / "embed" / "run.json").write_text(json.dumps(embed_meta))
    (run / "run.json").write_text(json.dumps({"source": source}))
    return run


def test_recorded_source_is_preferred(tmp_path):
    run = _archive(tmp_path)
    live = tmp_path / "live"
    live.mkdir()
    (live / "embeddings.jsonl").write_text("")
    (live / "run.json").write_text(json.dumps(
        {"model": "facebook/dinov3-vitb16-pretrain-lvd1689m", "image_size": "512x512"}))
    (run / "run.json").write_text(json.dumps(
        {"source": str(live / "embeddings.jsonl")}))
    ident = embedding_identity(clustering_source(run, None))
    assert embedding_slug(ident) == "dinov3-512"


def test_moved_archive_falls_back_to_its_own_sibling(tmp_path):
    """./clean moves a run root, stranding the absolute path it recorded.

    The sibling embed/ under the same root is the same file in its new home, so
    it is the right answer -- and crucially it is *this* run's, not the live one.
    """
    run = _archive(tmp_path)
    assert embedding_slug(embedding_identity(clustering_source(run, None))) == "bioclip-224crop"


def test_no_recoverable_source_yields_no_name(tmp_path):
    """A missing name is a small problem; a confidently wrong one is not."""
    run = tmp_path / "lonely" / "mcs3"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"source": "/gone/embeddings.jsonl"}))
    assert clustering_source(run, None) is None
    assert embedding_identity(None) == {}


def test_no_run_json_at_all_yields_no_name(tmp_path):
    run = tmp_path / "bare" / "mcs3"
    run.mkdir(parents=True)
    assert clustering_source(run, None) is None


def test_explicit_embeddings_wins(tmp_path):
    run = _archive(tmp_path)
    named = tmp_path / "elsewhere.jsonl"
    assert clustering_source(run, named) == named


def test_slug_keeps_the_resize_mode_so_two_sets_cannot_collide():
    crop = embedding_slug({"model": "hf-hub:imageomics/bioclip-2.5-vith14",
                           "image_size": "224x224/crop"})
    squash = embedding_slug({"model": "hf-hub:imageomics/bioclip-2.5-vith14",
                             "image_size": "224x224/squash"})
    assert crop != squash, "two different vector sets must not share a folder name"
    assert crop == "bioclip-224crop" and squash == "bioclip-224squash"
