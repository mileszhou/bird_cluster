"""Tests for the medoid-spread audit -- run from within test/: `pytest tools/`.

The tool exists to test a claim that turned out to be false (`findings/04`), so
what is pinned here is the arithmetic the verdict rested on: size-weighted modal
purity, the scale-free spread measure, and the width check that stops a run being
described with another backbone's vectors.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from audit_medoid_spread import load_vectors, medoid_spread, purity  # noqa: E402


def test_purity_is_weighted_by_group_size():
    # A pure group of 1 must not offset an impure group of 9. Averaging the
    # per-group shares would give 0.5*1 + 0.5*(5/9) = 0.78 instead of 0.6.
    groups = ["big"] * 9 + ["small"]
    values = ["a"] * 5 + ["b"] * 4 + ["c"]
    assert purity(groups, values) == pytest.approx(6 / 10)


def test_purity_of_a_perfect_partition_is_one():
    assert purity(["x", "x", "y"], ["a", "a", "b"]) == 1.0


def test_purity_of_nothing_is_nan():
    assert np.isnan(purity([], []))


def test_spread_ignores_vector_scale():
    # Vectors are L2-normalised first, so a rescaled copy is the same geometry.
    rng = np.random.default_rng(0)
    v = rng.normal(size=(20, 8))
    assert medoid_spread(v) == pytest.approx(medoid_spread(v * 37.0))


def test_the_cosine_cv_is_inflated_by_a_low_mean_not_a_wide_spread():
    """The mistake `findings/04` made, in eight dimensions.

    Two clouds with the *same* spread of cosines, one centred near 0 and one on
    a high similarity floor -- the shape a contrastive space has. The cosine CV
    says the concentrated one is far more evenly spaced. It is not: the standard
    deviations match, and only the denominator moved.
    """
    rng = np.random.default_rng(7)
    wide = rng.normal(size=(60, 24))                    # cosines around 0
    cone = wide + 2.6 * np.array([1.0] + [0.0] * 23)    # same cloud, shifted
    a, b = medoid_spread(wide), medoid_spread(cone)
    assert b["cos_mean"] > 4 * a["cos_mean"]
    assert b["cos_sd"] == pytest.approx(a["cos_sd"], rel=0.25)
    assert b["cos_cv"] < a["cos_cv"] / 3


def test_the_two_cvs_can_disagree_in_direction():
    # Which is the whole reason both are reported: the euclidean one is what
    # Ward is handed, and on this pair it orders the clouds the other way round.
    rng = np.random.default_rng(7)
    wide = rng.normal(size=(60, 24))
    cone = wide + 2.6 * np.array([1.0] + [0.0] * 23)
    a, b = medoid_spread(wide), medoid_spread(cone)
    assert b["cos_cv"] < a["cos_cv"]
    assert b["dist_cv"] > a["dist_cv"]


def jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_load_vectors_keeps_the_order_asked_for(tmp_path):
    p = tmp_path / "e.jsonl"
    jsonl(p, [{"key": "b", "embedding": [0.0, 1.0]},
              {"key": "a", "embedding": [1.0, 0.0]}])
    assert load_vectors(p, ["a", "b"], 2).tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_load_vectors_refuses_the_wrong_width(tmp_path):
    # 768-dim DINOv3 vectors and 1024-dim BioCLIP ones share a key space and
    # nothing else, so a stale recorded `source` is otherwise undetectable.
    p = tmp_path / "e.jsonl"
    jsonl(p, [{"key": "a", "embedding": [1.0, 0.0, 0.0]}])
    with pytest.raises(SystemExit, match="3-dimensional.*clustered 2"):
        load_vectors(p, ["a"], 2)


def test_load_vectors_names_a_missing_medoid(tmp_path):
    p = tmp_path / "e.jsonl"
    jsonl(p, [{"key": "a", "embedding": [1.0, 0.0]}])
    with pytest.raises(SystemExit, match="1 medoids are not in"):
        load_vectors(p, ["a", "gone"], 2)
