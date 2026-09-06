"""Tests for the modified-total-variation criterion -- run from within test/.

The criterion is only meaningful if two identities hold exactly: Huygens
(W + B = T for every partition), and J_f = T at both extremes. Everything else
the tool reports is a comparison of numbers that assume those.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from audit_partition_reward import f_weights, labels_for, score  # noqa: E402

VARIANTS = ["unit", "sqrt"]


def two_masses(n, d=1.0):
    """n points, half at the origin and half at distance d. T = n*d^2/4."""
    return np.array([[0.0, 0.0]] * (n // 2) + [[d, 0.0]] * (n // 2))


def test_huygens_holds_for_an_arbitrary_partition():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    s = score(X, rng.integers(0, 7, size=60), VARIANTS)
    assert s["W"] + s["B"] == pytest.approx(s["T"])


def test_both_extremes_score_exactly_one():
    # The whole point of the construction: trivial has no inter-variation to
    # discount, discrete has f(1)=1 so nothing is discounted. Neither is rewarded.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 6))
    trivial = score(X, np.zeros(40, dtype=int), VARIANTS)
    discrete = score(X, np.arange(40), VARIANTS)
    for v in VARIANTS + ["pair"]:
        assert trivial[v] == pytest.approx(1.0)
        assert discrete[v] == pytest.approx(1.0)


def test_a_random_partition_earns_only_sampling_noise():
    """A grouping with no structure still scores below 1, and that is not a bug.

    Random centroids sit O(sigma/sqrt(n_i)) from the grand mean by chance, so the
    reward is the finite-sample fluctuation and scales as k/N. The check is that
    it *decays* with N at fixed k -- a criterion whose null did not shrink with
    more data would be measuring the partition rather than the structure.
    """
    rng = np.random.default_rng(2)
    got = []
    for n in (400, 1600, 6400):
        X = rng.normal(size=(n, 8))
        got.append(1 - score(X, rng.integers(0, 9, size=n), VARIANTS)["unit"])
    assert got[0] > got[1] > got[2]
    assert got[0] < 0.05                      # small to begin with
    assert got[2] < got[0] / 3                # and shrinking roughly as k/N


@pytest.mark.parametrize("n", [4, 10, 50, 200])
def test_the_natural_partition_of_two_point_masses(n):
    # Closed form: J_unit = d^2/2 against T = n*d^2/4, so J/T = 2/n exactly.
    X = two_masses(n)
    lab = np.array([0] * (n // 2) + [1] * (n // 2))
    assert score(X, lab, VARIANTS)["unit"] == pytest.approx(2.0 / n)


def test_the_natural_partition_beats_both_extremes_iff_n_over_two():
    # n > 2 is the exact threshold, not an approximation.
    for n, wins in ((2, False), (4, True), (100, True)):
        X = two_masses(n)
        lab = np.array([0] * (n // 2) + [1] * (n // 2))
        assert (score(X, lab, VARIANTS)["unit"] < 1.0) is wins


def test_splitting_a_point_mass_is_penalised():
    # Nothing is gained in W -- the points are identical -- and one more cluster
    # must pay its displacement.
    X = two_masses(20)
    natural = np.array([0] * 10 + [1] * 10)
    split = np.array([0] * 5 + [1] * 5 + [2] * 10)
    assert score(X, split, VARIANTS)["unit"] > score(X, natural, VARIANTS)["unit"]


def test_sqrt_rewards_less_than_unit_everywhere():
    # f(n) = sqrt(n) >= 1 = f_unit(n), so the discount n - f(n) is smaller and
    # J is higher. That is what makes it the more split-averse of the two.
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 4))
    s = score(X, rng.integers(0, 12, size=80), VARIANTS)
    assert s["sqrt"] >= s["unit"]


@pytest.mark.parametrize("spec", ["unit", "sqrt", "pow=0.3", "pow=1"])
def test_every_admissible_f_fixes_f_of_one(spec):
    # f(1) = 1 is not a convention -- it is what makes the discrete partition
    # score T rather than winning outright.
    assert f_weights(spec, np.array([1]))[0] == pytest.approx(1.0)


def test_f_never_exceeds_n():
    n = np.arange(1, 500)
    for spec in ("unit", "sqrt", "pow=0.9"):
        assert np.all(f_weights(spec, n) <= n)


def test_unknown_f_is_refused():
    with pytest.raises(SystemExit, match="unknown f variant"):
        f_weights("log", np.array([4]))


ROWS = [{"key": "a", "cluster_id": "0", "is_noise": "0"},
        {"key": "b", "cluster_id": "0", "is_noise": "0"},
        {"key": "c", "cluster_id": "-1", "is_noise": "1"}]
INDEX = {"a": 0, "b": 1, "c": 2}


def test_singleton_convention_keeps_noise_as_its_own_cluster():
    take, lab = labels_for(ROWS, set(INDEX), "singleton", INDEX)
    assert take.tolist() == [0, 1, 2]
    assert len(set(lab.tolist())) == 2 and lab[0] == lab[1] and lab[2] != lab[0]


def test_exclude_convention_drops_noise():
    take, lab = labels_for(ROWS, set(INDEX), "exclude", INDEX)
    assert take.tolist() == [0, 1] and lab.tolist() == [0, 0]


def test_two_noise_points_do_not_share_a_cluster():
    rows = ROWS + [{"key": "d", "cluster_id": "-1", "is_noise": "1"}]
    index = dict(INDEX, d=3)
    _, lab = labels_for(rows, set(index), "singleton", index)
    assert lab[2] != lab[3]
