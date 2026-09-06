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


# --- the closed-form differences -------------------------------------------
# R_f depends on centroids and counts alone, so every primitive move has an
# exact O(d) delta. These are the formulae `project/theory/01` states; the tests
# check them against recomputing R_f from scratch, which is the only guarantee
# that the document and the code have not drifted apart.

from audit_partition_reward import (credibility, merge_delta,  # noqa: E402
                                    relocate_delta)


def reward(X, lab, spec):
    """R_f = T - J_f, recomputed the slow way.

    Labels are made contiguous first: `score` sizes its one-hot by lab.max()+1,
    so a gap left by a merge would be an empty cluster and a division by zero.
    """
    codes = {c: i for i, c in enumerate(dict.fromkeys(lab.tolist()))}
    s = score(X, np.array([codes[c] for c in lab.tolist()]), [spec])
    return s["T"] * (1 - s[spec])


def displacements(X, lab, c):
    m = lab == c
    return int(m.sum()), X[m].mean(axis=0) - X.mean(axis=0)


@pytest.mark.parametrize("spec", ["unit", "sqrt", "pow=0.4"])
def test_merge_delta_matches_recomputation(spec):
    rng = np.random.default_rng(11)
    for _ in range(50):
        X = rng.normal(size=(rng.integers(15, 45), 4))
        lab = rng.integers(0, 4, size=len(X))
        if len(np.unique(lab)) < 4:
            continue
        na, ua = displacements(X, lab, 0)
        nb, ub = displacements(X, lab, 1)
        direct = reward(X, np.where(lab == 1, 0, lab), spec) - reward(X, lab, spec)
        assert merge_delta(spec, na, ua, nb, ub) == pytest.approx(direct)


@pytest.mark.parametrize("spec", ["unit", "sqrt", "pow=0.4"])
def test_relocate_delta_matches_recomputation(spec):
    rng = np.random.default_rng(12)
    done = 0
    for _ in range(400):
        X = rng.normal(size=(rng.integers(15, 45), 4))
        lab = rng.integers(0, 4, size=len(X))
        if (lab == 0).sum() < 2 or (lab == 1).sum() < 1:
            continue                       # check counts before taking any mean
        na, ua = displacements(X, lab, 0)
        nb, ub = displacements(X, lab, 1)
        done += 1
        y = int(np.flatnonzero(lab == 0)[0])
        moved = lab.copy()
        moved[y] = 1
        direct = reward(X, moved, spec) - reward(X, lab, spec)
        v = X[y] - X.mean(axis=0)
        assert relocate_delta(spec, na, ua, nb, ub, v) == pytest.approx(direct)
    assert done > 20


def test_relocate_refuses_to_empty_a_cluster():
    with pytest.raises(ValueError, match="different move"):
        relocate_delta("unit", 1, np.zeros(3), 5, np.zeros(3), np.zeros(3))


def test_credibility_is_zero_at_one_and_rises():
    for spec in ("unit", "sqrt", "pow=0.4"):
        g = credibility(spec, np.arange(1, 40))
        assert g[0] == pytest.approx(0.0)
        assert np.all(np.diff(g) > 0)


def test_the_merge_rule_reproduces_the_two_mass_threshold():
    """Proposition 2, recovered from the merge rule rather than from J_f.

    Two point masses must not merge exactly when n > 2 -- the same threshold,
    derived along a different route, which is the check that the two agree.
    """
    for n, merges in ((2, False), (4, False), (100, False)):
        X = two_masses(n)
        lab = np.array([0] * (n // 2) + [1] * (n // 2))
        na, ua = displacements(X, lab, 0)
        nb, ub = displacements(X, lab, 1)
        assert (merge_delta("unit", na, ua, nb, ub) > 0) is merges
    # and at n = 2 it is exactly a tie: both partitions score T.
    X = two_masses(2)
    assert merge_delta("unit", 1, X[0] - X.mean(0), 1, X[1] - X.mean(0)) == pytest.approx(0.0)


def test_absorbing_a_lone_point_is_just_a_merge():
    """Noise is not a primitive: a declined point is a cluster of size one.

    `merge_delta` with n_a = 1 is the whole rule -- no separate formula, no
    threshold, no special case. g(1) = 0 means such a point earns nothing, so
    leaving it out is never free.
    """
    rng = np.random.default_rng(21)
    done = 0
    for _ in range(300):
        X = rng.normal(size=(rng.integers(12, 30), 3))
        lab = np.concatenate(([0], rng.integers(1, 4, size=len(X) - 1)))
        if (lab == 1).sum() < 2:
            continue
        done += 1
        nb, ub = displacements(X, lab, 1)
        v = X[0] - X.mean(axis=0)
        absorbed = lab.copy()
        absorbed[0] = 1
        direct = reward(X, absorbed, "unit") - reward(X, lab, "unit")
        assert merge_delta("unit", 1, v, nb, ub) == pytest.approx(direct)
    assert done > 20


def test_the_residual_class_is_central_not_outlying():
    """The consequence that inverts a density method's notion of noise.

    A point far beyond the cluster is absorbed -- it carries displacement mass
    worth crediting to a group. A point sitting at the grand mean is left alone --
    it carries none. The boundary is the bisector of mu and mu_b.
    """
    ub = np.array([1.0, 0.0])
    for x, absorbed in ((5.0, True), (1.0, True), (0.7, True),
                        (0.3, False), (0.0, False), (-1.0, False)):
        d = merge_delta("unit", 1, np.array([x, 0.0]), 200, ub)
        assert (d > 0) is absorbed, f"point at x={x}"


# --- the credibility family ------------------------------------------------

def test_kappa_one_is_exactly_the_unit_weight():
    n = np.arange(1, 60)
    assert f_weights("kappa=1", n) == pytest.approx(f_weights("unit", n))


@pytest.mark.parametrize("k", ["0.25", "1", "2", "8"])
def test_the_kappa_family_is_admissible(k):
    n = np.arange(1, 200)
    f = f_weights(f"kappa={k}", n)
    assert f[0] == pytest.approx(1.0)            # f(1) = 1
    assert np.all(f <= n)                        # f(n) <= n
    g = credibility(f"kappa={k}", n)
    assert g[0] == pytest.approx(0.0)
    assert np.all(np.diff(g) > 0)                # gamma non-decreasing


def test_kappa_must_be_positive():
    with pytest.raises(SystemExit, match="kappa must be positive"):
        f_weights("kappa=0", np.array([4]))


def test_larger_kappa_discounts_less_and_so_prefers_coarser():
    # gamma is the credibility; a bigger kappa means more shrinkage, a smaller
    # reward, and -- via the split threshold -- a coarser selected partition.
    n = np.arange(2, 100)
    assert np.all(credibility("kappa=4", n) < credibility("kappa=1", n))
    assert np.all(credibility("kappa=1", n) < credibility("kappa=0.25", n))
