"""Tests for the criterion descent -- run from within test/: `pytest lib/`.

What has to hold, from `project/theory/01`: R_f never falls (every move is
verified exactly before it is applied), both extremes score J_f = T, and the
closed-form deltas agree with recomputing R_f from scratch. If a delta and its
application ever disagree the search silently stops being a descent, so these
are the load-bearing tests.
"""
import numpy as np
import pytest

from code.cluster.descend import (Partition, descend, f_weight, gamma,
                                  merge_sweep, relocate_sweep)


def centred(X):
    return X - X.mean(0)


def two_masses(n, d=1.0):
    return centred(np.array([[0.0, 0.0]] * (n // 2) + [[d, 0.0]] * (n // 2)))


@pytest.mark.parametrize("spec", ["unit", "sqrt", "kappa=1", "kappa=4", "pow=0.5"])
def test_f_is_admissible(spec):
    n = np.arange(1, 300)
    assert f_weight(spec, np.array([1]))[0] == pytest.approx(1.0)   # f(1) = 1
    assert np.all(f_weight(spec, n) <= n)                           # f(n) <= n
    g = gamma(spec, n)
    assert g[0] == pytest.approx(0.0) and np.all(np.diff(g) > -1e-12)


def test_kappa_one_is_the_unit_weight():
    n = np.arange(1, 80)
    assert f_weight("kappa=1", n) == pytest.approx(f_weight("unit", n))


@pytest.mark.parametrize("spec", ["unit", "sqrt", "kappa=3"])
def test_both_extremes_score_J_equal_T(spec):
    rng = np.random.default_rng(0)
    V = centred(rng.normal(size=(40, 5)))
    trivial = Partition(V, np.zeros(40, dtype=int), spec)
    discrete = Partition(V, np.arange(40), spec)
    assert trivial.reward() == pytest.approx(0.0)      # J = T - 0 = T
    assert discrete.reward() == pytest.approx(0.0)


@pytest.mark.parametrize("spec", ["unit", "sqrt", "kappa=3"])
def test_merge_delta_matches_recomputation(spec):
    rng = np.random.default_rng(1)
    for _ in range(40):
        V = centred(rng.normal(size=(50, 4)))
        P = Partition(V, rng.integers(0, 5, size=50), spec)
        if (P.n == 0).any():
            continue
        before = P.reward()
        predicted = P.merge_delta(0, 1)
        P.do_merge(0, 1)
        assert P.reward() - before == pytest.approx(predicted)


@pytest.mark.parametrize("spec", ["unit", "sqrt", "kappa=3"])
def test_relocate_delta_matches_recomputation(spec):
    rng = np.random.default_rng(2)
    done = 0
    for _ in range(60):
        V = centred(rng.normal(size=(50, 4)))
        P = Partition(V, rng.integers(0, 5, size=50), spec)
        i = int(np.flatnonzero(P.lab == 0)[0]) if (P.lab == 0).any() else None
        if i is None or P.n[0] < 2 or P.n[1] == 0:
            continue
        done += 1
        before = P.reward()
        predicted = P.relocate_delta(i, 1)
        P.do_relocate(i, 1)
        assert P.reward() - before == pytest.approx(predicted)
    assert done > 10


def test_relocate_refuses_to_empty_a_cluster():
    V = centred(np.random.default_rng(3).normal(size=(6, 3)))
    P = Partition(V, np.array([0, 1, 1, 1, 2, 2]), "unit")
    assert P.relocate_delta(0, 1) == -np.inf     # cluster 0 would vanish


def test_a_sweep_never_lowers_the_reward():
    """The property the whole search rests on: propose in batch, verify exactly."""
    rng = np.random.default_rng(4)
    for spec in ("unit", "sqrt", "kappa=6"):
        V = centred(rng.normal(size=(200, 6)))
        P = Partition(V, rng.integers(0, 20, size=200), spec)
        for _ in range(6):
            before = P.reward()
            relocate_sweep(P)
            assert P.reward() >= before - 1e-9
            before = P.reward()
            merge_sweep(P)
            assert P.reward() >= before - 1e-9


def test_it_finds_two_point_masses():
    """Proposition 2's case: the natural partition is the optimum for n > 2."""
    V = two_masses(200)
    labels = np.random.default_rng(5).integers(0, 30, size=200)
    P = descend(V, labels, "unit", max_sweeps=40)
    assert P.k == 2
    left = {tuple(np.round(V[i], 6)) for i in np.flatnonzero(P.lab == P.lab[0])}
    assert len(left) == 1                       # each cluster is one mass, exactly


def test_larger_kappa_gives_fewer_clusters():
    """kappa shrinks harder, rewards less, and selects coarser -- monotonically."""
    rng = np.random.default_rng(6)
    V = centred(np.vstack([rng.normal(c, 0.35, size=(60, 4))
                           for c in (-6, -2, 2, 6)]))
    ks = [descend(V, np.arange(len(V)) % 40, s, 40).k
          for s in ("kappa=1", "kappa=10", "kappa=200")]
    assert ks[0] >= ks[1] >= ks[2]
    assert ks[0] > ks[2]


def test_noise_becomes_singletons_not_a_class():
    """The criterion has no noise class; a declined point is a cluster of one."""
    import csv, io
    from code.cluster.descend import initial_labels
    rows = [{"key": f"k{i}"} for i in range(5)]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["key", "cluster_id"]); w.writeheader()
    for i, cid in enumerate(["0", "0", "-1", "-1", "1"]):
        w.writerow({"key": f"k{i}", "cluster_id": cid})
    import tempfile, pathlib
    p = pathlib.Path(tempfile.mkdtemp()) / "assignments.csv"
    p.write_text(buf.getvalue())

    class A:
        init, init_from, init_k, seed = "from", p, 0, 0
    lab = initial_labels(np.zeros((5, 2)), A, rows)
    assert lab[0] == lab[1]                     # a real cluster stays together
    assert len({lab[2], lab[3], lab[0], lab[4]}) == 4   # each noise point alone
