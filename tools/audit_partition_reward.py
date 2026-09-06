#!/usr/bin/env python3
"""Score a partition by the modified total variation, and rank a sweep by it.

Huygens' identity makes the total variation a constant of the *data*, not of the
partition: for every partition of S,

    T  =  W(P)  +  B(P)
       =  sum_i sum_{x in C_i} ||x - mu_i||^2   +   sum_i n_i ||mu_i - mu||^2

The two extremes sit at opposite ends of that split -- the trivial partition puts
everything in W, the discrete partition everything in B -- and both total T.

Now discount the *inter* term by a weight f(n) <= n with f(1) = 1:

    J_f(P)  =  W(P) + sum_i f(n_i) ||mu_i - mu||^2  =  T - R_f(P)
    R_f(P)  =  sum_i ( n_i - f(n_i) ) ||mu_i - mu||^2   >=  0

J_f <= T everywhere, and equals T at *both* extremes for different reasons: the
trivial partition has mu_1 = mu, so there is no inter-variation to discount; the
discrete partition has f(1) = 1, so nothing is discounted. A partition is
rewarded only for between-variation it has captured in groups larger than one,
which is what makes the minimum interior and the criterion non-degenerate.

The point of the flexibility in f is that the split threshold it induces,

    split a cluster of n iff  (rise in mean displacement) > g(n) / 2g(n/2),  g = n - f

differs by 2x to 10x in its excess over 1 across n = 10..500 between f = 1 and
f = sqrt(n) -- the range real leaves occupy -- even though both tend to 1.

Three inter-term variants are reported:

    unit   f(n) = 1        every cluster pays its displacement once
    sqrt   f(n) = sqrt(n)  a milder discount, harsher on splitting
    pair   the same deformation applied to the *pairwise* form of B,
           (1/2N) sum_ij ||mu_i - mu_j||^2, which equals (k/N) sum_i ||mu_i - mubar||^2

`pair` is worth carrying because the two expressions of B are equal but their
deformations are not. It is not centre-free -- the identity holds only when mu is
the mass centre, so the centre is implied rather than absent -- but the count k
multiplies the dispersion, which is what stops a cluster sitting at the centre
being free. The anchored form also carries a hidden term k||mubar - mu||^2 that
charges for size correlating with position, which nothing here asked for.

**On noise, which is not a detail.** HDBSCAN returns a partial partition -- 15% of
the population unassigned at mcs3, 44% at mcs40 -- and J_f is defined over a
partition of all of S. The convention decides the answer, so all three are
reported and none is a default worth trusting alone:

    singleton  each noise point is its own cluster. One population for the whole
               sweep, so the numbers are comparable -- but a partition that
               declines more points is charged for every one of them, which will
               favour small min_cluster_size whatever the structure says.
    exclude    drop noise. Each mcs is then scored over its *own* population, so
               T differs per row and the ratios are not strictly comparable.
    common     restrict to the images every mcs in the arm assigns. One
               population, no charge for declining -- but clusters are truncated,
               so the n_i are not the ones the clustering chose.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from code.lib.csv_post import embeddings_for  # noqa: E402


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_vectors(path: Path, keys: set[str], dims: int | None) -> dict:
    got = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["key"] in keys:
                got[row["key"]] = row["embedding"]
    if not got:
        raise SystemExit(f"error: none of the {len(keys)} keys are in {path}")
    width = len(next(iter(got.values())))
    if dims is not None and width != dims:
        raise SystemExit(f"error: {path} is {width}-dimensional, but the run "
                         f"clustered {dims}-dimensional vectors.")
    return got


def f_weights(spec: str, n: np.ndarray) -> np.ndarray:
    """f(n) for a named variant. Must satisfy f(1) = 1 and f(n) <= n."""
    if spec == "unit":
        return np.ones_like(n, dtype=float)
    if spec == "sqrt":
        return np.sqrt(n.astype(float))
    m = re.fullmatch(r"pow=([0-9.]+)", spec)
    if m:
        return n.astype(float) ** float(m.group(1))
    raise SystemExit(f"error: unknown f variant {spec!r}; "
                     f"use unit, sqrt or pow=<alpha>")


def score(X: np.ndarray, lab: np.ndarray, variants: list[str]) -> dict:
    """T, W, B and J_f/T for one partition. `lab` is contiguous 0..k-1."""
    N, k = len(X), int(lab.max()) + 1
    onehot = csr_matrix((np.ones(N), (lab, np.arange(N))), shape=(k, N))
    n = np.asarray(onehot.sum(axis=1)).ravel()
    mu_i = (onehot @ X) / n[:, None]
    mu = X.mean(axis=0)

    T = float(((X - mu) ** 2).sum())
    disp = ((mu_i - mu) ** 2).sum(axis=1)          # ||mu_i - mu||^2
    B = float((n * disp).sum())
    W = T - B

    out = {"k": k, "N": N, "T": T, "W": W, "B": B}
    for spec in variants:
        out[spec] = (W + float((f_weights(spec, n) * disp).sum())) / T
    # The pairwise deformation: (1/2N) sum_ij ||mu_i-mu_j||^2 = (k/N) sum ||mu_i-mubar||^2
    mubar = mu_i.mean(axis=0)
    out["pair"] = (W + (k / N) * float(((mu_i - mubar) ** 2).sum())) / T
    return out


def labels_for(rows: list[dict], keep: set[str], convention: str,
               index: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """Row indices into X, and a contiguous cluster label for each."""
    ids, taken = [], []
    for r in rows:
        if r["key"] not in keep:
            continue
        noise = r["is_noise"] not in ("0", "False", "")
        if noise and convention != "singleton":
            continue
        taken.append(index[r["key"]])
        ids.append(f"n{r['key']}" if noise else r["cluster_id"])
    if not ids:
        raise SystemExit("error: the convention left no rows to score")
    codes = {c: i for i, c in enumerate(dict.fromkeys(ids))}
    return np.asarray(taken), np.asarray([codes[c] for c in ids])


def mcs_order(p: Path) -> tuple:
    m = re.search(r"\d+", p.name)
    return (int(m.group()) if m else 0, p.name)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", nargs=2, action="append", required=True,
                    metavar=("LABEL", "CLUSTER_DIR"),
                    help="a name and a level-1 run directory holding mcs*/ . "
                         "Repeatable")
    ap.add_argument("--embeddings", type=Path, action="append", default=None,
                    help="vectors for the matching --arm, when the run's own "
                         "recorded source has gone stale. Positional against --arm")
    ap.add_argument("--f", action="append", default=None, metavar="SPEC",
                    help="inter-term weight: unit, sqrt, or pow=<alpha>. "
                         "Repeatable (default: unit and sqrt)")
    ap.add_argument("--convention", action="append", default=None,
                    choices=("singleton", "exclude", "common"),
                    help="how HDBSCAN's noise enters. Repeatable (default: all "
                         "three, because the choice decides the answer)")
    ap.add_argument("--check", action="store_true",
                    help="verify W+B=T and that both extremes score J/T = 1")
    args = ap.parse_args()

    variants = args.f or ["unit", "sqrt"]
    conventions = args.convention or ["singleton", "exclude", "common"]
    vecs = list(args.embeddings or [])
    vecs += [None] * (len(args.arm) - len(vecs))

    for (label, cdir), vec in zip(args.arm, vecs):
        runs = sorted(Path(cdir).glob("mcs*"), key=mcs_order)
        runs = [r for r in runs if (r / "assignments.csv").is_file()]
        if not runs:
            raise SystemExit(f"error: no mcs*/assignments.csv under {cdir}")
        rows = {r.name: read_csv(r / "assignments.csv") for r in runs}
        dims = json.loads((runs[0] / "run.json").read_text()).get("dims")
        keys = {x["key"] for rs in rows.values() for x in rs}
        store = load_vectors(embeddings_for(runs[0], vec), keys, dims)
        order = sorted(store)
        index = {k: i for i, k in enumerate(order)}
        X = np.asarray([store[k] for k in order], dtype=np.float64)

        assigned = [{x["key"] for x in rs if x["is_noise"] in ("0", "False", "")}
                    for rs in rows.values()]
        common = set.intersection(*assigned) if assigned else set()

        print(f"\n  {label}: {len(runs)} partitions, {len(X):,} vectors, "
              f"{X.shape[1]} dims  |  common to all: {len(common):,}")
        if args.check:
            run_checks(X, variants)

        cols = variants + ["pair"]
        for convention in conventions:
            keep = common if convention == "common" else keys
            print(f"\n  convention: {convention}")
            print(f"    {'partition':<11}{'clusters':>9}{'scored':>9}{'W/T':>9}"
                  + "".join(f"{'J_' + c + '/T':>12}" for c in cols))
            table = []
            for run in runs:
                take, lab = labels_for(rows[run.name], keep, convention, index)
                table.append((run.name, score(X[take], lab, variants)))
            best = {c: min(range(len(table)), key=lambda i: table[i][1][c])
                    for c in cols}
            for i, (name, s) in enumerate(table):
                line = (f"    {name:<11}{s['k']:>9,}{s['N']:>9,}"
                        f"{s['W'] / s['T']:>9.4f}")
                for c in cols:
                    mark = "*" if best[c] == i else " "
                    line += f"{s[c]:>11.5f}{mark}"
                print(line)
            print("    lower is better; 1.00000 is what both extremes score. "
                  "* is the best in its column.")


def run_checks(X: np.ndarray, variants: list[str]) -> None:
    """The two identities the criterion rests on, on this arm's own vectors."""
    N = len(X)
    trivial = score(X, np.zeros(N, dtype=int), variants)
    discrete = score(X, np.arange(N), variants)
    mid = score(X, (np.arange(N) % 7), variants)
    ok = True
    for name, s in (("trivial", trivial), ("discrete", discrete), ("7 groups", mid)):
        huygens = abs(s["W"] + s["B"] - s["T"]) / s["T"]
        ok &= huygens < 1e-9
        print(f"    check {name:<10} W+B-T = {huygens:.2e} of T   "
              + "  ".join(f"J_{c}/T={s[c]:.6f}" for c in variants + ["pair"]))
    ok &= all(abs(trivial[c] - 1) < 1e-9 and abs(discrete[c] - 1) < 1e-9
              for c in variants + ["pair"])
    print(f"    checks {'pass' if ok else 'FAIL'}: Huygens holds and both "
          f"extremes score exactly 1")


if __name__ == "__main__":
    main()
