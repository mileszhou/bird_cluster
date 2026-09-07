#!/usr/bin/env python3
"""Cluster by descending on the variation criterion in `project/theory/01`.

Huygens makes T = W + B a constant of the data, so both extreme partitions spend
it in opposite halves and no undeformed combination separates them. Discount the
*between* term by an admissible f (f(n) <= n, f(1) = 1) and

    J_f = W + sum_i f(n_i)||mu_i - mu||^2  =  T - R_f
    R_f = sum_i g(n_i) ||u_i||^2,   g = n - f,   u_i = mu_i - mu

is at or below T everywhere and equal at *both* extremes, so the minimiser is
interior. Minimising J_f is maximising R_f, which depends on centroids and counts
alone -- the points never enter, and W is never computed.

**Why this needs no k, no min_cluster_size and no noise class.** All three are
consequences of one inequality here. A point left out is a cluster of size one,
g(1) = 0 means it earns nothing, and the merge rule decides its fate like any
other cluster's. There is nothing to declare because there is nothing to exempt.

**Why it never splits.** Of the three primitive moves, relocate and merge are
*exhaustive* -- every candidate can be enumerated and scored exactly -- while
"split C_a" is 2^(n_a-1)-1 moves rather than one, so it needs an outside
heuristic to propose the bisection. Starting from many small clusters and only
merging avoids the question: k falls to where the criterion stops it, and the
one unprincipled choice is confined to the initialisation.

The cost of that, stated plainly: **the starting k bounds the final k from
above**, and the answer is a local maximum that depends on where it began.
Whether several starting points converge is a property of the data, and running
--init-k over a range is how you find out. `theory/01` records that no bound is
known on the gap to the lattice optimum.

Every move is verified exactly before it is applied, using the closed forms
derived in the theory document, so R_f increases monotonically and the search
terminates on a finite lattice. The proposal matrices are only that -- proposals.
"""
import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from code.cluster.cluster import (  # noqa: E402
    capture_of, cluster_centres, drop_duplicate_captures, label_agreement, load,
    write_assignments)
from code.lib.csv_post import embeddings_for  # noqa: E402

logger = logging.getLogger("descend")


# --- the criterion ---------------------------------------------------------

def f_weight(spec: str, n):
    """An admissible f: f(n) <= n and f(1) = 1, both forced by Proposition 1."""
    n = np.asarray(n, dtype=np.float64)
    if spec == "unit":
        return np.ones_like(n)
    if spec == "sqrt":
        return np.sqrt(n)
    if spec.startswith("kappa="):
        k = float(spec.split("=", 1)[1])
        if k <= 0:
            raise SystemExit("error: kappa must be positive")
        return k * n / (n - 1 + k)
    if spec.startswith("pow="):
        return n ** float(spec.split("=", 1)[1])
    raise SystemExit(f"error: unknown f {spec!r}; use unit, sqrt, kappa=<k>, pow=<a>")


def gamma(spec: str, n):
    """Per-point credibility g(n)/n = 1 - f(n)/n. gamma(1) = 0, rising with n."""
    n = np.asarray(n, dtype=np.float64)
    return (n - f_weight(spec, n)) / n


class Partition:
    """Centroids and counts in displacement coordinates, plus exact move deltas.

    `U[c]` is mu_c - mu and `n[c]` the count, which between them are all R_f
    depends on. Sums are carried rather than means so a move is an O(d) update
    with no accumulated drift.
    """

    def __init__(self, V, labels, spec):
        self.V, self.spec = V, spec                    # V: points, mean-centred
        self.k = int(labels.max()) + 1
        self.lab = labels.astype(np.int64)
        self.S = np.zeros((self.k, V.shape[1]))        # sum of members
        np.add.at(self.S, self.lab, V)
        self.n = np.bincount(self.lab, minlength=self.k).astype(np.float64)
        self.T = float((V * V).sum())

    @property
    def U(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.n[:, None] > 0, self.S / np.maximum(self.n, 1)[:, None], 0.0)

    def reward(self) -> float:
        live = self.n > 0
        g = self.n[live] - f_weight(self.spec, self.n[live])
        return float((g * (self.U[live] ** 2).sum(1)).sum())

    # -- exact deltas, the closed forms from theory/01 ----------------------

    def merge_delta(self, a, b) -> float:
        na, nb = self.n[a], self.n[b]
        if na == 0 or nb == 0:
            return -np.inf
        ua, ub = self.S[a] / na, self.S[b] / nb
        n = na + nb
        gn, ga, gb = (gamma(self.spec, x) for x in (n, na, nb))
        d = ua - ub
        return float(na * (gn - ga) * ua @ ua + nb * (gn - gb) * ub @ ub
                     - gn * (na * nb / n) * d @ d)

    def relocate_delta(self, i, b) -> float:
        a = self.lab[i]
        if a == b:
            return 0.0
        na, nb, v = self.n[a], self.n[b], self.V[i]
        if na < 2 or nb == 0:            # emptying a cluster is a merge, not this
            return -np.inf
        ua, ub = self.S[a] / na, self.S[b] / nb
        ga_, gb_ = gamma(self.spec, na - 1), gamma(self.spec, nb + 1)
        da, db = ua - v, ub - v
        return float(na * (ga_ - gamma(self.spec, na)) * ua @ ua
                     + nb * (gb_ - gamma(self.spec, nb)) * ub @ ub
                     + (gb_ - ga_) * v @ v
                     + ga_ * (na / (na - 1)) * da @ da
                     - gb_ * (nb / (nb + 1)) * db @ db)

    # -- applying one -------------------------------------------------------

    def do_merge(self, a, b):
        self.S[a] += self.S[b]; self.n[a] += self.n[b]
        self.S[b] = 0.0; self.n[b] = 0
        self.lab[self.lab == b] = a

    def do_relocate(self, i, b):
        a = self.lab[i]
        self.S[a] -= self.V[i]; self.n[a] -= 1
        self.S[b] += self.V[i]; self.n[b] += 1
        self.lab[i] = b

    def compact(self):
        live = np.flatnonzero(self.n > 0)
        remap = -np.ones(self.k, dtype=np.int64)
        remap[live] = np.arange(len(live))
        self.lab, self.S, self.n, self.k = remap[self.lab], self.S[live], self.n[live], len(live)


# --- the two sweeps --------------------------------------------------------

def merge_sweep(P) -> int:
    """Every pair proposed at once; each verified exactly before it is applied."""
    live = np.flatnonzero(P.n > 0)
    if len(live) < 2:
        return 0
    U, n = P.U[live], P.n[live]
    g = gamma(P.spec, n)
    sq = (U ** 2).sum(1)
    D = np.maximum(sq[:, None] + sq[None, :] - 2 * U @ U.T, 0.0)   # ||u_a-u_b||^2
    ntot = n[:, None] + n[None, :]
    gn = gamma(P.spec, ntot)
    delta = (n[:, None] * (gn - g[:, None]) * sq[:, None]
             + n[None, :] * (gn - g[None, :]) * sq[None, :]
             - gn * (n[:, None] * n[None, :] / ntot) * D)
    iu = np.triu_indices(len(live), k=1)
    flat = delta[iu]                    # hoisted: k^2/2 entries, built once
    pos = np.flatnonzero(flat > 0)
    order = pos[np.argsort(-flat[pos])]
    # At most k/2 merges can apply in one sweep, since each consumes two
    # clusters, so a cap costs nothing -- whatever is skipped is re-proposed
    # next sweep against centroids that have actually moved.
    order = order[:max(10_000, 20 * len(live))]
    done, applied = set(), 0
    for a, b in zip(live[iu[0][order]], live[iu[1][order]]):
        if a in done or b in done:
            continue                    # its centroid moved; next sweep re-proposes
        if P.merge_delta(a, b) > 0:     # verify exactly -- monotone by construction
            P.do_merge(a, b); done |= {a, b}; applied += 1
    return applied


def relocate_sweep(P, chunk=4096) -> int:
    """Best target per point proposed by one matmul, then each verified exactly."""
    live = np.flatnonzero(P.n > 0)
    U, n = P.U[live], P.n[live]
    g, gplus = gamma(P.spec, n), gamma(P.spec, n + 1)
    sq = (U ** 2).sum(1)
    B = n * (gplus - g) * sq                       # target-only term
    scale = gplus * (n / (n + 1))
    applied = 0
    for s in range(0, len(P.V), chunk):
        V = P.V[s:s + chunk]
        D = np.maximum((V ** 2).sum(1)[:, None] + sq[None, :] - 2 * V @ U.T, 0.0)
        prop = B[None, :] + gplus[None, :] * (V ** 2).sum(1)[:, None] - scale[None, :] * D
        prop[np.arange(len(V)), np.searchsorted(live, P.lab[s:s + chunk])] = -np.inf
        best = prop.argmax(1)
        for j in np.argsort(-prop[np.arange(len(V)), best]):
            i, b = s + int(j), int(live[best[j]])
            if P.relocate_delta(i, b) > 0:         # verify exactly
                P.do_relocate(i, b); applied += 1
    return applied


def descend(V, labels, spec, max_sweeps=50):
    P = Partition(V, labels, spec)
    logger.info("  start   k=%-6d R/T=%.5f", int((P.n > 0).sum()), P.reward() / P.T)
    for sweep in range(1, max_sweeps + 1):
        before = P.reward()
        moved = relocate_sweep(P)
        merged = merge_sweep(P)
        after = P.reward()
        logger.info("  sweep %-2d k=%-6d R/T=%.5f  (%d moved, %d merged)",
                    sweep, int((P.n > 0).sum()), after / P.T, moved, merged)
        if after < before - 1e-9:
            raise SystemExit(f"error: R_f fell {before:.6f} -> {after:.6f}; "
                             f"a delta and its application disagree")
        if moved == 0 and merged == 0:
            break
    P.compact()
    return P


# --- initialisation --------------------------------------------------------

def expand_duplicates(labels, kept_rows, all_rows):
    """Put the dropped alternate edits back, in their capture's cluster.

    Deduplication is done for the *fitter* -- HDBSCAN's density and this
    criterion's counts are both distorted by the same photograph appearing
    several times -- but it is not a reason for the image to be missing from the
    result. The dropped rows are alternate edits of one capture, sharing a
    sidecar, so the representative's cluster is their answer exactly rather than
    approximately; nothing is inferred.

    `capture_of()` and the min-key representative rule are the same ones
    `drop_duplicate_captures()` used, so this inverts it rather than guessing at
    it.
    """
    of = {r["key"]: i for i, r in enumerate(kept_rows)}
    rep = {}
    for r in all_rows:
        c = capture_of(r)
        if r["key"] in of:
            rep[c] = of[r["key"]]
    out = []
    for r in all_rows:
        i = of.get(r["key"], rep.get(capture_of(r)))
        if i is None:
            raise SystemExit(f"error: {r['key']!r} has no representative; "
                             f"the dedup rule and its inverse disagree")
        out.append(labels[i])
    return np.asarray(out)


def initial_labels(V, args, rows):
    if args.init == "from":
        import csv
        by_key = {}
        with open(args.init_from, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                by_key[r["key"]] = r["cluster_id"]
        missing = [r["key"] for r in rows if r["key"] not in by_key]
        if missing:
            raise SystemExit(f"error: {len(missing)} keys are not in "
                             f"{args.init_from}, first {missing[0]!r}")
        # noise (-1) enters as its own singleton: the criterion has no noise class
        codes, out, nxt = {}, [], 0
        for r in rows:
            cid = by_key[r["key"]]
            if cid == "-1":
                out.append(nxt); nxt += 1
                continue
            if cid not in codes:
                codes[cid] = nxt; nxt += 1
            out.append(codes[cid])
        return np.asarray(out)

    from sklearn.cluster import kmeans_plusplus
    k = min(args.init_k, len(V))
    centres, _ = kmeans_plusplus(V, n_clusters=k, random_state=args.seed)
    lab = np.empty(len(V), dtype=np.int64)
    for s in range(0, len(V), 4096):
        Vc = V[s:s + 4096]
        lab[s:s + 4096] = (((Vc ** 2).sum(1)[:, None] + (centres ** 2).sum(1)[None, :]
                            - 2 * Vc @ centres.T)).argmin(1)
    _, lab = np.unique(lab, return_inverse=True)      # drop unused seeds
    return lab


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--embeddings", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=Path("./output/descend"))
    ap.add_argument("--f", default="unit",
                    help="inter-term weight: unit, sqrt, kappa=<k>, pow=<a> "
                         "(default unit, which is kappa=1)")
    ap.add_argument("--init", choices=("kmeans++", "from"), default="kmeans++")
    ap.add_argument("--init-k", type=int, default=2000,
                    help="starting cluster count for kmeans++ (default 2000). "
                         "It bounds the final k from above -- sweep it")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="an assignments.csv to start from; its noise enters as "
                         "singletons, since the criterion has no noise class")
    ap.add_argument("--max-sweeps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="first N vectors only, for a quick trial")
    ap.add_argument("--keep-duplicate-captures", action="store_true",
                    help="fit on every row, including alternate edits of one "
                         "capture. They distort the counts the criterion reads")
    ap.add_argument("--no-expand-duplicates", action="store_true",
                    help="leave the dropped alternate edits out of "
                         "assignments.csv. By default they are put back, in "
                         "their capture's cluster, which is exact")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.init == "from" and not args.init_from:
        raise SystemExit("error: --init from needs --init-from PATH")
    src = args.embeddings or embeddings_for(args.output_dir, None)
    X, rows = load(src)
    all_rows = rows
    dropped = 0
    if not args.keep_duplicate_captures:
        X, rows, dropped = drop_duplicate_captures(X, rows)
        logger.info("%d duplicate captures dropped for fitting", dropped)
    if args.limit:
        X, rows = X[:args.limit], rows[:args.limit]
    logger.info("%d vectors, %d dims, from %s", len(X), X.shape[1], src)

    V = X.astype(np.float64)
    V -= V.mean(0)                          # displacement coordinates; mu is the origin
    labels = initial_labels(V, args, rows)
    t0 = time.time()
    P = descend(V, labels, args.f, args.max_sweeps)
    took = time.time() - t0

    out = args.output_dir / args.f.replace("=", "")
    out.mkdir(parents=True, exist_ok=True)
    probs = np.ones(len(rows))              # hard assignment; no probability model
    # Centres come from the fitted set: the medoid must be a member of what was
    # actually optimised, and duplicate captures would re-weight the centroid.
    centres = cluster_centres(X, P.lab, probs, rows)
    names = {c["cluster_id"]: c["name"] for c in centres}
    written_rows, written_lab = rows, P.lab
    if dropped and not args.no_expand_duplicates and not args.limit:
        written_lab = expand_duplicates(P.lab, rows, all_rows)
        written_rows = all_rows
        logger.info("%d dropped edits put back in their capture's cluster", dropped)
    write_assignments(out / "assignments.csv", written_lab,
                      np.ones(len(written_rows)), written_rows, names)
    with open(out / "centers.jsonl", "w", encoding="utf-8") as fh:
        for c in centres:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    sizes = np.bincount(P.lab)
    summary = {
        "method": "descend", "f": args.f, "init": args.init,
        "init_k": args.init_k if args.init == "kmeans++" else None,
        "init_from": str(args.init_from) if args.init_from else None,
        "vectors": len(X), "dims": int(X.shape[1]),
        "rows_written": len(written_rows), "duplicate_captures": dropped,
        "clusters": int(P.k), "noise": 0, "noise_fraction": 0.0,
        "reward_over_T": round(P.reward() / P.T, 6),
        "J_over_T": round(1 - P.reward() / P.T, 6),
        "largest_clusters": sorted(sizes.tolist(), reverse=True)[:10],
        "median_cluster_size": int(np.median(sizes)),
        "seconds": round(took, 1),
        "label_agreement": label_agreement(P.lab, rows),
        "source": str(src),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True).stdout.strip(),
    }
    (out / "run.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("%d clusters, J/T=%.5f, %.0fs -> %s",
                P.k, summary["J_over_T"], took, out)


if __name__ == "__main__":
    main()
