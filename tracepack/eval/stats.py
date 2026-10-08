"""tracepack.eval.stats -- cluster-robust inference for paired contrasts over 10 sessions.

Why not the obvious percentile cluster bootstrap: we already measured it.  On the kvmemory E2 line
(`RESULTS_rA_e2.md` §5), with the SAME cluster structure -- G=10, uneven cluster sizes -- the
pre-registered percentile cluster bootstrap rejected a TRUE NULL **11.0%** of the time at a
nominal 5%, while the wild cluster bootstrap-t came in at **5.2%**.  TracePack has exactly ten
sessions, so importing that ruler unchanged would import a doubled false-positive rate with it.

So the primary interval here is the **wild cluster bootstrap-t** (Rademacher weights, null
imposed, ALL 2^G sign vectors enumerated at G<=16 so there is no Monte-Carlo error), and the
percentile interval is printed beside it for comparison only.  ``selfcheck()`` re-measures both
sizes on this project's own cluster sizes rather than citing the old number.

Ported from kvmemory/kv_e2stats.py; the closed form is the same, the packaging is different.
"""
from __future__ import annotations

import itertools

import numpy as np


def crve(S, ng, dhat):
    """Cluster-robust SE of the pooled mean.  c = G/(G-1); K=1 regressor."""
    G, n = len(S), ng.sum()
    V = (G / (G - 1.0)) * np.sum((S - ng * dhat) ** 2) / n ** 2
    return float(np.sqrt(V))


def sign_matrix(G, rng=None, nboot=9999):
    """All 2^G Rademacher vectors when cheap (exact), else a random sample."""
    if G <= 16:
        return np.array(list(itertools.product([-1.0, 1.0], repeat=G)))
    rng = rng or np.random.default_rng(1234)
    return rng.choice([-1.0, 1.0], size=(nboot, G))


def wcr_p(S, ng, d0, signs, dhat, se):
    """Wild cluster bootstrap-t p-value for H0: delta == d0, null imposed (closed form)."""
    n = ng.sum()
    A = S - ng * d0
    B = signs @ A / n
    resid = signs * A[None, :] - ng[None, :] * B[:, None]
    G = len(S)
    V = (G / (G - 1.0)) * np.sum(resid ** 2, axis=1) / n ** 2
    ok = V > 0
    if not ok.any():
        # every cluster difference is exactly d0 -> 0/0.  Must not return nan: a gate written
        # `p > 0.05` reads nan as False, i.e. as SIGNIFICANT, the exact wrong direction.
        return 1.0 if abs(dhat - d0) < 1e-12 else 0.0
    t_star = np.abs(B[ok]) / np.sqrt(V[ok])
    t_obs = (abs(dhat - d0) / se) if se > 0 else (0.0 if abs(dhat - d0) < 1e-12 else np.inf)
    return float(np.mean(t_star >= t_obs))


def informative(S, d0=0.0, ng=None):
    """How many clusters can actually vote at ``d0``.

    Under the imposed null the restricted residual is ``A_g = S_g - n_g*d0``.  A cluster with
    ``A_g == 0`` contributes nothing whichever sign it draws, so the 2^G sign vectors collapse
    onto 2^k distinct bootstrap samples.  k is that count.
    """
    A = np.asarray(S, dtype=float) - (0.0 if ng is None else np.asarray(ng, dtype=float) * d0)
    return int(np.sum(A != 0))


def p_floor(k):
    """Smallest p-value the WCR can EVER return with k informative clusters.

    The observed statistic is reproduced by the all-+1 sign vector and by its mirror, so at least
    2 of the 2^k distinct draws are >= t_obs.  Hence p >= 2^(1-k).  If this exceeds alpha the
    contrast is UNDECIDABLE by this ruler no matter how large the effect is -- reporting
    "p = 0.125, not significant" would then be a statement about the ruler's granularity, not
    about the data.  Red team round 2, finding (9).
    """
    return 1.0 if k < 1 else float(2.0 ** (1 - k))


def wcr_ci(S, ng, signs, dhat, se, alpha=0.05, half_width=6.0, npts=241):
    """Interval by INVERTING the wild bootstrap test: {d0 : p(d0) > alpha}.

    The grid ALWAYS contains 0.0 and ``dhat``.  Without that, p(d0) can accept a value the grid
    never visits, and the reported interval then excludes a point its own test does not reject --
    which is exactly how "the CI upper bound is pressed right against 0" got written about a
    contrast whose p-value was 0.125.  Red team round 2, finding (10).
    """
    if not (se > 0) or not np.isfinite(se):
        return (float(dhat), float(dhat))
    grid = sorted(set(np.linspace(dhat - half_width * se, dhat + half_width * se, npts).tolist())
                  | {0.0, float(dhat)})
    keep = [d0 for d0 in grid if wcr_p(S, ng, d0, signs, dhat, se) > alpha]
    if not keep:
        return (float(dhat), float(dhat))
    return (float(min(keep)), float(max(keep)))


def percentile_ci(S, ng, rng, B=4000, alpha=0.05):
    """Resample whole clusters with replacement.  Comparison only -- see the module docstring."""
    G = len(S)
    idx = rng.integers(0, G, size=(B, G))
    num, den = S[idx].sum(axis=1), ng[idx].sum(axis=1)
    good = den > 0
    m = np.sort(num[good] / den[good])
    if m.size == 0:
        return (float("nan"), float("nan"))
    return (float(m[int(alpha / 2 * m.size)]), float(m[int((1 - alpha / 2) * m.size) - 1]))


def paired_by_cluster(diffs_by_cluster):
    """(S, ng, dhat, se) from {cluster: [per-item differences]}."""
    keys = sorted(diffs_by_cluster)
    S = np.array([float(sum(diffs_by_cluster[k])) for k in keys])
    ng = np.array([float(len(diffs_by_cluster[k])) for k in keys])
    n = ng.sum()
    if n == 0:
        raise ValueError("no items")
    dhat = float(S.sum() / n)
    return S, ng, dhat, crve(S, ng, dhat)


def contrast(diffs_by_cluster, alpha=0.05, with_percentile=True, seed=1234):
    """Point estimate + WCR p and CI (+ the percentile CI for comparison)."""
    S, ng, dhat, se = paired_by_cluster(diffs_by_cluster)
    signs = sign_matrix(len(S))
    p = wcr_p(S, ng, 0.0, signs, dhat, se)
    lo, hi = wcr_ci(S, ng, signs, dhat, se, alpha=alpha)
    k = informative(S)
    floor = p_floor(k)
    out = {"n": int(ng.sum()), "G": int(len(S)), "delta": dhat, "se": se,
           "wcr_p": p, "wcr_lo": lo, "wcr_hi": hi,
           "k_informative": k, "p_floor": floor, "decidable": bool(floor <= alpha)}
    # The interval and the test must agree about zero.  They can only disagree through a grid
    # artefact, and a silent disagreement is exactly the failure this guard exists to stop.
    if (p > alpha) != (lo <= 0.0 <= hi):
        raise AssertionError(
            "CI/p disagree about zero: p=%.4f alpha=%.3f CI=[%+.5f,%+.5f] -- the interval is not "
            "the acceptance region of its own test" % (p, alpha, lo, hi))
    if with_percentile:
        plo, phi = percentile_ci(S, ng, np.random.default_rng(seed))
        out["pct_lo"], out["pct_hi"] = plo, phi
    return out


def selfcheck(cluster_sizes=(21, 22, 16, 17, 18, 18, 19, 20, 21, 18), reps=400, seed=7,
              discordance=0.20, tau=0.02):
    """Measure BOTH rulers' size on THIS project's cluster structure before trusting either.

    The generator is McNemar-shaped, which is what a paired forced-choice contrast actually looks
    like: on ``1 - discordance`` of the items the two arms agree (difference exactly 0) and only
    the discordant remainder carries signal, split ``(1 +- delta/discordance)/2``.  ``tau`` is a
    per-cluster random effect on delta.  Using a generator whose differences are noisy on EVERY
    item (as a first version here did) makes the size estimate fine but the power estimate
    meaningless, because real per-item differences are mostly zero.

    delta=0 measures SIZE (nominal 0.05); 0.05 and 0.10 measure power.
    """
    rng = np.random.default_rng(seed)
    ng = np.array([float(c) for c in cluster_sizes])
    signs = sign_matrix(len(ng))
    out = {}
    for delta in (0.0, 0.05, 0.10):
        rej_w = rej_p = 0
        for _ in range(reps):
            S = np.empty(len(ng))
            for g, k in enumerate(ng):
                d_g = delta + rng.normal(0, tau)
                p_plus = float(np.clip((1 + d_g / discordance) / 2.0, 0.0, 1.0))
                disc = rng.random(int(k)) < discordance
                sgn = np.where(rng.random(int(k)) < p_plus, 1.0, -1.0)
                S[g] = float((disc * sgn).sum())
            dhat = float(S.sum() / ng.sum())
            se = crve(S, ng, dhat)
            rej_w += int(wcr_p(S, ng, 0.0, signs, dhat, se) <= 0.05)
            lo, hi = percentile_ci(S, ng, rng, B=1500)
            rej_p += int(not (lo <= 0.0 <= hi))
        out[delta] = {"wcr": rej_w / reps, "percentile": rej_p / reps}
    return out


def selfcheck_sparse(cluster_sizes=(21, 22, 16, 17, 18, 18, 19, 20, 21, 18), reps=300, seed=11):
    """The regime the main selfcheck cannot reach: MOST clusters net exactly zero.

    ``selfcheck()`` draws ~20 items per cluster at 20% discordance, so a cluster score of exactly
    0 essentially never happens and the p-floor never binds.  TracePack's real auto-closure
    contrast has 4 of 10 sessions with a non-zero net difference.  This measures what the ruler
    can do there: the answer is that at k=4 the smallest attainable p is 0.125, so alpha=.05 is
    unreachable and the contrast is undecidable however large the effect.
    """
    rng = np.random.default_rng(seed)
    ng = np.array([float(c) for c in cluster_sizes])
    signs = sign_matrix(len(ng))
    out = {}
    for k_target in (2, 4, 6, 10):
        rej, floors = 0, []
        for _ in range(reps):
            S = np.zeros(len(ng))
            live = rng.choice(len(ng), size=k_target, replace=False)
            S[live] = rng.choice([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], size=k_target)
            S[live] = np.abs(S[live])          # a HUGE, perfectly consistent one-sided effect
            dhat = float(S.sum() / ng.sum())
            se = crve(S, ng, dhat)
            p = wcr_p(S, ng, 0.0, signs, dhat, se)
            rej += int(p <= 0.05)
            floors.append(p_floor(informative(S)))
        out[k_target] = {"power_at_a_huge_consistent_effect": rej / reps,
                         "attainable_p_floor": float(np.mean(floors))}
    return out


if __name__ == "__main__":
    res = selfcheck()
    print("cluster structure = TracePack's 10 sessions; nominal alpha = 0.05, reps = 400")
    print("  planted delta   WCR      percentile")
    for d, r in res.items():
        tag = "  <- SIZE" if d == 0.0 else "  (power)"
        print("      %.2f        %.3f    %.3f%s" % (d, r["wcr"], r["percentile"], tag))
    size = res[0.0]
    assert size["wcr"] <= 0.09, "WCR size out of tolerance: %.3f" % size["wcr"]

    print("\nsparse regime (most clusters net exactly zero) -- the regime above cannot reach it.")
    print("  effect planted is huge AND one-sided, so any shortfall in power is the RULER:")
    print("  informative clusters   power    attainable p-floor")
    sp = selfcheck_sparse()
    for k, r in sp.items():
        print("        %2d/10             %.3f          %.4f%s"
              % (k, r["power_at_a_huge_consistent_effect"], r["attainable_p_floor"],
                 "   <- floor > alpha: UNDECIDABLE" if r["attainable_p_floor"] > 0.05 else ""))
    assert sp[4]["power_at_a_huge_consistent_effect"] == 0.0, (
        "expected zero power at k=4; the floor argument is wrong")
    assert sp[10]["power_at_a_huge_consistent_effect"] > 0.9, "k=10 should be easy"

    # the guard added in contrast(): the interval must not exclude a value the test accepts
    boundary = {"s%d" % i: ([1] * 0) for i in range(10)}
    boundary = {"s0": [-1] + [0] * 20, "s1": [0] * 21, "s2": [-1, -1] + [0] * 14,
                "s3": [0] * 17, "s4": [-1] + [0] * 17, "s5": [-1] + [0] * 17,
                "s6": [0] * 19, "s7": [0] * 20, "s8": [0] * 21, "s9": [0] * 18}
    st = contrast(boundary)                       # would have raised before the grid fix
    assert st["wcr_lo"] <= 0.0 <= st["wcr_hi"], st
    assert not st["decidable"], st
    print("\n  boundary case (TracePack's own auto-closure shape): delta=%+.4f p=%.4f "
          "CI[%+.4f,%+.4f] k=%d floor=%.4f decidable=%s"
          % (st["delta"], st["wcr_p"], st["wcr_lo"], st["wcr_hi"],
             st["k_informative"], st["p_floor"], st["decidable"]))
    print("STATS_SELFCHECK_OK -- WCR is the primary ruler; percentile printed for comparison")
