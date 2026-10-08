"""tracepack.eval.attack_surface -- the no-evidence attack on the two-choice items.

A forced-choice benchmark is only measuring retrieval if the two options are indistinguishable
WITHOUT the trace.  The red-team pass on the first build (2026-09-01) showed they were not:

    chance (two-choice)                            acc=0.5000
    surface format rules only                      acc=0.5421   (fired on 16/190, precision 1.000)
    distractor-recycling rule only (transductive)  acc=0.7263   (fired on 108/190)
    surface -> recycling                           acc=0.7632   (fired on 122/190)
    path items only (n=128)                        acc=0.8047

Two artefacts produced that: `re.sub(r"\\D","",v)` dropped the decimal point (so a decimal gold got
integer distractors, and the hex bump turned an all-digit hash gold into a mixed-alnum one), and
path distractors recycled basenames out of a fixed per-session pool (distractor basenames recurred
2.77x on average, gold basenames 1.05x -- "pick the rarer string" is then a classifier).

This module is the standing check.  It plays the adversary: an attacker who gets the published
items file and NOTHING ELSE -- no trace, no context -- and may pick, per rule, whichever polarity
works better on this very dataset (that is deliberately generous; it is the transductive attacker
the red team ran).  Every rule is scored on all items, abstentions counting as a coin flip, which
is why a rule that never fires scores exactly 0.500.

Because the attacker may choose among many rules, the max over rules is inflated under the null.
So the gate is read against a label-permutation null: the same rule family is re-scored with the
gold/distractor sides randomly swapped, and the observed max must sit inside that null band as
well as under the absolute ceiling.

    python3 tracepack/eval/attack_surface.py [items.jsonl]
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

CEILING = 0.55          # the red team's own bar
MIN_ITEMS = 20          # below this the estimate is too noisy to gate on


def _base(v):
    return v.rsplit("/", 1)[-1]


def _shape(v):
    out = []
    for c in v:
        out.append("d" if c.isdigit() else "l" if c.islower() else "u" if c.isupper() else c)
    return "".join(out)


def corpus_stats(pairs):
    """Counts an attacker can compute from the published file alone (the transductive part)."""
    slots = Counter()
    bases = Counter()
    shapes = Counter()
    for g, d in pairs:
        for v in (g, d):
            slots[v] += 1
            bases[_base(v)] += 1
            shapes[_shape(v)] += 1
    return dict(slots=slots, bases=bases, shapes=shapes)


# Each feature maps a candidate string to a number; the rule predicts "higher = gold" (and the
# attacker is also allowed the opposite polarity, so both are scored).
FEATURES = {
    "len": lambda v, st: len(v),
    "n_digits": lambda v, st: sum(c.isdigit() for c in v),
    "n_letters": lambda v, st: sum(c.isalpha() for c in v),
    "digit_frac": lambda v, st: sum(c.isdigit() for c in v) / max(1, len(v)),
    "has_dot": lambda v, st: float("." in v),
    "n_dots": lambda v, st: v.count("."),
    "leading_zero": lambda v, st: float(v[:1] == "0"),
    "all_digits": lambda v, st: float(v.replace(".", "").isdigit()),
    "has_letter": lambda v, st: float(any(c.isalpha() for c in v)),
    "n_underscore": lambda v, st: v.count("_"),
    "base_len": lambda v, st: len(_base(v)),
    "shape_freq": lambda v, st: st["shapes"][_shape(v)],
    # --- transductive: counting how often a string / basename is reused across the whole file ---
    "slot_count": lambda v, st: st["slots"][v],
    "base_count": lambda v, st: st["bases"][_base(v)],
}
SURFACE = [k for k in FEATURES if k not in ("slot_count", "base_count", "shape_freq")]
TRANSDUCTIVE = ["slot_count", "base_count", "shape_freq"]


def score_rule(pairs, feat, sign, st):
    """accuracy of 'higher feature value is the gold', abstentions = 0.5; also #decisive"""
    f = FEATURES[feat]
    hit = fired = 0.0
    for g, d in pairs:
        a, b = sign * f(g, st), sign * f(d, st)
        if a == b:
            hit += 0.5
        else:
            fired += 1
            hit += 1.0 if a > b else 0.0
    return hit / max(1, len(pairs)), int(fired)


def best_over(pairs, feats, st):
    out = []
    for feat in feats:
        for sign in (1, -1):
            acc, fired = score_rule(pairs, feat, sign, st)
            out.append((acc, feat, sign, fired))
    out.sort(reverse=True)
    return out


def compose(pairs, first, second, st):
    """use `first` where it is decisive, else `second` -- the red team's 'surface -> recycling'"""
    facc, ffeat, fsign, _ = first
    sacc, sfeat, ssign, _ = second
    f1, f2 = FEATURES[ffeat], FEATURES[sfeat]
    hit = fired = 0.0
    for g, d in pairs:
        a, b = fsign * f1(g, st), fsign * f1(d, st)
        if a == b:
            a, b = ssign * f2(g, st), ssign * f2(d, st)
        if a == b:
            hit += 0.5
        else:
            fired += 1
            hit += 1.0 if a > b else 0.0
    return hit / max(1, len(pairs)), int(fired)


def attack_best(pairs, st):
    """The strongest no-evidence arm available: best single rule, or surface->recycling."""
    surf = best_over(pairs, SURFACE, st)
    tran = best_over(pairs, TRANSDUCTIVE, st)
    allr = best_over(pairs, list(FEATURES), st)
    comp = compose(pairs, surf[0], tran[0], st)
    return max(allr[0][0], comp[0]), surf, tran, allr, comp


def permutation_null(pairs, st, B=200, seed=7):
    """The same attacker, run against randomly swapped sides.

    The attacker picks the best of 14 features x 2 polarities plus a composition, so the observed
    max is inflated even when nothing leaks.  This is that inflation, measured on this very
    dataset -- the observed max must sit inside it, not just under the absolute ceiling.
    """
    rng = random.Random(seed)
    out = []
    for _ in range(B):
        flipped = [(d, g) if rng.random() < 0.5 else (g, d) for g, d in pairs]
        out.append(attack_best(flipped, st)[0])
    out.sort()
    return out[int(0.95 * len(out))], out[-1]


def run(items, verbose=True):
    """Returns (ok, report_lines).  ok=False means a no-evidence arm beats the ceiling."""
    pairs = [(it["gold"], it["distractors"][0]) for it in items if it.get("distractors")]
    st = corpus_stats(pairs)
    lines = []
    all_feats = list(FEATURES)
    worst, surf, tran, allr, comp = attack_best(pairs, st)
    p95, pmax = permutation_null(pairs, st)

    lines.append("NO-EVIDENCE ARMS  (attacker sees the items file only, best polarity per rule)")
    lines.append("  chance (two-choice)                     acc=0.5000")
    lines.append("  best surface rule   %-18s acc=%.4f   (decisive on %d/%d)"
                 % ("[%s%s]" % ("-" if surf[0][2] < 0 else "+", surf[0][1]),
                    surf[0][0], surf[0][3], len(pairs)))
    lines.append("  best recycling rule %-18s acc=%.4f   (decisive on %d/%d)"
                 % ("[%s%s]" % ("-" if tran[0][2] < 0 else "+", tran[0][1]),
                    tran[0][0], tran[0][3], len(pairs)))
    lines.append("  surface -> recycling                    acc=%.4f   (decisive on %d/%d)"
                 % (comp[0], comp[1], len(pairs)))
    lines.append("  MAX over all %d rules x 2 polarities    acc=%.4f  [%s%s]"
                 % (len(all_feats), allr[0][0], "-" if allr[0][2] < 0 else "+", allr[0][1]))
    lines.append("  permutation null for the strongest arm: p95=%.4f  max=%.4f  (B=200)"
                 % (p95, pmax))
    for typ in sorted({it["type"] for it in items}):
        sub = [(it["gold"], it["distractors"][0]) for it in items
               if it["type"] == typ and it.get("distractors")]
        if len(sub) >= MIN_ITEMS:
            sst = corpus_stats(sub)
            lines.append("    %-10s n=%3d  strongest arm acc=%.4f  (null p95=%.4f)"
                         % (typ, len(sub), attack_best(sub, sst)[0],
                            permutation_null(sub, sst, B=100)[0]))
    lines.append("  runners-up: %s" % ", ".join(
        "%s%s=%.3f" % ("-" if s < 0 else "+", f, a) for a, f, s, _ in allr[1:5]))

    ok = worst <= CEILING and worst <= max(p95, CEILING)
    lines.append("  VERDICT: worst no-evidence arm %.4f vs ceiling %.2f -> %s"
                 % (worst, CEILING, "OK" if ok else "LEAKS"))
    if verbose:
        print("\n".join(lines))
    return ok, lines


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "data/tracepack_items.jsonl")
    items = [json.loads(l) for l in open(path, encoding="utf-8")]
    print("[attack] %d items from %s" % (len(items), os.path.basename(path)))
    ok, _ = run(items)
    print("ATTACK_OK" if ok else "ATTACK_LEAKS")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
