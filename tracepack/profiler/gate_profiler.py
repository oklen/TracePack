"""tracepack.profiler.gate_profiler -- the pre-registered Profiler gate.

DESIGN_FROZEN §5: "人工注入已知故障样本上报 macro-F1 与混淆矩阵；允许 ambiguous".

Running `attribute()` on the ten canonical synthetic cases and reporting F1=1.0 would prove
nothing -- those cases are the definitions.  So the injected set is built in three families:

  KEEP    the canonical case plus a perturbation that provably CANNOT change the diagnosis:
          an extra non-diagnostic arm, a token count jittered strictly under budget, an unrelated
          stats key.  Ground truth = the canonical label.  A miss here is brittleness.
  FLIP    a perturbation that SHOULD change the diagnosis, with the new label stated up front
          (e.g. in the edge_coverage_miss pattern, make A3 correct -> that is routing_miss).
          A miss here means the taxonomy does not actually separate two failures it claims to.
  NOISE   a reader flake on an arm the label does not rest on.  Ground truth unchanged.

Reports the confusion matrix and macro-F1 over the labels that actually occur.

    python3 tracepack/profiler/gate_profiler.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tracepack.profiler.attribution import (ARM_A0_NULL, ARM_A0_SHUFFLED, ARM_A1, ARM_A2, ARM_A3,
                                            ARM_A4, ARM_B1, ARM_B2, ARM_B3, ARM_B4, LABELS,
                                            ReadResult, _case, attribute, confusion)

R_OK, R_NO = ReadResult(correct=True, raw=""), ReadResult(correct=False, raw="")


def _clone(case):
    per_arm, stats = case
    return dict(per_arm), {k: dict(v) for k, v in stats.items()}


def build_samples():
    """[(true_label, per_arm, packet_stats, family, note)]"""
    out = []
    for label in LABELS:
        try:
            base = _case(label)
        except AssertionError:
            continue
        out.append((label, base[0], base[1], "CANON", "as defined"))

        # ---- KEEP: perturbations that cannot move the diagnosis -----------------------------
        pa, st = _clone(base)
        if ARM_A0_SHUFFLED not in pa:
            pa[ARM_A0_SHUFFLED] = R_NO           # a shuffled-context arm is never diagnostic here
            out.append((label, pa, st, "KEEP", "+A0_shuffled=wrong"))

        pa, st = _clone(base)
        for k in st:
            st[k].setdefault("total_tokens", 1500)
            st[k]["total_tokens"] = min(2000, st[k].get("total_tokens", 1500))
            st[k].setdefault("budget", 2048)
        out.append((label, pa, st, "KEEP", "tokens jittered under budget"))

        pa, st = _clone(base)
        for k in st:
            st[k]["irrelevant_key"] = 42
        out.append((label, pa, st, "KEEP", "+unrelated stats key"))

    # ---- FLIP: the taxonomy must actually separate neighbouring failures --------------------
    pa, st = _clone(_case("edge_coverage_miss"))     # A1,A2,A3 wrong / A4 right
    pa[ARM_A3] = R_OK                                 # a perfect seed now suffices
    out.append(("routing_miss", pa, st, "FLIP", "edge_coverage_miss -> A3 correct"))

    pa, st = _clone(_case("routing_miss"))            # A1,A2 wrong / A3,A4 right
    pa[ARM_A2] = R_OK                                 # closure over the retrieved seed suffices
    out.append(("closure_miss", pa, st, "FLIP", "routing_miss -> A2 correct"))

    pa, st = _clone(_case("closure_miss"))
    pa[ARM_A1] = R_OK                                 # nothing was missing after all
    out.append(("ok", pa, st, "FLIP", "closure_miss -> A1 correct"))

    pa, st = _clone(_case("carrier_loss"))            # B2 wrong, probe negative
    st[ARM_B2]["probe_positive"] = True               # the value IS in the carrier -> reader's fault
    out.append(("reader_miss", pa, st, "FLIP", "carrier_loss -> probe positive"))

    pa, st = _clone(_case("reader_miss"))
    st[ARM_B2]["probe_positive"] = False              # and back again
    out.append(("carrier_loss", pa, st, "FLIP", "reader_miss -> probe negative"))

    pa, st = _clone(_case("ok"))
    pa[ARM_A0_NULL] = R_OK                            # answerable with NO context at all
    out.append(("ok", pa, st, "KEEP", "prior-answerable item is still ok"))

    pa, st = _clone(_case("stale_conflict"))
    st[ARM_B4]["stale_hit"] = False                   # the answer does not carry the old value
    out.append(("ok", pa, st, "FLIP", "stale_conflict -> no stale hit"))

    # ---- NOISE: a flake on an arm the label does not rest on --------------------------------
    pa, st = _clone(_case("budget_overflow"))
    pa[ARM_A0_NULL] = R_OK                            # the null arm guessing right changes nothing
    out.append(("budget_overflow", pa, st, "NOISE", "A0_null flaked correct"))

    pa, st = _clone(_case("routing_miss"))
    pa[ARM_B4] = R_NO                                 # B4 is not part of this diagnosis
    out.append(("routing_miss", pa, st, "NOISE", "+B4 wrong"))

    return out


def main():
    samples = build_samples()
    true, pred, misses = [], [], []
    for label, per_arm, stats, family, note in samples:
        got = attribute(per_arm, stats).label
        true.append(label)
        pred.append(got)
        if got != label:
            misses.append((family, label, got, note))

    c = confusion(true, pred)
    print("=" * 78)
    print("[Profiler gate] injected faults: %d samples over %d labels"
          % (len(samples), len(set(true))))
    print("  families: %s" % {f: sum(1 for s in samples if s[3] == f)
                              for f in ("CANON", "KEEP", "FLIP", "NOISE")})
    print("\n  %-24s %6s %6s %6s %5s" % ("label", "prec", "rec", "F1", "n"))
    for lab in sorted(c["labels"]):
        m = c["labels"][lab]
        print("  %-24s %6.3f %6.3f %6.3f %5d"
              % (lab, m["precision"], m["recall"], m["f1"], m["support"]))
    print("\n  macro-F1 = %.4f   over %d labels   accuracy = %.4f"
          % (c["macro_f1"], len(c["label_set"]), c["accuracy"]))
    if misses:
        print("\n  MISSES:")
        for f, want, got, note in misses:
            print("    [%s] %-22s -> %-22s (%s)" % (f, want, got, note))
    ok = c["macro_f1"] >= 0.90 and not any(m[0] == "FLIP" for m in misses)
    print("\n  gate: macro-F1 >= 0.90 and every FLIP separated -> %s" % ("PASS" if ok else "FAIL"))
    print("PROFILER_GATE_DONE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
