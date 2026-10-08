"""tracepack.eval.datasets -- build the dependency slice from REAL Claude Code traces.

Design (pre-registered in tracepack/DESIGN_FROZEN.md §3):

Every item is a two-choice question about a concrete value that provably lives in the trace, so
scoring needs no judge: the reader picks gold vs a same-type distractor and we read forced-choice
log-odds (the register validated across EC0-EC4/EC1b in kvmemory/RESULTS_ca_*.md).

Five slices, each defined by the DEPENDENCY SHAPE around the answer, not by wording:

  direct_fact      value V occurs in exactly one tool_result E.  gold = {E} (+ its call, atomic).
                   Closure must NOT expand beyond that -- this slice measures over-expansion.
  explicit_ref     same shape, but the question names a resolvable reference (step id / tool name /
                   file path), so a pinning router should win.  Tests §4.4 pinning.
  decision_source  V occurs in tool_result E; a later assistant event A has a DEPENDS_ON edge to E
                   but does NOT contain V.  The retriever tends to find A (the conclusion);
                   answering needs E.  gold = {E}; seed-only usually fails.  THE core slice.
  tool_chain       V occurs in tool_result E2 whose call C2 quotes content from an earlier
                   tool_result E1 (>=40 char literal, the frozen DEPENDS_ON threshold).
                   gold = {E1, C2, E2}: two hops plus an atomic group.
  correction_stale two events state different values for the same target, the later one
                   SUPERSEDES the earlier.  Question asks for the CURRENT value; the stale value
                   is the distractor -- so answering with the old value is a measurable
                   `stale_conflict`, not just a wrong guess.

Constraints enforced at build time (each drop is counted, never silent):
  * the gold minimal packet must FIT the primary budget (2048) -- otherwise we would be measuring
    budget overflow instead of closure quality (see the end-to-end note in DESIGN_FROZEN);
  * the distractor must be absent from the whole gold packet AND from the served text of the
    A4 arm (else the question is answerable by elimination);
  * gold must NOT appear in the carrier/summary text for carrier-testable items unless the item is
    explicitly labelled carrier_positive (surface leak, §5.3 rule 5);
  * values that occur in >=3 sessions are dropped as "generic" (the EC1 df filter);
  * per-session caps so one big session cannot dominate a slice.

    python3 tracepack/eval/datasets.py --out data/tracepack_items.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tracepack.adapters.claude_code import ClaudeCodeAdapter
from tracepack.adapters.registry import adapter_for
from tracepack.core.schema import SchemaError

PRIMARY_BUDGET = 2048
DF_GENERIC = 3
PER_SESSION_CAP = 26
SLICE_TARGET = {"direct_fact": 40, "explicit_ref": 40, "decision_source": 40,
                "tool_chain": 40, "correction_stale": 30}

# The value extractor (PATTERNS / MIN_V / MAX_V / plausible_path / extract_values) lives in
# core/values.py now (phase 2, WP3a): the adapter's value-source edges use the SAME definition,
# and that shared origin must stay visible (PLAN_phase2 §3.5, outcome 3d).  Moved verbatim.
from tracepack.core.values import MIN_V, MAX_V, PATTERNS, plausible_path, extract_values  # noqa: E402,F401

# The boundary rule lives in ONE place now (red-team finding #3): the profiler used to carry its
# own copy with "/" in the boundary class and was blind to 68% of the path golds.
from tracepack.core.textmatch import BOUNDARY  # noqa: E402,F401
from tracepack.core.textmatch import boundary_clean as _boundary_clean  # noqa: E402,F401
from tracepack.core.textmatch import wb  # noqa: E402


def char_class(c):
    """The class a character may be swapped within: digit / lower / upper / literal."""
    if c.isdigit():
        return "d"
    if "a" <= c <= "z":
        return "l"
    if "A" <= c <= "Z":
        return "u"
    return c


def shape(v):
    """Per-position class profile -- the fingerprint every surface classifier can see."""
    return "".join(char_class(c) for c in v)


_CLASS_POOL = {"d": "0123456789", "l": "abcdefghijklmnopqrstuvwxyz",
               "u": "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}


def perturb_same_shape(v, rng, n_out=6, span=None, max_changes=2, tries=80):
    """Variants of ``v`` with the SAME length and the SAME per-position character class.

    Red-team finding #2 (2026-09-01), part (b): the old rules changed shape.  ``re.sub(r"\D","",v)``
    dropped the decimal point, so a decimal gold got integer distractors; the hex bump turned an
    all-digit hash gold into a mixed-alnum distractor.  Either one is decidable with ZERO evidence
    served -- 16/190 items at precision 1.000.  Swapping a digit only for another digit and a
    letter only for another letter of the same case keeps length, digit/letter counts, punctuation
    positions and the leading character identical, so no surface feature separates the classes.

    ``span`` restricts edits to ``v[span[0]:span[1]]`` (used to keep a file:line's head intact).
    Index 0 of the span is never edited, which preserves a leading zero (or its absence).
    """
    lo, hi = span or (0, len(v))
    idx = [i for i in range(max(lo + 1, 1), hi) if char_class(v[i]) in _CLASS_POOL]
    out = []
    if not idx:
        return out
    seen = {v}
    for _ in range(tries):
        if len(out) >= n_out:
            break
        k = 1 if len(idx) == 1 else rng.randint(1, min(max_changes, len(idx)))
        cand = list(v)
        for i in rng.sample(idx, k):
            pool = _CLASS_POOL[char_class(v[i])]
            cand[i] = rng.choice([c for c in pool if c != v[i]])
        c = "".join(cand)
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def make_distractors(typ, v, rng):
    """Same-type, same-SHAPE surrogates.

    ``path`` returns nothing here on purpose: its distractors are assigned globally after the
    dataset is selected (``assign_path_distractors``), because the property that has to hold --
    every distractor basename used at most once across the whole dataset -- is not decidable one
    session at a time.
    """
    if typ == "path":
        return []
    if typ == "fileline":
        head, ln = v.rsplit(":", 1)
        outs = perturb_same_shape(v, rng, span=(len(head) + 1, len(v)))
    else:  # number / hash / version -- whole value, shape preserved
        outs = perturb_same_shape(v, rng)
    uniq = []
    for o in outs:
        if o != v and o not in uniq:
            uniq.append(o)
    return uniq[:6]


def basename(p):
    return p.rsplit("/", 1)[-1]


def ext_of(b):
    return b.rsplit(".", 1)[-1] if "." in b else ""


def surface_profile(b):
    """The features a no-evidence attacker can read off a basename."""
    return (len(b), b.count("_"), b.count("-"), sum(c.isdigit() for c in b),
            sum(c.isalpha() for c in b))


def profile_distance(a, b):
    return sum(abs(x - y) for x, y in zip(surface_profile(a), surface_profile(b)))


def assign_path_distractors(items, graphs, all_basenames, rng, inv):
    """Give every `path` item distractors drawn ONCE each from a global pool of real basenames.

    Red-team finding #2, part (a): the old rule recycled basenames out of a fixed per-session
    pool, so distractor basenames recurred (mean 2.77 uses; >=2 uses in 80% of items) while gold
    basenames were near-unique (mean 1.05).  "Pick the rarer of the two strings" is then a
    no-evidence classifier, and it alone scored 0.726 with nothing served.  Three rules kill it:

      * a basename is used as a distractor AT MOST ONCE in the whole dataset;
      * no distractor basename is ever some item's gold basename (else it recurs across slots);
      * the extension matches gold, so extension frequency cannot separate the classes either.

    The directory is gold's own, so the pair differs exactly where the answer lives.
    """
    gold_bases = {basename(it["gold"]) for it in items if it["type"] == "path"}
    by_ext = defaultdict(list)
    for sess, b in all_basenames:
        if b not in gold_bases:
            by_ext[ext_of(b)].append((sess, b))
    for v in by_ext.values():
        rng.shuffle(v)
    used = set()
    kept, lost = [], set()
    for it in items:
        if it["type"] != "path":
            kept.append(it)
            continue
        g = graphs[it["transcript"]]
        served = "\n".join(g.event(i).text for i in it["required_sources"])
        head = it["gold"].rsplit("/", 1)[0] if "/" in it["gold"] else ""
        gb = basename(it["gold"])
        pool = by_ext.get(ext_of(gb), [])
        # Order by SURFACE DISTANCE to the gold basename first, same session second.  The
        # extension was already matched; the first attack pass on the rebuilt file showed the
        # residual tell is basename shape -- "fewer underscores is gold" scored .550 on its own
        # and .574 composed.  Matching length, underscores, dashes, digits and letters makes
        # those rules abstain instead of fire.  The names stay REAL names out of real traces,
        # which a shape-preserving nonce could not have been.
        order = [b for _, b in sorted(
            pool, key=lambda sb: (profile_distance(sb[1], gb), sb[0] != it["session"]))]
        cands = []
        for b in order:
            if len(cands) >= 4:
                break
            if b in used or b == gb:
                continue
            c = (head + "/" + b) if head else b
            if c == it["gold"] or wb(c).search(served):
                continue
            used.add(b)
            cands.append(c)
        # a correction_stale item already carries the stale value as its first choice -- that one
        # is deliberately real, and must stay first (audit A5)
        forced = [d for d in it.get("distractors") or [] if d]
        if not forced and not cands:
            inv["no_path_distractor"] += 1
            lost.add(it["item_id"])
            continue
        it["distractors"] = (forced + [c for c in cands if c not in forced])[:4]
        kept.append(it)
    return kept, lost


def enforce_global_uniqueness(items, inv):
    """No string -- and no basename -- may appear twice anywhere in the published file.

    Second attack pass (2026-09-01).  Once distractor basenames are used at most once, ANY value
    that still recurs must be a gold, so "pick the one that occurs more often in the file" becomes
    a valid no-evidence rule again: it was decisive on 38/190 items and pushed the composed arm to
    .5632, above the .55 ceiling.  Making every string globally unique makes both counting rules
    abstain everywhere (every count is exactly 1) instead of merely weakening them.
    """
    seen = set()
    kept, lost = [], set()

    def keys_of(v):
        return {v, basename(v)}

    for it in items:
        fixed = [it["gold"]] + ([it["stale_value"]] if it.get("stale_value") else [])
        if set().union(*[keys_of(v) for v in fixed]) & seen:
            # gold (or the stale value, which must stay a listed choice) is the collision -- the
            # item itself has to go
            inv["duplicate_gold_across_items"] += 1
            lost.add(it["item_id"])
            continue
        taken = set().union(*[keys_of(v) for v in fixed])
        dists = []
        for d in it.get("distractors") or []:
            if d in fixed:
                dists.append(d)
                continue
            k = keys_of(d)
            if k & (seen | taken):
                inv["duplicate_distractor_redrawn"] += 1
                continue
            taken |= k
            dists.append(d)
        if not dists:
            inv["no_distractor_after_dedup"] += 1
            lost.add(it["item_id"])
            continue
        it["distractors"] = dists
        seen |= taken
        kept.append(it)
    return kept, lost


def gold_fits(graph, ids, budget=PRIMARY_BUDGET):
    return sum(graph.event(i).cost_of("raw_text") for i in ids) <= budget


def build_session(path, rng, inv):
    """Return candidate items for one transcript."""
    adapter = adapter_for(path)
    graph = adapter.normalize(path)
    sess = os.path.basename(path)[:8]
    events = list(graph.events)
    by_id = {e.event_id: e for e in events}
    results = [e for e in events if e.kind == "tool_result"]
    asst = [e for e in events if e.kind == "assistant"]
    summaries = [e for e in events if e.kind == "summary"]

    # value -> the tool_result events containing it (whole session)
    where = defaultdict(list)
    typed = {}
    for e in results:
        for typ, v in extract_values(e.text):
            where[v].append(e.event_id)
            typed[v] = typ
    path_pool = [v for v, t in typed.items() if t == "path"]
    session_basenames = sorted({basename(v) for v in path_pool})

    def occurs_in_n_results(v):
        """Uniqueness by OCCURRENCE, not by extraction (audit A3): a value can appear in another
        tool_result without the extractor emitting it there (overlapping pattern, length cap).
        The audit uses this definition, so the builder must too or every rebuild fails A3."""
        pat = wb(v)
        return sum(1 for e in results if pat.search(e.text))

    # text corpus for leak screening
    asst_text = "\n".join(e.text for e in asst)
    summ_text = "\n".join(e.text for e in summaries)

    items = []
    per_slice = Counter()

    def emit(slice_name, gold_v, typ, gold_seed, required, extra=None):
        if per_slice[slice_name] >= PER_SESSION_CAP:
            inv["cap_" + slice_name] += 1
            return
        if not gold_fits(graph, required):
            inv["gold_too_big"] += 1
            return
        if slice_name != "correction_stale" and occurs_in_n_results(gold_v) > 1:
            inv["not_unique_by_occurrence"] += 1
            return
        cands = make_distractors(typ, gold_v, rng)
        served = "\n".join(by_id[i].text for i in required)
        cands = [c for c in cands if not wb(c).search(served)]
        forced = (extra or {}).get("force_distractor")
        if forced:
            # the stale value is deliberately IN the trace (that is the point of the slice); it is
            # only screened against the SERVED gold packet, which must not contain it.
            if wb(forced).search(served):
                inv["stale_leaks_into_gold"] += 1
                return
            cands = [forced] + [c for c in cands if c != forced]
        if not cands and typ != "path":
            inv["no_distractor"] += 1
            return
        it = dict(
            item_id="%s#%s#%d" % (sess, slice_name, len(items)),
            session=sess, transcript=path, slice=slice_name,
            query_mode={"direct_fact": "lookup", "explicit_ref": "lookup",
                        "decision_source": "why", "tool_chain": "why",
                        "correction_stale": "state"}[slice_name],
            type=typ, gold=gold_v, distractors=cands[:4],
            gold_seed=gold_seed, required_sources=list(required),
            carrier_ids=[e.event_id for e in summaries][:2],
            gold_in_summary=int(bool(wb(gold_v).search(summ_text))),
            gold_in_assistant=int(bool(wb(gold_v).search(asst_text))),
            n_required=len(required),
            gold_tokens=sum(by_id[i].cost_of("raw_text") for i in required),
        )
        if extra:
            it.update(extra)
        items.append(it)
        per_slice[slice_name] += 1

    # ---------- direct_fact / explicit_ref ----------
    for e in results:
        for typ, v in extract_values(e.text)[:6]:
            if len(where[v]) != 1:
                continue
            call = None
            for ed in graph.parents(e.event_id, "RESULT_OF"):
                call = ed.dst_id
            req = [x for x in (call, e.event_id) if x]
            ev = by_id[e.event_id]
            named = ev.step_id
            if call and by_id[call].text and len(by_id[call].text) < 400:
                emit("explicit_ref", v, typ, e.event_id, req,
                     extra=dict(explicit_ref=dict(step_id=named, tool_call_id=ev.tool_call_id)))
            else:
                emit("direct_fact", v, typ, e.event_id, req)

    # ---------- decision_source ----------
    # The "downstream event that used the source" is whatever carries a DEPENDS_ON edge.  In this
    # adapter that is an Edit/Write tool_call quoting an earlier tool_result (>=40 chars), NOT an
    # assistant text event -- checked against the real graph rather than assumed.
    for a in [e for e in events if graph.parents(e.event_id, "DEPENDS_ON")]:
        deps = [ed.dst_id for ed in graph.parents(a.event_id, "DEPENDS_ON")]
        if not deps:
            continue
        for src in deps[:3]:
            se = by_id.get(src)
            if se is None or se.kind != "tool_result":
                continue
            for typ, v in extract_values(se.text)[:5]:
                if len(where[v]) != 1:
                    continue
                if wb(v).search(a.text):      # the conclusion already states it -> not this slice
                    continue
                call = None
                for ed in graph.parents(src, "RESULT_OF"):
                    call = ed.dst_id
                req = [x for x in (call, src) if x]
                emit("decision_source", v, typ, a.event_id, req,
                     extra=dict(decision_event=a.event_id))

    # ---------- tool_chain ----------
    for c2 in [e for e in events if e.kind == "tool_call"]:
        ups = [ed.dst_id for ed in graph.parents(c2.event_id, "DEPENDS_ON")]
        if not ups:
            continue
        res2 = [ed.src_id for ed in graph.children(c2.event_id, "RESULT_OF")]
        if not res2:
            continue
        e2 = by_id[res2[0]]
        e1 = by_id.get(ups[0])
        if e1 is None or e1.kind != "tool_result":
            continue
        for typ, v in extract_values(e2.text)[:4]:
            if len(where[v]) != 1:
                continue
            req = [e1.event_id, c2.event_id, e2.event_id]
            emit("tool_chain", v, typ, e2.event_id, req,
                 extra=dict(hops=2, upstream=e1.event_id))

    # ---------- correction_stale ----------
    for ed in graph.edges:
        if ed.edge_type != "SUPERSEDES":
            continue
        new_e, old_e = by_id.get(ed.src_id), by_id.get(ed.dst_id)
        if not new_e or not old_e:
            continue
        newv = extract_values(new_e.text)
        old_by_type = defaultdict(list)
        for t_, v_ in extract_values(old_e.text):
            old_by_type[t_].append(v_)
        oldv = {v for vs in old_by_type.values() for v in vs}
        newset = {v for _, v in newv}
        for typ, v in newv[:4]:
            # the answer must be NEW (absent from the superseded event) and locally unique
            if v in oldv or len(where.get(v, [])) > 1:
                continue
            # the distractor is a value that ONLY the stale event states: answering with it is a
            # measurable stale_conflict rather than an ordinary miss
            # same type as gold (audit A9) -- a path distractor for a numeric question is free
            stale = next((o for o in sorted(old_by_type.get(typ, [])) if o not in newset), None)
            if stale is None:
                inv["stale_no_distinct_old_value"] += 1
                continue
            emit("correction_stale", v, typ, new_e.event_id, [new_e.event_id],
                 extra=dict(stale_value=stale, stale_event=old_e.event_id,
                            force_distractor=stale))

    return graph, items, session_basenames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("data/tracepack_items.jsonl"))
    ap.add_argument("--max-sessions", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--transcripts", default=None,
                    help="glob of transcript paths (any format the registry knows); default = the local Claude Code sessions")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    files = sorted(glob.glob(os.path.expanduser(args.transcripts or "~/.claude/projects/*/*.jsonl")),
                   key=os.path.getsize, reverse=True)[:args.max_sessions]
    inv = Counter()
    pool = []
    graphs = {}
    all_basenames = []
    df_sessions = defaultdict(set)
    for p in files:
        try:
            graph, items, bases = build_session(p, rng, inv)
        except (SchemaError, ValueError) as e:
            inv["session_failed"] += 1
            print("  !! %s: %r" % (os.path.basename(p)[:10], e)); continue
        graphs[p] = graph
        sess = os.path.basename(p)[:8]
        all_basenames += [(sess, b) for b in bases]
        for it in items:
            df_sessions[it["gold"]].add(it["session"])
        pool.extend(items)
        print("  %s -> %d candidates %s" % (
            os.path.basename(p)[:10], len(items),
            dict(Counter(i["slice"] for i in items))))

    # cross-session generic filter (the EC1 df rule)
    pool = [it for it in pool if len(df_sessions[it["gold"]]) < DF_GENERIC]

    def downsample(cand_pool, seed):
        """stratified round-robin over sessions so no session dominates a slice"""
        r = random.Random(seed)
        out = []
        for slc, tgt in SLICE_TARGET.items():
            by_s = defaultdict(list)
            for it in cand_pool:
                if it["slice"] == slc:
                    by_s[it["session"]].append(it)
            for v in by_s.values():
                r.shuffle(v)
            order = sorted(by_s)
            take, i = [], 0
            while len(take) < tgt and any(by_s.values()):
                k = order[i % len(order)]
                if by_s[k]:
                    take.append(by_s[k].pop())
                i += 1
            # A7: no session may hold >40% of a slice.  Round-robin gives that automatically only
            # while every session still has candidates; correction_stale runs out on most sessions
            # (a supersede pair is rare), so the share has to be enforced by trimming.
            while len(take) > 2:
                c = Counter(it["session"] for it in take)
                top, ntop = c.most_common(1)[0]
                if ntop <= 0.40 * len(take):
                    break
                for j in range(len(take) - 1, -1, -1):
                    if take[j]["session"] == top:
                        take.pop(j)
                        break
            out.extend(take)
        return out

    # Path distractors are assigned globally (red-team finding #2): a basename may be used at most
    # once in the WHOLE dataset, which is only decidable after selection.  An item that cannot be
    # given one is dropped and its slice refilled, so a drop never silently shrinks a slice.
    banned = set()
    for _round in range(12):
        cand = downsample([it for it in pool if it["item_id"] not in banned], args.seed)
        final, lost = assign_path_distractors(cand, graphs, all_basenames, rng, inv)
        final, lost2 = enforce_global_uniqueness(final, inv)
        lost |= lost2
        if not lost:
            break
        banned |= lost
    for slc in SLICE_TARGET:
        inv["slice_" + slc] = sum(1 for it in final if it["slice"] == slc)
    rng.shuffle(final)

    with open(os.path.expanduser(args.out), "w", encoding="utf-8") as fo:
        for it in final:
            fo.write(json.dumps(it, ensure_ascii=False) + "\n")
    print("\n[datasets] wrote %d items -> %s" % (len(final), args.out))
    print("[datasets] per slice: %s" % {k[6:]: v for k, v in inv.items() if k.startswith("slice_")})
    print("[datasets] drops: %s" % {k: v for k, v in inv.items() if not k.startswith("slice_")})
    print("[datasets] sessions: %s" % dict(Counter(i["session"] for i in final)))
    print("[datasets] gold tokens p50/p90: %s" % (
        sorted(i["gold_tokens"] for i in final)[len(final) // 2] if final else 0,))
    print("CA_DATASETS_DONE")


if __name__ == "__main__":
    main()
