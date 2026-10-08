"""tracepack.core.router -- seed routing (proposal §4.4; stage 1 of the §4.3 main flow).

The router turns ``(query, graph)`` into a ranked, *attributed* list of ``Seed`` records.
It never rewrites the query, never calls an LLM, never touches the network, and never
decides what ends up in the packet -- closure (§3.3) and the budget assembler (§3.4) do
that.  The claim of §4.4 is not "a better hybrid retriever" but "the router is a swappable
module, and explicit addressing is measured separately from semantic retrieval".

Design decisions the rest of the package depends on
---------------------------------------------------

1. **Graph is duck-typed.**  ``core/graph.py`` does not exist yet, so every router accepts
   anything that exposes ``.events`` (attribute or zero-arg method), a mapping
   ``{event_id: TraceEvent}``, or a bare sequence of ``TraceEvent``.  The *only* contract
   this module needs from ``TraceGraph`` is "I can enumerate TraceEvents".  Edges are
   deliberately not read here: seeding is a text/address problem, expansion is closure's job.

2. **Total order, always.**  Ranking ties break on ``(-score, event_id)``, so two runs over
   the same graph give byte-identical seeds (test contract #1).  Seed ranks start at 0 and
   are contiguous in the returned order.

3. **Non-finite scores raise instead of ranking.**  NaN silently poisons ``sorted`` (it
   compares False against everything, so the "top-k" becomes insertion-order dependent) --
   exactly the class of bug that fakes results.  ``_check_scores`` runs *before* any sort,
   and raises ``SchemaError``.

4. **Pins are additive and never displaced (test contract #5).**  ``PinnedRefParser`` only
   emits a Seed when a reference *resolves* to an event that actually exists in the graph;
   an unresolvable "step 99" produces nothing rather than a dangling seed.  Pins are kept
   even when they exceed ``k``: ``PinnedHybridRouter.retrieve`` returns
   ``max(k, n_pins)`` seeds.  Dropping an explicitly addressed event to honour a retrieval
   hyper-parameter would be the exact failure #5 forbids; the *token* budget is enforced
   later, by the assembler, which knows the real cost.

5. **Lexical indexing reads only ``event.text``** -- not ``event_id`` / ``step_id`` /
   ``tool_call_id``.  If ids were in the BM25 document, the "explicit address" arm and the
   "semantic retrieval" arm would leak into each other and §4.4's separation would be
   unmeasurable.  Identifiers are reachable *only* through ``PinnedRefParser``.

6. **Offline deterministic dense arm.**  Python's builtin ``hash()`` is salted per process,
   so it can never be used here.  ``hashing_embed`` uses keyed BLAKE2b (signed hashing
   trick, sublinear tf, L2 normalised) and is reproducible across processes and machines.
   A real embedder can be injected via ``embed_fn``; its output is validated (count, width,
   finiteness) before use.

7. **Seed.source is a closed set in the frozen schema** (lexical|dense|rrf|pin|oracle), and
   there is no "recency" member.  ``LastNRouter`` therefore reports
   ``RECENCY_SEED_SOURCE = "lexical"``; the baseline arm is identified by the *router*
   (every router carries ``.name``, e.g. ``"last_n"``), not by ``seed.source``.  Anything
   that labels experiment arms in the manifest should read ``router.name``.

8. **BM25 backend is pinnable.**  ``rank_bm25`` is used when importable, otherwise an
   internal Okapi implementation with a strictly positive idf.  Because the two can rank
   differently, ``LexicalRouter(backend="internal")`` forces the dependency-free path when
   a result has to reproduce across machines that may not have ``rank_bm25``.
   ``PacketManifest.digest`` excludes float scores, so a backend swap only shows up if the
   *order* changes.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence, runtime_checkable

try:  # normal package import
    from .schema import SchemaError, Seed, TraceEvent
except ImportError:  # pragma: no cover - running the file directly (python3 .../router.py)
    import os as _os
    import sys as _sys

    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
        _os.path.abspath(__file__)))))
    from tracepack.core.schema import SchemaError, Seed, TraceEvent  # type: ignore

try:  # optional, never required
    from rank_bm25 import BM25Okapi as _RankBM25  # type: ignore
except Exception:  # pragma: no cover - dependency absent
    _RankBM25 = None

__all__ = [
    "RouterConfig", "SeedRouter", "LastNRouter", "LexicalRouter", "DenseRouter",
    "HybridRouter", "PinnedRefParser", "PinnedHybridRouter", "make_router",
    "hashing_embed", "ROUTER_NAMES", "RECENCY_SEED_SOURCE",
]

#: names accepted by :func:`make_router`
ROUTER_NAMES = ("last_n", "lexical", "dense", "hybrid", "hybrid_pin")

#: see design note 7 -- the frozen schema has no "recency" seed source.
RECENCY_SEED_SOURCE = "lexical"

_BM25_K1 = 1.5
_BM25_B = 0.75


# ---------------------------------------------------------------- config


@dataclass(frozen=True)
class RouterConfig:
    """Retrieval hyper-parameters.  Frozen so a config can be hashed into an eval run id."""

    k: int = 8
    rrf_k: int = 60
    dense_dim: int = 256
    lexical_weight: float = 1.0
    dense_weight: float = 1.0
    pin_first: bool = True

    def __post_init__(self):
        if not isinstance(self.k, int) or isinstance(self.k, bool) or self.k <= 0:
            raise SchemaError("RouterConfig.k must be a positive int, got %r" % (self.k,))
        if not isinstance(self.rrf_k, int) or self.rrf_k <= 0:
            raise SchemaError("RouterConfig.rrf_k must be a positive int, got %r" % (self.rrf_k,))
        if not isinstance(self.dense_dim, int) or self.dense_dim <= 0:
            raise SchemaError("RouterConfig.dense_dim must be a positive int, got %r"
                              % (self.dense_dim,))
        for field_name in ("lexical_weight", "dense_weight"):
            w = getattr(self, field_name)
            if not isinstance(w, (int, float)) or isinstance(w, bool):
                raise SchemaError("RouterConfig.%s must be a number, got %r" % (field_name, w))
            if not math.isfinite(float(w)):
                raise SchemaError("RouterConfig.%s must be finite, got %r" % (field_name, w))
            if float(w) < 0.0:
                raise SchemaError("RouterConfig.%s must be >= 0, got %r" % (field_name, w))
        if float(self.lexical_weight) == 0.0 and float(self.dense_weight) == 0.0:
            raise SchemaError("lexical_weight and dense_weight cannot both be 0 (RRF would be flat)")


@runtime_checkable
class SeedRouter(Protocol):
    """§4.2 interface.  Implementations additionally carry a ``name`` for arm labelling."""

    def retrieve(self, query: str, graph, k: int) -> "list[Seed]":
        ...


# ---------------------------------------------------------------- helpers


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_IDENT_KEEP_RE = re.compile(r"[^a-z0-9]+")


def _tokenize(text: str) -> "list[str]":
    """Lowercase alphanumeric tokens.  ``step_7`` -> ``["step", "7"]`` so that a query
    written "step 7" and a text written "step_7" share tokens."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def _norm_ident(s: str) -> str:
    """Identifier normal form: lowercase, alphanumerics only.  ``tc-42`` == ``TC_42``."""
    return _IDENT_KEEP_RE.sub("", (s or "").lower())


def _check_query(query: str) -> str:
    if query is None or not isinstance(query, str):
        raise SchemaError("query must be a str, got %r" % (type(query).__name__,))
    return query


def _check_k(k: int) -> int:
    if not isinstance(k, int) or isinstance(k, bool):
        raise SchemaError("k must be an int, got %r" % (type(k).__name__,))
    if k <= 0:
        raise SchemaError("k must be positive, got %d" % k)
    return k


def _check_scores(ids: "Sequence[str]", scores: "Sequence[float]", where: str) -> "list[float]":
    """Guard *before* sorting: NaN compares False with everything and would silently make
    top-k depend on insertion order (the MPS-NaN lesson)."""
    out = []
    for eid, s in zip(ids, scores):
        try:
            v = float(s)
        except (TypeError, ValueError):
            raise SchemaError("%s produced a non-numeric score for %s: %r" % (where, eid, s))
        if not math.isfinite(v):
            raise SchemaError("%s produced a non-finite score (%r) for event %s" % (where, v, eid))
        out.append(v)
    return out


def _graph_events(graph) -> "list[TraceEvent]":
    """Accept a TraceGraph-like object, a mapping id->event, or a bare sequence."""
    if graph is None:
        raise SchemaError("graph is None")
    raw = getattr(graph, "events", None)
    if callable(raw):
        raw = raw()
    if raw is None:
        if isinstance(graph, Mapping):
            raw = list(graph.values())
        elif isinstance(graph, (list, tuple)):
            raw = graph
    if raw is None:
        raise SchemaError("graph %r exposes no .events (and is not a sequence/mapping of events)"
                          % (type(graph).__name__,))
    try:
        events = list(raw)
    except TypeError:
        raise SchemaError("graph.events is not iterable (%r)" % (type(raw).__name__,))
    for e in events:
        if not hasattr(e, "event_id") or not hasattr(e, "text") or not hasattr(e, "timestamp"):
            raise SchemaError("graph contains a non-TraceEvent object: %r" % (type(e).__name__,))
        if not e.event_id:
            raise SchemaError("graph contains an event with an empty event_id")
    seen = set()
    for e in events:
        if e.event_id in seen:
            raise SchemaError("duplicate event_id in graph: %s" % e.event_id)
        seen.add(e.event_id)
    return events


def _fingerprint(events: "Sequence[TraceEvent]") -> str:
    h = hashlib.sha1()
    for e in events:
        h.update(e.event_id.encode("utf-8"))
        h.update(b"\x01")
        h.update((e.text or "").encode("utf-8"))
        h.update(b"\x01")
        h.update(str(e.timestamp).encode("ascii"))
        h.update(b"\x00")
    return h.hexdigest()


class _Index:
    """Tokenised view of the graph.  Pure function of event (id, text, timestamp)."""

    __slots__ = ("events", "ids", "texts", "tokens", "docfreq", "avgdl", "n")

    def __init__(self, events: "Sequence[TraceEvent]"):
        self.events = list(events)
        self.ids = [e.event_id for e in self.events]
        self.texts = [(e.text or "") for e in self.events]
        self.tokens = [_tokenize(t) for t in self.texts]
        self.n = len(self.events)
        df: "dict[str, int]" = {}
        total = 0
        for toks in self.tokens:
            total += len(toks)
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self.docfreq = df
        self.avgdl = (total / self.n) if self.n else 0.0


class _IndexCache:
    """One-slot content-keyed memo.  Never changes results, only avoids re-tokenising."""

    def __init__(self):
        self._fp = None
        self._idx = None
        self._ident = None

    def get(self, events: "Sequence[TraceEvent]") -> _Index:
        # Fast path: the very same event objects as last time (the cached index holds them, so their
        # ids cannot be recycled). Skips hashing the whole trace on repeated queries over one graph.
        ident = (len(events), id(events[0]), id(events[len(events) // 2]), id(events[-1])) if events else None
        if ident is not None and ident == self._ident and self._idx is not None:
            return self._idx
        fp = _fingerprint(events)
        if fp != self._fp or self._idx is None:
            self._fp = fp
            self._idx = _Index(events)
        self._ident = ident
        return self._idx


def _seeds_from_scores(ids: "Sequence[str]", scores: "Sequence[float]", source: str,
                       k: int, where: str) -> "list[Seed]":
    vals = _check_scores(ids, scores, where)
    order = sorted(range(len(ids)), key=lambda i: (-vals[i], ids[i]))
    out = []
    for rank, i in enumerate(order[: min(k, len(order))]):
        out.append(Seed(event_id=ids[i], score=vals[i], source=source, rank=rank, pinned=False))
    return out


def _renumber(seeds: "Sequence[Seed]") -> "list[Seed]":
    """Ranks must be contiguous from 0 in the *returned* order."""
    return [Seed(event_id=s.event_id, score=s.score, source=s.source, rank=i, pinned=s.pinned)
            for i, s in enumerate(seeds)]


# ---------------------------------------------------------------- last-n baseline


class LastNRouter:
    """Baseline: the last ``k`` events by ``(timestamp, event_id)``.  No query is read.

    This is the "no retrieval at all" control arm of §6.2; keeping it in the same interface
    means the eval harness can swap it in without a second code path.
    """

    name = "last_n"

    def __init__(self, cfg: RouterConfig = RouterConfig()):
        if not isinstance(cfg, RouterConfig):
            raise SchemaError("cfg must be a RouterConfig")
        self.cfg = cfg

    def retrieve(self, query: str, graph, k: int) -> "list[Seed]":
        _check_query(query)
        _check_k(k)
        events = _graph_events(graph)
        if not events:
            return []
        ordered = sorted(events, key=lambda e: (-int(e.timestamp), e.event_id))
        ids = [e.event_id for e in ordered]
        scores = [1.0 / (1.0 + i) for i in range(len(ordered))]
        return _seeds_from_scores(ids, scores, RECENCY_SEED_SOURCE, k, "LastNRouter")


# ---------------------------------------------------------------- lexical / BM25


def _bm25_internal(index: _Index, q_tokens: "Sequence[str]") -> "list[float]":
    """Okapi BM25 with the ``ln(1 + (N-df+0.5)/(df+0.5))`` idf, which is strictly positive
    (no epsilon floor hack, no negative-weight surprises for very common terms)."""
    n = index.n
    if n == 0 or not q_tokens or index.avgdl <= 0.0:
        return [0.0] * n
    idf = {}
    for t in set(q_tokens):
        df = index.docfreq.get(t, 0)
        idf[t] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
    scores = []
    for toks in index.tokens:
        dl = len(toks)
        if dl == 0:
            scores.append(0.0)
            continue
        tf: "dict[str, int]" = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        denom_norm = _BM25_K1 * (1.0 - _BM25_B + _BM25_B * dl / index.avgdl)
        for t in q_tokens:
            f = tf.get(t)
            if not f:
                continue
            s += idf[t] * (f * (_BM25_K1 + 1.0)) / (f + denom_norm)
        scores.append(s)
    return scores


class LexicalRouter:
    """BM25 over ``event.text``.  ``backend``: ``auto`` (rank_bm25 if importable) |
    ``rank_bm25`` | ``internal``.  Pin it to ``internal`` for cross-machine reproducibility."""

    name = "lexical"

    def __init__(self, cfg: RouterConfig = RouterConfig(), backend: str = "auto"):
        if not isinstance(cfg, RouterConfig):
            raise SchemaError("cfg must be a RouterConfig")
        if backend not in ("auto", "rank_bm25", "internal"):
            raise SchemaError("unknown BM25 backend: %r" % (backend,))
        if backend == "rank_bm25" and _RankBM25 is None:
            raise SchemaError("backend='rank_bm25' requested but rank_bm25 is not importable")
        self.cfg = cfg
        self.backend = backend
        self._cache = _IndexCache()

    @property
    def effective_backend(self) -> str:
        if self.backend == "auto":
            return "rank_bm25" if _RankBM25 is not None else "internal"
        return self.backend

    def scores(self, query: str, graph) -> "tuple[list[str], list[float]]":
        _check_query(query)
        events = _graph_events(graph)
        index = self._cache.get(events)
        q = _tokenize(query)
        if index.n == 0:
            return [], []
        if not q or index.avgdl <= 0.0:
            return list(index.ids), [0.0] * index.n
        if self.effective_backend == "rank_bm25":
            model = _RankBM25(index.tokens)
            raw = model.get_scores(q)
            vals = [float(x) for x in raw]
        else:
            vals = _bm25_internal(index, q)
        return list(index.ids), vals

    def retrieve(self, query: str, graph, k: int) -> "list[Seed]":
        _check_k(k)
        ids, vals = self.scores(query, graph)
        if not ids:
            return []
        return _seeds_from_scores(ids, vals, "lexical", k, "LexicalRouter")


# ---------------------------------------------------------------- dense


def hashing_embed(texts: "Sequence[str]", dim: int = 256, seed: int = 0) -> "list[list[float]]":
    """Offline, deterministic bag-of-words embedding (signed hashing trick).

    Keyed BLAKE2b, *not* ``hash()``: the builtin string hash is salted per process, so a
    router built on it would silently return different seeds on every run.  Sublinear tf,
    then L2 normalisation; an all-zero document stays all-zero (cosine 0, never 0/0 NaN).
    """
    if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
        raise SchemaError("dim must be a positive int, got %r" % (dim,))
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise SchemaError("seed must be an int, got %r" % (seed,))
    key = str(seed).encode("ascii")[:64]
    out = []
    for text in texts:
        vec = [0.0] * dim
        tf: "dict[str, int]" = {}
        for t in _tokenize(text or ""):
            tf[t] = tf.get(t, 0) + 1
        for t, f in tf.items():
            h = int.from_bytes(
                hashlib.blake2b(t.encode("utf-8"), digest_size=8, key=key).digest(), "big")
            idx = (h >> 1) % dim
            sign = 1.0 if (h & 1) else -1.0
            vec[idx] += sign * (1.0 + math.log(f))
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]
        out.append(vec)
    return out


def _l2_normalise(vec: "Sequence[float]", where: str) -> "list[float]":
    vals = []
    for v in vec:
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise SchemaError("%s returned a non-numeric embedding component: %r" % (where, v))
        if not math.isfinite(f):
            raise SchemaError("%s returned a non-finite embedding component: %r" % (where, f))
        vals.append(f)
    norm = math.sqrt(sum(v * v for v in vals))
    if norm > 0.0:
        vals = [v / norm for v in vals]
    return vals


class DenseRouter:
    """Cosine similarity over unit vectors.  Default embedding is the offline hashing trick;
    ``embed_fn(list[str]) -> list[list[float]]`` injects a real encoder (validated, then
    re-normalised, so the caller does not have to promise unit norm)."""

    name = "dense"

    def __init__(self, cfg: RouterConfig = RouterConfig(),
                 embed_fn: "Callable[[list[str]], list[list[float]]] | None" = None,
                 seed: int = 0):
        if not isinstance(cfg, RouterConfig):
            raise SchemaError("cfg must be a RouterConfig")
        if embed_fn is not None and not callable(embed_fn):
            raise SchemaError("embed_fn must be callable or None")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise SchemaError("seed must be an int, got %r" % (seed,))
        self.cfg = cfg
        self.embed_fn = embed_fn
        self.seed = seed
        self._cache = _IndexCache()
        self._emb_fp = None
        self._emb = None

    def _embed(self, texts: "list[str]") -> "list[list[float]]":
        if self.embed_fn is None:
            return hashing_embed(texts, dim=self.cfg.dense_dim, seed=self.seed)
        raw = self.embed_fn(list(texts))
        try:
            rows = list(raw)
        except TypeError:
            raise SchemaError("embed_fn must return a sequence of vectors")
        if len(rows) != len(texts):
            raise SchemaError("embed_fn returned %d vectors for %d texts"
                              % (len(rows), len(texts)))
        out = [_l2_normalise(r, "embed_fn") for r in rows]
        widths = {len(r) for r in out}
        if len(widths) > 1:
            raise SchemaError("embed_fn returned ragged vectors: widths=%s" % sorted(widths))
        return out

    def _doc_vectors(self, index: _Index) -> "list[list[float]]":
        if self._emb is not None and getattr(self, "_emb_index", None) is index:
            return self._emb                     # the same cached index object: same texts
        fp = _fingerprint(index.events)
        if fp != self._emb_fp or self._emb is None:
            self._emb = self._embed(list(index.texts))
            self._emb_fp = fp
        self._emb_index = index
        return self._emb

    def scores(self, query: str, graph) -> "tuple[list[str], list[float]]":
        _check_query(query)
        events = _graph_events(graph)
        index = self._cache.get(events)
        if index.n == 0:
            return [], []
        docs = self._doc_vectors(index)
        qv = self._embed([query])[0]
        if docs and len(qv) != len(docs[0]):
            raise SchemaError("query embedding width %d != document width %d"
                              % (len(qv), len(docs[0])))
        vals = [sum(a * b for a, b in zip(qv, d)) for d in docs]
        return list(index.ids), vals

    def retrieve(self, query: str, graph, k: int) -> "list[Seed]":
        _check_k(k)
        ids, vals = self.scores(query, graph)
        if not ids:
            return []
        return _seeds_from_scores(ids, vals, "dense", k, "DenseRouter")


# ---------------------------------------------------------------- hybrid (RRF)


class HybridRouter:
    """Reciprocal Rank Fusion of the lexical and dense arms (§4.4 step 3).

    ``score(e) = w_lex/(rrf_k + rank_lex(e)+1) + w_dense/(rrf_k + rank_dense(e)+1)``.

    Fusion runs over the *full* ranked list of each arm by default (``depth=None``): with a
    trace-sized graph this removes a cutoff hyper-parameter, and deep ranks contribute
    ~equally negligible mass.  ``depth=d`` truncates each arm to its top ``d`` (events
    outside an arm's list contribute 0 from that arm), which is the textbook RRF.
    """

    name = "hybrid"

    def __init__(self, cfg: RouterConfig = RouterConfig(),
                 lexical: "LexicalRouter | None" = None,
                 dense: "DenseRouter | None" = None,
                 depth: "int | None" = None):
        if not isinstance(cfg, RouterConfig):
            raise SchemaError("cfg must be a RouterConfig")
        if depth is not None and (not isinstance(depth, int) or isinstance(depth, bool)
                                  or depth <= 0):
            raise SchemaError("depth must be a positive int or None, got %r" % (depth,))
        self.cfg = cfg
        self.lexical = lexical if lexical is not None else LexicalRouter(cfg)
        self.dense = dense if dense is not None else DenseRouter(cfg)
        self.depth = depth

    def retrieve(self, query: str, graph, k: int) -> "list[Seed]":
        _check_query(query)
        _check_k(k)
        events = _graph_events(graph)
        if not events:
            return []
        n = len(events)
        limit = n if self.depth is None else min(self.depth, n)
        lex = self.lexical.retrieve(query, graph, limit)
        den = self.dense.retrieve(query, graph, limit)
        rrf_k = float(self.cfg.rrf_k)
        fused: "dict[str, float]" = {e.event_id: 0.0 for e in events}
        for arm, weight in ((lex, float(self.cfg.lexical_weight)),
                            (den, float(self.cfg.dense_weight))):
            if weight == 0.0:
                continue
            for s in arm:
                fused[s.event_id] = fused.get(s.event_id, 0.0) + weight / (rrf_k + s.rank + 1.0)
        ids = [e.event_id for e in events]
        vals = [fused[i] for i in ids]
        return _seeds_from_scores(ids, vals, "rrf", k, "HybridRouter")


# ---------------------------------------------------------------- explicit refs


_REF_PATTERNS = (
    ("step", re.compile(r"\bstep[\s_\-#:]*([A-Za-z0-9][A-Za-z0-9_\-.]*)", re.IGNORECASE)),
    ("tool_call", re.compile(r"\btool[\s_\-]?call[\s_\-#:]*([A-Za-z0-9][A-Za-z0-9_\-.]*)",
                             re.IGNORECASE)),
    ("event", re.compile(r"\bevent[\s_\-#:]*([A-Za-z0-9][A-Za-z0-9_\-.]*)", re.IGNORECASE)),
)
_QUOTED_RE = re.compile(r"[\"'`]([^\"'`\n]{1,300})[\"'`]")
_BARE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-./]*")
_PATH_EXT_RE = re.compile(r"\.[A-Za-z0-9]{1,6}$")


def _is_identifierish(raw: str) -> bool:
    """A bare query token is only treated as an address if it *looks* like one -- otherwise
    an ordinary English word colliding with an id would silently pin an event."""
    if len(raw) < 2:
        return False
    if any(c.isdigit() for c in raw):
        return True
    if any(c in "_-./:" for c in raw):
        return True
    return len(raw) >= 8


def _looks_like_path(raw: str) -> bool:
    return ("/" in raw) or bool(_PATH_EXT_RE.search(raw))


@dataclass(frozen=True)
class _Ref:
    kind: str      # step | tool_call | event | quoted | bare
    raw: str
    pos: int


class _RefIndex:
    __slots__ = ("by_step", "by_tool", "by_event", "events")

    def __init__(self, events: "Sequence[TraceEvent]"):
        self.events = list(events)
        self.by_step: "dict[str, list[TraceEvent]]" = {}
        self.by_tool: "dict[str, list[TraceEvent]]" = {}
        self.by_event: "dict[str, list[TraceEvent]]" = {}
        for e in self.events:
            self.by_event.setdefault(_norm_ident(e.event_id), []).append(e)
            sid = getattr(e, "step_id", None)
            if sid:
                self.by_step.setdefault(_norm_ident(sid), []).append(e)
            tid = getattr(e, "tool_call_id", None)
            if tid:
                self.by_tool.setdefault(_norm_ident(tid), []).append(e)


class PinnedRefParser:
    """Resolve EXPLICIT references in the query text against the graph (§4.4 step 4).

    Handles: ``step 7`` / ``step_7`` / ``step-7``, ``tool_call tc_42``, ``event e12``, bare
    identifier-looking tokens that equal a step/tool/event id, and quoted file paths
    (matched against ``native_ref``, ``meta`` values, or verbatim occurrence in event text).

    Two deliberate boundaries:

    * a reference that does not resolve to an event in the graph produces **no seed** --
      a pin is an address, and an address that points nowhere is a bug, not evidence;
    * quoted *prose* is not searched.  Free-text matching is the lexical arm's job; letting
      the pin path do it would collapse the very separation §4.4 asks us to measure.

    Multiple events can legitimately answer one reference (a step contains several events);
    matches are ordered ``(timestamp, event_id)`` and capped at ``max_per_ref``.
    """

    name = "pin"

    def __init__(self, max_per_ref: int = 8):
        if not isinstance(max_per_ref, int) or isinstance(max_per_ref, bool) or max_per_ref <= 0:
            raise SchemaError("max_per_ref must be a positive int, got %r" % (max_per_ref,))
        self.max_per_ref = max_per_ref

    # -- ref extraction ------------------------------------------------
    def _refs(self, query: str) -> "list[_Ref]":
        refs: "list[_Ref]" = []
        for kind, pat in _REF_PATTERNS:
            for m in pat.finditer(query):
                refs.append(_Ref(kind, m.group(1), m.start(1)))
        for m in _QUOTED_RE.finditer(query):
            refs.append(_Ref("quoted", m.group(1).strip(), m.start(1)))
        for m in _BARE_RE.finditer(query):
            raw = m.group(0)
            if _is_identifierish(raw):
                refs.append(_Ref("bare", raw, m.start(0)))
        refs.sort(key=lambda r: (r.pos, r.kind, r.raw))
        seen = set()
        out = []
        for r in refs:
            key = (r.kind, r.raw.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    # -- resolution ----------------------------------------------------
    def _path_hits(self, raw: str, index: _RefIndex) -> "list[TraceEvent]":
        if not _looks_like_path(raw):
            return []
        needle = raw.strip()
        if needle.startswith("./"):
            needle = needle[2:]
        if not needle:
            return []
        hits = []
        for e in index.events:
            ref = getattr(e, "native_ref", None)
            if ref and (ref == raw or ref == needle or ref.endswith("/" + needle)):
                hits.append(e)
                continue
            meta = getattr(e, "meta", None) or {}
            matched = False
            try:
                values = list(meta.values())
            except AttributeError:
                values = []
            for v in values:
                if isinstance(v, str) and (v == raw or v == needle or v.endswith("/" + needle)):
                    hits.append(e)
                    matched = True
                    break
            if matched:
                continue
            if needle in (e.text or ""):
                hits.append(e)
        return hits

    def _resolve(self, ref: _Ref, index: _RefIndex) -> "list[TraceEvent]":
        key = _norm_ident(ref.raw)
        hits: "list[TraceEvent]" = []
        if not key and ref.kind != "quoted":
            return []
        if ref.kind in ("step", "bare", "quoted"):
            for cand in (key, "step" + key):
                hits.extend(index.by_step.get(cand, []))
        if ref.kind in ("tool_call", "bare", "quoted"):
            for cand in (key, "toolcall" + key, "tc" + key):
                hits.extend(index.by_tool.get(cand, []))
        if ref.kind in ("event", "bare", "quoted"):
            for cand in (key, "event" + key, "ev" + key, "e" + key):
                hits.extend(index.by_event.get(cand, []))
        if ref.kind in ("quoted", "bare"):
            hits.extend(self._path_hits(ref.raw, index))
        uniq: "dict[str, TraceEvent]" = {}
        for e in hits:
            uniq.setdefault(e.event_id, e)
        ordered = sorted(uniq.values(), key=lambda e: (int(e.timestamp), e.event_id))
        return ordered[: self.max_per_ref]

    def parse(self, query: str, graph) -> "list[Seed]":
        _check_query(query)
        events = _graph_events(graph)
        if not events:
            return []
        index = _RefIndex(events)
        picked: "list[str]" = []
        seen = set()
        for ref in self._refs(query):
            for e in self._resolve(ref, index):
                if e.event_id in seen:
                    continue
                seen.add(e.event_id)
                picked.append(e.event_id)
        # score 1.0 for every pin: pins are addresses, not similarities.  Order is carried by
        # `rank` (query order, then chronological within one reference).
        return [Seed(event_id=eid, score=1.0, source="pin", rank=i, pinned=True)
                for i, eid in enumerate(picked)]


class PinnedHybridRouter:
    """Pins first, then RRF fills the rest (§4.4 steps 3+4; test contract #5).

    ``cfg.pin_first=True`` puts the pins at the head in query order.  With ``False`` the
    pins are merged into the ranking by ``(-score, event_id)`` -- they still cannot be
    dropped, only reordered (their score is 1.0, well above any RRF mass, so in practice
    they stay at the top; the flag exists so the eval harness can prove that).

    Pins are never truncated: the returned list can be longer than ``k`` when the query
    addresses more events than ``k``.  The hard budget is the assembler's contract (§3.4),
    not the router's.
    """

    name = "hybrid_pin"

    def __init__(self, cfg: RouterConfig = RouterConfig(),
                 hybrid: "HybridRouter | None" = None,
                 parser: "PinnedRefParser | None" = None):
        if not isinstance(cfg, RouterConfig):
            raise SchemaError("cfg must be a RouterConfig")
        self.cfg = cfg
        self.hybrid = hybrid if hybrid is not None else HybridRouter(cfg)
        self.parser = parser if parser is not None else PinnedRefParser()

    def retrieve(self, query: str, graph, k: int) -> "list[Seed]":
        _check_query(query)
        _check_k(k)
        events = _graph_events(graph)
        if not events:
            return []
        pins = self.parser.parse(query, graph)
        pinned_ids = {s.event_id for s in pins}
        room = max(0, k - len(pins))
        fill: "list[Seed]" = []
        if room:
            # ask for enough candidates that removing the pins still leaves `room` of them
            want = min(len(events), room + len(pins))
            for s in self.hybrid.retrieve(query, graph, want):
                if s.event_id in pinned_ids:
                    continue
                fill.append(s)
                if len(fill) >= room:
                    break
        if self.cfg.pin_first:
            merged = list(pins) + fill
        else:
            merged = sorted(list(pins) + fill, key=lambda s: (-s.score, s.event_id))
        return _renumber(merged)


# ---------------------------------------------------------------- factory


def make_router(name: str, cfg: RouterConfig = RouterConfig(), **kw) -> SeedRouter:
    """Build a router by arm name: ``last_n | lexical | dense | hybrid | hybrid_pin``.

    Extra kwargs go to the concrete class (``backend=`` for lexical, ``embed_fn=``/``seed=``
    for dense, ``depth=`` for hybrid, ``parser=`` for hybrid_pin)."""
    if not isinstance(name, str):
        raise SchemaError("router name must be a str, got %r" % (type(name).__name__,))
    if not isinstance(cfg, RouterConfig):
        raise SchemaError("cfg must be a RouterConfig")
    key = name.strip().lower()
    if key not in ROUTER_NAMES:
        raise SchemaError("unknown router %r (known: %s)" % (name, ", ".join(ROUTER_NAMES)))
    try:
        if key == "last_n":
            return LastNRouter(cfg, **kw)
        if key == "lexical":
            return LexicalRouter(cfg, **kw)
        if key == "dense":
            return DenseRouter(cfg, **kw)
        if key == "hybrid":
            return HybridRouter(cfg, **kw)
        return PinnedHybridRouter(cfg, **kw)
    except TypeError as exc:
        raise SchemaError("bad kwargs for router %r: %s" % (key, exc))


# ---------------------------------------------------------------- selfcheck


class _TinyGraph:
    """Stand-in for core/graph.py: the router only needs `.events`."""

    def __init__(self, events):
        self.events = list(events)
        self.edges = ()


def _selfcheck() -> None:
    def ev(eid, text, ts, **kw):
        return TraceEvent(event_id=eid, kind=kw.pop("kind", "assistant"), text=text,
                          timestamp=ts, **kw)

    # The pinned event (e_step7) is deliberately a POOR lexical match for the query, while
    # e_decoy is stuffed with the query's content words.
    events = [
        ev("e1", "user asks about the release", 1, kind="user"),
        ev("e_step7", "applied the shard planner change and moved on", 2, step_id="step_7"),
        ev("e_tool", "ls -la on the build dir", 3, kind="tool_call", tool_call_id="tc_42"),
        ev("e_path", "wrote patch to src/planner/shard.py after review", 4,
           native_ref="src/planner/shard.py"),
        ev("e_decoy", "rollback rollback budget cap break why rollback budget", 5),
        ev("e6", "unrelated chatter about lunch", 6),
    ]
    graph = _TinyGraph(events)
    query = 'why did the rollback in step 7 break the budget cap?'
    cfg = RouterConfig(k=3)

    # -- 1. the lexical winner is NOT the pinned event ------------------
    lex = LexicalRouter(cfg)
    lex_top = lex.retrieve(query, graph, 3)
    assert lex_top[0].event_id == "e_decoy", "lexical winner should be the stuffed decoy, got %r" \
        % (lex_top[0].event_id,)
    assert lex_top[0].source == "lexical" and lex_top[0].rank == 0
    assert [s.rank for s in lex_top] == [0, 1, 2], "ranks must start at 0 and be contiguous"

    # design note 5: identifiers must NOT be in the lexical document, otherwise the
    # "explicit address" arm and the "semantic retrieval" arm leak into each other and
    # §4.4's separation becomes unmeasurable.
    ids_, vals_ = lex.scores("step 7", graph)
    by_id = dict(zip(ids_, vals_))
    assert by_id["e_step7"] == 0.0, \
        "step_id leaked into the BM25 document (score %r)" % (by_id["e_step7"],)
    ids_, vals_ = lex.scores("tc_42 event e1", graph)
    by_id = dict(zip(ids_, vals_))
    assert by_id["e_tool"] == 0.0 and by_id["e1"] == 0.0, \
        "tool_call_id/event_id leaked into the BM25 document"

    # -- 2. the pin survives top-k (test contract #5) -------------------
    parser = PinnedRefParser()
    pins = parser.parse(query, graph)
    assert [s.event_id for s in pins] == ["e_step7"], [s.event_id for s in pins]
    assert pins[0].pinned is True and pins[0].source == "pin"

    ph = PinnedHybridRouter(RouterConfig(k=1))
    got = ph.retrieve(query, graph, 1)
    assert [s.event_id for s in got] == ["e_step7"], \
        "pin was displaced by top-k: %r" % ([s.event_id for s in got],)
    assert got[0].pinned is True

    wide = ph.retrieve(query, graph, 4)
    assert wide[0].event_id == "e_step7" and wide[0].pinned is True
    assert "e_decoy" in [s.event_id for s in wide[1:]], "hybrid should still fill the rest"
    assert all(not s.pinned for s in wide[1:])
    assert [s.rank for s in wide] == list(range(len(wide)))
    assert len({s.event_id for s in wide}) == len(wide), "no duplicate seeds"

    # pins are never truncated, even below k
    many = ph.retrieve('check step 7 and tool_call tc_42 and "src/planner/shard.py"', graph, 1)
    assert [s.event_id for s in many] == ["e_step7", "e_tool", "e_path"], \
        [s.event_id for s in many]
    assert all(s.pinned for s in many)

    # an unresolvable address yields nothing (no dangling seeds)
    assert parser.parse("what happened in step 99?", graph) == []
    assert parser.parse('and "no/such/file.py"?', graph) == []
    # bare id reference, no keyword
    assert [s.event_id for s in parser.parse("did tc_42 succeed?", graph)] == ["e_tool"]
    # ordinary English words must not be treated as addresses
    assert parser.parse("why did the rollback break?", graph) == []

    # -- 3. determinism across two calls --------------------------------
    for name in ROUTER_NAMES:
        r1 = make_router(name, cfg)
        r2 = make_router(name, cfg)
        a = r1.retrieve(query, graph, 3)
        b = r2.retrieve(query, graph, 3)
        c = r1.retrieve(query, graph, 3)   # same instance, warm cache
        assert a == b == c, "router %s is not deterministic" % name
        assert all(math.isfinite(s.score) for s in a)
        assert [s.rank for s in a] == list(range(len(a)))
        assert isinstance(r1, SeedRouter)

    # -- 4. baselines behave --------------------------------------------
    last = make_router("last_n", cfg).retrieve("anything", graph, 2)
    assert [s.event_id for s in last] == ["e6", "e_decoy"], [s.event_id for s in last]
    hyb = make_router("hybrid", cfg).retrieve(query, graph, 3)
    assert all(s.source == "rrf" for s in hyb)
    dense_seeds = make_router("dense", cfg).retrieve(query, graph, 3)
    assert all(s.source == "dense" for s in dense_seeds)
    # empty query: no crash, still a total order (tie-break on event_id)
    assert [s.event_id for s in lex.retrieve("", graph, 2)] == ["e1", "e6"]
    # k larger than the graph
    assert len(lex.retrieve(query, graph, 999)) == len(events)
    # empty graph
    assert make_router("hybrid_pin", cfg).retrieve(query, _TinyGraph([]), 3) == []
    # graph given as a bare list / mapping
    assert lex.retrieve(query, events, 1)[0].event_id == "e_decoy"
    assert lex.retrieve(query, {e.event_id: e for e in events}, 1)[0].event_id == "e_decoy"

    # -- 5. embedding determinism + NaN guard (fault injection) ---------
    v1 = hashing_embed(["shard planner"], dim=32, seed=0)
    v2 = hashing_embed(["shard planner"], dim=32, seed=0)
    assert v1 == v2 and abs(sum(x * x for x in v1[0]) - 1.0) < 1e-9
    assert hashing_embed([""], dim=8)[0] == [0.0] * 8, "empty text must give a zero vector, not NaN"
    # Pinned constant: in-process equality cannot detect a hash-randomised implementation
    # (builtin hash() is salted PER PROCESS, identical within one run).  Freezing one
    # coordinate makes any such regression fail on essentially every run, not on some.
    assert [(i, x) for i, x in enumerate(hashing_embed(["shard"], dim=32, seed=0)[0]) if x] \
        == [(26, 1.0)], "hashing_embed is no longer reproducible across processes"
    assert hashing_embed(["shard"], dim=32, seed=1) != hashing_embed(["shard"], dim=32, seed=0), \
        "the seed must actually change the projection"

    def nan_embed(texts):
        return [[float("nan")] * 4 for _ in texts]

    try:
        DenseRouter(cfg, embed_fn=nan_embed).retrieve(query, graph, 2)
    except SchemaError:
        pass
    else:
        raise AssertionError("NaN embedding must be rejected, not ranked")

    def inf_embed(texts):
        return [[float("inf")] + [0.0] * 3 for _ in texts]

    try:
        DenseRouter(cfg, embed_fn=inf_embed).retrieve(query, graph, 2)
    except SchemaError:
        pass
    else:
        raise AssertionError("infinite embedding must be rejected")

    def short_embed(texts):
        return [[1.0, 0.0]]  # wrong count

    try:
        DenseRouter(cfg, embed_fn=short_embed).retrieve(query, graph, 2)
    except SchemaError:
        pass
    else:
        raise AssertionError("embed_fn returning the wrong number of vectors must be rejected")

    # a NaN reaching the ranker directly must raise rather than sort arbitrarily
    try:
        _seeds_from_scores(["a", "b"], [1.0, float("nan")], "lexical", 2, "test")
    except SchemaError:
        pass
    else:
        raise AssertionError("NaN score must be rejected before sorting")

    # -- 6. broken inputs are rejected ----------------------------------
    for bad_k in (0, -1):
        try:
            RouterConfig(k=bad_k)
        except SchemaError:
            pass
        else:
            raise AssertionError("RouterConfig(k=%r) must be rejected" % (bad_k,))
    try:
        RouterConfig(lexical_weight=float("nan"))
    except SchemaError:
        pass
    else:
        raise AssertionError("non-finite weight must be rejected")
    try:
        make_router("bm42", cfg)
    except SchemaError:
        pass
    else:
        raise AssertionError("unknown router name must be rejected")
    try:
        lex.retrieve(query, object(), 3)
    except SchemaError:
        pass
    else:
        raise AssertionError("a graph without .events must be rejected")
    try:
        dup = _TinyGraph([ev("dup", "a", 1), ev("dup", "b", 2)])
        lex.retrieve(query, dup, 2)
    except SchemaError:
        pass
    else:
        raise AssertionError("duplicate event_id must be rejected")
    try:
        lex.retrieve(query, graph, 0)
    except SchemaError:
        pass
    else:
        raise AssertionError("k=0 must be rejected")
    try:
        lex.retrieve(None, graph, 2)
    except SchemaError:
        pass
    else:
        raise AssertionError("non-str query must be rejected")
    try:
        LexicalRouter(cfg, backend="lucene")
    except SchemaError:
        pass
    else:
        raise AssertionError("unknown BM25 backend must be rejected")

    # -- 7. both BM25 backends agree on this fixture --------------------
    internal = LexicalRouter(cfg, backend="internal").retrieve(query, graph, 3)
    assert internal[0].event_id == "e_decoy"
    if _RankBM25 is not None:
        rb = LexicalRouter(cfg, backend="rank_bm25").retrieve(query, graph, 3)
        assert rb[0].event_id == "e_decoy"

    print("router selfcheck OK (bm25 backend=%s, rank_bm25=%s)"
          % (lex.effective_backend, _RankBM25 is not None))


if __name__ == "__main__":
    _selfcheck()
