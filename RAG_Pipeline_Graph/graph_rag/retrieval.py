"""
Query-time retrieval across the three tiers, followed by priority fusion.

    query --> entity linking (graph node names + known_entities + optional spaCy)
          --> predicate intent ("on call" -> on_call_engineer, "owns" -> owner_team, ...)
          --> [live DB refresh of Priority-1 facts for the linked entities]
          --> Priority 1 / 2 / 3 graph facts touching the entities
          --> multi-hop expansion (e.g. service -> owner team -> manager)
          --> semantic search over wiki passages + PDF chunks
          --> fusion.resolve_facts (priority rules, conflict detection)
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .fusion import DEFAULT_SINGLE_VALUED, resolve_facts
from .graph_store import QuadStore
from .models import CTX_CMDB, CTX_PDF, CTX_WIKI, Quad, ResolvedFact, Retrieval
from .sources.database import DatabaseSource, facts_for
from .text import content_tokens, find_ids, flat

logger = logging.getLogger("graph_rag.retrieval")

#: Question keywords -> predicate the question is asking about.
PREDICATE_INTENTS: Sequence[Tuple[str, re.Pattern]] = [
    ("on_call_engineer", re.compile(r"\bon[\s-]?call\b|\bpager\b|\bpaged\b", re.I)),
    ("escalation_contact", re.compile(r"\bescalat", re.I)),
    ("owner_team", re.compile(r"\bown(?:s|ed|er|ers|ership|ing)?\b|\bwhich team\b|\bresponsible for\b", re.I)),
    ("status", re.compile(r"\bstatus\b|\bhealth(?:y)?\b|\bdegraded\b", re.I)),
    ("version", re.compile(r"\bversion\b|\brelease\b", re.I)),
    ("environment", re.compile(r"\benvironment\b|\benv\b|\bstaging\b|\bproduction\b", re.I)),
    ("sla_tier", re.compile(r"\bsla\b", re.I)),
    ("last_deployed_at", re.compile(r"\blast(?:ly)? deploy|\bwhen was .* deployed\b|\bdeployed (?:at|on)\b", re.I)),
    ("depends_on", re.compile(r"\bdepend|\bdependenc|\bupstream\b", re.I)),
    ("manager", re.compile(r"\bmanag(?:e|es|er|ed|ing)\b|\bheads?\b|\bleads?\b", re.I)),
    ("slack_channel", re.compile(r"\bslack\b|\bchannel\b", re.I)),
    ("member", re.compile(r"\bmembers?\b|\bwho (?:is|are|works?) (?:in|on) (?:the )?\w+ team\b", re.I)),
]

#: Procedural / explanatory questions are answered from passages first, facts second.
NARRATIVE_RE = re.compile(r"^\s*(?:how\s+(?:do|does|can|should|to|did)|why|explain|describe|what\s+happened|"
                          r"what\s+(?:are|were)\s+the\s+steps|steps|procedure|summari[sz]e)\b", re.I)

_NLP = None


def load_spacy(model: str = "en_core_web_sm"):
    """spaCy pipeline or None (cached; a missing package/model only logs once)."""
    global _NLP
    if _NLP is None:
        import importlib.util
        if importlib.util.find_spec(model) is None:     # avoid the slow spaCy import when the model is absent
            logger.info("spaCy model %s not installed; using graph-node matching only", model)
            _NLP = False
            return None
        try:
            import spacy
            _NLP = spacy.load(model)
        except Exception as e:
            logger.info("spaCy NER unavailable (%s); using graph-node matching only", type(e).__name__)
            _NLP = False
    return _NLP or None


def _entity_regex(name: str) -> re.Pattern:
    parts = [re.escape(p) for p in re.split(r"[^A-Za-z0-9]+", name) if p]
    return re.compile(r"(?<![A-Za-z0-9])" + r"[^A-Za-z0-9]+".join(parts) + r"(?![A-Za-z0-9])", re.I)


def extract_query_entities(query: str, known_entities: Iterable[str] = (), nlp=None) -> List[str]:
    """
    Link the query to graph entities: case/hyphen-insensitive matches of known names (longest
    match wins, so `ledger-service` beats `ledger`), plus spaCy entities if a model is given.
    Returned in order of appearance.
    """
    spans: List[Tuple[int, int, str]] = []
    for ent in sorted({e for e in known_entities if len(flat(e)) >= 3}, key=lambda e: -len(e)):
        for m in _entity_regex(ent).finditer(query):
            if not any(s <= m.start() < e or s < m.end() <= e for s, e, _ in spans):
                spans.append((m.start(), m.end(), ent))
    if nlp is not None:
        for ent in nlp(query).ents:
            if not any(s <= ent.start_char < e for s, e, _ in spans):
                spans.append((ent.start_char, ent.end_char, ent.text))
    seen, out = set(), []
    for _, _, ent in sorted(spans):
        if flat(ent) not in seen:
            seen.add(flat(ent))
            out.append(ent)
    return out


def detect_predicates(query: str, available: Iterable[str] = ()) -> List[str]:
    """
    Predicates the question asks about, via keyword intents plus a generic rule: a predicate
    whose words all appear in the question (e.g. `escalation_contact`, `sla_tier`).
    """
    found = [p for p, rx in PREDICATE_INTENTS if rx.search(query)]
    q_tokens = set(content_tokens(query))
    for p in sorted(set(available)):
        toks = content_tokens(p.replace("_", " "))
        if toks and set(toks) <= q_tokens and p not in found and p not in ("type", "mentioned_in"):
            found.append(p)
    return found


def is_narrative_question(query: str) -> bool:
    return bool(NARRATIVE_RE.search(query))


def _rank_facts(facts: List[Quad], predicates: Sequence[str], entities: Set[str]) -> List[Quad]:
    pred_rank = {p: i for i, p in enumerate(predicates)}
    return sorted(facts, key=lambda q: (q.predicate not in pred_rank, pred_rank.get(q.predicate, 0),
                                        q.predicate in ("type", "mentioned_in"),
                                        flat(q.subject) not in entities))


def relevant_facts(r: Retrieval) -> List[ResolvedFact]:
    """Resolved facts touching the query entities; restricted to the asked predicates when any match."""
    ents = {flat(e) for e in r.entities}
    rel = [rf for rf in r.resolved if (flat(rf.subject) in ents or flat(rf.value) in ents)
           and rf.predicate not in ("type", "mentioned_in")]
    if r.predicates:
        asked = [rf for rf in rel if rf.predicate in r.predicates]
        # multi-hop: facts about the values of asked facts (team -> manager)
        values = {flat(rf.value) for rf in asked}
        asked += [rf for rf in r.resolved if flat(rf.subject) in values and rf.predicate in r.predicates
                  and rf not in asked]
        rel = asked or rel
    return rel


def refresh_live_facts(store: QuadStore, db: DatabaseSource, specs: List[Dict[str, Any]],
                       entities: Iterable[str]) -> Dict[str, int]:
    """Replace cached Priority-1 facts for `entities` with the database's current rows."""
    entities = list(entities)
    live = {flat(k): v for k, v in facts_for(db, specs, entities).items()}
    refreshed = {}
    for e in entities:
        fresh = live.get(flat(e), [])
        if fresh or store.about(e, context=CTX_CMDB):
            store.replace_subject_facts(e, CTX_CMDB, fresh)
            refreshed[e] = len(fresh)
    return refreshed


def retrieve(query: str, store: QuadStore, vstore, config: Dict[str, Any], nlp=None,
             db: Optional[DatabaseSource] = None, db_specs: Optional[List[Dict[str, Any]]] = None) -> Retrieval:
    t0 = time.perf_counter()
    top_k = int(config["graph_top_k"])
    known = set(config.get("known_entities") or []) | store.node_names()
    entities = extract_query_entities(query, known, nlp)
    ent_flat = {flat(e) for e in entities}
    predicates = detect_predicates(query, {q.predicate for q in store})
    unknown = [i for i in find_ids(query) if flat(i) not in ent_flat]
    r = Retrieval(entities=entities, predicates=predicates, unknown_entities=unknown)
    trace: Dict[str, Any] = {"entities": entities, "predicates": predicates, "unknown_entities": unknown,
                             "narrative_question": is_narrative_question(query)}

    if entities and db is not None and config.get("live_db_refresh", True):
        try:
            trace["live_refresh"] = refresh_live_facts(store, db, db_specs or [], entities)
        except Exception as e:                           # stale cache is better than no answer
            logger.warning("live DB refresh failed, using cached Priority-1 facts: %s", e)
            trace["live_refresh_error"] = str(e)

    if entities:
        r.p1 = _rank_facts(store.query(entities, CTX_CMDB), predicates, ent_flat)[:top_k]
        r.p2 = _rank_facts(store.query(entities, CTX_WIKI), predicates, ent_flat)[:top_k]
        pdf_all = store.query(entities, CTX_PDF)
        r.pdf_facts = _rank_facts([q for q in pdf_all if q.predicate != "mentioned_in"],
                                  predicates, ent_flat)[:top_k]
        trace["pdf_mentions"] = sorted({q.object for q in pdf_all if q.predicate == "mentioned_in"})
        hops = int(config.get("graph_hops", 0))
        if hops > 0:
            direct = set(r.p1) | set(r.p2) | set(pdf_all)
            hop_facts = [(h, q) for h, q in store.expand_with_hops(entities, hops, config["traversal_predicates"])
                         if q not in direct and q.predicate != "mentioned_in"]
            pred_set = set(predicates)
            hop_facts.sort(key=lambda hq: (hq[1].predicate not in pred_set, hq[0], hq[1].priority,
                                           hq[1].predicate == "type"))
            r.related = [q for _, q in hop_facts[:top_k]]
            trace["hops"] = [{"hop": h, "fact": q.as_text()} for h, q in hop_facts[:top_k]]

    if vstore is not None:
        hits = vstore.search(query, int(config["vector_fallback_top_k"]), config.get("vector_max_distance"),
                             min_overlap=int(config.get("min_passage_overlap", 0)),
                             dedupe_jaccard=float(config.get("dedupe_jaccard", 1.01)))
        if unknown and not entities:
            # The question names something the graph does not know: only passages that
            # literally mention it can be relevant.
            hits = [h for h in hits if any(u in h["text"].lower() for u in unknown)]
        r.wiki_passages = [h for h in hits if h.get("tier") == CTX_WIKI]
        r.p3 = [h for h in hits if h.get("tier") != CTX_WIKI]

    r.resolved, r.conflicts = resolve_facts(
        r.p1 + r.p2 + r.pdf_facts + r.related,
        config.get("single_valued_predicates") or DEFAULT_SINGLE_VALUED)
    trace["counts"] = {"p1_facts": len(r.p1), "p2_facts": len(r.p2), "p3_facts": len(r.pdf_facts),
                       "related_facts": len(r.related), "wiki_passages": len(r.wiki_passages),
                       "pdf_passages": len(r.p3), "conflicts": len(r.conflicts)}
    trace["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    r.trace = trace
    return r
