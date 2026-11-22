"""
Query-time orchestration: retrieve -> fuse -> single prompt -> generate -> validate -> provenance.

Python API:

    from graph_rag import GraphRAGPipeline
    pipe = GraphRAGPipeline.from_config()          # sample data, offline defaults
    result = pipe.ask("Who is on-call for payments-service?")
    print(result.answer, result.conflicts)
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, List, Optional

from .config import load_config
from .fusion import conflict_dicts, explain_sources
from .generation import ExtractiveGenerator, GenerationOutput, make_generator
from .graph_store import QuadStore
from .index import GraphIndex, build_graph_index
from .models import CTX_CMDB, CTX_PDF, CTX_WIKI, REFUSAL, TIER_LABEL, TIER_PRIORITY, QueryResult, ResolvedFact, Retrieval
from .prompting import fit_prompt
from .retrieval import load_spacy, relevant_facts, retrieve
from .sources.database import DatabaseSource
from .text import content_tokens, flat
from .validation import validate_graph_answer

logger = logging.getLogger("graph_rag.pipeline")

_REFUSAL_MARKERS = ("don't know", "do not know", "not enough information", "cannot answer")


# ---- result helpers ------------------------------------------------------------------------
def _tier(r: Retrieval) -> str:
    if r.p1:
        return CTX_CMDB
    if r.p2 or r.wiki_passages:
        return CTX_WIKI
    if r.pdf_facts or r.p3:
        return CTX_PDF
    return "none"


def _chunk_dicts(r: Retrieval) -> List[Dict[str, Any]]:
    out = []
    for c in sorted(r.passages, key=lambda c: c["distance"]):
        out.append({"chunk_id": c["chunk_id"], "page": c["page"], "source_type": c.get("tier", CTX_PDF),
                    "tier": c.get("tier", CTX_PDF), "source": c["source"], "title": c.get("title", ""),
                    "section": c.get("section", ""), "distance": c["distance"], "score": c.get("score")})
    return out


def _citation_for_fact(rf: ResolvedFact, n: int) -> Dict[str, Any]:
    q = rf.winner
    return {"id": n, "tier": q.context, "priority": TIER_PRIORITY.get(q.context), "fact": q.as_text(),
            "source": q.source, "timestamp": q.timestamp}


def _citation_for_passage(p: Dict[str, Any], n: int) -> Dict[str, Any]:
    c = {"id": n, "tier": p.get("tier", CTX_PDF), "priority": TIER_PRIORITY.get(p.get("tier", CTX_PDF)),
         "source": p["source"], "title": p.get("title", ""), "chunk_id": p["chunk_id"]}
    if p.get("page"):
        c["page"] = p["page"]
    if p.get("section"):
        c["section"] = p["section"]
    return c


def _evidence(facts: List[ResolvedFact], passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ev = [{"type": "fact", "tier": rf.winner.context, "label": TIER_LABEL.get(rf.winner.context),
           "statement": f"{rf.subject} {rf.predicate} {rf.value}", "source": rf.winner.source,
           "timestamp": rf.winner.timestamp, "resolution": rf.reason} for rf in facts]
    ev += [{"type": "passage", "tier": p.get("tier"), "label": TIER_LABEL.get(p.get("tier", "")),
            "text": p["text"][:500], "source": p["source"], "page": p.get("page") or None,
            "section": p.get("section") or None, "score": p.get("score")} for p in passages]
    return ev


def _attribute(answer: str, r: Retrieval):
    """
    For free-text LLM answers: the resolved facts whose value the answer states, and the passages
    that contain most of the answer's words. Falls back to all relevant facts / passages.
    """
    a = f" {flat(answer)} "
    facts = relevant_facts(r)
    stated = [rf for rf in facts if f" {flat(rf.value)} " in a]
    toks = set(content_tokens(answer))
    passages = [p for p in r.passages if toks and len(toks & set(content_tokens(p["text"]))) >= 0.6 * len(toks)]
    if stated:
        return stated, passages
    return ([], passages) if passages else (facts, list(r.passages))


def _qtype(r: Retrieval) -> str:
    return "GRAPH_FACT" if r.has_graph_facts else "NARRATIVE" if r.passages else "UNKNOWN"


def _refuse(query: str, reason: str, r: Retrieval, prompt: Optional[str], config: Dict[str, Any],
            validation: Optional[Dict[str, bool]] = None, trace: Optional[Dict[str, Any]] = None) -> QueryResult:
    return QueryResult(
        query=query, query_type=_qtype(r), answer=REFUSAL, grounded=False, grounding_status="refused",
        confidence=0.0, citations=[], retrieved_chunks=_chunk_dicts(r), validation=validation or {},
        refusal_reason=reason, priority_tier=_tier(r), vector_fallback_used=r.vector_fallback_used,
        entities=r.entities, conflicts=conflict_dicts(r.resolved),
        source_explanation=f"Refused: {reason}", trace=trace or r.trace,
        prompt=prompt if config.get("include_prompt") else None)


# ---- main entry points ---------------------------------------------------------------------
def answer_query(query: str, store: QuadStore, vstore, config: Dict[str, Any], llm, nlp=None,
                 db: Optional[DatabaseSource] = None, db_specs: Optional[List[Dict[str, Any]]] = None) -> QueryResult:
    """Answer one query against an already-built index."""
    t0 = time.perf_counter()
    r = retrieve(query, store, vstore, config, nlp, db, db_specs)
    trace = dict(r.trace)
    if r.is_empty:
        return _refuse(query, "No evidence retrieved from any priority tier.", r, None, config, trace=trace)

    prompt = fit_prompt(query, r, config, getattr(llm, "count_tokens", None))
    t1 = time.perf_counter()
    if hasattr(llm, "answer"):
        out: GenerationOutput = llm.answer(r, query)
    else:
        text = (llm.generate(prompt) or "").strip()
        out = GenerationOutput(text, *_attribute(text, r), "llm")
    trace["generator"] = getattr(llm, "name", type(llm).__name__)
    trace["generation_mode"] = out.mode
    trace["generation_ms"] = round((time.perf_counter() - t1) * 1000, 2)

    raw = out.text
    if not raw or any(m in raw.lower() for m in _REFUSAL_MARKERS):
        return _refuse(query, "Model reported insufficient evidence.", r, prompt, config, trace=trace)

    grounded, validation, reason = validate_graph_answer(raw, r, float(config["grounding_min_coverage"]))
    if not grounded:
        return _refuse(query, reason, r, prompt, config, validation, trace)

    used_passages = out.used_passages or ([] if out.used_facts else list(r.passages))
    citations = [_citation_for_fact(rf, i) for i, rf in enumerate(out.used_facts, 1)]
    citations += [_citation_for_passage(p, len(citations) + i) for i, p in enumerate(used_passages, 1)]
    used_keys = {(flat(rf.subject), rf.predicate) for rf in out.used_facts}
    conflicts = conflict_dicts(r.resolved)
    for c in conflicts:
        c["affects_answer"] = (flat(c["subject"]), c["predicate"]) in used_keys
    used_conflicts = [c for c in conflicts if c["affects_answer"]]
    answer_tiers = [rf.winner.context for rf in out.used_facts] + [p.get("tier", CTX_PDF) for p in used_passages]
    best = min(answer_tiers, key=lambda t: TIER_PRIORITY.get(t, 99)) if answer_tiers else _tier(r)
    trace["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return QueryResult(
        query=query, query_type=_qtype(r), answer=raw, grounded=True, grounding_status="grounded",
        confidence=float(config["priority_weights"].get(best, 0.0)),   # tier-based, not calibrated
        citations=citations, retrieved_chunks=_chunk_dicts(r), validation=validation,
        priority_tier=best, vector_fallback_used=r.vector_fallback_used, entities=r.entities,
        conflicts=conflicts, evidence=_evidence(out.used_facts, used_passages),
        source_explanation=explain_sources(out.used_facts, used_passages, used_conflicts),
        trace=trace, prompt=prompt if config.get("include_prompt") else None)


class GraphRAGPipeline:
    """Build (or load) the index once, then answer any number of questions."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, generator=None, embedding_fn=None, nlp=None):
        self.config = config or load_config()
        self.generator = generator if generator is not None else make_generator(self.config)
        self.embedding_fn = embedding_fn
        self._nlp = nlp
        self.index: Optional[GraphIndex] = None

    @classmethod
    def from_config(cls, config_path: Optional[str] = None, **overrides: Any) -> "GraphRAGPipeline":
        return cls(load_config(config_path, **overrides))

    def build(self, rebuild: bool = False) -> "GraphRAGPipeline":
        self.index = build_graph_index(self.config, rebuild, self.embedding_fn)
        if self._nlp is None and self.config.get("use_spacy", True):
            self._nlp = load_spacy(self.config.get("spacy_model", "en_core_web_sm"))
        return self

    def ask(self, query: str) -> QueryResult:
        if self.index is None:
            self.build()
        idx = self.index
        return answer_query(query, idx.store, idx.vstore, self.config, self.generator, self._nlp or None,
                            idx.db, idx.db_specs)

    def ask_many(self, queries: Iterable[str]) -> List[QueryResult]:
        return [self.ask(q) for q in queries]

    def stats(self) -> Dict[str, Any]:
        if self.index is None:
            self.build()
        s = dict(self.index.stats)
        s.update({"graph": self.index.store.stats(), "passages": self.index.vstore.count(),
                  "vector_backend": self.index.vstore.backend, "from_cache": self.index.from_cache,
                  "generator": getattr(self.generator, "name", type(self.generator).__name__)})
        return s


def run_pipeline(query: str, config: Dict[str, Any], rebuild: bool = False) -> QueryResult:
    """Convenience wrapper mirroring rag_pipeline.run_pipeline (builds/loads the index every call)."""
    return GraphRAGPipeline(config).build(rebuild).ask(query)


__all__ = ["GraphRAGPipeline", "answer_query", "run_pipeline", "ExtractiveGenerator"]
