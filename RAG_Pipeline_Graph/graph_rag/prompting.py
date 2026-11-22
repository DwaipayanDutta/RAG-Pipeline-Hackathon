"""
Single fused prompt: priority rules + facts and passages grouped by tier + resolved conflicts +
the key (already priority-resolved) facts for the question, placed right before the question so
small seq2seq models attend to them.

Used verbatim by the FLAN-T5 backend; recorded for audit (`--show-prompt`) with any backend.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from .models import REFUSAL, TIER_PRIORITY, Quad, ResolvedFact, Retrieval
from .retrieval import relevant_facts

PRIORITY_RULES = f"""You are an enterprise assistant. Answer ONLY from the context below.

SOURCE PRIORITY RULES:
- Priority 1 (CMDB) is absolute and always current. If it conflicts with any other source, use Priority 1.
- Priority 2 (Wiki) is supplementary. Use it only if Priority 1 does not cover the question.
- Priority 3 (PDFs) is last-resort narrative context.
- Facts listed under RESOLVED CONFLICTS have already been settled: use the winning value only.

If the context does not contain enough information to answer, say exactly: "{REFUSAL}\""""


def _section(title: str, lines: Sequence[str]) -> str:
    return f"=== {title} ===\n" + ("\n".join(lines) if lines else "(none)")


def _passage_line(c: Dict[str, Any]) -> str:
    where = f"p.{c.get('page')}" if c.get("page") else (c.get("section") or "")
    return f"[{c.get('source', 'doc')} {where}] {c['text']}"


def _key_fact_line(rf: ResolvedFact) -> str:
    return (f"{rf.subject} {rf.predicate.replace('_', ' ')}: {rf.value} "
            f"(Priority {TIER_PRIORITY.get(rf.winner.context, '?')}, {rf.winner.source})")


def build_system_prompt(question: str, p1: Sequence[Quad], p2: Sequence[Quad],
                        p3: Sequence[Dict[str, Any]], related: Sequence[Quad] = (),
                        resolved: Sequence[ResolvedFact] = (), wiki_passages: Sequence[Dict[str, Any]] = (),
                        pdf_facts: Sequence[Quad] = (), key_facts: Sequence[ResolvedFact] = ()) -> str:
    def facts(qs: Sequence[Quad]) -> List[str]:
        return [q.as_text() for q in qs if q.predicate != "type"]

    parts = [PRIORITY_RULES,
             _section("PRIORITY 1: LIVE DATABASE FACTS", facts(p1)),
             _section("PRIORITY 2: WIKI FACTS (may be stale)", facts(p2))]
    if wiki_passages:
        parts.append(_section("PRIORITY 2: WIKI / RUNBOOK PASSAGES", [_passage_line(c) for c in wiki_passages]))
    if related:
        parts.append(_section("RELATED GRAPH FACTS", facts(related)))
    if pdf_facts:
        parts.append(_section("PRIORITY 3: FACTS EXTRACTED FROM PDFS", facts(pdf_facts)))
    parts.append(_section("PRIORITY 3: PDF NARRATIVE CONTEXT", [_passage_line(c) for c in p3]))
    conflicts = [f"{rf.subject} {rf.predicate}: use '{rf.value}' ({rf.winner.context}); ignore "
                 + ", ".join(f"'{q.object}' ({q.context})" for q in rf.overridden)
                 for rf in resolved if rf.overridden]
    if conflicts:
        parts.append(_section("RESOLVED CONFLICTS", conflicts))
    if key_facts:
        parts.append(_section("KEY FACTS FOR THIS QUESTION (conflicts already resolved)",
                              [_key_fact_line(rf) for rf in key_facts]))
    parts.append(f"=== QUESTION ===\n{question}\n\nGive a concise answer.\nAnswer:")
    return "\n\n".join(parts)


def fit_prompt(question: str, r: Retrieval, config: Dict[str, Any],
               count_tokens: Optional[Callable[[str], int]] = None) -> str:
    """
    Keep the prompt inside the model's input window (the question is last, so truncation would
    cut it first). Drops lowest-priority material first: related facts, PDF passages, PDF facts,
    wiki passages, wiki facts. Priority-1 facts, resolved conflicts and key facts are never
    dropped. Uses the generator's tokenizer when available, else a conservative chars/3 estimate.
    """
    budget = int(config["max_context_tokens"]) + 200
    count = count_tokens or (lambda s: len(s) // 3)
    lists: List[list] = [list(r.p2), list(r.wiki_passages), list(r.pdf_facts), list(r.p3), list(r.related)]
    p2, wiki, pdf_facts, p3, rel = lists
    key = relevant_facts(r)[:8]

    def build() -> str:
        return build_system_prompt(question, r.p1, p2, p3, rel, r.resolved, wiki, pdf_facts, key)

    prompt = build()
    while count(prompt) > budget and any(lists):
        next(lst for lst in reversed(lists) if lst).pop()
        prompt = build()
    return prompt
