"""Post-generation grounding checks (adapted from rag_pipeline.validate_answer)."""
from __future__ import annotations

import re
from typing import Dict, Tuple

from .generation import PREDICATE_LABELS, TEMPLATE_VOCAB
from .models import Retrieval
from .retrieval import relevant_facts
from .text import STOP, flat

# Capitalised multi-word names, hyphenated ids, and standalone numbers (not digits inside "SEV1").
_NAME_RE = re.compile(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+|\b\d[\d,.]*\d\b|\b\d\b")


def _claims(answer: str):
    """Names/ids/numbers asserted by the answer; leading stop words ("The Slack ...") are dropped."""
    for c in _NAME_RE.findall(answer):
        words = c.split()
        while len(words) > 1 and words[0].lower() in STOP:
            words.pop(0)
        c = " ".join(words)
        if len(words) == 1 and c.isalpha():          # a single ordinary capitalised word is not a claim
            continue
        yield c


def evidence_text(r: Retrieval) -> str:
    facts = r.p1 + r.p2 + r.related + r.pdf_facts
    parts = [f"{q.subject} {q.predicate} {q.object}" for q in facts]
    parts += [c["text"] for c in r.passages]
    parts += [PREDICATE_LABELS.get(q.predicate, "") for q in facts]
    return " ".join(parts)


def validate_graph_answer(answer: str, r: Retrieval,
                          min_coverage: float = 0.8) -> Tuple[bool, Dict[str, bool], str]:
    """
    Returns (grounded, validation, reason), the same shape as rag_pipeline.validate_answer.

      no_priority_conflict  the answer must not assert a value that a higher-priority source overrides
      entity_grounding      every name / hyphenated id / number in the answer appears in the evidence
      token_coverage        share of the answer's content words found in the evidence >= min_coverage
    """
    validation = {"no_priority_conflict": True, "entity_grounding": False, "token_coverage": False}
    a = f" {flat(answer)} "

    for winner, stale in r.conflicts:
        if f" {flat(stale.object)} " in a and f" {flat(winner.object)} " not in a:
            validation["no_priority_conflict"] = False
            return False, validation, (f"Contradicts Priority {winner.priority} ({winner.context}): "
                                       f"{winner.subject} {winner.predicate} is {winner.object}, "
                                       f"not {stale.object} ({stale.context})")

    # The answer must state a value for the attribute that was asked (when the graph has one):
    # "Web Commerce" is grounded text but not an answer to "which version is deployed?".
    asked = [rf for rf in relevant_facts(r) if rf.predicate in r.predicates] if r.predicates else []
    if asked:
        ents = {flat(e) for e in r.entities}
        answers = {flat(rf.value) for rf in asked} | {flat(rf.subject) for rf in asked if flat(rf.subject) not in ents}
        if not any(f" {v} " in a for v in answers if v):
            validation["answers_question"] = False
            return False, validation, (f"Answer states no retrieved value for the asked attribute "
                                       f"({', '.join(r.predicates)})")
        validation["answers_question"] = True

    eflat = f" {flat(evidence_text(r))} "
    etokens = set(eflat.split()) | TEMPLATE_VOCAB

    missing = [c for c in _claims(answer) if flat(c) and f" {flat(c)} " not in eflat]
    if missing:
        return False, validation, f"Entity not found in retrieved evidence: {missing[0]}"
    validation["entity_grounding"] = True

    content = [t for t in a.split() if t not in STOP]
    coverage = (sum(t in etokens for t in content) / len(content)) if content else 0.0
    if coverage < min_coverage:
        return False, validation, f"Answer not grounded in evidence (coverage {coverage:.2f})"
    validation["token_coverage"] = True
    return True, validation, ""
