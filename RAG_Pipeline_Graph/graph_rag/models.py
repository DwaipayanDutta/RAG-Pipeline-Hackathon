"""Core data structures: SPOC quads, document chunks, retrieval bundles and query results."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

REFUSAL = "I don't know from the document."

# Source tiers. The tier name is stored in Quad.context.
CTX_CMDB, CTX_WIKI, CTX_PDF = "cmdb", "wiki", "pdf"
TIER_PRIORITY = {CTX_CMDB: 1, CTX_WIKI: 2, CTX_PDF: 3}
TIER_LABEL = {
    CTX_CMDB: "Priority 1 (live database)",
    CTX_WIKI: "Priority 2 (wiki/runbook)",
    CTX_PDF: "Priority 3 (PDF document)",
}


@dataclass(frozen=True)
class Quad:
    """
    Subject-Predicate-Object-Context fact.

    `context` is the source tier (cmdb | wiki | pdf). The remaining context fields
    (timestamp, confidence, evidence) are provenance metadata: they do NOT take part in
    equality/hashing, so the same statement from the same source is stored once.
    """
    subject: str
    predicate: str
    object: str
    context: str
    source: str = ""
    timestamp: str = field(default="", compare=False)
    confidence: float = field(default=1.0, compare=False)
    evidence: str = field(default="", compare=False)

    @property
    def priority(self) -> int:
        """1 = most authoritative. Unknown tiers sort last."""
        return TIER_PRIORITY.get(self.context, 99)

    def as_text(self) -> str:
        return f"({self.context}) {self.subject} {self.predicate}: {self.object}."

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    """A text chunk (PDF page window or wiki section) with provenance metadata."""
    chunk_id: int
    page: int
    section: Optional[str]
    source_type: str                  # "text" (PDF) | "wiki"
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolvedFact:
    """The outcome of priority-based conflict resolution for one (subject, predicate[, value])."""
    subject: str
    predicate: str
    value: str
    winner: Quad
    agreeing: List[Quad] = field(default_factory=list)     # other sources stating the same value
    overridden: List[Quad] = field(default_factory=list)   # sources stating a different value
    reason: str = ""

    @property
    def has_conflict(self) -> bool:
        return bool(self.overridden)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject, "predicate": self.predicate, "value": self.value,
            "winner": {"tier": self.winner.context, "source": self.winner.source,
                       "timestamp": self.winner.timestamp},
            "agreeing": [{"tier": q.context, "source": q.source} for q in self.agreeing],
            "overridden": [{"tier": q.context, "source": q.source, "value": q.object,
                            "timestamp": q.timestamp} for q in self.overridden],
            "reason": self.reason,
        }


@dataclass
class Retrieval:
    """Everything retrieved for one query, grouped by source tier."""
    entities: List[str]
    p1: List[Quad] = field(default_factory=list)                      # live DB facts
    p2: List[Quad] = field(default_factory=list)                      # wiki facts
    related: List[Quad] = field(default_factory=list)                 # multi-hop graph facts
    p3: List[Dict[str, Any]] = field(default_factory=list)            # PDF passages
    conflicts: List[Tuple[Quad, Quad]] = field(default_factory=list)  # (winner, overridden)
    wiki_passages: List[Dict[str, Any]] = field(default_factory=list)  # wiki/runbook passages
    pdf_facts: List[Quad] = field(default_factory=list)               # facts extracted from PDFs
    resolved: List[ResolvedFact] = field(default_factory=list)
    predicates: List[str] = field(default_factory=list)               # what the question asks for
    unknown_entities: List[str] = field(default_factory=list)         # ids not found in the graph
    trace: Dict[str, Any] = field(default_factory=dict)

    @property
    def facts(self) -> List[Quad]:
        return self.p1 + self.p2 + self.pdf_facts

    @property
    def passages(self) -> List[Dict[str, Any]]:
        return self.wiki_passages + self.p3

    @property
    def has_graph_facts(self) -> bool:
        return bool(self.p1 or self.p2 or self.pdf_facts)

    @property
    def vector_fallback_used(self) -> bool:
        """True when no graph fact matched and the answer can only come from passages."""
        return not self.has_graph_facts and bool(self.passages)

    @property
    def is_empty(self) -> bool:
        return not (self.has_graph_facts or self.related or self.passages)


@dataclass
class QueryResult:
    """
    Output record. The first ten fields match the sibling pipelines' QueryResult
    (RAG_Pipeline_Simple / RAG_Pipeline_Jev); the rest are Graph-RAG provenance fields.
    """
    query: str
    query_type: str
    answer: str
    grounded: bool
    grounding_status: str
    confidence: float
    citations: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    validation: Dict[str, bool]
    refusal_reason: Optional[str] = None
    # graph-specific
    priority_tier: str = "none"           # highest tier that contributed evidence
    vector_fallback_used: bool = False
    entities: List[str] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    source_explanation: str = ""
    trace: Dict[str, Any] = field(default_factory=dict)
    prompt: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
