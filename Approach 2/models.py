from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    page_start: int
    page_end: int
    section: Optional[str]
    subsection: Optional[str]
    source_type: str  # "text" or "table"
    table_id: Optional[str]
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Table:
    table_id: str
    page: int
    title: Optional[str]
    section: Optional[str]
    headers: List[str]
    rows: List[List[str]]
    units: Optional[str]
    fiscal_years: List[str]
    raw_text: str

@dataclass
class RetrievalCandidate:
    chunk: Chunk
    dense_score: Optional[float] = None
    lexical_score: Optional[float] = None
    rrf_score: Optional[float] = None
    reranker_score: Optional[float] = None
    final_rank: Optional[int] = None

@dataclass
class ValidationResult:
    status: str  # grounded, partially_grounded, unsupported, refused, validation_failed
    confidence: float
    claims: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class Answer:
    text: str
    citations: List[Dict[str, Any]]
    validation: ValidationResult
    grounded: bool