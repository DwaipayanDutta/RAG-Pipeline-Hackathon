import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

#: Repository root (parent of this package) - default paths are resolved against it.
REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@dataclass
class RAGConfig:
    # Paths
    pdf_path: str = str(REPO_ROOT / "Data and Question" / "Titan AR 2026_0.pdf")
    queries_path: str = str(REPO_ROOT / "Data and Question" / "queries.json")
    output_path: str = "results_structured.json"
    index_dir: str = str(REPO_ROOT / "RAG_Pipeline_Structured" / ".index")

    # Models
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    generation_model: str = "google/flan-t5-base"

    # Chunking
    chunk_size: int = 400          # words
    chunk_overlap: int = 80

    # Retrieval
    dense_top_k: int = 20
    lexical_top_k: int = 20
    rrf_k: int = 60
    reranker_top_k: int = 10
    final_top_k: int = 5
    use_lexical: bool = True
    use_reranker: bool = True

    # Generation
    context_token_budget: int = 512
    max_output_tokens: int = 150
    temperature: float = 0.0
    seed: int = 42

    # Validation
    enable_numeric_validation: bool = True
    enable_grounding_validation: bool = True

    # Logging
    log_level: str = "INFO"

    # Device ("" = auto-detect on first use; torch is not imported at class-definition time)
    device: str = ""

    def __post_init__(self):
        if not self.device:
            self.device = _default_device()

    def validate(self) -> "RAGConfig":
        """Raise ValueError for inconsistent settings."""
        if self.chunk_size <= 0 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_size must be > 0 and 0 <= chunk_overlap < chunk_size")
        if self.final_top_k <= 0 or self.dense_top_k < self.final_top_k:
            raise ValueError("final_top_k must be > 0 and <= dense_top_k")
        return self

    @classmethod
    def from_json(cls, path: Optional[str] = None) -> "RAGConfig":
        """Defaults, overridden by the keys of a JSON file (unknown keys are rejected)."""
        if not path:
            return cls().validate()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown RAGConfig keys in {path}: {sorted(unknown)}")
        return cls(**data).validate()
