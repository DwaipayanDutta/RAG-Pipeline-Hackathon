from dataclasses import dataclass, field
from typing import Optional

@dataclass
class RAGConfig:
    # Paths
    pdf_path: str = "data/Titan AR 2026_0.pdf"
    queries_path: str = "queries.json"
    output_path: str = "results.json"
    index_dir: str = "index"

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

    # Device
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    def validate(self):
        # Add validation logic
        pass