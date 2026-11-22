import hashlib
import json
from pathlib import Path
from typing import Dict

def compute_document_fingerprint(pdf_path: str) -> str:
    """SHA-256 of full PDF binary."""
    with open(pdf_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def get_index_metadata(pdf_path: str, config) -> Dict:
    return {
        "document_sha256": compute_document_fingerprint(pdf_path),
        "embedding_model": config.embedding_model,
        "embedding_dimension": 384,  # fixed for MiniLM
        "parser_version": "v2",
        "chunking_version": "v3",
        "schema_version": "v1",
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap
    }