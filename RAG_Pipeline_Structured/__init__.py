"""
RAG_Pipeline_Structured - a modular (multi-file) layout of the Titan annual-report RAG pipeline.

Run from the repository root:

    python -m RAG_Pipeline_Structured query --question "What was Titan's revenue in FY2025-26?"
    python -m RAG_Pipeline_Structured evaluate

This package is deliberately import-light: heavy dependencies (torch, transformers,
sentence-transformers, faiss) are only imported by the modules that need them
(indexing, retrieval, reranking, prompting, generation, pipeline).
"""
from .config import RAGConfig

__all__ = ["RAGConfig"]
