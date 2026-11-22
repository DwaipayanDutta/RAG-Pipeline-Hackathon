"""
graph_rag - three-tier enterprise Graph RAG.

    Priority 1  live structured database (SQLite service registry)  -> SPOC quads, context "cmdb"
    Priority 2  wiki pages / runbooks (Markdown)                     -> quads "wiki" + passages
    Priority 3  PDF documents                                        -> chunks in a vector store + quads "pdf"

Quick start:

    from graph_rag import GraphRAGPipeline
    result = GraphRAGPipeline.from_config().ask("Who is on-call for payments-service?")
"""
from .config import DEFAULT_CONFIG, PACKAGE_ROOT, load_config
from .graph_store import QuadStore
from .models import (CTX_CMDB, CTX_PDF, CTX_WIKI, REFUSAL, Chunk, Quad, QueryResult, ResolvedFact,
                     Retrieval)
from .pipeline import GraphRAGPipeline, answer_query, run_pipeline

__all__ = ["GraphRAGPipeline", "answer_query", "run_pipeline", "load_config", "DEFAULT_CONFIG",
           "PACKAGE_ROOT", "QuadStore", "Quad", "Chunk", "QueryResult", "ResolvedFact", "Retrieval",
           "REFUSAL", "CTX_CMDB", "CTX_WIKI", "CTX_PDF"]
__version__ = "2.0.0"
