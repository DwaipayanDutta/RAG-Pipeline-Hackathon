#!/usr/bin/env python3
"""
Graph-RAG pipeline - CLI entry point and backward-compatible module facade.

    python RAG_Pipeline_Graph/rag_pipeline_graph.py --query "Who is on-call for payments-service?" --format text

The implementation lives in the `graph_rag` package next to this file; this module re-exports
the names the original single-file version provided, so `import rag_pipeline_graph as g` keeps
working.
"""
import sys
from dataclasses import asdict  # noqa: F401  (re-exported for callers that used g.asdict)
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from graph_rag.cli import main  # noqa: E402
from graph_rag.config import DEFAULT_CONFIG, load_config  # noqa: E402,F401
from graph_rag.demo_data import generate_sample_pdfs, seed_demo_data  # noqa: E402,F401
from graph_rag.fusion import find_conflicts, resolve_facts  # noqa: E402,F401
from graph_rag.generation import ExtractiveGenerator, FlanT5Generator, get_llm, make_generator  # noqa: E402,F401
from graph_rag.graph_store import QuadStore  # noqa: E402,F401
from graph_rag.index import GraphIndexManager, build_graph_index, build_index  # noqa: E402,F401
from graph_rag.models import (CTX_CMDB, CTX_PDF, CTX_WIKI, REFUSAL, Chunk, Quad, QueryResult,  # noqa: E402,F401
                              ResolvedFact, Retrieval)
from graph_rag.pipeline import GraphRAGPipeline, answer_query, run_pipeline  # noqa: E402,F401
from graph_rag.prompting import PRIORITY_RULES, build_system_prompt, fit_prompt  # noqa: E402,F401
from graph_rag.retrieval import detect_predicates, extract_query_entities, load_spacy, retrieve  # noqa: E402,F401
from graph_rag.sources.database import ingest_database  # noqa: E402,F401
from graph_rag.sources.pdf import chunk_text, extract_pdf_pages  # noqa: E402,F401
from graph_rag.sources.wiki import extract_wiki_facts, extract_wiki_facts_llm, ingest_wiki_dir  # noqa: E402,F401
from graph_rag.text import STOP, compute_sha256, normalize_text  # noqa: E402,F401
from graph_rag.text import flat as _flat, slug as _slug  # noqa: E402,F401
from graph_rag.validation import validate_graph_answer  # noqa: E402,F401
from graph_rag.vector_store import ChromaVectorStore, InMemoryVectorStore  # noqa: E402,F401

if __name__ == "__main__":
    sys.exit(main())
