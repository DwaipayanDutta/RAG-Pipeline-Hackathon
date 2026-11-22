import pytest
from rag_pipeline.chunking import chunk_document
from rag_pipeline.models import Chunk

def test_chunking_creates_chunks():
    pages = [{"page": 1, "raw_text": "First paragraph. Second paragraph.", "tables": []}]
    chunks = chunk_document(pages, "doc1", config)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)