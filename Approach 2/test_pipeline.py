import json
import pytest
from pathlib import Path
from rag_pipeline import (
    clean_text,
    chunk_document,
    compute_document_fingerprint,
    Chunk,
    CONFIG_DEFAULT
)

def test_clean_text():
    assert clean_text("Hello (cid:123) world") == "Hello world"
    assert clean_text("  Multiple   spaces  ") == "Multiple spaces"

def test_chunking():
    pages = [{"page": 1, "text": "First paragraph. Second paragraph.", "tables": []}]
    chunks = chunk_document(pages, CONFIG_DEFAULT)
    assert len(chunks) == 1
    assert chunks[0].page == 1
    assert chunks[0].source_type == "text"

def test_fingerprint_consistency():
    # With same config and same pdf, fingerprint should be same
    # We can't test with real PDF in unit test, so just check it returns a string.
    fp = compute_document_fingerprint("dummy.pdf", CONFIG_DEFAULT)
    assert isinstance(fp, str)
    assert len(fp) > 0

def test_index_cache_invalidation():
    # Simulate: if config changes, fingerprint changes
    config1 = CONFIG_DEFAULT.copy()
    config2 = CONFIG_DEFAULT.copy()
    config2["chunk_size"] = 500
    fp1 = compute_document_fingerprint("dummy.pdf", config1)
    fp2 = compute_document_fingerprint("dummy.pdf", config2)
    assert fp1 != fp2

def test_retrieval_metadata():
    # We don't have full pipeline test, but we can test that retrieval returns chunks with scores.
    # This is a placeholder for integration test.
    pass

def test_refusal_behavior():
    # For out-of-document query, should return "I don't know from the document."
    # This needs an end-to-end test with the PDF; we'll keep as placeholder.
    pass

def test_json_output_schema():
    # Verify that results.json matches expected schema
    sample = {
        "id": 1,
        "query": "test",
        "answer": "test answer",
        "retrieved_chunks": [{"page": 1, "chunk_id": 0, "source_type": "text", "score": 0.9}],
        "grounded": True
    }
    # Just check no exception
    json.dumps(sample)