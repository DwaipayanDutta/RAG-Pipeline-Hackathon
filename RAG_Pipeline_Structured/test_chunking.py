"""Chunking and ingestion helpers of the structured package (no model downloads)."""
from RAG_Pipeline_Structured.chunking import chunk_document
from RAG_Pipeline_Structured.config import RAGConfig
from RAG_Pipeline_Structured.ingestion import normalize_text
from RAG_Pipeline_Structured.models import Chunk


def test_normalize_text_removes_cid_and_whitespace():
    assert normalize_text("Hello (cid:123) world") == "Hello world"
    assert normalize_text("  Multiple   spaces  ") == "Multiple spaces"


def test_chunking_creates_text_and_table_row_chunks():
    pages = [{"page": 1, "raw_text": "First paragraph.\n\nSecond paragraph.",
              "tables": [[["Metric", "FY2025-26"], ["Revenue", "51,084"]]]}]
    chunks = chunk_document(pages, "doc1", RAGConfig(device="cpu"))
    assert all(isinstance(c, Chunk) for c in chunks)
    table = [c for c in chunks if c.source_type == "table"]
    text = [c for c in chunks if c.source_type == "text"]
    assert len(table) == 1 and table[0].text == "Metric: Revenue | FY2025-26: 51,084" and table[0].table_id == "tbl_1_0"
    assert [c.text for c in text] == ["First paragraph.", "Second paragraph."]
    assert len({c.chunk_id for c in chunks}) == len(chunks) and all(c.page_start == 1 for c in chunks)


def test_long_paragraph_uses_word_window_with_overlap():
    cfg = RAGConfig(chunk_size=50, chunk_overlap=10, device="cpu")
    words = [f"w{i}" for i in range(120)]
    chunks = chunk_document([{"page": 2, "raw_text": " ".join(words), "tables": []}], "d", cfg)
    assert [len(c.text.split()) for c in chunks] == [50, 50, 40]
    assert chunks[1].text.split()[:10] == chunks[0].text.split()[-10:]
