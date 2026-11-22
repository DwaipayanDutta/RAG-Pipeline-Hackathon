"""Fusion and fingerprinting of the structured package (no model downloads)."""
import importlib.util

import pytest

from RAG_Pipeline_Structured.config import RAGConfig
from RAG_Pipeline_Structured.fingerprint import compute_document_fingerprint, get_index_metadata
from RAG_Pipeline_Structured.models import Chunk

needs_faiss = pytest.mark.skipif(importlib.util.find_spec("faiss") is None, reason="faiss not installed")


def _chunk(cid):
    return Chunk(cid, "doc", 1, 1, None, None, "text", None, f"text {cid}")


def test_fingerprint_and_metadata(tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")
    fp = compute_document_fingerprint(str(pdf))
    assert isinstance(fp, str) and len(fp) == 64
    meta = get_index_metadata(str(pdf), RAGConfig(device="cpu"))
    assert meta["document_sha256"] == fp and meta["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
    pdf.write_bytes(b"%PDF-1.4 changed")
    assert compute_document_fingerprint(str(pdf)) != fp


@needs_faiss
def test_rrf_fusion_orders_by_combined_rank():
    from RAG_Pipeline_Structured.retrieval import reciprocal_rank_fusion
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    fused = reciprocal_rank_fusion([(a, 0.9), (b, 0.8)], [(b, 0.7), (c, 0.6)], rrf_k=60)
    assert [ch.chunk_id for ch, _ in fused] == ["b", "a", "c"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_bm25_scores_keyword_matches():
    rank_bm25 = pytest.importorskip("rank_bm25")
    # "apple" must be rare (in < half the docs), otherwise BM25Okapi's IDF is negative.
    texts = ["apple orange", "banana kiwi", "grape melon", "banana grape", "kiwi apple"]
    scores = rank_bm25.BM25Okapi([d.split() for d in texts]).get_scores(["apple"])
    assert scores[0] > 0 and scores[4] > 0 and scores[1] == 0
